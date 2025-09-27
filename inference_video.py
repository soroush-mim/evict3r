import argparse
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
import time
from datetime import datetime
from functools import partial
from pathlib import Path

# Add src to path
sys.path.append("src/")

from streamvggt.models.streamvggt import StreamVGGT
from streamvggt.utils.load_fn import load_and_preprocess_images
from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
from streamvggt.utils.geometry import unproject_depth_map_to_point_map
from visual_util import predictions_to_glb

import torch
import torch.nn as nn


class AttentionMapExtractor:
    """
    A class to extract attention maps from a Vision Transformer.

    This class uses PyTorch hooks to capture the output of the attention
    mechanism from specified layers. It's designed to be used as a
    context manager.

    Args:
        model (nn.Module): The Vision Transformer model.
        target_module_name (str): The name of the module within each transformer
                                  block whose output is the attention map.
                                  e.g., 'attn_drop'.
    """
    def __init__(self, model: nn.Module, target_module_name: str, layer_indices: list[int], num_frames: int):
        self.model = model
        self.target_module_name = target_module_name
        # self.attention_maps = {}
        self._hooks = []
        self.curr_frame_index = 0
        self.num_forward_passes = 0
        self.layer_indices = layer_indices
        self.layer_num = len(layer_indices)
        self.num_frames = num_frames

    def _get_attention_map(self, layer_index: int, module: nn.Module, inp: tuple, out: torch.Tensor):
        """Hook function to capture the attention map."""
        # The attention map is the output of the target module.
        # We detach and clone it to move it to the CPU and avoid holding onto the computation graph.
        self.attention_maps[self.curr_frame_index][layer_index] = out.squeeze(0).mean(dim=0).detach().cpu()
        self.num_forward_passes += 1
        self.curr_frame_index = self.num_forward_passes // self.layer_num

    def __enter__(self):
        """Register hooks for the specified layers."""
        # self.attention_maps = {} # Clear previous maps
        self._hooks = []
        self.attention_maps =[{} for _ in range(self.num_frames)]

        # Find the transformer blocks (assuming they are in a Modu)leList named 'blocks')
        try:
            transformer_blocks = self.model.aggregator.global_blocks
            print(len(transformer_blocks))
        except AttributeError:
            raise ValueError("The model must have a 'blocks' attribute which is a ModuleList of transformer blocks.")

        for layer_idx in self.layer_indices:
            if layer_idx >= len(transformer_blocks):
                raise ValueError(f"Layer index {layer_idx} is out of bounds for model with {len(transformer_blocks)} layers.")

            block = transformer_blocks[layer_idx]
            
            # Find the target module within the block
            target_module_found = False
            for name, module in block.named_modules():
                if name == self.target_module_name:
                    # Register the hook using a partial function to pass the layer_index
                    hook = module.register_forward_hook(
                        partial(self._get_attention_map, layer_idx)
                    )
                    self._hooks.append(hook)
                    target_module_found = True
                    break
            
            if not target_module_found:
                raise ValueError(f"Target module '{self.target_module_name}' not found in block {layer_idx}.")
        
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def get_maps(self):
        """Returns the collected attention maps."""
        return self.attention_maps


def build_full_attention_matrices_averaged(list_of_attention_dicts: list[dict]) -> dict:
    """
    Builds full, square attention matrices by averaging over the attention heads.

    Args:
        list_of_attention_dicts: A list where each element is a dictionary
            of attention maps from a single frame's forward pass. Assumes
            each tensor has a shape of (1, H, N, ...).

    Returns:
        A dictionary where keys are layer indices and values are the full
        2D square attention tensors of shape (N*T, N*T), averaged over heads.
    """
    if not list_of_attention_dicts:
        return {}

    # --- 1. Determine parameters ---
    num_frames = len(list_of_attention_dicts)
    layer_indices = list(list_of_attention_dicts[-1].keys())
    first_map_4d = list(list_of_attention_dicts[0].values())[0]
    tokens_per_frame, _ = first_map_4d.shape
    total_tokens = tokens_per_frame * num_frames

    # --- 2. Initialize output dictionary with 2D zero tensors ---
    final_attention_maps = {}
    for layer_idx in layer_indices:
        final_attention_maps[layer_idx] = torch.zeros(
            (total_tokens, total_tokens),
            dtype=first_map_4d.dtype,
            device=first_map_4d.device
        )

    # --- 3. Fill the matrices with averaged maps ---
    for layer_idx in layer_indices:
        for frame_idx, frame_attention_dict in enumerate(list_of_attention_dicts):
            # Shape: (1, H, N, N * (frame_idx + 1))
            partial_map_4d = frame_attention_dict[layer_idx]
            
            # CRITICAL CHANGE: Squeeze batch dim and average over head dim
            # Shape becomes: (N, N * (frame_idx + 1))
            # averaged_map_2d = partial_map_4d.squeeze(0).mean(dim=0)

            # Define the 2D slice to place this partial map
            row_start = frame_idx * tokens_per_frame
            row_end = (frame_idx + 1) * tokens_per_frame
            col_end = (frame_idx + 1) * tokens_per_frame

            # Place the 2D averaged map into the 2D full matrix
            final_attention_maps[layer_idx][row_start:row_end, :col_end] = partial_map_4d

    return final_attention_maps


def save_all_attention_plots(
    full_attention_matrices: dict,
    tokens_per_frame: int=1374,
    output_folder: str = "attention_plots_scaled",
    apply_scale: bool = False
):
    """
    Plots, scales, and saves the head-averaged attention map for every layer.
    """
    os.makedirs(output_folder, exist_ok=True)
    print(f"Saving scaled plots to '{output_folder}/' directory...")

    for layer_idx, attn_map_tensor in full_attention_matrices.items():
        total_tokens = attn_map_tensor.shape[0]
        num_frames = total_tokens // tokens_per_frame
        
        # Create a copy to avoid modifying the original dictionary in-place
        scaled_map_tensor = attn_map_tensor.clone()

        if apply_scale:
            # --- SCALING BLOCK: Apply the scaling as requested ---
            for i in range(num_frames):
                row_start = i * tokens_per_frame
                row_end = (i + 1) * tokens_per_frame
                scale_factor = i + 1
                
                # Multiply the rows corresponding to frame 'i' by 'i+1'
                scaled_map_tensor[row_start:row_end, :] *= scale_factor
            # ----------------------------------------------------

        attn_map_to_plot = scaled_map_tensor.cpu().numpy()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(attn_map_to_plot, cmap='viridis', interpolation='nearest')

        # Draw frame boundaries
        for i in range(1, num_frames):
            boundary = i * tokens_per_frame - 0.5
            ax.axhline(y=boundary, color='white', linestyle='--', linewidth=1.5)
            ax.axvline(x=boundary, color='white', linestyle='--', linewidth=1.5)

        ax.set_title(f'Scaled Attention Map - Layer {layer_idx+1}', fontsize=16)
        ax.set_xlabel('Key Tokens (All Frames)', fontsize=12)
        ax.set_ylabel('Query Tokens (All Frames)', fontsize=12)
        
        tick_locations = [i * tokens_per_frame + tokens_per_frame / 2 for i in range(num_frames)]
        tick_labels = [f'Frame {i+1}' for i in range(num_frames)]
        ax.set_xticks(tick_locations)
        ax.set_xticklabels(tick_labels, rotation=45)
        ax.set_yticks(tick_locations)
        ax.set_yticklabels(tick_labels)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()

        save_path = os.path.join(output_folder, f"scaled_attention_layer_{layer_idx}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
    print("All scaled plots saved successfully. ✅")

def save_frame_level_plots(
    full_attention_matrices: dict,
    tokens_per_frame: int=1374,
    output_folder: str = "attention_plots_frame_level"
    ):
    """
    Creates and saves plots of frame-to-frame average attention.
    """
    os.makedirs(output_folder, exist_ok=True)
    print(f"\nSaving frame-level plots to '{output_folder}/'...")

    for layer_idx, full_map in full_attention_matrices.items():
        total_tokens = full_map.shape[0]
        num_frames = total_tokens // tokens_per_frame
        
        # Initialize the new (num_frames, num_frames) map
        frame_level_map = torch.zeros((num_frames, num_frames))

        # Calculate the average attention between each pair of frames
        for i in range(num_frames): # Query frame
            for j in range(num_frames): # Key frame
                row_start, row_end = i * tokens_per_frame, (i + 1) * tokens_per_frame
                col_start, col_end = j * tokens_per_frame, (j + 1) * tokens_per_frame
                
                # Extract the block corresponding to (frame_i -> frame_j) attention
                attention_block = full_map[row_start:row_end, col_start:col_end]
                
                # Calculate the mean and store it
                frame_level_map[i, j] = attention_block.mean()

        # Plotting logic
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(frame_level_map.cpu().numpy(), cmap='plasma', interpolation='nearest')
        
        ax.set_title(f'Frame-to-Frame Attention - Layer {layer_idx}\n(Averaged over Tokens & Heads)', fontsize=14)
        ax.set_xlabel('Key Frames', fontsize=12)
        ax.set_ylabel('Query Frames', fontsize=12)
        
        tick_labels = [f'Frame {i}' for i in range(num_frames)]
        ax.set_xticks(range(num_frames))
        ax.set_xticklabels(tick_labels, rotation=45)
        ax.set_yticks(range(num_frames))
        ax.set_yticklabels(tick_labels)

        # Add text labels for values in each cell
        for i in range(num_frames):
            for j in range(num_frames):
                ax.text(j, i, f'{frame_level_map[i, j]:.3f}', ha='center', va='center', color='white', fontsize=10)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        
        save_path = os.path.join(output_folder, f"frame_attention_layer_{layer_idx}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
    print("Frame-level plots saved successfully. ✅")

def save_layer_averaged_plot(
    full_attention_matrices: dict,
    tokens_per_frame: int=1374,
    output_folder: str = "attention_plots_layer_averaged"
):
    """
    Averages attention maps across all layers and saves the single resulting plot.
    """
    os.makedirs(output_folder, exist_ok=True)
    print(f"\nSaving layer-averaged plot to '{output_folder}/'...")

    if not full_attention_matrices:
        print("Dictionary is empty, nothing to average.")
        return

    # --- 1. Stack all layer maps and average them ---
    # Create a new dimension for layers, then compute the mean over it
    stacked_maps = torch.stack(list(full_attention_matrices.values()), dim=0)
    layer_averaged_map = stacked_maps.mean(dim=0)

    # --- 2. Plotting (similar to the original plotting function) ---
    attn_map_to_plot = layer_averaged_map.cpu().numpy()
    total_tokens = attn_map_to_plot.shape[0]
    num_frames = total_tokens // tokens_per_frame

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(attn_map_to_plot, cmap='viridis', interpolation='nearest')

    for i in range(1, num_frames):
        boundary = i * tokens_per_frame - 0.5
        ax.axhline(y=boundary, color='white', linestyle='--', linewidth=1.5)
        ax.axvline(x=boundary, color='white', linestyle='--', linewidth=1.5)
    
    layer_keys = list(full_attention_matrices.keys())
    ax.set_title(f'Layer-Averaged Attention Map (Layers {layer_keys})', fontsize=16)
    ax.set_xlabel('Key Tokens (All Frames)', fontsize=12)
    ax.set_ylabel('Query Tokens (All Frames)', fontsize=12)
    
    tick_locations = [i * tokens_per_frame + tokens_per_frame / 2 for i in range(num_frames)]
    tick_labels = [f'Frame {i}' for i in range(num_frames)]
    ax.set_xticks(tick_locations)
    ax.set_xticklabels(tick_labels, rotation=45)
    ax.set_yticks(tick_locations)
    ax.set_yticklabels(tick_labels)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    
    save_path = os.path.join(output_folder, "attention_averaged_over_all_layers.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print("Layer-averaged plot saved successfully. ✅")

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    """Load StreamVGGT model following the same logic as demo_gradio.py"""
    print("Initializing and loading StreamVGGT model...")
    
    local_ckpt_path = "/data/soroush/StreamVGGT/ckpt/checkpoints.pth"
    if os.path.exists(local_ckpt_path):
        print(f"Loading local checkpoint from {local_ckpt_path}")
        model = StreamVGGT()
        ckpt = torch.load(local_ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt, strict=True)
        model.eval()
        del ckpt
    else:
        print("Local checkpoint not found, downloading from Hugging Face...")
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id="lch01/StreamVGGT",
            filename="checkpoints.pth",
            revision="main",
            force_download=True
        )
        model = StreamVGGT()
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt, strict=True)
        model.eval() 
        del ckpt
    
    return model

def extract_frames_from_video(video_path, output_dir, fps_interval=1):
    """Extract frames from video following the same logic as demo_gradio.py"""
    os.makedirs(output_dir, exist_ok=True)
    
    vs = cv2.VideoCapture(video_path)
    fps = vs.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * fps_interval)  # Extract 1 frame per second by default
    
    count = 0
    video_frame_num = 0
    image_paths = []
    
    while True:
        gotit, frame = vs.read()
        if not gotit:
            break
        count += 1
        if count % frame_interval == 0:
            image_path = os.path.join(output_dir, f"{video_frame_num:06}.png")
            cv2.imwrite(image_path, frame)
            image_paths.append(image_path)
            video_frame_num += 1
    
    vs.release()
    return sorted(image_paths)

def run_model_on_images(image_paths, model, eviction=True, P=0.5, temp=0.5):
    """Run the model on images following the same logic as demo_gradio.py"""
    print(f"Processing {len(image_paths)} images")
    
    # Device check
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Check your environment.")
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Load and preprocess images using the same function as demo
    images = load_and_preprocess_images(image_paths).to(device)
    print(f"Preprocessed images shape: {images.shape}")
    
    # Create frames list following the same structure as demo
    frames = []
    for i in range(images.shape[0]):
        image = images[i].unsqueeze(0) 
        frame = {
            "img": image
        }
        frames.append(frame)
    

    # frames is a list of dicts, each dict has a key "img" with a value of shape (1, 3, H, W)
    # Run inference with same dtype handling as demo
    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            
            output = model.inference(frames, eviction=eviction, P=P, temp=temp)
    
    # Process outputs following the same logic as demo
    all_pts3d = []
    all_conf = []
    all_depth = []
    all_depth_conf = []
    all_camera_pose = []
    
    for res in output.ress:
        all_pts3d.append(res['pts3d_in_other_view'].squeeze(0))
        all_conf.append(res['conf'].squeeze(0))
        all_depth.append(res['depth'].squeeze(0))
        all_depth_conf.append(res['depth_conf'].squeeze(0))
        all_camera_pose.append(res['camera_pose'].squeeze(0))
    
    predictions = {}
    predictions["world_points"] = torch.stack(all_pts3d, dim=0)  # (S, H, W, 3)
    predictions["world_points_conf"] = torch.stack(all_conf, dim=0)  # (S, H, W)
    predictions["depth"] = torch.stack(all_depth, dim=0)  # (S, H, W, 1)
    predictions["depth_conf"] = torch.stack(all_depth_conf, dim=0)  # (S, H, W)
    predictions["pose_enc"] = torch.stack(all_camera_pose, dim=0)  # (S, 9)
    predictions["images"] = images  # (S, 3, H, W)
    
    print("World points shape:", predictions["world_points"].shape)
    print("World points confidence shape:", predictions["world_points_conf"].shape)
    print("Depth map shape:", predictions["depth"].shape)
    print("Depth confidence shape:", predictions["depth_conf"].shape)
    print("Pose encoding shape:", predictions["pose_enc"].shape)
    print(f"Images shape: {images.shape}")
    
    # Convert pose encoding to extrinsic and intrinsic matrices
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"].unsqueeze(0) if predictions["pose_enc"].ndim == 2 else predictions["pose_enc"], 
        images.shape[-2:]
    )
    predictions["extrinsic"] = extrinsic.squeeze(0)  # (S, 3, 4)
    predictions["intrinsic"] = intrinsic.squeeze(0) if intrinsic is not None else None  # (S, 3, 3) or None
    print("Extrinsic shape:", predictions["extrinsic"].shape)
    print("Intrinsic shape:", predictions["intrinsic"].shape)
    
    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy()
    
    # Generate world points from depth map
    print("Computing world points from depth map...")
    predictions["world_points_from_depth"] = predictions["world_points"]
    
    # Clean up
    torch.cuda.empty_cache()
    
    return predictions

def save_results(predictions, out_dir):
    """Save results as numpy files"""
    os.makedirs(out_dir, exist_ok=True)
    
    # Save predictions as npz file
    prediction_save_path = os.path.join(out_dir, "predictions.npz")
    np.savez(prediction_save_path, **predictions)
    print(f"Predictions saved to {prediction_save_path}")
    
    # Save individual frame results
    for i in range(predictions["world_points"].shape[0]):
        np.save(os.path.join(out_dir, f"frame_{i:04d}_pts3d.npy"), predictions["world_points"][i])
        np.save(os.path.join(out_dir, f"frame_{i:04d}_depth.npy"), predictions["depth"][i])

def create_3d_visualization(predictions, out_dir, conf_thres=3.0, show_cam=True, mask_black_bg=False, mask_white_bg=False, mask_sky=False, prediction_mode="Pointmap Regression"):
    """Create 3D GLB visualization from predictions"""
    print("Creating 3D visualization...")
    
    # Create GLB file using the same logic as demo_gradio.py
    glbfile = os.path.join(
        out_dir,
        f"glbscene_{conf_thres}_all_frames_maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}_sky{mask_sky}_pred{prediction_mode.replace(' ', '_')}.glb",
    )
    
    # Convert predictions to GLB using the same function as demo
    glbscene = predictions_to_glb(
        predictions,
        conf_thres=conf_thres,
        filter_by_frames="All",
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        show_cam=show_cam,
        mask_sky=mask_sky,
        target_dir=out_dir,
        prediction_mode=prediction_mode,
    )
    
    # Export GLB file
    glbscene.export(file_obj=glbfile)
    print(f"3D visualization saved to {glbfile}")
    
    return glbfile

import os
import cv2
import numpy as np
from pathlib import Path

def _tokenj_to_box(token_j: int, H: int, W: int, patch: int):
    """
    Map token_j (including specials) to pixel bbox, ignoring specials (handled upstream).
    Returns (x0, y0, x1, y1).
    """
    H_p, W_p = H // patch, W // patch  # expected 37x37 for 518/14
    p = token_j - 5  # strip 5 specials
    r0 = p // W_p
    c0 = p %  W_p
    x0, y0 = c0 * patch, r0 * patch
    x1, y1 = min(W, (c0 + 1) * patch), min(H, (r0 + 1) * patch)
    return x0, y0, x1, y1



def main():
    parser = argparse.ArgumentParser(description="Run StreamVGGT inference on a video and create 3D visualizations.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file.")
    parser.add_argument("--ckpt", type=str, default="/data/soroush/StreamVGGT/ckpt/checkpoints.pth", help="Path to StreamVGGT checkpoint (optional).")
    parser.add_argument("--out_dir", type=str, default="output_streamvggt", help="Directory to save results.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on.")
    parser.add_argument("--attn_layers", type=str, default=None, help="Comma-separated list of attention layer indices to plot.")
    parser.add_argument("--fps_interval", type=float, default=2.5, help="Extract 1 frame every N seconds (default: 1.0).")
    parser.add_argument("--conf_thres", type=float, default=3.0, help="Confidence threshold for 3D visualization (default: 3.0).")
    parser.add_argument("--show_cam", action="store_true", help="Show cameras in 3D visualization.")
    parser.add_argument("--mask_black_bg", action="store_true", help="Mask black background in 3D visualization.")
    parser.add_argument("--mask_white_bg", action="store_true", help="Mask white background in 3D visualization.")
    parser.add_argument("--mask_sky", action="store_true", help="Apply sky segmentation mask.")
    parser.add_argument("--prediction_mode", type=str, default="Pointmap Regression", help="Prediction mode for visualization.")
    parser.add_argument("--no_3d_viz", action="store_true", help="Skip 3D visualization creation.")
    parser.add_argument("--eviction", action="store_true", help="Use eviction.")
    parser.add_argument("--P", type=float, default=0.5, help="Eviction probability.")
    parser.add_argument("--temp", type=float, default=0.5, help="Temperature for softmax.")

    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Extract frames from video
    print("Extracting frames from video...")
    temp_dir = os.path.join(args.out_dir, "temp_frames")
    image_paths = extract_frames_from_video(args.video, temp_dir, args.fps_interval)
    print(f"Extracted {len(image_paths)} frames to {temp_dir}")
    
    # Load model
    model = load_model()

    
    
    # Parse requested attention layers
    requested_attn_layers = None
    if args.attn_layers:
        requested_attn_layers = [int(x) for x in args.attn_layers.split(",")]
        print(f"Will extract attention maps from layers: {requested_attn_layers}")
        extractor = AttentionMapExtractor(model, target_module_name="attn.attn_drop", layer_indices=requested_attn_layers, num_frames=len(image_paths))
        with extractor:

        
            # Perform the forward pass
            print("Running StreamVGGT inference...")
            predictions = run_model_on_images(image_paths, model, args.eviction, args.P, args.temp)

            # The attention maps are now stored in the extractor instance
            attention_maps = extractor.get_maps()
            attention_maps = build_full_attention_matrices_averaged(attention_maps)
            save_all_attention_plots(attention_maps, apply_scale=True)
            # save_frame_level_plots(full_attention_matrices=attention_maps)
            # Plot 2: A single plot averaged over all layers
            # save_layer_averaged_plot(full_attention_matrices=attention_maps)


    else:
        # Run model inference
        print("Running StreamVGGT inference...")
        predictions = run_model_on_images(image_paths, model, args.eviction, args.P, args.temp)




    
    # Save results
    print("Saving results...")
    save_results(predictions, args.out_dir)
    
    
    # Create 3D visualization if not skipped
    if not args.no_3d_viz:
        print("Creating 3D visualization...")
        glb_file = create_3d_visualization(
            predictions, 
            args.out_dir, 
            conf_thres=args.conf_thres,
            show_cam=args.show_cam,
            mask_black_bg=args.mask_black_bg,
            mask_white_bg=args.mask_white_bg,
            mask_sky=args.mask_sky,
            prediction_mode=args.prediction_mode
        )
        print(f"3D visualization created: {glb_file}")
    
    # Clean up temp directory
    # import shutil
    # shutil.rmtree(temp_dir)
    # print(f"Temporary frames removed from {temp_dir}")
    
    print(f"All results saved to {args.out_dir}")
    print("\nTo view the 3D visualization:")
    print("1. Download the .glb file from the output directory")
    print("2. Open it in a 3D viewer like:")
    print("   - Online: https://gltf-viewer.donmccurdy.com/")
    print("   - Blender (free)")
    print("   - Windows 3D Viewer")
    print("   - Or any GLB-compatible viewer")

if __name__ == "__main__":
    main() 