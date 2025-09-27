import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from torch.optim import rmsprop  # used for model hub
import torch.nn.functional as F

from streamvggt.models.aggregator import Aggregator
from streamvggt.heads.camera_head import CameraHead
from streamvggt.heads.dpt_head import DPTHead
from streamvggt.heads.track_head import TrackHead
from transformers.file_utils import ModelOutput
from typing import Optional, Tuple, List, Any
from dataclasses import dataclass

@dataclass
class StreamVGGTOutput(ModelOutput):
    ress: Optional[List[dict]] = None
    views: Optional[torch.Tensor] = None

class StreamVGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size)
    


    def forward(
        self,
        views,
        query_points: torch.Tensor = None,
        history_info: Optional[dict] = None,
        past_key_values=None,
        use_cache=False,
        past_frame_idx=0
    ):
        images = torch.stack(
            [view["img"] for view in views], dim=0
        ).permute(1, 0, 2, 3, 4)    # B S C H W

        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        if history_info is None:
            history_info = {"token": None}

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

            if self.track_head is not None and query_points is not None:
                track_list, vis, conf = self.track_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
                )
                predictions["track"] = track_list[-1]  # track of the last iteration
                predictions["vis"] = vis
                predictions["conf"] = conf
            predictions["images"] = images

            B, S = images.shape[:2]
            ress = []
            for s in range(S):
                res = {
                    'pts3d_in_other_view': predictions['world_points'][:, s],  # [B, H, W, 3]
                    'conf': predictions['world_points_conf'][:, s],  # [B, H, W]

                    'depth': predictions['depth'][:, s],  # [B, H, W, 1]
                    'depth_conf': predictions['depth_conf'][:, s],  # [B, H, W]
                    'camera_pose': predictions['pose_enc'][:, s, :],  # [B, 9]

                    **({'valid_mask': views[s]["valid_mask"]}
                    if 'valid_mask' in views[s] else {}),  # [B, H, W]

                    **({'track': predictions['track'][:, s],  # [B, N, 2]
                        'vis': predictions['vis'][:, s],  # [B, N]
                        'track_conf': predictions['conf'][:, s]}
                    if 'track' in predictions else {})
                }
                ress.append(res)
            return StreamVGGTOutput(ress=ress, views=views)  # [S] [B, C, H, W]
        
    def inference(self, frames, query_points: torch.Tensor = None, past_key_values=None, eviction=True, P=0.5, temp=0.5, merge_tokens=False, merge_threshold=0.7):        
        past_key_values = [None] * self.aggregator.depth
        past_key_values_camera = [None] * self.camera_head.trunk_depth
        
        all_ress = []
        processed_frames = []

        for i, frame in enumerate(frames):
            print('proccesing frame: ', i+1)
            images = frame["img"].unsqueeze(0) 
            aggregator_output = self.aggregator(
                images, 
                past_key_values=past_key_values,
                use_cache=True, 
                past_frame_idx=i
            )
            
            if isinstance(aggregator_output, tuple) and len(aggregator_output) == 4:
                aggregated_tokens, patch_start_idx, past_key_values, attn_maps_global_layers = aggregator_output
                # attn_maps_global_layers is a list with len=#layers. for each layer we have attention map averaged on heads between current frame tokens and
                # tokens from previous frames + current frame
                if eviction:
                    print('eviction is ON with P: ', P)
                    if i==0:
                        # first frame att = inf to keep all tokens from first frame
                        cum_attn_maps = [torch.full((A.size(0),), float('inf'), device=A.device, dtype=torch.float16)
                                for A in attn_maps_global_layers]
                        # initialize exposure counters (how many frames each key column has been exposed to)
                        exposure_counts_layers = [
                            torch.ones(map.size(0), device=map.device, dtype=torch.int64)
                            for map in attn_maps_global_layers
                        ]
                        frame_token_num = attn_maps_global_layers[0].shape[0]
                        del attn_maps_global_layers

                    elif i>0 and i < len(frames) - 1:
                        tk_rm_num_per_layer = self.compute_layer_alphas(attn_maps_global_layers, frame_token_num, len(frames), S=i, L=self.aggregator.depth, temp=temp, P=P)
                        
                        # Optimize attention map updates to reduce memory usage
                        for j, attn_map in enumerate(attn_maps_global_layers):
                            # Get current cumulative map
                            curr_cum_map = cum_attn_maps[j]
                            key_len = attn_map.size(0)
                            
                            # Create new tensor for updated cumulative map
                            new_cum_map = torch.empty(key_len, device=attn_map.device, dtype=torch.float16)
                            
                            # Scale current attention map
                            new_cum_map.copy_(attn_map * float(key_len))
                            
                            # Add previous cumulative values in-place
                            old_size = min(curr_cum_map.size(0), key_len)
                            if old_size > 0:
                                new_cum_map[:old_size].add_(curr_cum_map[:old_size])
                            
                            # Update exposure counts efficiently
                            prev_exp = exposure_counts_layers[j]
                            new_exp = torch.ones(key_len, device=attn_map.device, dtype=torch.int64)
                            old_exp_size = min(prev_exp.size(0), key_len)
                            if old_exp_size > 0:
                                new_exp[:old_exp_size].copy_(prev_exp[:old_exp_size]).add_(1)
                            
                            # Set special tokens attention to inf to protect from eviction
                            new_cum_map[-frame_token_num:-frame_token_num+patch_start_idx] = float('inf')
                            
                            # Replace old tensors
                            cum_attn_maps[j] = new_cum_map
                            exposure_counts_layers[j] = new_exp
                            
                            # Explicit cleanup
                            del curr_cum_map, prev_exp

                        del attn_maps_global_layers
                        
                        if any(x > 0 for x in tk_rm_num_per_layer):
                            kv_remove_indices = self.get_remove_indices(cum_attn_maps,frame_token_num, i, tk_rm_num_per_layer, exposure_counts_layers)
                            past_key_values, keep_idx_list, cum_attn_maps, exposure_counts_layers = self.compact_kv_cache_per_layer(
                                past_key_values,
                                kv_remove_indices,
                                cum_attn_maps,
                                exposure_counts_layers,
                                merge_tokens=merge_tokens,
                                merge_threshold=merge_threshold
                            )
                            # Explicit cleanup of removal indices
                            del kv_remove_indices, keep_idx_list
                            
                            # Force memory cleanup after eviction
                            torch.cuda.empty_cache()

                    else:
                        del attn_maps_global_layers
                        del past_key_values
            else:
                aggregated_tokens, patch_start_idx = aggregator_output
            
            with torch.cuda.amp.autocast(enabled=False):
                if self.camera_head is not None:
                    pose_enc, past_key_values_camera = self.camera_head(aggregated_tokens, past_key_values_camera=past_key_values_camera, use_cache=True)
                    pose_enc = pose_enc[-1]
                    camera_pose = pose_enc[:, 0, :]

                if self.depth_head is not None:
                    depth, depth_conf = self.depth_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx
                    )
                    depth = depth[:, 0] 
                    depth_conf = depth_conf[:, 0]
                
                if self.point_head is not None:
                    pts3d, pts3d_conf = self.point_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx
                    )
                    pts3d = pts3d[:, 0] 
                    pts3d_conf = pts3d_conf[:, 0]

                if self.track_head is not None and query_points is not None:
                    track_list, vis, conf = self.track_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx, query_points=query_points
                )
                    track = track_list[-1][:, 0]  
                    query_points = track
                    vis = vis[:, 0]
                    track_conf = conf[:, 0]

            all_ress.append({
                'pts3d_in_other_view': pts3d,
                'conf': pts3d_conf,
                'depth': depth,
                'depth_conf': depth_conf,
                'camera_pose': camera_pose,
                **({'valid_mask': frame["valid_mask"]}
                    if 'valid_mask' in frame else {}),  

                **({'track': track, 
                    'vis': vis,  
                    'track_conf': track_conf}
                if query_points is not None else {})
            })
            processed_frames.append(frame)
        
        output = StreamVGGTOutput(ress=all_ress, views=processed_frames)
        return output

    def get_remove_indices(self, attn_maps_global_layers,token_per_frame_num, S, tk_rm_num_per_layer, exposure_counts_layers, rm_percentage=5, rand_rm= False):
        '''returns a list where item i contains a tensor of indices to evict in layer i'''

        rm_indices_per_layer=[]
        device = attn_maps_global_layers[0].device
        num_evicted = 0
        for layer_idx, (attn_map, k) in enumerate(zip(attn_maps_global_layers, tk_rm_num_per_layer)):
            all_token_num = attn_map.shape[0]
            #rm percentage is per layer
            rm_num = k - (token_per_frame_num*(S+1) - all_token_num)
            if rm_num>0:
                num_evicted = num_evicted + rm_num
                if rand_rm:
                    #select random indices
                    flat_indices = torch.randperm(all_token_num - 5, device=device)[:rm_num] + 5

                else:
                    # exposure normalization: divide by how many times each key column has been exposed so far
                    exp = exposure_counts_layers[layer_idx]
                    # More memory-efficient scoring without cloning
                    scores = attn_map / (exp + 1e-6)
                    # Using topk directly without intermediate tensors
                    _, flat_indices = torch.topk(scores, rm_num, largest=False)
                    del scores
            else:
                flat_indices = None
            rm_indices_per_layer.append(flat_indices)
        print(f'num tokens to evict: {num_evicted}')
        return rm_indices_per_layer


    @torch.no_grad()
    def compact_kv_cache_per_layer(self,
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        per_layer_remove: List[torch.Tensor],
        cum_attn_maps,
        exposure_counts_layers,
        merge_tokens: bool = False,
        merge_threshold: float = 0.7,
    ):
        """
        Evict token columns from a KV cache *layer by layer*.

        Args
        ----
        kv_cache : list of length L
            Each item is a tuple (K, V, pos)
            - K, V: [B, H, T, D]  (remove along dim=2)
            - pos : [B, T, 2]     (remove along dim=1)
        per_layer_remove : list of length L
            Each item i is a 1D LongTensor of column indices to remove for layer i.
            (Same indices applied to K, V, and pos of that layer.)

        Returns
        -------
        new_kv_cache : list of length L
            Layer-wise compacted (K, V, pos).
        keep_idx_list : list of length L
            For each layer i, a 1D LongTensor mapping new -> old column indices.
        """
        # assert len(kv_cache) == len(per_layer_remove), "one removal tensor per layer"

        # new_kv_cache = []
        keep_idx_list = []
        new_cum_maps = []
        new_exposure_counts_layers = []
        
        for layer_idx, ((K, V, pos), rem, attn_map) in enumerate(zip(kv_cache, per_layer_remove, cum_attn_maps)):
            # Shapes & device
            B, H, T, D = K.shape
            device = K.device

            # --- build keep indices (complement of rem), preserving original order ---
            if rem is None or rem.numel() == 0:
                keep_idx = torch.arange(T, device=device)
            else:
                rem = rem.to(device).long()
                if rem.numel() == 0:
                    keep_idx = torch.arange(T, device=device)
                elif rem.numel() == T:
                    # everything removed -> keep nothing
                    keep_idx = torch.empty(0, dtype=torch.long, device=device)
                else:
                    keep_mask = torch.ones(T, dtype=torch.bool, device=device)
                    keep_mask[rem] = False
                    keep_idx = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)  # sorted ascending

            # --- slice tensors once ---
            K_new = K.index_select(2, keep_idx)
            del K
            V_new = V.index_select(2, keep_idx)
            del V
            pos_new = pos.index_select(1, keep_idx)
            del pos
            attn_map_new = attn_map.index_select(0, keep_idx)
            del attn_map
            # propagate exposure counts
            exp_vec = exposure_counts_layers[layer_idx]
            exp_new = exp_vec.index_select(0, keep_idx)
            
            # Simplified token merging to reduce memory usage
            if merge_tokens and (rem is not None) and (rem.numel() > 0) and (keep_idx.numel() > 0):
                pass

            kv_cache[layer_idx] = (K_new, V_new, pos_new)
            del K_new, V_new, pos_new
            # new_kv_cache.append((K_new, V_new, pos_new))
            keep_idx_list.append(keep_idx)
            new_cum_maps.append(attn_map_new)
            new_exposure_counts_layers.append(exp_new)
            
        return kv_cache, keep_idx_list, new_cum_maps, new_exposure_counts_layers

    def compute_layer_alphas(self, attn_maps, frame_token_num, len_frames, S, L, P=0.8, epsilon=1e-6, temp=1.0):
        """
        Computes inverse-variance softmax layer allocation weights α_l.
        
        Args:
            attn_maps (list of torch.Tensor): each of shape [num_heads, seq_len, seq_len],
                or with batch dim [batch, num_heads, seq_len, seq_len].
            epsilon (float): small value to avoid division by zero.
            temp (float): temperature scale for softmax (optional).
        
        Returns:
            alphas (Tensor): shape [num_layers], normalized weights summing to 1.
            variances (Tensor): shape [num_layers], raw variance values per layer.
        """
        variances = []
        for A in attn_maps:
            # Compute per-token cumulative attention, e.g., sum over queries (rows)
            # Compute variance across token_sums
            var = A.var(unbiased=False)  # scalar
            variances.append(var)
        
        variances = torch.stack(variances)  # shape: [num_layers]
        alphas = F.softmax(-variances / temp, dim=0)
        total_budget = len_frames* L * frame_token_num
        tk_rm_num_per_layer = (frame_token_num* (S+1)) - (total_budget * P * alphas)

        return (tk_rm_num_per_layer.int()).tolist()