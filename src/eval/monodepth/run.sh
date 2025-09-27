#!/bin/bash
set -e

workdir='..'
model_name='StreamVGGT'
ckpt_name='checkpoints'
model_weights="/data/soroush/StreamVGGT/ckpt/${ckpt_name}.pth"
datasets=('sintel' 'bonn' 'kitti' 'nyu')
t_list=("1.0")
p_list=("0.5")

for data in "${datasets[@]}"; do
    for temp in "${t_list[@]}"; do
        for P in "${p_list[@]}"; do
            echo "Running with t=${temp} and p=${P}"
            output_dir="${workdir}/eval_results/monodepth/${data}_${model_name}_P${P}_temp${temp}"
            echo "$output_dir"
            CUDA_LAUNCH_BLOCKING=1 python ./eval/monodepth/launch.py \
                --weights "$model_weights" \
                --output_dir "$output_dir" \
                --eval_dataset "$data" \
                --eviction \
                --P "$P" \
                --temp "$temp"
        done
    done
done

for data in "${datasets[@]}"; do
    for temp in "${t_list[@]}"; do
        for P in "${p_list[@]}"; do
            output_dir="${workdir}/eval_results/monodepth/${data}_${model_name}_P${P}_temp${temp}"
            CUDA_LAUNCH_BLOCKING=1 python ./eval/monodepth/eval_metrics.py \
                --output_dir "$output_dir" \
                --eval_dataset "$data"
        done
    done
done

