#!/bin/bash

set -e

workdir='..'
model_name='streamvggt'
ckpt_name='checkpoints'
model_weights="/data/soroush/StreamVGGT/ckpt/${ckpt_name}.pth"
datasets=('kitti' 'sintel')
t_list=("1.5")
p_list=("0.9" "0.8" "0.7" "0.6" "0.5" "0.4" "0.3" "0.2" "0.1")

for data in "${datasets[@]}"; do
    for temp in "${t_list[@]}"; do
        for P in "${p_list[@]}"; do
            output_dir="${workdir}/eval_results/video_depth/${data}_${model_name}_P${P}_temp${temp}_optmem4"
            echo "$output_dir"
            CUDA_LAUNCH_BLOCKING=1 accelerate launch --num_processes 1  ../src/eval/video_depth/launch.py \
                --weights "$model_weights" \
                --output_dir "$output_dir" \
                --eval_dataset "$data" \
                --size 518 \
                --eviction \
                --P "$P" \
                --temp "$temp"
            python ../src/eval/video_depth/eval_depth.py \
            --output_dir "$output_dir" \
            --eval_dataset "$data" \
            --align "scale"

            rm -r ${output_dir}/*/
        done
    done
done
