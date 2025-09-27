#!/bin/bash

set -e
workdir='..'
model_name='StreamVGGT'
ckpt_name='checkpoints'
model_weights="/data/soroush/StreamVGGT/ckpt/${ckpt_name}.pth"
# P="0.5"
# temp="0.5"


# output_dir="${workdir}/eval_results/mv_recon/${model_name}_${ckpt_name}"
# echo "$output_dir"
# accelerate launch --num_processes 1 --main_process_port 29602 ./eval/mv_recon/launch.py \
#     --weights "$model_weights" \
#     --output_dir "$output_dir" \
#     --model_name "$model_name" \
    # --eviction \
    # --P "$P" \
    # --temp "$temp"

#!/bin/bash

# Define your lists
t_list=("1.5")
p_list=("0.5" "0.4" "0.3" "0.2" "0.1")
for temp in "${t_list[@]}"; do
    for P in "${p_list[@]}"; do
        echo "Running with t=${temp} and p=${P}"
        output_dir="${workdir}/eval_results/mv_recon/${model_name}_${ckpt_name}_P${P}_temp${temp}"
        echo "$output_dir"
        accelerate launch --num_processes 1 --main_process_port 29602 ./eval/mv_recon/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --model_name "$model_name" \
        --eviction \
        --P "$P" \
        --temp "$temp"
        
    done
done
