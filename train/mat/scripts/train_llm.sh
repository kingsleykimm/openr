#!/bin/sh
#
#SBATCH -t 24:00:00
#SBATCH -A _
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:3
#SBATCH --constraint=a100_80gb
#SBATCH --ntasks=1
#SBATCH -o out.txt

/scratch/bjb3az/.conda/envs/open_reasoner/bin/python -u train_math.py --seed 10 \
                --dataset_name "prealgebra" \
                --dataset_path "../envs/math/data/math_500.jsonl" \
                --model_name_or_path "../../../models/Qwen2.5-1.5B-Instruct/" \
                --prm_type "LLEMMA" \
                --prm_model_name_or_path "../../../models/llemma_34b/" \
                --algorithm_name "GRPO" \
                --experiment_name "llemma_grpo_single_epoch" \
                --num_mini_batch 4 \
                --ppo_epoch 1



# above is Rivanna version
# below is one in the original repo

# CUDA_VISIBLE_DEVICES=0 python -u train_math.py --seed 10 \
#                         --dataset_name "prealgebra" \
#                         --dataset_path "../envs/math/data/math_500.jsonl" \
#                         --model_name_or_path "MODEL_PATH" \
#                         --prm_type "MS" \
#                         --prm_model_name_or_path "PRM_PATH" \
#                         --prm_checkpoint_path "CHECKPOINT_PATH" \
#                         --algorithm_name "APPO" \
#                         --experiment_name "ms_single" \
#                         --num_mini_batch 4 \
#                         --ppo_epoch 1

