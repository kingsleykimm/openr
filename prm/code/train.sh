#!/bin/bash
#
#SBATCH -t 24:00:00
#SBATCH -A cral
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb
#SBATCH --ntasks=1
#SBATCH -o dump.txt

/scratch/bjb3az/.conda/envs/open_reasoner/bin/python finetune_llama.py