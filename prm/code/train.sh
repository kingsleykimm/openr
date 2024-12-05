#!/bin/bash
#
#SBATCH -t 24:00:00
#SBATCH -A cywlab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --constraint=a100_80gb
#SBATCH --ntasks=1
#SBATCH 
#SBATCH -o dump.txt

module load cuda
module load apptainer pytorch

/scratch/bjb3az/.conda/envs/open_reasoner/bin/python finetune_llama.py