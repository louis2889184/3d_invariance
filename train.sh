#!/bin/bash

#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32g
#SBATCH -p volta-gpu
#SBATCH -t 05-00:00:00
#SBATCH --qos gpu_access
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end
#SBATCH --mail-user=ylsung@email.unc.edu

source /nas/longleaf/home/ylsung/envs/3d/bin/activate

python run.py --config experiments/train_celeba.yml --num_workers 4 --model Unsup3D_Classifier