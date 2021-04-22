#!/bin/bash

#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32g
#SBATCH -p gpu
#SBATCH -t 05-00:00:00
#SBATCH --qos gpu_access
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end
#SBATCH --mail-user=ylsung@email.unc.edu

source /nas/longleaf/home/ylsung/envs/3d/bin/activate

for angle in 0 5 15 30 45 90
    do
    for scale in 0 0.1 0.3 0.5
    do
        python run.py \
            --config experiments/test_celeba.yml \
            --model Unsup3D_Classifier \
            --rotated_angle $angle \
            --jitter_scale $scale
    done
done


# python run.py --config experiments/test_celeba.yml --model Unsup3D_Classifier --rotated_angle 90