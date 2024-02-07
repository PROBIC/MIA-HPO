#!/bin/bash
#SBATCH --job-name=cifar10_full
#SBATCH --account=project_2003275
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=150000
#SBATCH --gres=gpu:v100:1
#SBATCH -o /scratch/project_2003275/gpradhan_temp/dp_hp_tuning/logs/%j.log

module purge
export PATH="/projappl/project_2003275/dp-fsl-root/dp-fsl-env/bin:$PATH"
export TORCH_HOME="/scratch/project_2003275/yuan_temp/torchhome"

cd /projappl/project_2003275/gpradhan/dp_hp_tuning/feature_space_cache
python3 -m map_to_feature_space --feature_extractor BiT-M-R50x1 --feature_dim_path /scratch/project_2003275/gpradhan_temp/feature_space_cache --dataset cifar10 --examples_per_class -1 --download_path_for_tensorflow_datasets /scratch/project_2003275/tobaben_temp/tensorflow_datasets