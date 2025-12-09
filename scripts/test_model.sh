#!/bin/bash

#SBATCH --account=deep_learning
#SBATCH --output=logs/test_%j.out
#SBATCH --time=01:00
#SBATCH --job-name=test-model
#SBATCH --gpus 1080ti:1

. /etc/profile.d/modules.sh
module add cuda/12.6

source "$HOME/venvs/mtbreaker/bin/activate"
cd "/work/scratch/mbehanzin/DeepLearning/"
mkdir -p logs

# python -m src.scorers
# python test_scripts/test_model_inference.py
# "$HOME/ckpt/stage-x/"
python test_scripts/test_model_inference.py --checkpoint "$HOME/ckpt/stage-x/" --base-model "Qwen/Qwen3-0.6B-Base"