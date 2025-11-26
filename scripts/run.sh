#!/bin/bash

#SBATCH --account=deep_learning
#SBATCH --output=logs/mtbreaker_%j.out
#SBATCH --job-name=setup-RL
#SBATCH --gpus 1080ti:1


. /etc/profile.d/modules.sh
module add cuda/12.6

source "$HOME/venvs/mtbreaker/bin/activate"
cd "$HOME/DeepLearning/"
mkdir -p logs

python -m src.RL_ppo_training \
  --seeds data/seeds.txt \
  --k 500 \
  --x 0.7 \
  --y 0.3 \
  --z 0.3 \
  --steps 500 \
  --batch_size 2 \
  --temperature 0.4
