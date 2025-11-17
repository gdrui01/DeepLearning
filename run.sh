#!/bin/bash

#SBATCH --account=deep_learning
#SBATCH --output=logs/mtbreaker_%j.out
#SBATCH --time=10:00

. /etc/profile.d/modules.sh
module add cuda/12.1

source "$HOME/venvs/mtbreaker/bin/activate"
cd "$HOME/DeepLearning/"
mkdir -p logs

python -m src.RL_ppo_training \
  --seeds data/seeds.txt \
  --k 100 \
  --steps 200 \
  --batch_size 8
