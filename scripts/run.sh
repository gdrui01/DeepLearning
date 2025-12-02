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
  --k 800 \
  # delta difficulty
  --x 1.0 \
  # verifier score
  --y 2.0 \
  # constraint score
  --z 2.0 \
  --steps 600 \
  --batch_size 4 \
  --temperature 0.4 \
  --save_dir "checkpoints/stage-1" \
  --resume_from_checkpoint "checkpoints/stage-2/"
