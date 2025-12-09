#!/bin/bash

#SBATCH --account=deep_learning
#SBATCH --output=logs/mtbreaker_%j.out
#SBATCH --job-name=mtbreaker
#SBATCH --gpus 1080ti:1


. /etc/profile.d/modules.sh
module add cuda/12.6

source "$HOME/venvs/mtbreaker/bin/activate"
cd "/work/scratch/mbehanzin/DeepLearning/"
mkdir -p logs

python -m src.RL_ppo_training \
  --seeds data/seeds.txt \
  --k 800 \
  --gen_max_new_tokens 90 \
  --x 5.0 \
  --y 0.3 \
  --z 0.3 \
  --steps 400 \
  --batch_size 8 \
  --temperature 0.4 \
  --save_dir "checkpoints/stage-z" \
  --resume_from_checkpoint "checkpoints/stage-0/"
