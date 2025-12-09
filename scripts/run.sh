#!/bin/bash

#SBATCH --account=deep_learning
#SBATCH --output=logs/mtbreaker_%j.out
#SBATCH --job-name=mtbreaker
#SBATCH --gpus 5060ti:1


. /etc/profile.d/modules.sh
module add cuda/12.8  # Blackwell (sm_120) requires CUDA 12.8+

source "$HOME/venvs/mtbreaker/bin/activate"
cd "/work/scratch/mbehanzin/DeepLearning/"
mkdir -p logs

python -m src.RL_ppo_training \
  --k 200 \
  --gen_max_new_tokens 400 \
  --x 1.0 \
  --y 1.0 \
  --z 1.0 \
  --steps 100 \
  --batch_size 4 \
  --temperature 0.4 \
  --trainable_layers 4 \
  --freeze_embeddings \
  --save_dir "checkpoints/stage-alpha"
