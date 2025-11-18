#!/bin/bash

#SBATCH --account=deep_learning
#SBATCH --output=logs/verify_%j.out
#SBATCH --time=01:00
#SBATCH --job-name=verify-models
#SBATCH --gpus 1080ti:1

. /etc/profile.d/modules.sh
module add cuda/12.1

source "$HOME/venvs/mtbreaker/bin/activate"
cd "$HOME/DeepLearning/"
mkdir -p logs

# Run model comparison: gpt2 vs google/flan-t5-small
python verify_gpt2_generation.py \
  --model_path facebook/opt-iml-1.3b \
  --seed_sentence "They walked towards the park's entrance." \
  --num_samples 3 \
  --max_new_tokens 60 \
  --temperature 0.9 \
  --top_p 0.95
