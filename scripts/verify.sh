#!/bin/bash

#SBATCH --account=deep_learning
#SBATCH --output=logs/verify_%j.out
#SBATCH --time=01:00
#SBATCH --job-name=verify-models
#SBATCH --gpus rtx5060ti:1
# NOTE: GPU specification syntax may vary by cluster. Check with admin for correct identifier.

. /etc/profile.d/modules.sh
module add cuda/12.8  # Blackwell (sm_120) requires CUDA 12.8+

source "$HOME/venvs/mtbreaker/bin/activate"
cd "$HOME/DeepLearning/"
mkdir -p logs

# Run model comparison: gpt2 vs google/flan-t5-small
python test_scripts/verify_gpt2_generation.py \
  --compare \
  --compare_models google/flan-t5-base facebook/opt-iml-1.3b \
  --seed_sentence "They walked towards the park's entrance." \
  --num_samples 3 \
  --max_new_tokens 2000 \
  --temperature 0.9 \
  --top_p 0.95
