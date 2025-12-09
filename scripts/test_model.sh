#!/bin/bash

#SBATCH --account=deep_learning
#SBATCH --output=logs/test_%j.out
#SBATCH --time=01:00
#SBATCH --job-name=test-model
#SBATCH --gpus rtx5060ti:1
# NOTE: GPU specification syntax may vary by cluster. Check with admin for correct identifier.

. /etc/profile.d/modules.sh
module add cuda/12.8  # Blackwell (sm_120) requires CUDA 12.8+

source "$HOME/venvs/mtbreaker/bin/activate"
cd "$HOME/DeepLearning/"
mkdir -p logs

python test_scripts/test_model_inference.py