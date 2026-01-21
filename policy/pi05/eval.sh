#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 # ensure GPU < 24G

policy_name=pi05
task_name=${1}
task_config=${2}
train_config_name=${3}
model_name=${4}
seed=${5}
gpu_id=${6}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

# Add uv to PATH if not already there
export PATH="$HOME/.local/bin:$PATH"

# Activate pi05's virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "Activated .venv environment"
fi

# Set LD_LIBRARY_PATH to include RoboTwin conda environment libraries (for ffmpeg, etc.)
# This must be done AFTER activating venv to ensure it's not overridden
if [ -n "$CONDA_PREFIX" ] && [ -d "$CONDA_PREFIX/lib" ]; then
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
fi
# Always add RoboTwin conda env lib path if it exists (for ffmpeg)
if [ -d "/root/miniconda3/envs/RoboTwin/lib" ]; then
    export LD_LIBRARY_PATH=/root/miniconda3/envs/RoboTwin/lib:$LD_LIBRARY_PATH
fi

cd ../.. # move to root

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --ckpt_setting ${model_name} \
    --seed ${seed} \
    --policy_name ${policy_name} 
