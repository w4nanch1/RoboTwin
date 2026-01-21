export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
train_config_name=$1
model_name=$2
gpu_use=$3
enable_wandb=${4:-false} 

export CUDA_VISIBLE_DEVICES=$gpu_use
echo $CUDA_VISIBLE_DEVICES

wandb_arg="--no-wandb-enabled"
if [ "$enable_wandb" = "true" ] || [ "$enable_wandb" = "1" ]; then
    wandb_arg=""
    echo "Wandb logging is enabled"
else
    echo "Wandb logging is disabled (default)"
fi

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py $train_config_name --exp-name=$model_name --overwrite $wandb_arg