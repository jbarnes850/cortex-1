#!/bin/bash

# Run GRPO training with the specified dataset size
# Usage: ./run_training.sh [small|medium|large] [enable_wandb]

# Default to small dataset if not specified
DATASET_SIZE=${1:-"small"}
ENABLE_WANDB=${2:-"false"}

# Set environment variables
if [ "$ENABLE_WANDB" = "true" ]; then
    echo "Enabling W&B logging"
    export WANDB_MODE="online"
else
    echo "Disabling W&B logging"
    export WANDB_MODE="disabled"
fi

# Run the training script
echo "Starting GRPO training with $DATASET_SIZE dataset..."
python scripts/train_grpo_native.py \
    --config configs/grpo_config.yaml \
    --dataset-size $DATASET_SIZE \
    --verbose

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code $?"
fi 