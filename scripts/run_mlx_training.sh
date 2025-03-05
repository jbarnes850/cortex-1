#!/bin/bash

# Run MLX-based GRPO training on Apple Silicon
# Usage: ./run_mlx_training.sh [small|medium|large] [skip_sft] [huggingface_token]

# Default to small dataset for initial validation run
DATASET_SIZE=${1:-"small"}
SKIP_SFT=${2:-"false"}
HF_TOKEN=${3:-""}

# Activate the Python virtual environment
if [ -d "cortex_mlx_env" ]; then
    echo "Activating cortex_mlx_env virtual environment..."
    source cortex_mlx_env/bin/activate
else
    echo "Virtual environment not found. Please create it first with:"
    echo "python -m venv cortex_mlx_env"
    echo "source cortex_mlx_env/bin/activate"
    echo "pip install mlx mlx-lm huggingface_hub python-dotenv transformers datasets"
    exit 1
fi

# Check if MLX is installed
if ! python -c "import mlx.core" &> /dev/null; then
    echo "MLX not found. Installing MLX..."
    pip install mlx mlx-lm
fi

# Check if huggingface_hub is installed
if ! python -c "import huggingface_hub" &> /dev/null; then
    echo "huggingface_hub not found. Installing huggingface_hub..."
    pip install huggingface_hub
fi

# Check for dotenv
if ! python -c "import dotenv" &> /dev/null; then
    echo "python-dotenv not found. Installing..."
    pip install python-dotenv
fi

# Create log directory
mkdir -p logs

# Set log file name with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/mlx_training_${DATASET_SIZE}_${TIMESTAMP}.log"

echo "Starting MLX-based GRPO training with $DATASET_SIZE dataset..."
echo "This implementation trains the model to use <thinking>...</thinking> XML tags for reasoning processes"
echo "Logs will be saved to: $LOG_FILE"

# Create token argument if supplied
TOKEN_ARG=""
if [ -n "$HF_TOKEN" ]; then
    TOKEN_ARG="--hf-token $HF_TOKEN"
    echo "Using provided Hugging Face token for authentication"
elif [ -n "$HUGGINGFACE_TOKEN" ]; then
    TOKEN_ARG="--hf-token $HUGGINGFACE_TOKEN"
    echo "Using Hugging Face token from environment variable"
else
    echo "WARNING: No Hugging Face token provided. You may not be able to access gated models like Phi-4."
fi

# Run the training script
if [ "$SKIP_SFT" = "true" ]; then
    python scripts/train_mlx_grpo.py \
        --config configs/mlx_grpo_config.yaml \
        --dataset-size $DATASET_SIZE \
        --skip-sft \
        --verbose \
        $TOKEN_ARG 2>&1 | tee $LOG_FILE
else
    python scripts/train_mlx_grpo.py \
        --config configs/mlx_grpo_config.yaml \
        --dataset-size $DATASET_SIZE \
        --verbose \
        $TOKEN_ARG 2>&1 | tee $LOG_FILE
fi

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
    echo "Model saved to: $(grep 'Saved final model to' $LOG_FILE | tail -n 1 | awk '{print $NF}')"
else
    echo "Training failed with exit code $?"
fi

# Make the script executable
chmod +x scripts/run_mlx_training.sh 