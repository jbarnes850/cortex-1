#!/bin/bash
# Script to download and process historical cryptocurrency data for NEAR Cortex-1
# Part of Phase 1 implementation of the hybrid dataset approach

set -e

# Define variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/historical_data_download_$(date +%Y%m%d_%H%M%S).log"

# Make sure logs directory exists
mkdir -p "$LOG_DIR"

# Create a function to print messages with timestamps
log() {
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] $1" | tee -a "$LOG_FILE"
}

# Print welcome message
log "Starting historical data download for NEAR Cortex-1"
log "Project root: $PROJECT_ROOT"
log "Data directory: $DATA_DIR"
log "Log file: $LOG_FILE"

# Load environment variables from .env file
if [ -f "$PROJECT_ROOT/.env" ]; then
    log "Loading environment variables from .env file"
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
    log "Environment variables loaded successfully"
else
    log "Warning: .env file not found at $PROJECT_ROOT/.env"
fi

# Check for Kaggle API credentials
if [ ! -f ~/.kaggle/kaggle.json ]; then
    log "Kaggle API credentials not found at ~/.kaggle/kaggle.json"
    
    # Check if environment variables are set
    if [ -z "$KAGGLE_USERNAME" ] || [ -z "$KAGGLE_KEY" ]; then
        log "KAGGLE_USERNAME and KAGGLE_KEY environment variables not set"
        log "Please set up Kaggle API credentials by:"
        log "1. Creating a Kaggle account at https://www.kaggle.com"
        log "2. Go to Account -> Create API Token to download kaggle.json"
        log "3. Place kaggle.json in ~/.kaggle/ or set KAGGLE_USERNAME and KAGGLE_KEY environment variables"
        exit 1
    else
        log "Found Kaggle credentials in environment variables"
        
        # Create kaggle directory if it doesn't exist
        mkdir -p ~/.kaggle
        
        # Create kaggle.json file
        echo "{\"username\":\"$KAGGLE_USERNAME\",\"key\":\"$KAGGLE_KEY\"}" > ~/.kaggle/kaggle.json
        chmod 600 ~/.kaggle/kaggle.json
        log "Created ~/.kaggle/kaggle.json from environment variables"
    fi
fi

# Check for python virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    log "No active virtual environment detected"
    
    # Check if venv directory exists
    if [ -d "$PROJECT_ROOT/venv" ]; then
        log "Activating existing virtual environment"
        source "$PROJECT_ROOT/venv/bin/activate"
    elif [ -d "$PROJECT_ROOT/cortex_mlx_env" ]; then
        log "Activating existing MLX environment"
        source "$PROJECT_ROOT/cortex_mlx_env/bin/activate"
    elif [ -d "$PROJECT_ROOT/deepseek-mlx/cortex-mlx-env" ]; then
        log "Activating existing DeepSeek MLX environment"
        source "$PROJECT_ROOT/deepseek-mlx/cortex-mlx-env/bin/activate"
    else
        log "Creating new virtual environment"
        python -m venv "$PROJECT_ROOT/venv"
        source "$PROJECT_ROOT/venv/bin/activate"
        log "Installing required packages"
        pip install -q -U pip
        pip install -q -r "$PROJECT_ROOT/requirements.txt"
    fi
else
    log "Using active virtual environment: $VIRTUAL_ENV"
fi

# Parse command line arguments
COINS=""
FORCE_DOWNLOAD=false
SKIP_METRICS=false
ALL_COINS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --coins)
            COINS="$2"
            shift 2
            ;;
        --force-download)
            FORCE_DOWNLOAD=true
            shift
            ;;
        --skip-derived-metrics)
            SKIP_METRICS=true
            shift
            ;;
        --all-coins)
            ALL_COINS=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --coins COINS          Comma-separated list of coins to process (e.g. BTC,ETH,XPR)"
            echo "  --force-download       Force download even if data exists locally"
            echo "  --skip-derived-metrics Skip calculation of derived metrics"
            echo "  --all-coins            Process all available coins"
            echo "  --help                 Display this help message"
            echo ""
            echo "Note: Use ticker symbols (BTC, ETH, XRP) instead of full names (bitcoin, ethereum)."
            exit 0
            ;;
        *)
            log "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Print a helpful message about coin naming
log "Note: Use ticker symbols (BTC, ETH, XPR) instead of full names (bitcoin, ethereum)."

# Prepare command
CMD="python $SCRIPT_DIR/process_historical_data.py"

if [ -n "$COINS" ]; then
    CMD="$CMD --coins $COINS"
elif [ "$ALL_COINS" = true ]; then
    CMD="$CMD --all-coins"
fi

if [ "$FORCE_DOWNLOAD" = true ]; then
    CMD="$CMD --force-download"
fi

if [ "$SKIP_METRICS" = true ]; then
    CMD="$CMD --skip-derived-metrics"
fi

# Pass environment variables explicitly
CMD="KAGGLE_USERNAME=\"$KAGGLE_USERNAME\" KAGGLE_KEY=\"$KAGGLE_KEY\" $CMD"

# Run the command
log "Running command: $CMD"
eval "$CMD"

# Check if command was successful
if [ $? -eq 0 ]; then
    log "Historical data processing completed successfully"
else
    log "Error: Historical data processing failed with exit code $?"
    exit 1
fi

log "Done!" 