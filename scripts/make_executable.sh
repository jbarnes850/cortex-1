#!/bin/bash
# Make all Python scripts in the scripts directory executable

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Find all Python scripts in the scripts directory
PYTHON_SCRIPTS=$(find "$SCRIPT_DIR" -name "*.py")

# Make each script executable
for script in $PYTHON_SCRIPTS; do
    echo "Making executable: $script"
    chmod +x "$script"
done

echo "All Python scripts are now executable."
echo "You can run them directly, e.g.:"
echo "./scripts/train_grpo.py --config configs/grpo_config.yaml" 