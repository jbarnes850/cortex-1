# NEAR Cortex-1: Crypto Reasoning AI Model

A specialized AI model that reasons about and predicts crypto market movements using cross-chain data and advanced machine learning techniques.

## Overview

NEAR Cortex-1 uses Llama 3.3 70B as its base model, enhanced with GRPO (Group Policy Optimization) fine-tuning and synthetic data generation to create a powerful crypto market analysis system.

## Requirements

- Python 3.10+
- CUDA-compatible GPU(s) for training
- 192GB+ RAM for data preprocessing
- Cloud GPU access (A100/H100) for full model training

## Setup

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables:

```bash
cp .env.example .env
# Edit .env with your API keys and configurations
```

## Project Structure

```bash
├── data/                  # Data storage
│   ├── raw/              # Raw data from Flipside
│   ├── processed/        # Processed datasets
│   └── synthetic/        # Generated synthetic data
├── src/
│   ├── data/            # Data processing modules
│   ├── model/           # Model architecture and training
│   ├── utils/           # Utility functions
│   └── inference/       # Inference and deployment code
├── configs/             # Configuration files
├── scripts/             # Training and utility scripts
└── notebooks/          # Development notebooks
```

## Training Pipeline

1. **Data Collection**:
   - Fetch historical data from Flipside Crypto API
   - Generate synthetic reasoning data
   - Preprocess and validate datasets

2. **Model Training**:
   - Initial supervised fine-tuning
   - GRPO reinforcement learning
   - Distributed training support

3. **Evaluation**:
   - Prediction accuracy metrics
   - Reasoning quality assessment
   - Performance benchmarking

## Usage

1. Data Collection:

```bash
python scripts/collect_data.py --days 365
```

2. Generate Synthetic Data:

```bash
python scripts/generate_synthetic.py --input-file data/raw/market_data.csv
```

3. Train Model:

```bash
python scripts/train.py --config configs/training_config.yaml
```

4. Inference:

```bash
python scripts/inference.py --model-path models/latest --input "market analysis query"
```

## Configuration

The project uses YAML configuration files in the `configs/` directory:

- `training_config.yaml`: Training hyperparameters
- `model_config.yaml`: Model architecture settings
- `data_config.yaml`: Data processing parameters

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- Unsloth Team for GRPO implementation
- Flipside Crypto for market data access
- NEAR Foundation for support
