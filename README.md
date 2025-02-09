# NEAR Cortex-1: Crypto Reasoning AI Model

A specialized AI model that reasons about and predicts crypto market movements using cross-chain data and advanced machine learning techniques.

## Overview

NEAR Cortex-1 uses Llama 3.3 70B as its base model, enhanced with GRPO (Group Policy Optimization) fine-tuning and synthetic data generation to create a powerful crypto market analysis system. The model specializes in:

- Historical Market Analysis & Q&A
- Cross-Chain Correlation Analysis
- DeFi Protocol Performance Prediction
- Risk Assessment & Opportunity Detection
- Technical & Fundamental Analysis

## Requirements

- Python 3.10+
- CUDA-compatible GPU(s) for training
- 192GB+ RAM for data preprocessing
- Cloud GPU access (A100/H100) for full model training
- OpenAI API key for synthetic data generation
- Flipside Crypto API key for market data

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
# Edit .env with your API keys:
# - OPENAI_API_KEY
# - FLIPSIDE_API_KEY
```

## Project Structure

```bash
├── data/                  # Data storage
│   ├── raw/              # Raw data from Flipside
│   ├── processed/        # Processed datasets
│   └── synthetic/        # Generated synthetic data
├── src/
│   ├── data/            # Data processing modules
│   │   ├── flipside_client.py     # Flipside API client
│   │   └── synthetic_generator.py  # Synthetic data generation
│   ├── model/           # Model architecture and training
│   │   ├── openai_client.py       # OpenAI API client
│   │   └── reward_function.py     # GRPO reward function
│   └── utils/           # Utility functions
├── scripts/             # Training and utility scripts
│   ├── generate_synthetic.py   # Generate training data
│   └── test_synthetic.py      # Test data generation
└── configs/             # Configuration files
```

## Training Pipeline

1. **Data Collection & Generation**:

```bash
# Generate synthetic training data
python scripts/generate_synthetic.py \
    --days 180 \
    --samples-per-day 10 \
    --chains ethereum \
    --protocols uniswap \
    --model o3-mini \
    --output-dir data/synthetic
```

The synthetic data generator creates:

- Chain-of-thought reasoning examples
- Market prediction scenarios
- Cross-chain analysis problems
- Protocol performance analysis
- Risk assessment cases

2. **Dataset Features**:

- Balanced market conditions (bullish, bearish, sideways, volatile)
- Diverse prompt types (prediction, analysis, Q&A)
- Quality metrics for each example
- Incremental saving and progress tracking
- Comprehensive reward scoring

3. **Reward Function Components**:

- Prediction accuracy
- Reasoning depth
- Technical analysis quality
- Market understanding
- Risk assessment
- Data usage efficiency
- Cross-chain analysis
- Group policy bonus

4. **Model Training**:

- Initial supervised fine-tuning
- GRPO reinforcement learning
- Distributed training support
- Quality-based filtering

## Usage

1. Generate Synthetic Data:

```bash
python scripts/generate_synthetic.py --days 180 --samples-per-day 10
```

2. Test Data Generation:

```bash
python scripts/test_synthetic.py
```

3. Train Model (coming soon):

```bash
python scripts/train.py --config configs/training_config.yaml
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
- OpenAI for synthetic data generation support
