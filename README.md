# NEAR Cortex-1: Advanced Crypto Market Reasoning AI

A specialized AI model that combines chain-of-thought reasoning with cross-chain data analysis to understand and predict crypto market dynamics. Built on Microsoft's Phi-4 (14B) and enhanced through GRPO (Group Policy Optimization), Cortex-1 aims to reason about market dynamics the way experienced traders do, but at a massive scale and with perfect recall of historical patterns.

## 💡 Open Source Commitment

We believe in the power of open collaboration and are committed to making Cortex-1 fully accessible to the developer community:

- **Open Source Dataset**: Our synthetic training dataset will be publicly available, providing developers with high-quality, labeled examples of crypto market reasoning
- **Open Model Weights**: Once trained, the complete model weights will be open-sourced for the community
- **Transparent Development**: All training code, reward functions, and benchmarking tools are open source
- **Developer-First**: Built as a tool for developers to integrate advanced market reasoning into their applications

Our goal is to create a foundation for the community to build upon, whether you're developing trading strategies, market analysis tools, or educational platforms.

## 🌟 Key Features

- **Chain-of-Thought Reasoning**: Detailed step-by-step analysis of market conditions
- **Cross-Chain Analysis**: Deep understanding of relationships between different blockchain networks
- **Quantitative Predictions**: Data-driven forecasting with confidence intervals
- **Risk Assessment**: Comprehensive evaluation of technical, market, and systemic risks
- **Opportunity Detection**: Identification of market inefficiencies and arbitrage opportunities

## 🏗 Architecture

```mermaid
graph TB
    subgraph Data Collection
        FC[Flipside Client] --> |Raw Data| DP[Data Processing]
        MC[Market Conditions] --> DP
        PC[Protocol Collection] --> DP
    end

    subgraph Synthetic Generation
        DP --> |Processed Data| SG[Synthetic Generator]
        DR1[DeepSeek R1] --> |Reasoning| SG
        SG --> |Examples| DS[Dataset]
        RF[Reward Functions] --> |Quality Metrics| SG
    end

    subgraph Model Training
        DS --> |Training Data| MT[Model Training]
        PHI4[Phi-4 14B] --> |Base Model| MT
        MT --> |Fine-tuned Model| MD[Model Deployment]
    end

    subgraph Testing & Evaluation
        TRF[Test Reward Functions] --> |Validation Results| RF
        MT --> |Checkpoints| EV[Evaluation Pipeline]
        EV --> |Metrics| MT
    end

    style Data Collection fill:#f9f,stroke:#333,stroke-width:2px
    style Synthetic Generation fill:#bbf,stroke:#333,stroke-width:2px
    style Model Training fill:#bfb,stroke:#333,stroke-width:2px
    style Testing & Evaluation fill:#fbb,stroke:#333,stroke-width:2px
```

### Component Details

1. **Data Collection Layer**
   - Flipside Client: Fetches raw blockchain data
   - Market Conditions: Analyzes and labels market states
   - Protocol Collection: Gathers DeFi protocol metrics

2. **Synthetic Generation Layer**
   - DeepSeek R1 Integration: Uses R1's reasoning capabilities to generate high-quality examples
   - Quality-Focused: Applies reward functions to verify example quality
   - Multi-chain Data: Integrates data from various blockchain sources
   - Template-based Prompts: Uses structured prompts to elicit detailed reasoning

3. **Reward System**
   - Modular Design: Separate reward components for different aspects of quality
   - Finance-Specific: Rewards for calculation accuracy, confidence intervals, and investment insights
   - Format Quality: Rewards for citation format, structure, and completeness
   - Composite Framework: Weighted combination of individual rewards

4. **Model Training Layer**
   - Phi-4 Base Model: 14B parameters, 16K context window, strong reasoning capabilities
   - GRPO (Group Policy Optimization): Optimizes for reward maximization
   - 4-bit Quantization: Enables deployment on consumer hardware
   - Local Training Capability: Can be fine-tuned on consumer GPUs (16GB+ VRAM)

### Data Pipeline

Our data pipeline is designed with a clear separation between the main generation system and the testing components. For detailed information, see [Data Pipeline Documentation](docs/DATA_PIPELINE.md).

1. **Main Pipeline:**
   - Fetches real market data from Flipside
   - Generates detailed reasoning using DeepSeek R1
   - Applies validated reward functions for quality verification
   - Creates standardized training examples with reasoning traces

2. **Testing System:**
   - Uses mock examples to validate reward functions
   - Provides controlled test cases of varying quality
   - Ensures reward functions correctly differentiate quality levels
   - Operates independently from the main pipeline

## 🎯 Core Capabilities

1. **Market Analysis & Prediction**
   - Historical pattern recognition
   - Cross-chain correlation analysis
   - Transaction volume forecasting
   - User behavior analysis

2. **Protocol Analysis**
   - Performance metrics evaluation
   - Growth trajectory analysis
   - Risk factor assessment
   - Optimization recommendations

3. **Risk Management**
   - Technical risk quantification
   - Market exposure analysis
   - Systemic risk assessment
   - Mitigation strategy development

4. **Opportunity Discovery**
   - Arbitrage opportunity detection
   - Yield optimization strategies
   - Market inefficiency analysis
   - Entry/exit point identification

## 🛠 Technical Architecture

### Base Model: Microsoft Phi-4

We chose Microsoft's Phi-4 (14B) as our base model for several key reasons:

- **Accessibility**: 14B parameters can run on consumer hardware (32GB+ RAM, 16GB+ VRAM)
- **Reasoning Capabilities**: Strong performance on mathematics and logical reasoning tasks
- **Context Window**: 16K tokens is sufficient for financial analysis scenarios
- **Quantization-Friendly**: Works efficiently with 4-bit quantization for memory optimization
- **Developer-First**: Enables more contributors to run and fine-tune locally

This choice reinforces our commitment to creating a truly accessible, open-source model that developers can run on their own hardware.

### Training Pipeline

1. **Synthetic Data Generation**
   - DeepSeek R1 reasoning integration
   - Market condition balancing
   - Cross-chain correlation scenarios
   - Protocol performance analysis cases
   - Risk assessment simulations

2. **Reward Function Components**
   - Finance-specific metrics (calculation accuracy, confidence intervals)
   - Format quality (structure, completeness)
   - Citation quality (metric citations, historical references)
   - Composite reward framework with flexible weighting

3. **Benchmarking Framework**
   - Historical prediction accuracy
   - Reasoning quality metrics
   - Cross-chain correlation accuracy
   - Protocol analysis precision
   - Real-world performance testing

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (16GB+ VRAM recommended)
- 32GB+ RAM for data preprocessing and training
- 100GB disk space for datasets and model weights

### Installation

1. Clone the repository:
```bash
git clone https://github.com/near/cortex-1.git
cd cortex-1
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys:
# - OPENROUTER_API_KEY (for DeepSeek R1 access)
# - FLIPSIDE_API_KEY (for market data)
# - WANDB_API_KEY (for experiment tracking)
# - HUGGINGFACE_TOKEN (for model downloading)
```

## 📊 Data Pipeline

### Market Data Collection

```bash
python scripts/test_flipside.py --days 30 --chains ethereum near
```

### Synthetic Data Generation
```bash
python scripts/generate_synthetic.py \
    --dataset-size medium \
    --chains market \
    --verify-all
```

### Reward Function Testing

```bash
python scripts/test_rewards.py --verbose
```

### Model Training with Phi-4 and GRPO

```bash
python scripts/train_grpo.py \
    --config configs/grpo/financial_reasoning.json \
    --verbose
```

## 🔍 Hardware Requirements

Cortex-1 is designed to run on accessible hardware:

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| RAM | 32GB | 64GB | Required for data preprocessing |
| GPU | 16GB VRAM | 24GB+ VRAM | A single NVIDIA RTX 3090/4090 or A6000 is sufficient |
| CPU | 8 cores | 16+ cores | For data preparation tasks |
| Storage | 100GB SSD | 250GB+ SSD | For datasets and model weights |

With 4-bit quantization, the Phi-4 model requires ~8GB of VRAM for inference and ~16GB for training with small batch sizes, making it feasible to run on a single consumer GPU.

## 🔍 Benchmarking

Our comprehensive benchmarking suite evaluates:

1. **Prediction Accuracy**
   - Transaction volume forecasting
   - User growth projections
   - Price movement predictions
   - Cross-chain correlation accuracy

2. **Reasoning Quality**
   - Chain-of-thought completeness
   - Logical consistency
   - Data citation accuracy
   - Technical analysis depth

3. **Real-World Performance**
   - Strategy backtesting
   - Market simulation
   - Live prediction tracking
   - Cross-chain arbitrage detection

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Code Contributions**
   - Fork the repository
   - Create a feature branch
   - Submit a pull request

2. **Data Contributions**
   - Historical market data
   - Protocol performance metrics
   - Cross-chain correlation data
   - Benchmark test cases

3. **Documentation**
   - Technical documentation
   - Use case examples
   - Benchmark results
   - Tutorial creation

4. **Model Development**
   - Fine-tuning improvements
   - Synthetic data generation
   - Reward function optimization
   - Benchmarking scenarios

## Project Structure

```
cortex-1/
├── configs/                  # Configuration files
│   ├── grpo/                 # GRPO configurations
│   ├── data_config.yaml      # Data configuration
│   └── model_config.yaml     # Model configuration
├── data/                     # Data directories
│   ├── raw/                  # Raw examples with reasoning
│   ├── training/             # Processed training examples
│   └── splits/               # Train/eval splits
├── docs/                     # Documentation
│   ├── DATA_PIPELINE.md      # Data pipeline documentation
│   └── TRAINING_PLAN.md      # Training strategy documentation
├── models/                   # Saved model weights
│   └── phi4_financial_reasoning/ # Trained Phi-4 model
├── scripts/                  # Execution scripts
│   ├── generate_synthetic.py # Data generation script
│   ├── test_rewards.py       # Reward testing script
│   └── train_grpo.py         # GRPO training script
├── src/                      # Source code
│   ├── data/                 # Data processing modules
│   ├── model/                # Model-related code
│   └── rewards/              # Reward function modules
└── README.md                 # This file
```

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NEAR Foundation for support and guidance
- Unsloth Team for GRPO implementation
- Flipside Crypto for market data access
- OpenRouter for DeepSeek R1 API access
- Microsoft for the Phi-4 model

## 📚 Documentation

For detailed documentation, visit our [Wiki](https://github.com/near/cortex-1/wiki).

## 🔗 Links

- [NEAR Foundation](https://near.foundation/)
- [Project Documentation](https://near-foundation.notion.site/NEAR-Cortex-1-AI-Reasoning-Model)
- [Microsoft Phi-4](https://huggingface.co/microsoft/phi-4)
- [Unsloth GRPO](https://github.com/unslothai/unsloth)
- [Training Plan](docs/TRAINING_PLAN.md)
- [Data Pipeline](docs/DATA_PIPELINE.md)
- [Contribution Guide](CONTRIBUTING.md)

## 📧 Contact

For questions or support, please open an issue or contact the NEAR Foundation team.
