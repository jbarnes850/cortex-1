# Financial Analysis Rewards Module

This module provides a comprehensive set of reward functions for evaluating and training language models on financial analysis tasks using GRPO (Generative Reinforcement Policy Optimization). The rewards are designed to incentivize accurate, well-structured, and insightful financial analysis with proper citation and reasoning.

## Architecture

The rewards module is structured as a hierarchical system of specialized evaluators:

```bash
BaseReward (abstract)
├── Finance Rewards
│   ├── CalculationAccuracyReward
│   ├── ConfidenceIntervalReward
│   └── InvestmentInsightReward
├── Format Rewards
│   ├── CitationFormatReward
│   ├── StructureReward
│   └── CompletenessReward
├── Citation Rewards
│   ├── MetricCitationReward
│   └── HistoricalReferenceReward
└── CompositeReward
    └── create_financial_reward()
```

## Reward Components

### Finance Rewards

- **CalculationAccuracyReward**: Evaluates the accuracy of financial calculations, checking for numerical results and proper formula usage.
- **ConfidenceIntervalReward**: Assesses the proper use of confidence intervals, including confidence levels and bounds.
- **InvestmentInsightReward**: Evaluates the quality of investment insights, including specific recommendations, quantitative support, and risk assessments.

### Format Rewards

- **CitationFormatReward**: Evaluates the proper citation of metrics in a specific format, rewarding well-formatted citations.
- **StructureReward**: Assesses the structure of financial analysis, checking for required sections and logical ordering.
- **CompletenessReward**: Evaluates the completeness of the analysis by checking for key components in various categories.

### Citation Rewards

- **MetricCitationReward**: Evaluates the proper citation of specific metrics, rewarding comprehensive citation and usage in calculations.
- **HistoricalReferenceReward**: Assesses proper reference to historical data, rewarding quality historical context.

### Composite Reward

- **CompositeReward**: Combines multiple individual rewards with appropriate weights.
- **create_financial_reward()**: Factory function that creates a standard composite reward for financial analysis.

## Usage

### Basic Usage

```python
from src.rewards import get_default_financial_reward

# Get the default composite reward function
reward_fn = get_default_financial_reward()

# Calculate reward for a prompt-response pair
prompt = "Analyze the following market data for NEAR protocol..."
response = "Based on the provided metrics, NEAR protocol shows..."
reward_score = reward_fn(prompt, response)

# Access individual component scores
component_scores = reward_fn.get_component_scores()
print(component_scores)
```

### Custom Weights

You can customize the weights of different reward components:

```python
from src.rewards.composite_reward import create_financial_reward

# Create a custom weighted composite reward
custom_reward = create_financial_reward(
    calculation_accuracy_weight=1.2,
    confidence_interval_weight=0.5,
    investment_insight_weight=1.5,
    citation_format_weight=0.8,
    structure_weight=0.7,
    completeness_weight=0.9,
    metric_citation_weight=1.0,
    historical_reference_weight=0.8
)
```

### Integration with GRPO Training

The rewards module is designed to integrate with Unsloth's GRPO training:

```python
from src.rewards import get_default_financial_reward
from unsloth import setup_trainer_for_grpo

# Setup GRPO trainer with custom reward function
trainer = setup_trainer_for_grpo(...)
trainer.set_reward_fn(get_default_financial_reward())
```

## Evaluation

You can evaluate the effectiveness of different reward functions using the `evaluate_rewards.py` script:

```bash
python scripts/evaluate_rewards.py --input data/synthetic/training/reasoning_training_1000.jsonl
```

This will generate visualizations showing the distributions of rewards, correlations between different reward components, and reward scores versus response length.

## Configuration

The weights and parameters for reward functions can be configured in the GRPO configuration file (`configs/grpo_config.yaml`):

```yaml
reward:
  weights:
    calculation_accuracy: 1.0
    confidence_interval: 0.8
    investment_insight: 1.0
    citation_format: 0.7
    structure: 0.6
    completeness: 0.8
    metric_citation: 0.9
    historical_reference: 0.7
```

## Extending the Rewards Module

To create a new reward function:

1. Inherit from `BaseReward` in `src.rewards.finance_rewards`.
2. Implement the `calculate` method to compute the reward score.
3. Add the new reward to the appropriate reward category file.
4. Update the `create_financial_reward` function in `composite_reward.py` to include your new reward.

## Best Practices

1. **Balanced Weighting**: Ensure weights are balanced to avoid over-optimization of one aspect at the expense of others.

2. **Response Calibration**: Use the `evaluate_rewards.py` script to analyze reward distributions and adjust weights accordingly.

3. **Targeted Rewards**: Design rewards that specifically target desired behaviors in financial analysis.

4. **Explainable Components**: Each reward component should be interpretable and measurable. 