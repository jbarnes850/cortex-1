#!/usr/bin/env python
"""
Script to evaluate reward functions on existing financial analysis responses.
This tool helps to analyze the effectiveness of different reward components
and to calibrate weights for GRPO training.
"""

import os
import sys
import argparse
import logging
import json
import yaml
import pandas as pd
from typing import Dict, List, Optional, Union, Any
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.rewards import get_default_financial_reward
from src.rewards.finance_rewards import CalculationAccuracyReward, ConfidenceIntervalReward, InvestmentInsightReward
from src.rewards.format_rewards import CitationFormatReward, StructureReward, CompletenessReward
from src.rewards.citation_rewards import MetricCitationReward, HistoricalReferenceReward
from src.rewards.composite_reward import CompositeReward

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_examples(file_path: str) -> List[Dict]:
    """Load examples from a JSONL file."""
    examples = []
    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            example = json.loads(line)
            examples.append(example)
    return examples

def extract_prompt_response_pairs(examples: List[Dict]) -> List[Dict]:
    """Extract prompt-response pairs from examples."""
    pairs = []
    for example in examples:
        messages = example.get('messages', [])
        
        # Find user prompt and assistant response
        prompt = None
        response = None
        
        for msg in messages:
            if msg.get('role') == 'user':
                prompt = msg.get('content', '')
            elif msg.get('role') == 'assistant':
                response = msg.get('content', '')
        
        if prompt and response:
            pairs.append({
                'prompt': prompt,
                'response': response
            })
    
    return pairs

def evaluate_rewards(prompt_response_pairs: List[Dict], reward_dict: Dict[str, Any]) -> pd.DataFrame:
    """Evaluate different reward functions on the given prompt-response pairs."""
    results = []
    
    for pair in tqdm(prompt_response_pairs, desc="Evaluating rewards"):
        prompt = pair['prompt']
        response = pair['response']
        
        row = {'prompt': prompt[:100] + '...', 'response_length': len(response)}
        
        # Evaluate each reward function
        for reward_name, reward_fn in reward_dict.items():
            reward_score = reward_fn(prompt, response)
            row[reward_name] = reward_score
            
            # If it's a composite reward, add component scores
            if hasattr(reward_fn, 'get_component_scores'):
                component_scores = reward_fn.get_component_scores()
                for component_name, component_score in component_scores.items():
                    row[f"{reward_name}_{component_name}"] = component_score
            
        results.append(row)
    
    return pd.DataFrame(results)

def plot_reward_distributions(df: pd.DataFrame, output_dir: str):
    """Plot the distributions of reward scores."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Identify reward columns
    reward_cols = [col for col in df.columns if col not in ['prompt', 'response_length']]
    
    # Plot distribution of all rewards
    plt.figure(figsize=(12, 8))
    for reward_col in reward_cols:
        sns.kdeplot(df[reward_col], label=reward_col)
    
    plt.title('Distribution of Reward Scores')
    plt.xlabel('Reward Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'reward_distributions.png'))
    
    # Plot correlation matrix
    plt.figure(figsize=(14, 10))
    correlation = df[reward_cols].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Reward Scores')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reward_correlations.png'))
    
    # Plot reward vs response length
    plt.figure(figsize=(12, 8))
    for reward_col in reward_cols:
        plt.scatter(df['response_length'], df[reward_col], alpha=0.5, label=reward_col)
    
    plt.title('Reward Scores vs Response Length')
    plt.xlabel('Response Length (chars)')
    plt.ylabel('Reward Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'reward_vs_length.png'))
    
    # For composite reward, plot component contributions
    composite_cols = [col for col in reward_cols if '_' in col and 'composite' in col]
    if composite_cols:
        plt.figure(figsize=(10, 6))
        component_avgs = df[composite_cols].mean().sort_values()
        component_avgs.plot(kind='barh')
        plt.title('Average Component Contribution to Composite Reward')
        plt.xlabel('Average Score')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'component_contributions.png'))

def create_reward_dict(weights: Dict[str, float] = None) -> Dict[str, Any]:
    """Create a dictionary of reward functions with specified weights."""
    if weights is None:
        weights = {
            'calculation_accuracy': 1.0,
            'confidence_interval': 0.8,
            'investment_insight': 1.0,
            'citation_format': 0.7,
            'structure': 0.6,
            'completeness': 0.8,
            'metric_citation': 0.9,
            'historical_reference': 0.7
        }
    
    return {
        'calculation_accuracy': CalculationAccuracyReward(weight=weights.get('calculation_accuracy', 1.0)),
        'confidence_interval': ConfidenceIntervalReward(weight=weights.get('confidence_interval', 0.8)),
        'investment_insight': InvestmentInsightReward(weight=weights.get('investment_insight', 1.0)),
        'citation_format': CitationFormatReward(weight=weights.get('citation_format', 0.7)),
        'structure': StructureReward(weight=weights.get('structure', 0.6)),
        'completeness': CompletenessReward(weight=weights.get('completeness', 0.8)),
        'metric_citation': MetricCitationReward(weight=weights.get('metric_citation', 0.9)),
        'historical_reference': HistoricalReferenceReward(weight=weights.get('historical_reference', 0.7)),
        'composite': get_default_financial_reward()
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate reward functions on existing responses')
    parser.add_argument('--input', type=str, default=None, help='Path to JSONL file with examples')
    parser.add_argument('--config', type=str, default='configs/grpo_config.yaml', help='Path to GRPO config file')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save evaluation results')
    parser.add_argument('--sample-size', type=int, default=100, help='Number of examples to evaluate')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed for sampling')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    reward_config = config.get('reward', {})
    
    # Determine input file path
    if args.input:
        input_path = args.input
    else:
        data_config = config.get('data', {})
        dataset_size = data_config.get('train_sizes', [1000])[0]
        input_path = data_config.get('train_path', 'data/synthetic/training/reasoning_training_{size}.jsonl').format(size=dataset_size)
    
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    # Determine output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(
        project_root, 
        "evaluation", 
        f"reward_eval_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Evaluating reward functions on examples from: {input_path}")
    logger.info(f"Sample size: {args.sample_size}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Load examples
        examples = load_examples(input_path)
        logger.info(f"Loaded {len(examples)} examples from {input_path}")
        
        # Sample if needed
        if args.sample_size and args.sample_size < len(examples):
            import random
            random.seed(args.random_seed)
            examples = random.sample(examples, args.sample_size)
            logger.info(f"Sampled {len(examples)} examples for evaluation")
        
        # Extract prompt-response pairs
        pairs = extract_prompt_response_pairs(examples)
        logger.info(f"Extracted {len(pairs)} prompt-response pairs")
        
        # Create reward functions
        weights = reward_config.get('weights', {})
        reward_dict = create_reward_dict(weights)
        logger.info(f"Created {len(reward_dict)} reward functions")
        
        # Evaluate rewards
        results_df = evaluate_rewards(pairs, reward_dict)
        
        # Save raw results
        results_path = os.path.join(output_dir, 'reward_evaluations.csv')
        results_df.to_csv(results_path, index=False)
        logger.info(f"Saved raw evaluation results to: {results_path}")
        
        # Plot distributions
        plot_reward_distributions(results_df, output_dir)
        logger.info(f"Saved reward distribution plots to: {output_dir}")
        
        # Print summary statistics
        reward_cols = [col for col in results_df.columns if col not in ['prompt', 'response_length']]
        summary = results_df[reward_cols].describe()
        summary_path = os.path.join(output_dir, 'reward_summary.csv')
        summary.to_csv(summary_path)
        
        logger.info("\nReward Function Summary Statistics:")
        print(summary)
        logger.info(f"Summary statistics saved to: {summary_path}")
        
    except Exception as e:
        logger.error(f"Error during reward evaluation: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    from datetime import datetime
    main() 