#!/usr/bin/env python
"""
Script to generate synthetic, chain-of-thought reasoning training data for NEAR Cortex-1.
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import json
import numpy as np
from typing import Dict, List
import time
import random
import uuid

from src.data.flipside_client import FlipsideClient
from src.data.synthetic_generator import SyntheticDataGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Number of days of historical data to collect"
    )
    parser.add_argument(
        "--samples-per-day",
        type=int,
        default=5,
        help="Number of synthetic examples to generate per day"
    )
    parser.add_argument(
        "--chains",
        nargs="+",
        default=["ethereum", "near", "solana", "avalanche"],
        help="Blockchains to collect data from"
    )
    parser.add_argument(
        "--protocols",
        nargs="+",
        default=["uniswap", "aave", "curve"],
        help="DeFi protocols to analyze"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/synthetic",
        help="Directory to save synthetic data"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="o3-mini",
        help="Model to use for synthetic data generation"
    )
    return parser.parse_args()

def collect_market_data(
    client: FlipsideClient,
    chains: list[str],
    start_date: datetime,
    end_date: datetime
) -> list[dict]:
    """Collect comprehensive market data for multiple chains."""
    market_data = []
    
    for chain in chains:
        logger.info(f"Collecting market data for {chain}...")
        try:
            # Get market data with retries
            max_retries = 3
            retry_delay = 5
            
            for attempt in range(max_retries):
                try:
                    chain_data = client.get_market_data(
                        chain=chain,
                        start_date=start_date.strftime("%Y-%m-%d"),
                        end_date=end_date.strftime("%Y-%m-%d")
                    )
                    
                    if chain_data.empty:
                        raise ValueError(f"No data returned for {chain}")
                    
                    # Group by day and create samples with comprehensive metrics
                    for date, day_data in chain_data.groupby(pd.Grouper(key='block_timestamp', freq='D')):
                        if not day_data.empty:
                            # Calculate additional derived metrics
                            avg_tx_value = day_data['avg_tx_value'].mean() if 'avg_tx_value' in day_data else 0
                            prev_day_value = market_data[-1]['avg_tx_value'] if market_data else avg_tx_value
                            
                            market_data.append({
                                'block_timestamp': str(date),
                                'network': day_data['network'].iloc[0] if 'network' in day_data else chain,
                                'blockchain': day_data['blockchain'].iloc[0] if 'blockchain' in day_data else chain,
                                'num_txs': int(day_data['num_txs'].iloc[0]) if 'num_txs' in day_data else 0,
                                'unique_senders': int(day_data['unique_senders'].iloc[0]) if 'unique_senders' in day_data else 0,
                                'unique_receivers': int(day_data['unique_receivers'].iloc[0]) if 'unique_receivers' in day_data else 0,
                                'success_rate': float(day_data['success_rate'].iloc[0]) if 'success_rate' in day_data else 0,
                                'avg_tx_value': avg_tx_value,
                                'avg_tx_value_change_pct': ((avg_tx_value - prev_day_value) / prev_day_value * 100) if prev_day_value else 0,
                                'avg_gas_used': float(day_data['avg_gas_used'].mean()) if 'avg_gas_used' in day_data else 0,
                                'avg_gas_price': float(day_data['avg_gas_price'].mean()) if 'avg_gas_price' in day_data else 0,
                                'smart_contract_calls': int(day_data['smart_contract_calls'].iloc[0]) if 'smart_contract_calls' in day_data else 0,
                                'bridge_volume': float(day_data['bridge_volume'].iloc[0]) if 'bridge_volume' in day_data else 0,
                                'total_volume': float(day_data['total_volume'].iloc[0]) if 'total_volume' in day_data else 0,
                                'txn_growth_pct_7d': float(day_data['txn_growth_pct_7d'].iloc[0]) if 'txn_growth_pct_7d' in day_data else 0,
                                'user_growth_pct_7d': float(day_data['user_growth_pct_7d'].iloc[0]) if 'user_growth_pct_7d' in day_data else 0,
                                'volume_growth_pct_7d': float(day_data['volume_growth_pct_7d'].iloc[0]) if 'volume_growth_pct_7d' in day_data else 0,
                                'tx_volatility_7d': float(day_data['tx_volatility_7d'].iloc[0]) if 'tx_volatility_7d' in day_data else 0
                            })
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    logger.warning(f"Attempt {attempt + 1} failed for {chain}: {str(e)}")
                    time.sleep(retry_delay)
                    
        except Exception as e:
            logger.error(f"Error collecting data for {chain}: {str(e)}")
            continue
            
    return market_data

def collect_protocol_data(
    client: FlipsideClient,
    protocols: list[str],
    start_date: datetime,
    end_date: datetime
) -> list[dict]:
    """Collect protocol data for multiple DeFi protocols.
    
    Args:
        client: Initialized Flipside client
        protocols: List of protocol names
        start_date: Start date for data collection
        end_date: End date for data collection
        
    Returns:
        List of protocol data dictionaries
    """
    protocol_data = []
    
    for protocol in protocols:
        logger.info(f"Collecting protocol data for {protocol}...")
        try:
            protocol_metrics = client.get_defi_metrics(
                protocol=protocol,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d")
            )
            
            # Group by day
            for date, day_data in protocol_metrics.groupby(pd.Grouper(key='block_timestamp', freq='D')):
                if not day_data.empty:
                    protocol_data.append({
                        'block_timestamp': str(date),
                        'network': day_data['network'].iloc[0] if 'network' in day_data else 'ethereum',
                        'blockchain': day_data['blockchain'].iloc[0] if 'blockchain' in day_data else 'ethereum',
                        'gas_used': day_data['gas_used'].mean() if 'gas_used' in day_data else None,
                        'size': day_data['size'].mean() if 'size' in day_data else None
                    })
                
        except Exception as e:
            logger.error(f"Error collecting data for {protocol}: {str(e)}")
            continue
            
    return protocol_data

def analyze_dataset_quality(dataset_path: str) -> Dict:
    """Analyze the quality of generated dataset.
    
    Args:
        dataset_path: Path to the generated dataset
        
    Returns:
        Dictionary of quality metrics
    """
    examples = []
    with open(dataset_path, 'r') as f:
        for line in f:
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    if not examples:
        return {
            'total_examples': 0,
            'prompt_type_distribution': {},
            'market_condition_distribution': {},
            'avg_reward': 0.0,
            'reward_std': 0.0,
            'avg_prediction_accuracy': 0.0,
            'avg_reasoning_score': 0.0,
            'prompt_type_rewards': {}
        }
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(examples)
    
    # Analyze prompt type distribution
    prompt_dist = df['type'].value_counts(normalize=True)
    
    # Analyze market conditions
    market_dist = df['market_condition'].value_counts(normalize=True)
    
    # Calculate quality metrics
    rewards = [ex['reward']['final_total'] for ex in examples if isinstance(ex.get('reward', {}), dict)]
    prediction_accuracies = [
        ex['reward'].get('prediction_accuracy', 0) 
        for ex in examples 
        if isinstance(ex.get('reward', {}), dict)
    ]
    reasoning_scores = [
        ex['reward'].get('reasoning_depth', 0) 
        for ex in examples
        if isinstance(ex.get('reward', {}), dict)
    ]
    
    # Group by prompt type and calculate rewards
    prompt_rewards = {}
    for prompt_type in df['type'].unique():
        type_examples = [
            ex for ex in examples 
            if ex['type'] == prompt_type and isinstance(ex.get('reward', {}), dict)
        ]
        if type_examples:
            rewards = [ex['reward']['final_total'] for ex in type_examples]
            prompt_rewards[prompt_type] = {
                'mean': float(np.mean(rewards)),
                'std': float(np.std(rewards))
            }
    
    quality_metrics = {
        'total_examples': len(examples),
        'prompt_type_distribution': prompt_dist.to_dict(),
        'market_condition_distribution': market_dist.to_dict(),
        'avg_reward': float(np.mean(rewards)) if rewards else 0.0,
        'reward_std': float(np.std(rewards)) if rewards else 0.0,
        'avg_prediction_accuracy': float(np.mean(prediction_accuracies)) if prediction_accuracies else 0.0,
        'avg_reasoning_score': float(np.mean(reasoning_scores)) if reasoning_scores else 0.0,
        'prompt_type_rewards': prompt_rewards
    }
    
    return quality_metrics

def generate_synthetic_data(
    market_data: list[dict],
    protocol_data: list[dict],
    num_samples: int,
    model: str
) -> list[dict]:
    """Generate synthetic training data using market and protocol data."""
    synthetic_data = []
    
    # Get market trends analysis
    market_trends = analyze_market_trends(market_data)
    
    # Define prompt types with weights based on market conditions
    prompt_types = {
        'prediction': 0.3,  # Higher weight for prediction tasks
        'analytical': 0.2,
        'correlation': 0.15,
        'market_qa': 0.15,
        'financial': 0.1,
        'protocol': 0.1
    }
    
    for _ in range(num_samples):
        # Select random market data point
        market_snapshot = random.choice(market_data)
        
        # Get market condition and relevant protocols
        condition = classify_market_condition(market_snapshot)
        relevant_protocols = [p for p in protocol_data if p['date'] == market_snapshot['block_timestamp']]
        
        # Select prompt type based on weights
        prompt_type = random.choices(
            list(prompt_types.keys()),
            weights=list(prompt_types.values())
        )[0]
        
        # Generate problem statement based on market condition and prompt type
        problem = generate_problem_statement(
            market_snapshot,
            condition,
            market_trends,
            relevant_protocols,
            prompt_type
        )
        
        # Generate synthetic example
        example = {
            'id': str(uuid.uuid4()),
            'timestamp': market_snapshot['block_timestamp'],
            'market_condition': condition,
            'prompt_type': prompt_type,
            'problem_statement': problem,
            'market_data': market_snapshot,
            'protocol_data': relevant_protocols,
            'market_trends': market_trends
        }
        
        synthetic_data.append(example)
    
    return synthetic_data

def generate_problem_statement(
    market_data: dict,
    condition: str,
    market_trends: dict,
    protocol_data: list[dict],
    prompt_type: str
) -> str:
    """Generate a problem statement based on market conditions and prompt type."""
    templates = {
        'prediction': [
            "Given the current {condition} market with {volatility_level} volatility and {growth_trajectory} growth, "
            "predict the likely changes in transaction volume and user growth over the next 24 hours. "
            "Consider the impact of {smart_contract_activity} smart contract calls and {bridge_volume} bridge volume.",
            
            "With {trend_strength:.0%} trend strength showing a {dominant_condition} market, "
            "forecast potential shifts in gas prices and transaction success rates. "
            "Account for the current {volatility_level} volatility environment."
        ],
        'analytical': [
            "Analyze how the {condition} market condition has affected user behavior, "
            "particularly in terms of transaction values (currently averaging {avg_tx_value:.2f}) "
            "and smart contract interactions ({smart_contract_calls} calls).",
            
            "Evaluate the market microstructure given {volatility_level} volatility and {growth_trajectory} growth. "
            "Focus on the relationship between transaction volume ({total_volume:.2f}) and unique users ({unique_senders})."
        ],
        'correlation': [
            "Investigate the correlation between bridge volume ({bridge_volume:.2f}) and "
            "smart contract activity ({smart_contract_calls} calls) in this {condition} market. "
            "Consider the impact on overall market stability.",
            
            "Examine the relationship between gas prices ({avg_gas_price:.2f}) and "
            "transaction success rates ({success_rate:.2%}) during this period of {volatility_level} volatility."
        ],
        'market_qa': [
            "What factors are driving the current {growth_trajectory} growth trajectory? "
            "Consider the {trend_strength:.0%} trend strength and {volatility_level} volatility level.",
            
            "How has the {condition} market condition influenced cross-chain activity, "
            "given the bridge volume of {bridge_volume:.2f} and {smart_contract_calls} smart contract calls?"
        ],
        'financial': [
            "Identify potential arbitrage opportunities given the current {condition} market, "
            "considering {bridge_volume:.2f} bridge volume and {total_volume:.2f} total volume. "
            "Factor in gas costs averaging {avg_gas_price:.2f}.",
            
            "Evaluate the cost-effectiveness of cross-chain transactions with current gas prices "
            "at {avg_gas_price:.2f} and a {success_rate:.2%} success rate. Consider the impact of "
            "{volatility_level} volatility."
        ],
        'protocol': [
            "Analyze protocol performance during this {condition} market phase. "
            "Focus on user adoption trends given {unique_senders} unique users and "
            "{smart_contract_calls} smart contract interactions.",
            
            "How have protocol metrics evolved with the current {growth_trajectory} growth trajectory? "
            "Consider changes in transaction volume ({total_volume:.2f}) and user activity ({unique_senders} unique users)."
        ]
    }
    
    # Select template based on prompt type
    template = random.choice(templates[prompt_type])
    
    # Format template with market data and trends
    problem = template.format(
        condition=condition,
        **market_data,
        **market_trends
    )
    
    return problem

def main():
    args = parse_args()
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    # Initialize clients
    flipside_client = FlipsideClient()
    synthetic_generator = SyntheticDataGenerator(model=args.model)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique timestamp for the output file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = output_dir / f"reasoning_data_{timestamp}.jsonl"
    
    # Collect data
    logger.info(f"Collecting {args.days} days of historical data...")
    market_data = []
    for blockchain in args.chains:
        logger.info(f"Collecting market data for {blockchain}...")
        try:
            chain_data = flipside_client.get_market_data(
                blockchain=blockchain,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d")
            )
            market_data.append(chain_data.to_dict('records'))
        except Exception as e:
            logger.error(f"Error collecting data for {blockchain}: {str(e)}")
    
    protocol_data = []
    for protocol in args.protocols:
        logger.info(f"Collecting protocol data for {protocol}...")
        try:
            protocol_metrics = flipside_client.get_defi_metrics(
                protocol=protocol,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d")
            )
            protocol_data.append(protocol_metrics.to_dict('records'))
        except Exception as e:
            logger.error(f"Error collecting data for {protocol}: {str(e)}")
    
    # Flatten data lists
    market_data = [item for sublist in market_data for item in sublist]
    protocol_data = [item for sublist in protocol_data for item in sublist]
    
    # Generate synthetic data
    logger.info("Generating synthetic training data...")
    synthetic_generator.generate_dataset(
        market_data=market_data,
        protocol_data=protocol_data,
        output_path=str(output_path),
        samples_per_prompt=args.samples_per_day
    )
    
    # Analyze dataset quality
    logger.info("Analyzing dataset quality...")
    quality_metrics = analyze_dataset_quality(str(output_path))
    
    # Save quality metrics
    metrics_path = output_dir / f"quality_metrics_{timestamp}.json"
    with open(metrics_path, 'w') as f:
        json.dump(quality_metrics, f, indent=2)
    
    logger.info(f"\nDataset Quality Metrics:\n{json.dumps(quality_metrics, indent=2)}")
    logger.info(f"\nDataset saved to: {output_path}")
    logger.info(f"Quality metrics saved to: {metrics_path}")

if __name__ == "__main__":
    main() 