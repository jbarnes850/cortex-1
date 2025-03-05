#!/usr/bin/env python3

"""
Combine historical and real-time datasets for NEAR Cortex-1 training.
This script implements Phase 3 of the hybrid dataset implementation plan,
merging historical data with real-time synthetic examples.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Any
import random
from tqdm import tqdm
import math

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import setup_logger

logger = setup_logger()

def calculate_temporal_weight(timestamp: str, current_time: datetime, alpha: float = 0.7) -> float:
    """
    Calculate weight based on temporal distance from current time.
    
    Args:
        timestamp: ISO format timestamp
        current_time: Current reference time
        alpha: Decay factor (higher values prioritize recent data more)
        
    Returns:
        float: Weight between 0 and 1
    """
    try:
        ts = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        time_diff = (current_time - ts).total_seconds() / (3600 * 24)  # days
        weight = math.exp(-alpha * time_diff / 365)  # Decay over a year
        return weight
    except Exception as e:
        logger.warning(f"Error calculating temporal weight for {timestamp}: {str(e)}")
        return 0.0

def load_jsonl(file_path: str) -> List[Dict]:
    """Load examples from a JSONL file."""
    examples = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                examples.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                logger.warning(f"Error parsing line in {file_path}: {str(e)}")
                continue
    return examples

def save_jsonl(examples: List[Dict], file_path: str) -> None:
    """Save examples to a JSONL file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    logger.info(f"Saved {len(examples)} examples to {file_path}")

def update_metadata(example: Dict[str, Any], source: str, timestamp: str = None) -> Dict[str, Any]:
    """Update example metadata with source and temporal information."""
    if 'metadata' not in example:
        example['metadata'] = {}
    
    example['metadata'].update({
        'source': source,
        'temporal_tag': 'historical' if source == 'historical' else 'current',
        'generation_date': timestamp or datetime.now().isoformat()
    })
    return example

def combine_datasets(
    historical_path: str,
    realtime_path: str,
    output_dir: str,
    train_split: float = 0.9,
    alpha: float = 0.7
) -> None:
    """
    Combine historical and real-time datasets with temporal weighting.
    
    Args:
        historical_path: Path to historical dataset directory
        realtime_path: Path to real-time dataset directory
        output_dir: Output directory for combined dataset
        train_split: Proportion of data for training (default: 0.9)
        alpha: Temporal weighting factor (default: 0.7)
    """
    current_time = datetime.now()
    
    # Load historical data
    logger.info("Loading historical dataset...")
    historical_train = load_jsonl(os.path.join(historical_path, 'train_large.jsonl'))
    historical_eval = load_jsonl(os.path.join(historical_path, 'eval_large.jsonl'))
    
    # Load real-time data
    logger.info("Loading real-time dataset...")
    realtime_file = None
    for file in os.listdir(os.path.join(realtime_path, 'training', 'financial_analysis')):
        if file.endswith('.jsonl'):
            realtime_file = os.path.join(realtime_path, 'training', 'financial_analysis', file)
            break
    
    if not realtime_file:
        raise FileNotFoundError("No real-time dataset found")
    
    realtime_data = load_jsonl(realtime_file)
    
    # Update metadata and calculate weights
    logger.info("Updating metadata and calculating temporal weights...")
    weighted_examples = []
    
    # Process historical data
    for example in tqdm(historical_train + historical_eval, desc="Processing historical data"):
        example = update_metadata(example, 'historical', example.get('metadata', {}).get('generation_date'))
        timestamp = example['metadata']['generation_date']
        weight = calculate_temporal_weight(timestamp, current_time, alpha)
        weighted_examples.append((example, weight))
    
    # Process real-time data
    for example in tqdm(realtime_data, desc="Processing real-time data"):
        example = update_metadata(example, 'realtime')
        timestamp = example['metadata']['generation_date']
        weight = calculate_temporal_weight(timestamp, current_time, alpha)
        weighted_examples.append((example, weight))
    
    # Normalize weights
    total_weight = sum(w for _, w in weighted_examples)
    if total_weight > 0:
        weighted_examples = [(ex, w/total_weight) for ex, w in weighted_examples]
    
    # Sort by weight for weighted sampling
    weighted_examples.sort(key=lambda x: x[1], reverse=True)
    
    # Create train/eval splits
    logger.info("Creating train/eval splits...")
    random.seed(42)  # For reproducibility
    
    # Calculate split index
    total_examples = len(weighted_examples)
    train_size = int(total_examples * train_split)
    
    # Split examples
    train_examples = [ex for ex, _ in weighted_examples[:train_size]]
    eval_examples = [ex for ex, _ in weighted_examples[train_size:]]
    
    # Save combined datasets
    logger.info("Saving combined datasets...")
    output_train = os.path.join(output_dir, 'train_combined.jsonl')
    output_eval = os.path.join(output_dir, 'eval_combined.jsonl')
    
    save_jsonl(train_examples, output_train)
    save_jsonl(eval_examples, output_eval)
    
    # Save dataset statistics
    stats = {
        'total_examples': total_examples,
        'train_examples': len(train_examples),
        'eval_examples': len(eval_examples),
        'historical_ratio': len(historical_train + historical_eval) / total_examples,
        'realtime_ratio': len(realtime_data) / total_examples,
        'temporal_weight_alpha': alpha,
        'generation_timestamp': current_time.isoformat(),
        'data_sources': {
            'historical': historical_path,
            'realtime': realtime_file
        }
    }
    
    with open(os.path.join(output_dir, 'dataset_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info("Dataset combination complete!")
    logger.info(f"Total examples: {total_examples}")
    logger.info(f"Training examples: {len(train_examples)}")
    logger.info(f"Evaluation examples: {len(eval_examples)}")
    logger.info(f"Historical ratio: {stats['historical_ratio']:.2%}")
    logger.info(f"Real-time ratio: {stats['realtime_ratio']:.2%}")

def main():
    parser = argparse.ArgumentParser(description="Combine historical and real-time datasets")
    parser.add_argument("--historical-dir", type=str, default="data/deepseek_ready",
                        help="Directory containing historical dataset")
    parser.add_argument("--realtime-dir", type=str, default="data/synthetic/hybrid",
                        help="Directory containing real-time dataset")
    parser.add_argument("--output-dir", type=str, default="data/training/combined",
                        help="Output directory for combined dataset")
    parser.add_argument("--train-split", type=float, default=0.9,
                        help="Proportion of data for training")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Temporal weighting factor")
    
    args = parser.parse_args()
    
    try:
        combine_datasets(
            args.historical_dir,
            args.realtime_dir,
            args.output_dir,
            args.train_split,
            args.alpha
        )
    except Exception as e:
        logger.error(f"Error combining datasets: {str(e)}")
        raise

if __name__ == "__main__":
    main() 