#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prepare training dataset for GRPO training from existing synthetic and market analysis data.
This script converts pre-existing examples into the format needed for GRPO training.
"""

import os
import sys
import json
import logging
import argparse
import random
import datetime
from tqdm import tqdm
from typing import Dict, List, Any, Optional
import copy

# Add the src directory to the path so we can import modules from there
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import setup_logger

logger = setup_logger()

def load_jsonl_file(file_path: str) -> List[Dict]:
    """
    Load a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries containing the data
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def convert_synthetic_example(example: Dict) -> Dict:
    """
    Convert a synthetic example to the training format.
    
    Args:
        example: Raw synthetic example
        
    Returns:
        Converted example in training format
    """
    # Extract fields
    prompt = example.get('prompt', '')
    response = example.get('response', '')
    
    # Create metadata
    metadata = {
        "source": "synthetic_reasoning",
        "model": "deepseek-r1",
        "generation_date": datetime.datetime.now().isoformat(),
        "has_reasoning": True,
        "quality_score": example.get("reward", {}).get("final_total", 0.0),
        "word_count": len(response.split())
    }
    
    # Return in training format
    return {
        "input": prompt,
        "output": response,
        "reasoning": response,  # For these examples, the response includes the reasoning
        "metadata": metadata
    }

def convert_market_analysis(example: Dict) -> Dict:
    """
    Convert a market analysis example to the training format.
    
    Args:
        example: Raw market analysis example
        
    Returns:
        Converted example in training format
    """
    # Extract fields
    chain = example.get('chain', '')
    market_data = example.get('market_data', {})
    reasoning = example.get('reasoning', '')
    
    # Create input prompt using a template similar to the synthetic generator
    input_prompt = f"""You are analyzing {chain} blockchain data for {market_data.get('date', 'Unknown date')}. Think through the analysis step-by-step, focusing on calculations and reasoning.

### KEY METRICS (always cite as [metric_name])

1. Activity Metrics:
   - Daily Transactions: {market_data.get('daily_txns', 0):,} [daily_txns]
   - Unique Users: {market_data.get('unique_users', 0):,} [unique_users]
   - Success Rate: {market_data.get('success_rate', 0):.1f}% [success_rate]

2. Volume & Value:
   - Total Volume: {market_data.get('total_volume', 0):,.2f} [total_volume]
   - Avg Transaction: {market_data.get('avg_tx_value', 0):,.2f} [avg_tx_value]
   - Gas Used: {market_data.get('avg_gas', 0):,.0f} [gas_used]

3. Growth & Volatility:
   - Transaction Growth (7d): {market_data.get('txn_growth_pct_7d', 0):.1f}% [txn_growth]
   - User Growth (7d): {market_data.get('user_growth_pct_7d', 0):.1f}% [user_growth]
   - Volatility (7d): {market_data.get('tx_volatility_7d', 0):.2f} [volatility]

### ANALYSIS REQUIREMENTS

Please provide a detailed analysis with your thinking process exposed at every step:

1. Network Activity Analysis
2. Value Flow Analysis
3. Growth Pattern Analysis
4. Investment Implications

For each calculation, show your work step-by-step. Use [metric_name] format for EVERY data citation. Include confidence intervals (90% CI) for all projections."""

    # Extract summary output
    summary_lines = []
    in_summary = False
    
    for line in reasoning.split('\n'):
        if "Summary" in line or "Conclusion" in line:
            in_summary = True
            summary_lines.append(line)
            continue
            
        if in_summary and line.strip() and not line.startswith('#'):
            summary_lines.append(line)
        
        # Stop after collecting enough summary content
        if in_summary and len(summary_lines) > 10:
            break
    
    # If no summary section found, use the last few lines
    if not summary_lines:
        summary_lines = reasoning.split('\n')[-10:]
    
    output = '\n'.join(summary_lines)
    
    # Create metadata
    metadata = {
        "source": "market_analysis",
        "chain": chain,
        "date": market_data.get('date', datetime.datetime.now().isoformat()),
        "generation_date": example.get('timestamp', datetime.datetime.now().isoformat()),
        "has_reasoning": True,
        "word_count": len(output.split())
    }
    
    # Return in training format
    return {
        "input": input_prompt,
        "output": output,
        "reasoning": reasoning,
        "metadata": metadata
    }

def save_examples(examples: List[Dict], output_path: str, format_type: str = "jsonl") -> None:
    """
    Save examples to a file in the specified format.
    
    Args:
        examples: List of examples to save
        output_path: Path to save the examples
        format_type: Format to save the examples (jsonl or json)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if format_type == "jsonl":
        with open(output_path, "w") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")
    else:  # json
        with open(output_path, "w") as f:
            json.dump(examples, f, indent=2)
    
    logger.info(f"Saved {len(examples)} examples to {output_path}")

def duplicate_examples_with_variation(examples: List[Dict], target_count: int) -> List[Dict]:
    """
    Duplicate examples with slight variations to reach the target count.
    
    Args:
        examples: List of examples to duplicate
        target_count: Target number of examples
        
    Returns:
        List of examples with duplicates added to reach the target count
    """
    if len(examples) >= target_count:
        return examples[:target_count]
    
    # Calculate how many duplicates we need
    num_duplicates = target_count - len(examples)
    logger.info(f"Need {num_duplicates} more examples to reach target count of {target_count}")
    
    # Create duplicates with slight variations
    duplicated_examples = []
    original_examples = examples.copy()
    
    # Keep duplicating until we have enough
    while len(duplicated_examples) < num_duplicates:
        for example in original_examples:
            if len(duplicated_examples) >= num_duplicates:
                break
                
            # Create a copy with a slight variation
            duplicate = copy.deepcopy(example)
            
            # Add a small variation to the metadata to make it unique
            duplicate["metadata"]["duplication_id"] = len(duplicated_examples) + 1
            duplicate["metadata"]["generation_date"] = datetime.datetime.now().isoformat()
            
            duplicated_examples.append(duplicate)
    
    # Combine original and duplicated examples
    all_examples = examples + duplicated_examples
    logger.info(f"Created {len(duplicated_examples)} duplicated examples with variations")
    
    return all_examples[:target_count]

def main():
    parser = argparse.ArgumentParser(description="Prepare training dataset from existing synthetic data")
    parser.add_argument("--output-dir", type=str, default="data", help="Directory to save the generated dataset")
    parser.add_argument("--dataset-size", type=str, default="small", 
                        choices=["small", "medium", "large", "xlarge"],
                        help="Size of dataset to generate (affects number of examples)")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    parser.add_argument("--force-target-size", action="store_true", 
                       help="Force the dataset to reach the target size by duplicating examples if needed")
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    
    # Define dataset sizes
    size_mapping = {
        "small": 50,
        "medium": 200,
        "large": 500,
        "xlarge": 1000
    }
    n_examples = size_mapping[args.dataset_size]
    
    logger.info(f"Preparing {args.dataset_size} dataset with up to {n_examples} examples")
    
    # Create output directories
    raw_output_path = os.path.join(args.output_dir, "raw", "financial_analysis")
    training_output_path = os.path.join(args.output_dir, "training", "financial_analysis")
    
    os.makedirs(raw_output_path, exist_ok=True)
    os.makedirs(training_output_path, exist_ok=True)
    
    # Load existing examples
    logger.info("Loading existing synthetic examples")
    synthetic_examples = []
    
    # Load from data/synthetic/ directory
    synthetic_dir = os.path.join(args.output_dir, "synthetic")
    if os.path.exists(synthetic_dir):
        for file_name in os.listdir(synthetic_dir):
            if file_name.endswith('.jsonl'):
                file_path = os.path.join(synthetic_dir, file_name)
                logger.info(f"Loading {file_path}")
                synthetic_examples.extend(load_jsonl_file(file_path))
    
    # Load market analysis examples
    logger.info("Loading existing market analysis examples")
    market_examples = []
    
    # Load from data/market_analysis/ directory
    market_dir = os.path.join(args.output_dir, "market_analysis")
    if os.path.exists(market_dir):
        for file_name in os.listdir(market_dir):
            if file_name.endswith('.jsonl'):
                file_path = os.path.join(market_dir, file_name)
                logger.info(f"Loading {file_path}")
                market_examples.extend(load_jsonl_file(file_path))
    
    # Convert examples to the desired format
    logger.info("Converting examples to training format")
    converted_synthetic = [convert_synthetic_example(ex) for ex in tqdm(synthetic_examples, desc="Converting synthetic examples")]
    converted_market = [convert_market_analysis(ex) for ex in tqdm(market_examples, desc="Converting market analysis examples")]
    
    # Combine examples and limit to desired size
    all_examples = converted_synthetic + converted_market
    random.seed(42)  # For reproducibility
    random.shuffle(all_examples)
    
    # Check if we need to duplicate examples to reach the target count
    if args.force_target_size and len(all_examples) < n_examples:
        all_examples = duplicate_examples_with_variation(all_examples, n_examples)
    # Limit to desired size
    elif len(all_examples) > n_examples:
        all_examples = all_examples[:n_examples]
    
    logger.info(f"Created {len(all_examples)} examples from existing data")
    
    # Save the raw examples
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    raw_output_file = os.path.join(
        raw_output_path, 
        f"financial_reasoning_{args.dataset_size}_{timestamp}.jsonl"
    )
    
    save_examples(all_examples, raw_output_file)
    
    # Save the training examples (in this case, they're the same)
    training_output_file = os.path.join(
        training_output_path, 
        f"financial_reasoning_{args.dataset_size}_{timestamp}.jsonl"
    )
    
    save_examples(all_examples, training_output_file)
    
    # Create train/eval split (90/10)
    if len(all_examples) > 0:
        logger.info("Creating train/eval splits (90/10)")
        train_eval_dir = os.path.join(args.output_dir, "splits", "financial_analysis")
        os.makedirs(train_eval_dir, exist_ok=True)
        
        # Split into train/eval
        split_index = int(len(all_examples) * 0.9)
        train_examples = all_examples[:split_index]
        eval_examples = all_examples[split_index:]
        
        # Save train/eval splits
        train_file = os.path.join(train_eval_dir, f"train_{args.dataset_size}_{timestamp}.jsonl")
        eval_file = os.path.join(train_eval_dir, f"eval_{args.dataset_size}_{timestamp}.jsonl")
        
        save_examples(train_examples, train_file)
        save_examples(eval_examples, eval_file)
        
        logger.info(f"Created train/eval split: {len(train_examples)} train, {len(eval_examples)} eval")
    
    logger.info("=" * 50)
    logger.info("DATASET PREPARATION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Examples processed: {len(all_examples)}")
    logger.info(f"Dataset size: {args.dataset_size}")
    logger.info(f"Files saved:")
    logger.info(f"  Raw examples: {raw_output_file}")
    logger.info(f"  Training examples: {training_output_file}")
    logger.info(f"  Train split: {train_file}")
    logger.info(f"  Eval split: {eval_file}")
    logger.info("=" * 50)
    
    logger.info("Done!")

if __name__ == "__main__":
    main() 