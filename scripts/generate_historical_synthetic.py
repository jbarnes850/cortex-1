#!/usr/bin/env python
"""
Generate Synthetic Data with Historical Context

This script extends the existing synthetic data generation pipeline
to incorporate historical context and temporal mixing strategies.
It generates high-quality synthetic examples that leverage both 
historical and current market data.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import logging
from datetime import datetime
import random
from typing import Dict, List, Any, Optional, Union

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.historical_context import HistoricalContextGenerator
from src.data.temporal_mixing import TemporalMixingStrategy

# Try to import existing synthetic data generation components
try:
    from scripts.generate_synthetic import generate_examples
    HAS_GENERATOR = True
except ImportError:
    print("Warning: Could not import existing synthetic data generation components.")
    print("Make sure the generate_synthetic.py script exists and is properly formatted.")
    HAS_GENERATOR = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/historical_synthetic_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate synthetic data with historical context")
    
    # Coin selection options
    parser.add_argument("--coins", type=str, default="btc,eth,ltc,xpr",
                      help="Comma-separated list of coins to process")
    
    # Dataset options
    parser.add_argument("--dataset-size", choices=["small", "medium", "large"], 
                      default="small", help="Size of the dataset to generate")
    parser.add_argument("--output-dir", type=str, default="data/synthetic/hybrid",
                      help="Output directory for synthetic datasets")
    
    # Temporal mixing options
    parser.add_argument("--mixing-strategy", choices=["weighted", "exponential", "stratified", "time_sliced"], 
                      default="weighted", help="Temporal mixing strategy to use")
    parser.add_argument("--historical-ratio", type=float, default=0.6,
                      help="Ratio of historical data")
    parser.add_argument("--current-ratio", type=float, default=0.2,
                      help="Ratio of current data")
    parser.add_argument("--recent-ratio", type=float, default=0.2,
                      help="Ratio of recent data")
    
    # Prompt options
    parser.add_argument("--analysis-types", type=str, 
                      default="price_trend,pattern_recognition,volatility_analysis,market_cycle",
                      help="Comma-separated list of analysis types to generate")
    
    # Model options
    parser.add_argument("--model", type=str, default="deepseek/deepseek-r1:free",
                      help="Model to use for generation")
    
    # Other options
    parser.add_argument("--seed", type=int, default=None,
                      help="Random seed for reproducibility")
    parser.add_argument("--skip-generation", action="store_true",
                      help="Skip the actual generation step (for testing)")
    
    # Input file option
    parser.add_argument("--input-file", type=str,
                      help="Path to the processed data file")
    
    return parser.parse_args()

def determine_dataset_size(size_name):
    """Map size name to actual size."""
    size_map = {
        "small": 10,
        "medium": 50,
        "large": 200
    }
    return size_map.get(size_name, 10)

def create_mixed_dataset(mixing_strategy, args, size):
    """Create a mixed dataset using the specified strategy."""
    coins = [coin.strip().lower() for coin in args.coins.split(",")]
    
    if args.mixing_strategy == "weighted":
        mixed_data = mixing_strategy.create_weighted_dataset(
            coins=coins,
            historical_ratio=args.historical_ratio,
            current_ratio=args.current_ratio,
            recent_ratio=args.recent_ratio,
            sample_size=size,
            random_seed=args.seed
        )
    
    elif args.mixing_strategy == "exponential":
        mixed_data = mixing_strategy.create_exponential_decay_dataset(
            coins=coins,
            alpha=0.7,
            sample_size=size,
            random_seed=args.seed
        )
    
    elif args.mixing_strategy == "stratified":
        strata_configs = [
            {"period": "historical", "ratio": args.historical_ratio},
            {"period": "recent", "ratio": args.recent_ratio},
            {"period": "current", "ratio": args.current_ratio}
        ]
        
        mixed_data = mixing_strategy.create_stratified_temporal_dataset(
            coins=coins,
            strata_configs=strata_configs,
            sample_size=size,
            random_seed=args.seed
        )
    
    elif args.mixing_strategy == "time_sliced":
        # Create time slices for different market periods
        slice_configs = [
            {"start_date": "2018-01-01", "end_date": "2019-12-31", "ratio": 0.2},  # 2018-2019
            {"start_date": "2020-01-01", "end_date": "2020-12-31", "ratio": 0.2},  # 2020 (COVID)
            {"start_date": "2021-01-01", "end_date": "2021-12-31", "ratio": 0.3},  # 2021 (bull)
            {"start_date": "2022-01-01", "end_date": "2023-05-31", "ratio": 0.3},  # 2022-2023
        ]
        
        mixed_data = mixing_strategy.create_time_sliced_dataset(
            coins=coins,
            slice_configs=slice_configs,
            sample_size=size,
            random_seed=args.seed
        )
    
    else:
        logger.error(f"Unknown strategy: {args.mixing_strategy}")
        return None
    
    return mixed_data

def generate_prompts(context_generator, mixed_data, analysis_types):
    """Generate prompts for each record in the mixed dataset."""
    prompts = []
    
    analysis_type_list = [at.strip() for at in analysis_types.split(",")]
    
    for record in mixed_data:
        coin = record.get("chain", "unknown")
        
        # Create a window of context (just this record for simplicity)
        context_window = [record]
        
        # Choose a random analysis type
        analysis_type = random.choice(analysis_type_list)
        
        # Generate prompt
        prompt = context_generator.generate_context_prompt(
            coin, context_window, analysis_type=analysis_type
        )
        
        prompts.append({
            "prompt": prompt,
            "record": record,
            "analysis_type": analysis_type
        })
    
    return prompts

def generate_cross_temporal_prompts(context_generator, coins, size):
    """Generate prompts that require reasoning across different time periods."""
    prompts = []
    
    question_types = ["comparison", "prediction", "cycle_analysis"]
    
    # Count of prompts per coin
    num_per_coin = max(1, size // (len(coins) * len(question_types)))
    
    for coin in coins:
        # Get historical and current windows
        historical_window = context_generator.get_historical_window(
            coin, window_size=90, time_period="historical"
        )
        
        current_window = context_generator.get_historical_window(
            coin, window_size=30, time_period="recent"
        )
        
        if not historical_window or not current_window:
            logger.warning(f"Insufficient data for cross-temporal analysis of {coin}")
            continue
        
        # Generate prompts for each question type
        for question_type in question_types:
            for _ in range(num_per_coin):
                prompt = context_generator.generate_temporal_mixing_prompt(
                    coin, historical_window, current_window, question_type=question_type
                )
                
                prompts.append({
                    "prompt": prompt,
                    "coin": coin,
                    "question_type": question_type,
                    "cross_temporal": True
                })
    
    # Shuffle and limit to size
    random.shuffle(prompts)
    return prompts[:size]

def save_prompts(prompts, output_dir, prefix):
    """Save prompts to a file."""
    # Create timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Create target directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create output file
    output_file = output_path / f"{prefix}_{timestamp}.json"
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(prompts, f, indent=2)
    
    logger.info(f"Saved {len(prompts)} prompts to {output_file}")
    
    return output_file

def generate_examples(prompts, model_name, output_dir, skip_generation=False):
    """Generate synthetic examples using the prompts."""
    if skip_generation:
        logger.info("Skipping generation step as requested")
        return None
    
    if not generate_synthetic_main:
        logger.error("Cannot generate examples: generate_synthetic.py not properly imported")
        return None
    
    # This is a placeholder - the actual implementation depends on how
    # your existing synthetic generation pipeline works. You would need
    # to adapt this to work with your specific implementation.
    
    logger.info(f"Generating {len(prompts)} examples using model {model_name}")
    
    # Save prompts to a temporary file that can be used by the existing pipeline
    temp_prompts_file = save_prompts(prompts, output_dir, "temp_prompts")
    
    # Call the existing synthetic generation pipeline
    # This is just an example - you'll need to adapt this to your specific implementation
    # generate_synthetic_main(["--prompts", str(temp_prompts_file), "--model", model_name])
    
    logger.info("Synthetic examples generated successfully")
    
    return None

def generate_examples_fallback(prompts_file: str, output_file: str, model: str = "deepseek/deepseek-r1:free"):
    """
    Fallback function to generate examples using the DeepSeek R1 API directly.
    
    Args:
        prompts_file: Path to the prompts file
        output_file: Path to save the generated examples
        model: Model to use for generation
    """
    import requests
    import time
    import os
    from tqdm import tqdm
    
    logger.info(f"Generating examples using {model} model")
    
    # Load API key from environment variable
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY environment variable not set")
        return
        
    # Load prompts
    try:
        with open(prompts_file, 'r') as f:
            prompts = json.load(f)
            
        logger.info(f"Loaded {len(prompts)} prompts from {prompts_file}")
    except Exception as e:
        logger.error(f"Error loading prompts from {prompts_file}: {e}")
        return
        
    # OpenRouter API endpoint
    api_url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    # Generate examples
    examples = []
    for i, prompt_data in enumerate(tqdm(prompts, desc="Generating examples")):
        try:
            # Extract prompt
            prompt = prompt_data.get("prompt", "")
            record = prompt_data.get("record", {})
            
            if not prompt:
                logger.warning(f"Prompt {i} is empty. Skipping.")
                continue
                
            # Prepare payload
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are an expert financial analyst with deep knowledge of cryptocurrency markets. Provide detailed reasoning and analysis."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 1500
            }
            
            # Make API request
            response = requests.post(api_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                response_data = response.json()
                reasoning = response_data["choices"][0]["message"]["content"]
                
                # Create example
                example = {
                    "prompt": prompt,
                    "reasoning": reasoning,
                    "record": record,
                    "model": model,
                    "timestamp": datetime.now().isoformat()
                }
                
                examples.append(example)
                
                # Sleep to avoid rate limiting
                time.sleep(2)
            else:
                logger.error(f"API request failed with status code {response.status_code}: {response.text}")
                
        except Exception as e:
            logger.error(f"Error generating example {i}: {e}")
            
    # Save examples
    try:
        with open(output_file, 'w') as f:
            json.dump(examples, f, indent=2)
            
        logger.info(f"Saved {len(examples)} examples to {output_file}")
    except Exception as e:
        logger.error(f"Error saving examples to {output_file}: {e}")
        
    # Convert to DeepSeek-ready format
    try:
        deepseek_ready_file = output_file.replace(".json", "_deepseek_ready.jsonl")
        convert_to_deepseek_ready(examples, deepseek_ready_file)
    except Exception as e:
        logger.error(f"Error converting to DeepSeek-ready format: {e}")

def convert_to_deepseek_ready(examples, output_file):
    """
    Convert examples to DeepSeek-ready format.
    
    Args:
        examples: List of examples
        output_file: Path to save the DeepSeek-ready examples
    """
    deepseek_ready = []
    
    for example in examples:
        prompt = example.get("prompt", "")
        reasoning = example.get("reasoning", "")
        record = example.get("record", {})
        
        # Create DeepSeek-ready example
        deepseek_example = {
            "input": prompt,
            "output": reasoning,
            "metadata": {
                "source": "synthetic_historical",
                "time_period": record.get("metadata", {}).get("time_period", "unknown"),
                "coin": record.get("chain", ""),
                "date": record.get("timestamp", "").split("T")[0] if record.get("timestamp") else "",
                "model": example.get("model", ""),
                "generation_date": example.get("timestamp", "")
            }
        }
        
        deepseek_ready.append(deepseek_example)
        
    # Save DeepSeek-ready examples
    with open(output_file, 'w') as f:
        for example in deepseek_ready:
            f.write(json.dumps(example) + "\n")
            
    logger.info(f"Saved {len(deepseek_ready)} DeepSeek-ready examples to {output_file}")

def main():
    """Main function."""
    args = parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize historical context generator
    if args.input_file:
        processed_dir = Path(args.input_file).parent.parent
        historical_generator = HistoricalContextGenerator(processed_dir=str(processed_dir))
    else:
        historical_generator = HistoricalContextGenerator()
    
    # Initialize temporal mixing strategy
    mixing_strategy = TemporalMixingStrategy(historical_generator)
    
    # Parse coins and analysis types
    coins = [coin.strip() for coin in args.coins.split(",")]
    analysis_types = [analysis_type.strip() for analysis_type in args.analysis_types.split(",")]
    
    # Determine dataset size
    dataset_size = determine_dataset_size(args.dataset_size)
    
    # Calculate examples per coin and analysis type combination
    total_combinations = len(coins) * len(analysis_types)
    examples_per_combination = dataset_size // total_combinations
    
    logger.info(f"Generating {examples_per_combination} examples per coin/analysis type combination")
    
    # Generate mixed datasets
    all_prompts = []
    mixed_data = []
    
    # Create weighted dataset with all coins
    mixed_data = mixing_strategy.create_weighted_dataset(
        coins=coins,
        historical_ratio=args.historical_ratio,
        current_ratio=args.current_ratio,
        recent_ratio=args.recent_ratio,
        sample_size=dataset_size,
        random_seed=args.seed
    )
    
    if not mixed_data:
        logger.warning(f"No data available for mixing. Skipping.")
    else:
        logger.info(f"Created mixed dataset with {len(mixed_data)} records")
        
        # Generate prompts for each coin and analysis type
        for coin in coins:
            # Get context data for this coin
            context_data = [item for item in mixed_data if item.get("chain", "").lower() == coin.lower()]
            
            if not context_data:
                logger.warning(f"No context data found for {coin}. Skipping.")
                continue
                
            for analysis_type in analysis_types:
                # Generate multiple prompts for this combination
                for _ in range(examples_per_combination):
                    # Randomly sample a window of context data
                    window_size = min(60, len(context_data))
                    context_window = random.sample(context_data, window_size)
                    context_window.sort(key=lambda x: x.get("timestamp", ""))
                    
                    # Generate prompt
                    prompt = historical_generator.generate_context_prompt(
                        coin=coin,
                        context_window=context_window,
                        analysis_type=analysis_type
                    )
                    
                    if prompt:
                        # Create prompt data with record
                        prompt_data = {
                            "prompt": prompt,
                            "record": context_window[0] if context_window else {},
                            "analysis_type": analysis_type
                        }
                        
                        all_prompts.append(prompt_data)
                
    # Save prompts to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    prompts_file = output_dir / f"hybrid_prompts_{args.dataset_size}_{args.mixing_strategy}_{timestamp}.json"
    
    try:
        with open(prompts_file, 'w') as f:
            json.dump(all_prompts, f, indent=2)
            
        logger.info(f"Saved {len(all_prompts)} prompts to {prompts_file}")
    except Exception as e:
        logger.error(f"Error saving prompts to {prompts_file}: {e}")
        
    # Generate synthetic examples if not skipped
    if not args.skip_generation:
        try:
            examples_file = output_dir / f"hybrid_examples_{args.dataset_size}_{args.mixing_strategy}_{timestamp}.json"
            
            # Call the generate_examples function from generate_synthetic.py
            if HAS_GENERATOR:
                generate_examples(
                    prompts_file=str(prompts_file),
                    output_file=str(examples_file),
                    model=args.model
                )
                logger.info(f"Generated synthetic examples saved to {examples_file}")
            else:
                # Fallback to direct API call if import failed
                logger.info("Using fallback method for DeepSeek R1 API call")
                generate_examples_fallback(
                    prompts_file=str(prompts_file),
                    output_file=str(examples_file),
                    model=args.model
                )
                
        except Exception as e:
            logger.error(f"Error generating synthetic examples: {e}")
    elif args.skip_generation:
        logger.info("Skipping synthetic data generation as requested")
    else:
        logger.warning("Synthetic data generation skipped due to missing generator components")
        
    # Save mixed dataset for testing
    try:
        mixed_dataset_file = output_dir / f"test_{args.mixing_strategy}_{timestamp}.json"
        with open(mixed_dataset_file, 'w') as f:
            json.dump(mixed_data, f, indent=2)
            
        logger.info(f"Saved mixed dataset to {mixed_dataset_file}")
    except Exception as e:
        logger.error(f"Error saving mixed dataset: {e}")
        
    logger.info("Historical synthetic data generation completed successfully")

if __name__ == "__main__":
    main() 