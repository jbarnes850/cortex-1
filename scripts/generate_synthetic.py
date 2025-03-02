#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate synthetic training data using DeepSeek R1's reasoning capabilities.
This script integrates real Flipside market data with DeepSeek R1's reasoning
to create high-quality financial analysis examples for training models.
"""

import os
import sys
import json
import logging
import argparse
import time
from typing import Dict, List, Any, Optional
import random
import datetime
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the src directory to the path so we can import modules from there
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.synthetic_generator import SyntheticDataGenerator
from src.utils.logger import setup_logger

logger = setup_logger()

def verify_model_config() -> bool:
    """
    Verify that the OpenRouter model is set to DeepSeek R1.
    
    Returns:
        True if the model is set correctly, False otherwise
    """
    model = os.environ.get("OPENROUTER_MODEL", "")
    if "deepseek-r1" not in model.lower():
        logger.warning(f"OpenRouter model is set to '{model}', not DeepSeek R1.")
        logger.warning("DeepSeek R1 is recommended for high-quality reasoning generation.")
        response = input("Continue anyway? (y/n): ")
        return response.lower() == 'y'
    return True

def create_training_pair(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a training pair from a raw example.
    
    Args:
        example: A raw example from the dataset
        
    Returns:
        A training pair with input prompt and expected output
    """
    # Extract the input, reasoning, and output
    input_prompt = example.get("input", "")
    reasoning = example.get("reasoning", "")
    output = example.get("output", "")
    
    # Create metadata with source information
    metadata = {
        "source": "synthetic",
        "model": "deepseek-r1",
        "generation_date": datetime.datetime.now().isoformat(),
        "has_reasoning": bool(reasoning),
        "word_count": len(output.split())
    }
    
    # Create the training pair
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

def main():
    # Track overall execution time
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description="Generate synthetic training data using DeepSeek R1")
    parser.add_argument("--output-dir", type=str, default="data", help="Directory to save the generated dataset")
    parser.add_argument("--n-examples", type=int, default=10, help="Number of examples to generate")
    parser.add_argument("--chains", type=str, nargs="+", default=["market"], 
                        help="Types of chains to generate (e.g., market, token, protocol)")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    parser.add_argument("--verify-all", action="store_true", help="Verify all examples for quality")
    parser.add_argument("--dataset-size", type=str, default="medium", 
                        choices=["small", "medium", "large", "xlarge"],
                        help="Size of dataset to generate (affects number of examples)")
    parser.add_argument("--target-model", type=str, default="llama-3.3-70b", 
                        help="Target model for fine-tuning (affects prompt engineering)")
    parser.add_argument("--skip-quality-check", action="store_true", 
                        help="Skip quality verification (faster but may reduce quality)")
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    
    # Verify the model configuration
    logger.info("Verifying DeepSeek R1 model configuration...")
    if not verify_model_config():
        logger.error("Exiting because DeepSeek R1 model is not configured properly.")
        sys.exit(1)
    
    # Map dataset size to number of examples
    size_mapping = {
        "small": 50,
        "medium": 200,
        "large": 500,
        "xlarge": 1000
    }
    
    # Override n_examples if dataset_size is specified
    if args.dataset_size != "medium":
        args.n_examples = size_mapping[args.dataset_size]
        logger.info(f"Generating {args.n_examples} examples for {args.dataset_size} dataset")
    
    # Create the output directories
    raw_output_path = os.path.join(args.output_dir, "raw", "financial_analysis")
    training_output_path = os.path.join(args.output_dir, "training", "financial_analysis")
    
    os.makedirs(raw_output_path, exist_ok=True)
    os.makedirs(training_output_path, exist_ok=True)
    
    logger.info(f"Output directories created at {raw_output_path} and {training_output_path}")
    
    # Generate synthetic data
    logger.info(f"Generating {args.n_examples} synthetic examples with chains: {args.chains}")
    logger.info(f"Quality verification is {'enabled' if not args.skip_quality_check else 'disabled'}")
    
    # Initialize the synthetic data generator
    logger.info("Initializing DeepSeek R1 synthetic data generator...")
    generator_start = time.time()
    generator = SyntheticDataGenerator(
        verify_quality=not args.skip_quality_check,
        target_model=args.target_model
    )
    logger.info(f"Generator initialized in {time.time() - generator_start:.2f} seconds")
    
    # Generate the dataset
    logger.info("Starting data collection from Flipside and generating examples...")
    generation_start = time.time()
    raw_examples = generator.generate_dataset(
        n_examples=args.n_examples, 
        chains=args.chains
    )
    generation_time = time.time() - generation_start
    
    # Calculate statistics
    passed_examples = len(raw_examples)
    quality_scores = [ex.get('quality_score', 0) for ex in raw_examples if 'quality_score' in ex]
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
    
    logger.info(f"Dataset generation completed in {generation_time:.2f} seconds")
    logger.info(f"Generated {passed_examples} examples that passed quality checks")
    logger.info(f"Average quality score: {avg_quality:.4f}")
    
    if args.verify_all and quality_scores:
        logger.info("Quality score distribution:")
        score_ranges = {
            "0.0-0.2": 0, 
            "0.2-0.4": 0, 
            "0.4-0.6": 0, 
            "0.6-0.8": 0, 
            "0.8-1.0": 0
        }
        for score in quality_scores:
            if score < 0.2:
                score_ranges["0.0-0.2"] += 1
            elif score < 0.4:
                score_ranges["0.2-0.4"] += 1
            elif score < 0.6:
                score_ranges["0.4-0.6"] += 1
            elif score < 0.8:
                score_ranges["0.6-0.8"] += 1
            else:
                score_ranges["0.8-1.0"] += 1
                
        for range_name, count in score_ranges.items():
            percentage = (count / len(quality_scores)) * 100 if quality_scores else 0
            logger.info(f"  {range_name}: {count} examples ({percentage:.1f}%)")
    
    # Save the raw examples
    logger.info("Saving raw examples...")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    raw_output_file = os.path.join(
        raw_output_path, 
        f"synthetic_{args.dataset_size}_{len(args.chains)}_chains_{timestamp}.jsonl"
    )
    
    save_examples(raw_examples, raw_output_file)
    
    # Convert the raw examples to training pairs
    logger.info("Converting raw examples to training pairs...")
    training_examples = []
    
    for example in tqdm(raw_examples, desc="Converting to training pairs"):
        # Skip examples without reasoning
        if not example.get("reasoning"):
            logger.warning(f"Skipping example without reasoning")
            continue
            
        training_pair = create_training_pair(example)
        training_examples.append(training_pair)
    
    logger.info(f"Created {len(training_examples)} training pairs from {len(raw_examples)} raw examples")
    
    # Save the training examples
    logger.info("Saving training examples...")
    training_output_file = os.path.join(
        training_output_path, 
        f"synthetic_{args.dataset_size}_{len(args.chains)}_chains_{timestamp}.jsonl"
    )
    
    save_examples(training_examples, training_output_file)
    
    # Create train/eval split (90/10)
    if len(training_examples) > 0:
        logger.info("Creating train/eval splits (90/10)...")
        train_eval_dir = os.path.join(args.output_dir, "splits", "financial_analysis")
        os.makedirs(train_eval_dir, exist_ok=True)
        
        # Shuffle examples
        random.seed(42)  # For reproducibility
        random.shuffle(training_examples)
        
        # Split into train/eval
        split_index = int(len(training_examples) * 0.9)
        train_examples = training_examples[:split_index]
        eval_examples = training_examples[split_index:]
        
        # Save train/eval splits
        train_file = os.path.join(train_eval_dir, f"train_{args.dataset_size}_{timestamp}.jsonl")
        eval_file = os.path.join(train_eval_dir, f"eval_{args.dataset_size}_{timestamp}.jsonl")
        
        save_examples(train_examples, train_file)
        save_examples(eval_examples, eval_file)
        
        logger.info(f"Created train/eval split: {len(train_examples)} train, {len(eval_examples)} eval")
    
    # Calculate and display overall statistics
    total_time = time.time() - start_time
    avg_example_time = generation_time / max(1, len(raw_examples))
    
    logger.info("=" * 50)
    logger.info("DATASET GENERATION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total time elapsed: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    logger.info(f"Examples generated: {len(raw_examples)} of {args.n_examples} requested")
    logger.info(f"Chains used: {', '.join(args.chains)}")
    logger.info(f"Average time per example: {avg_example_time:.2f} seconds")
    logger.info(f"Average quality score: {avg_quality:.4f}")
    logger.info(f"Files saved:")
    logger.info(f"  Raw examples: {raw_output_file}")
    logger.info(f"  Training examples: {training_output_file}")
    if len(training_examples) > 0:
        logger.info(f"  Train split: {train_file}")
        logger.info(f"  Eval split: {eval_file}")
    logger.info("=" * 50)
    
    logger.info("Done!")

if __name__ == "__main__":
    main() 