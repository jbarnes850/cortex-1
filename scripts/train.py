#!/usr/bin/env python
"""
Training script for the NEAR Cortex-1 crypto analysis model.
"""

import os
import argparse
from datetime import datetime, timedelta
import logging
from pathlib import Path

from src.data.flipside_client import FlipsideClient
from src.data.synthetic_generator import SyntheticDataGenerator
from src.model.grpo_trainer import CryptoGRPOTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train the NEAR Cortex-1 model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--data-days",
        type=int,
        default=90,
        help="Number of days of historical data to collect"
    )
    parser.add_argument(
        "--synthetic-samples-per-day",
        type=int,
        default=5,
        help="Number of synthetic examples to generate per day"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save outputs"
    )
    return parser.parse_args()

def collect_training_data(
    flipside_client: FlipsideClient,
    days: int,
    output_dir: Path
) -> tuple[Path, Path]:
    """Collect market and protocol data from Flipside.
    
    Args:
        flipside_client: Initialized Flipside client
        days: Number of days of historical data
        output_dir: Output directory for data
        
    Returns:
        Tuple of (market_data_path, protocol_data_path)
    """
    logger.info(f"Collecting {days} days of historical data from Flipside...")
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Collect market data for major chains
    chains = ["ethereum", "near", "solana", "avalanche"]
    market_data = []
    
    for chain in chains:
        logger.info(f"Collecting market data for {chain}...")
        chain_data = flipside_client.get_market_data(
            chain=chain,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )
        market_data.append(chain_data)
    
    # Collect protocol data
    protocols = ["uniswap", "aave", "curve"]
    protocol_data = []
    
    for protocol in protocols:
        logger.info(f"Collecting protocol data for {protocol}...")
        protocol_metrics = flipside_client.get_defi_metrics(
            protocol=protocol,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )
        protocol_data.append(protocol_metrics)
    
    # Save data
    market_data_path = output_dir / "market_data.csv"
    protocol_data_path = output_dir / "protocol_data.csv"
    
    import pandas as pd
    pd.concat(market_data).to_csv(market_data_path, index=False)
    pd.concat(protocol_data).to_csv(protocol_data_path, index=False)
    
    logger.info(f"Saved market data to {market_data_path}")
    logger.info(f"Saved protocol data to {protocol_data_path}")
    
    return market_data_path, protocol_data_path

def generate_synthetic_data(
    generator: SyntheticDataGenerator,
    market_data_path: Path,
    protocol_data_path: Path,
    samples_per_day: int,
    output_path: Path
) -> Path:
    """Generate synthetic training data with chain-of-thought reasoning.
    
    Args:
        generator: Initialized synthetic data generator
        market_data_path: Path to market data CSV
        protocol_data_path: Path to protocol data CSV
        samples_per_day: Number of synthetic examples per day
        output_path: Path to save synthetic data
        
    Returns:
        Path to generated synthetic data
    """
    logger.info("Generating synthetic training data...")
    
    import pandas as pd
    market_df = pd.read_csv(market_data_path)
    protocol_df = pd.read_csv(protocol_data_path)
    
    generator.augment_flipside_data(
        market_df=market_df,
        protocol_df=protocol_df,
        output_path=output_path,
        samples_per_day=samples_per_day
    )
    
    logger.info(f"Saved synthetic data to {output_path}")
    return output_path

def main():
    args = parse_args()
    
    # Create output directories
    output_dir = Path(args.output_dir)
    data_dir = output_dir / "data"
    model_dir = output_dir / "model"
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize clients
    flipside_client = FlipsideClient()
    synthetic_generator = SyntheticDataGenerator()
    
    # Collect training data
    market_data_path, protocol_data_path = collect_training_data(
        flipside_client=flipside_client,
        days=args.data_days,
        output_dir=data_dir
    )
    
    # Generate synthetic data
    synthetic_data_path = data_dir / "synthetic_data.jsonl"
    synthetic_data_path = generate_synthetic_data(
        generator=synthetic_generator,
        market_data_path=market_data_path,
        protocol_data_path=protocol_data_path,
        samples_per_day=args.synthetic_samples_per_day,
        output_path=synthetic_data_path
    )
    
    # Initialize and train model
    logger.info("Initializing GRPO trainer...")
    trainer = CryptoGRPOTrainer(config_path=args.config)
    
    logger.info("Starting model training...")
    trainer.train(
        train_dataset=str(synthetic_data_path),
        eval_dataset=None  # Could split dataset for eval
    )
    
    logger.info(f"Training complete! Model saved to {model_dir}")

if __name__ == "__main__":
    main() 