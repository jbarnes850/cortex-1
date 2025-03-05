#!/usr/bin/env python
"""
Historical Data Processor for NEAR Cortex-1

This script downloads historical cryptocurrency data from Kaggle,
normalizes it to the standard schema, applies temporal tagging,
and organizes it according to the data_plan.md specification.

Usage:
    python scripts/process_historical_data.py --coins BTC,ETH,XPR
    python scripts/process_historical_data.py --all-coins
"""

import os
import sys
import json
import argparse
from pathlib import Path
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.kaggle_loader import KaggleDatasetLoader
from src.data.data_normalizer import DataNormalizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/historical_data_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process historical cryptocurrency data from Kaggle")
    
    # Coin selection options
    coin_group = parser.add_mutually_exclusive_group()
    coin_group.add_argument("--coins", type=str, help="Comma-separated list of coins to process (e.g., BTC,ETH,XPR)")
    coin_group.add_argument("--all-coins", action="store_true", help="Process all available coins")
    
    # Processing options
    parser.add_argument("--force-download", action="store_true", 
                      help="Force download even if data exists locally")
    parser.add_argument("--skip-derived-metrics", action="store_true",
                      help="Skip calculation of derived metrics")
    parser.add_argument("--output-dir", type=str, default="data/processed",
                      help="Output directory for processed data")
    
    return parser.parse_args()

def create_directory_structure():
    """Create the directory structure according to the data plan."""
    # Define the directories to create
    directories = [
        "data/raw/kaggle/price_analysis",
        "data/raw/flipside",
        "data/processed/historical",
        "data/processed/current",
        "data/processed/merged",
        "data/synthetic/historical_based",
        "data/synthetic/current_based", 
        "data/synthetic/hybrid",
        "data/splits/base_training",
        "data/splits/fine_tuning",
        "data/splits/evaluation",
        "logs"
    ]
    
    # Create each directory
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def process_price_analysis_dataset(coins: Optional[List[str]], 
                                  force_download: bool, 
                                  skip_derived_metrics: bool,
                                  output_dir: str) -> List[Path]:
    """
    Process the price analysis dataset.
    
    Args:
        coins: List of specific coins to process (None for all)
        force_download: Whether to force download even if data exists
        skip_derived_metrics: Whether to skip calculation of derived metrics
        output_dir: Directory to save processed data
        
    Returns:
        List of paths to the processed files
    """
    # Initialize components
    loader = KaggleDatasetLoader(download_dir=f"data/raw/kaggle")
    normalizer = DataNormalizer(processed_dir=output_dir)
    
    # Download dataset if needed or forced
    if force_download:
        logger.info(f"Forcing download of price_analysis dataset")
        loader.download_dataset("price_analysis")
    
    # Load the data
    logger.info(f"Loading price_analysis dataset")
    data = loader.load_price_analysis_data(download_if_missing=True)
    
    logger.info(f"Loaded {len(data)} coins from price_analysis")
    
    # Filter coins if specified
    if coins:
        available_coins = set(data.keys())
        requested_coins = set(coins)
        valid_coins = list(requested_coins.intersection(available_coins))
        
        if not valid_coins:
            logger.warning(f"None of the requested coins {coins} are available in price_analysis")
            return []
        
        # Filter data to only include requested coins
        data = {coin: df for coin, df in data.items() if coin in valid_coins}
        logger.info(f"Filtered to {len(data)} requested coins: {', '.join(data.keys())}")
    
    # Process each coin
    processed_files = []
    for coin, df in data.items():
        try:
            logger.info(f"Processing {coin} from price_analysis ({len(df)} records)")
            
            # Normalize data
            normalized_records = normalizer.normalize_kaggle_price_analysis(df, coin)
            
            # Add temporal tagging
            for record in normalized_records:
                # Set time period based on timestamp
                timestamp = record.get("timestamp", "")
                record_date = datetime.fromisoformat(timestamp.split("T")[0] if "T" in timestamp else timestamp)
                current_date = datetime.now()
                
                # Calculate days difference
                days_diff = (current_date - record_date).days
                
                # Assign time period based on recency
                if days_diff < 30:
                    time_period = "current"
                elif days_diff < 365:
                    time_period = "recent"
                else:
                    time_period = "historical"
                
                # Update metadata
                record["metadata"]["time_period"] = time_period
            
            # Save normalized data
            file_path = normalizer.save_normalized_data(
                normalized_records, 
                'historical', 
                f"price_analysis_{coin}"
            )
            
            # Calculate derived metrics if not skipped
            if not skip_derived_metrics:
                logger.info(f"Calculating derived metrics for {coin}")
                file_path = normalizer.calculate_derived_metrics(file_path)
            
            processed_files.append(file_path)
            logger.info(f"Successfully processed {coin} ({len(normalized_records)} records)")
            
        except Exception as e:
            logger.error(f"Error processing {coin} from price_analysis: {e}")
    
    return processed_files

def main():
    """Main function to orchestrate the data processing."""
    args = parse_args()
    
    # Create directory structure
    create_directory_structure()
    
    # Determine which coins to process
    coins = None
    if args.coins:
        coins = [coin.strip() for coin in args.coins.split(",")]
    
    # Process the price analysis dataset
    logger.info(f"Starting processing of price_analysis dataset")
    processed_files = process_price_analysis_dataset(
        coins=coins,
        force_download=args.force_download,
        skip_derived_metrics=args.skip_derived_metrics,
        output_dir=args.output_dir
    )
    
    # Create a merged dataset if files were processed
    if processed_files:
        logger.info(f"Creating merged dataset from {len(processed_files)} files")
        normalizer = DataNormalizer(processed_dir=args.output_dir)
        merged_file = normalizer.merge_datasets(
            historical_files=processed_files,
            current_files=[]  # No current files yet
        )
        logger.info(f"Merged dataset created at {merged_file}")
    
    logger.info("Historical data processing complete")

if __name__ == "__main__":
    main() 