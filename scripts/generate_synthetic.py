#!/usr/bin/env python
"""
Script to generate synthetic training data using market data and chain-of-thought reasoning.
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
from typing import List
import yaml

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.data.flipside_client import FlipsideClient
from src.data.synthetic_generator import SyntheticDataGenerator
from src.data.market_conditions import MarketConditions

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

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic training data')
    parser.add_argument('--days', type=int, default=180, help='Number of days of historical data')
    parser.add_argument('--samples-per-day', type=int, default=10, help='Number of examples per day')
    parser.add_argument('--chains', nargs='+', default=['ethereum', 'near'], help='Chains to analyze')
    parser.add_argument('--config', type=str, default='configs/data_config.yaml', help='Path to config file')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    
    # Initialize clients
    logger.info("Initializing clients...")
    flipside_client = FlipsideClient()
    market_conditions = MarketConditions()
    generator = SyntheticDataGenerator()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(project_root, "data/synthetic")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"reasoning_data_{timestamp}.jsonl")
    
    logger.info(f"Starting data generation process...")
    logger.info(f"Output will be saved to: {output_path}")
    
    try:
        for chain in args.chains:
            logger.info(f"Processing {chain} chain data...")
            
            # Fetch market data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=args.days)
            
            logger.info(f"Fetching market data for {chain} from {start_date} to {end_date}")
            market_data = flipside_client.get_market_metrics(
                chain=chain,
                start_date=start_date,
                end_date=end_date,
                metrics=config['metrics']
            )
            
            logger.info(f"Retrieved {len(market_data)} data points for {chain}")
            
            # Enrich with market conditions
            logger.info("Analyzing market conditions...")
            for data_point in market_data:
                conditions = market_conditions.analyze(data_point)
                data_point.update(conditions)
            
            # Generate synthetic examples
            logger.info(f"Generating {args.samples_per_day} examples per day...")
            generator.generate_dataset(
                market_data=market_data,
                output_path=output_path,
                samples_per_day=args.samples_per_day
            )
            
            logger.info(f"Completed processing {chain} chain")
            
    except Exception as e:
        logger.error(f"Error generating synthetic data: {str(e)}")
        raise
        
    logger.info(f"Successfully generated synthetic data: {output_path}")

if __name__ == "__main__":
    main() 