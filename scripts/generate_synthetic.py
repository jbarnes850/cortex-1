#!/usr/bin/env python
"""
Script to generate synthetic, chain-of-thought reasoning training data for NEAR Cortex-1.
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

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
    """Collect market data for multiple chains.
    
    Args:
        client: Initialized Flipside client
        chains: List of blockchain names
        start_date: Start date for data collection
        end_date: End date for data collection
        
    Returns:
        List of market data dictionaries
    """
    market_data = []
    
    for chain in chains:
        logger.info(f"Collecting market data for {chain}...")
        try:
            chain_data = client.get_market_data(
                chain=chain,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d")
            )
            
            # Group by day and create samples
            for date, day_data in chain_data.groupby(pd.Grouper(key='block_timestamp', freq='D')):
                if not day_data.empty:
                    market_data.append({
                        'block_timestamp': str(date),
                        'network': day_data['network'].iloc[0] if 'network' in day_data else chain,
                        'blockchain': day_data['blockchain'].iloc[0] if 'blockchain' in day_data else chain,
                        'gas_used': day_data['gas_used'].mean() if 'gas_used' in day_data else None,
                        'size': day_data['size'].mean() if 'size' in day_data else None
                    })
                
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
    output_path = output_dir / f"reasoning_data_{end_date.strftime('%Y%m%d')}.jsonl"
    
    synthetic_generator.generate_dataset(
        market_data=market_data,
        protocol_data=protocol_data,
        output_path=str(output_path),
        samples_per_prompt=args.samples_per_day
    )

if __name__ == "__main__":
    main() 