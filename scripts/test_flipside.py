#!/usr/bin/env python3
"""Test Flipside API connection."""

import os
import sys
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.data.flipside_client import FlipsideClient
from dotenv import load_dotenv
from src.utils.logger import setup_logger

# Set up logging
logger = setup_logger()

def test_flipside_connection():
    """Test Flipside connection and data retrieval for all chains."""
    
    # Initialize client
    client = FlipsideClient()
    
    # Test period (last 7 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    # Format dates
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    # Test each chain
    for chain in ['bitcoin', 'ethereum', 'near']:
        logger.info(f'\nTesting {chain.upper()} data:')
        try:
            df = client.get_market_data(chain, start_str, end_str)
            logger.info(f'Successfully retrieved {len(df)} records')
            if not df.empty:
                logger.info('Sample metrics:')
                sample = df[['network', 'num_txs', 'unique_senders', 'total_volume']].head(1)
                logger.info(f'\n{sample.to_string()}')
        except Exception as e:
            logger.error(f'Error retrieving {chain} data: {str(e)}')

def main():
    try:
        # Load environment variables
        env_path = Path(project_root) / '.env'
        logger.info(f"Loading environment from: {env_path}")
        load_dotenv(env_path)
        
        # Print API key (first 8 chars)
        api_key = os.getenv("FLIPSIDE_API_KEY")
        if api_key:
            logger.info(f"Found API key (first 8 chars): {api_key[:8]}...")
        else:
            logger.error("No API key found in environment!")
            return
            
        logger.info("Initializing Flipside client...")
        client = FlipsideClient()
        
        logger.info("Testing connection with simple query...")
        result = client.test_connection()
        
        if result:
            logger.info("✅ Connection successful!")
            
            # Try a simple market data query
            logger.info("Testing market data query...")
            market_data = client.get_market_data(
                blockchain='ethereum',
                start_date='2025-02-01',
                end_date='2025-02-09'
            )
            logger.info(f"Retrieved {len(market_data)} rows of market data")
            
        else:
            logger.error("❌ Connection failed!")
            
        test_flipside_connection()
        
    except Exception as e:
        logger.error(f"Error testing connection: {str(e)}")
        raise

if __name__ == "__main__":
    main() 