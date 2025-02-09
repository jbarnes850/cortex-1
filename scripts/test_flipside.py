#!/usr/bin/env python
"""Test Flipside API connection."""

import os
import sys
import logging
from pathlib import Path

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.data.flipside_client import FlipsideClient
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
            
    except Exception as e:
        logger.error(f"Error testing connection: {str(e)}")
        raise

if __name__ == "__main__":
    main() 