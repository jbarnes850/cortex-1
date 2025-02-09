"""
Test script for the simplified data generator.
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
logging.info(f"Loading environment from: {dotenv_path}")

# Force reload of environment variables
if not load_dotenv(dotenv_path, override=True):
    raise RuntimeError(f"Failed to load .env file from {dotenv_path}")

# Get API keys
FLIPSIDE_API_KEY = os.getenv("FLIPSIDE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

logging.info(f"Loaded Flipside API Key: {FLIPSIDE_API_KEY}")
logging.info(f"Loaded OpenAI API Key: {OPENAI_API_KEY[:8]}... (length: {len(OPENAI_API_KEY) if OPENAI_API_KEY else 0})")

if not FLIPSIDE_API_KEY or not OPENAI_API_KEY:
    raise ValueError("Missing required environment variables. Please check your .env file.")

from src.simple_generator import SimpleDataGenerator

def test_flipside_connection(chain: str = 'ethereum'):
    """Test the connection to Flipside API."""
    logging.info(f"\nTesting {chain} chain...")
    
    try:
        generator = SimpleDataGenerator(flipside_key=FLIPSIDE_API_KEY)
        logging.info(f"Attempting to fetch {chain} market data...")
        
        # Test market data fetch
        market_data = generator.get_market_data(chain=chain, days=1)
        
        if market_data:
            logging.info(f"✅ Successfully fetched {chain} market data")
            logging.info(f"Retrieved {len(market_data)} rows")
            logging.info("Sample data:")
            logging.info(market_data[0])
            return True
        else:
            logging.error(f"❌ No data returned for {chain}")
            return False
            
    except Exception as e:
        logging.error(f"❌ Error testing {chain} connection: {str(e)}")
        return False

def test_reasoning_generation(chain: str = 'near'):
    """Test the reasoning generation."""
    logging.info(f"\nTesting {chain} reasoning generation...")
    
    try:
        generator = SimpleDataGenerator(
            flipside_key=FLIPSIDE_API_KEY,
            openai_key=OPENAI_API_KEY
        )
        
        # Get market data
        market_data = generator.get_market_data(chain=chain, days=1)
        
        if not market_data:
            logging.error(f"❌ No market data available for {chain}")
            return False
            
        # Generate reasoning for the first day
        data_point = market_data[0]
        reasoning = generator.generate_reasoning(data_point)
        
        if reasoning and len(reasoning.strip()) > 0:
            logging.info(f"✅ Successfully generated {chain} reasoning")
            logging.info("Sample reasoning preview:")
            logging.info("---")
            logging.info(reasoning[:200] + "...")
            logging.info("---")
            return True
        else:
            logging.error(f"❌ No {chain} reasoning generated")
            return False
            
    except Exception as e:
        logging.error(f"❌ Error testing {chain} reasoning generation: {str(e)}")
        return False

def main():
    """Run all tests."""
    try:
        # Test Flipside connection for both chains
        eth_success = test_flipside_connection('ethereum')
        near_success = test_flipside_connection('near')
        
        # Test reasoning generation
        if near_success:
            reasoning_success = test_reasoning_generation('near')
        else:
            reasoning_success = False
            
        if not (eth_success and near_success and reasoning_success):
            logging.error("\n❌ Some tests failed")
            return False
            
        logging.info("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        logging.error(f"Error running tests: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 