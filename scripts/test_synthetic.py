#!/usr/bin/env python
"""
Test script for synthetic data generation improvements.
"""

import logging
import json
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

from src.data.flipside_client import FlipsideClient
from src.data.synthetic_generator import SyntheticDataGenerator, MarketCondition

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_market_condition_labeling():
    """Test market condition labeling logic."""
    generator = SyntheticDataGenerator()
    
    # Test cases
    test_cases = [
        {
            'current': {
                'txn_growth_pct_7d': 15.0,
                'user_growth_pct_7d': 8.0,
                'tx_volatility_7d': 100.0
            },
            'historical': [{
                'txn_growth_pct_7d': 5.0,
                'user_growth_pct_7d': 3.0,
                'tx_volatility_7d': 80.0
            }],
            'expected': MarketCondition.BULLISH
        },
        {
            'current': {
                'txn_growth_pct_7d': -15.0,
                'user_growth_pct_7d': -8.0,
                'tx_volatility_7d': 100.0
            },
            'historical': [{
                'txn_growth_pct_7d': 5.0,
                'user_growth_pct_7d': 3.0,
                'tx_volatility_7d': 80.0
            }],
            'expected': MarketCondition.BEARISH
        },
        {
            'current': {
                'txn_growth_pct_7d': 2.0,
                'user_growth_pct_7d': 1.0,
                'tx_volatility_7d': 200.0
            },
            'historical': [{
                'txn_growth_pct_7d': 5.0,
                'user_growth_pct_7d': 3.0,
                'tx_volatility_7d': 80.0
            }],
            'expected': MarketCondition.VOLATILE
        }
    ]
    
    for i, test in enumerate(test_cases):
        result = generator._label_market_condition(test['current'], test['historical'])
        logger.info(f"Test {i+1}: Expected {test['expected']}, Got {result}")
        assert result == test['expected'], f"Test {i+1} failed"

def test_balanced_sampling():
    """Test balanced sampling across market conditions."""
    generator = SyntheticDataGenerator()
    
    # Create synthetic market data with different conditions
    market_data = []
    base_conditions = {
        MarketCondition.BULLISH: (15.0, 8.0, 100.0),
        MarketCondition.BEARISH: (-15.0, -8.0, 100.0),
        MarketCondition.SIDEWAYS: (2.0, 1.0, 80.0),
        MarketCondition.VOLATILE: (2.0, 1.0, 200.0)
    }
    
    # Generate equal samples for each condition
    samples_per_condition = 10
    for condition, (txn_growth, user_growth, volatility) in base_conditions.items():
        for i in range(samples_per_condition):
            # Add some random noise to create variants
            noise = np.random.normal(0, 0.1)
            market_data.append({
                'block_timestamp': datetime.now().isoformat(),
                'txn_growth_pct_7d': txn_growth * (1 + noise),
                'user_growth_pct_7d': user_growth * (1 + noise),
                'tx_volatility_7d': volatility * (1 + noise)
            })
    
    # Test balancing
    balanced_data = generator._balance_market_conditions(market_data)
    
    # Count conditions in balanced data
    condition_counts = {}
    for data in balanced_data:
        condition = generator._label_market_condition(data, [])
        condition_counts[condition] = condition_counts.get(condition, 0) + 1
    
    logger.info("Balanced condition counts:")
    for condition, count in condition_counts.items():
        logger.info(f"{condition}: {count}")
        
    # Group conditions by type
    trend_conditions = {MarketCondition.BULLISH, MarketCondition.BEARISH}
    stability_conditions = {MarketCondition.SIDEWAYS, MarketCondition.VOLATILE}
    
    # Calculate group totals
    trend_total = sum(condition_counts.get(c, 0) for c in trend_conditions)
    stability_total = sum(condition_counts.get(c, 0) for c in stability_conditions)
    
    logger.info(f"\nCondition group totals:")
    logger.info(f"Trend conditions (Bullish/Bearish): {trend_total}")
    logger.info(f"Stability conditions (Sideways/Volatile): {stability_total}")
    
    # Verify group balance
    group_imbalance = abs(trend_total - stability_total)
    assert group_imbalance <= 4, f"Condition groups not properly balanced (imbalance: {group_imbalance})"
    
    # Verify all base conditions are represented
    for condition in base_conditions.keys():
        assert condition in condition_counts, f"Missing condition: {condition}"
        
    # Verify minimum representation
    min_count = min(condition_counts.values())
    assert min_count >= 5, f"Some conditions underrepresented (min count: {min_count})"

def test_prompt_diversity():
    """Test prompt template diversity."""
    generator = SyntheticDataGenerator()
    
    # Create test market data
    market_data = {
        'block_timestamp': datetime.now().isoformat(),
        'num_txs': 1000000,
        'success_rate': 0.985,
        'txn_growth_pct_7d': 5.0,
        'user_growth_pct_7d': 3.0,
        'tx_volatility_7d': 100.0,
        'avg_tx_value': 1.5,
        'avg_tx_value_change_pct': -2.5,
        'avg_gas_price': 3.2,
        'smart_contract_calls': 950000,
        'network': 'ethereum',
        'unique_users': 50000,
        'bridge_volume': 25000000
    }
    
    # Create test protocol data
    protocol_data = {
        'ethereum': {
            'volume_usd': 1000000,
            'unique_users': 50000,
            'volume_growth_pct': 5.0,
            'volume_share': 15.0
        },
        'near': {
            'volume_usd': 500000,
            'unique_users': 25000,
            'volume_growth_pct': 8.0,
            'volume_share': 10.0
        }
    }
    
    # Create test chain data
    chain_data = {
        'network': 'ethereum',
        'num_txs': 1000000,
        'success_rate': 0.99,
        'txn_growth_pct_7d': 3.0,
        'user_growth_pct_7d': 2.0,
        'smart_contract_calls': 800000
    }
    
    # Expected sections for each prompt type
    expected_sections = {
        'prediction': ['Current Market State', 'Required Analysis'],
        'correlation': ['Correlation Analysis', 'Network Effects'],
        'protocol': ['Protocol Metrics', 'Chain Context'],
        'risk': ['Market Context', 'Risk Identification'],
        'opportunity': ['Market Context', 'Opportunity Identification'],
        'market_qa': ['Market Context', 'Question', 'Required Response Format'],
        'analytical': ['Market Context', 'Analytical Problem', 'Required Analysis Structure'],
        'financial': ['Market Context', 'Financial Analysis Problem', 'Required Analysis Framework']
    }
    
    # Test each prompt type
    for prompt_type, prompt_func in generator.prompt_templates.items():
        logger.info(f"\nTesting {prompt_type} prompt:")
        
        try:
            if prompt_type == 'prediction' or prompt_type == 'market_qa':
                prompt = prompt_func(market_data, market_data)  # Use same data for outcome
            elif prompt_type == 'protocol':
                prompt = prompt_func(protocol_data, chain_data)
            else:
                prompt = prompt_func(market_data)
            
            # Verify prompt structure
            assert '<reasoning>' in prompt, f"{prompt_type} prompt missing reasoning tag"
            
            # Verify required sections
            for section in expected_sections[prompt_type]:
                assert section in prompt, f"{prompt_type} prompt missing section: {section}"
            
            logger.info(f"✓ {prompt_type} prompt structure verified")
            
        except Exception as e:
            logger.error(f"Error testing {prompt_type} prompt: {str(e)}")
            raise

def main():
    """Run all tests."""
    logger.info("Testing market condition labeling...")
    test_market_condition_labeling()
    logger.info("✓ Market condition labeling tests passed\n")
    
    logger.info("Testing balanced sampling...")
    test_balanced_sampling()
    logger.info("✓ Balanced sampling tests passed\n")
    
    logger.info("Testing prompt diversity...")
    test_prompt_diversity()
    logger.info("✓ Prompt diversity tests passed\n")

if __name__ == "__main__":
    main() 