"""
Synthetic data generator for creating chain-of-thought reasoning examples.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from collections import defaultdict
import yaml

from src.data.flipside_client import FlipsideClient
from src.model.openai_client import OpenAIClient
from src.data.market_conditions import MarketConditions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SyntheticDataGenerator:
    """Generate synthetic chain-of-thought reasoning data using market data."""
    
    def __init__(self, config_path: str = "configs/data_config.yaml"):
        """Initialize the synthetic data generator."""
        self.openai_client = OpenAIClient()
        self.market_conditions = MarketConditions()
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _create_reasoning_prompt(self, market_data: Dict[str, Any]) -> str:
        """Create a focused prompt for market reasoning."""
        # Extract and format metrics
        metrics = {
            'network': market_data.get('network', 'Unknown'),
            'daily_txns': market_data.get('num_txs', 0),
            'total_volume': float(market_data.get('total_volume', 0)),
            'unique_users': int(market_data.get('unique_users', 0)),
            'avg_tx_value': float(market_data.get('avg_tx_value', 0)),
            'txn_growth': float(market_data.get('txn_growth_pct_7d', 0)),
            'user_growth': float(market_data.get('user_growth_pct_7d', 0)),
            'volatility': float(market_data.get('tx_volatility_7d', 0)),
            'gas_used': float(market_data.get('gas_used', 0)),
            'success_rate': float(market_data.get('success_rate', 0))
        }
        
        market_conditions = market_data.get('market_conditions', {})
        
        return f"""Analyze this {metrics['network']} blockchain data with detailed calculations:

Key Metrics (cite as [metric_name]):
1. Activity Metrics:
   - Daily Transactions: {metrics['daily_txns']:,} [daily_txns]
   - Unique Users: {metrics['unique_users']:,} [unique_users]
   - Success Rate: {metrics['success_rate']:.1f}% [success_rate]

2. Volume & Value:
   - Total Volume: {metrics['total_volume']:,.2f} [total_volume]
   - Avg Transaction: {metrics['avg_tx_value']:,.2f} [avg_tx_value]
   - Gas Used: {metrics['gas_used']:,.0f} [gas_used]

3. Growth & Volatility:
   - Transaction Growth (7d): {metrics['txn_growth']:.1f}% [txn_growth]
   - User Growth (7d): {metrics['user_growth']:.1f}% [user_growth]
   - Volatility (7d): {metrics['volatility']:.2f} [volatility]

Market Conditions:
{market_conditions}

Required Analysis:

1. Network Activity Analysis:
   - Calculate user engagement ratio: [daily_txns] / [unique_users]
   - Analyze success rate impact on volume
   - Project 30-day transaction growth with confidence interval

2. Value Flow Analysis:
   - Calculate value density: [total_volume] / [daily_txns]
   - Analyze gas efficiency: [gas_used] / [daily_txns]
   - Compare with network averages

3. Growth Patterns:
   - Calculate growth correlation: [txn_growth] vs [user_growth]
   - Project user acquisition rate
   - Estimate volatility impact on growth

Requirements:
1. Show ALL calculations step-by-step
2. Use [metric_name] format for EVERY metric citation
3. Include confidence intervals (90% CI) for projections
4. Explain the significance of each calculation
5. Consider market conditions in your analysis"""

    def generate_dataset(self,
                        market_data: List[Dict],
                        output_path: str,
                        samples_per_day: int = 5) -> None:
        """Generate synthetic examples from market data."""
        examples = []
        batch_size = self.config['generation']['batch_size']
        max_retries = self.config['generation']['max_retries']
        
        for data_point in market_data:
            successful_samples = 0
            retries = 0
            
            while successful_samples < samples_per_day and retries < max_retries:
                try:
                    # Generate reasoning
                    prompt = self._create_reasoning_prompt(data_point)
                    response = self.openai_client.generate_completion(
                        prompt=prompt,
                        system_prompt="""You are a quantitative analyst specializing in blockchain data.
Your task is to analyze market data with precise calculations and clear reasoning.
Always show your work and cite data sources using [metric_name] format.""",
                        model="o3-mini",
                        temperature=self.config['generation']['temperature'],
                        max_tokens=self.config['generation']['max_tokens']
                    )
                    
                    # Verify quality
                    passes_quality, quality_score = self.openai_client.verify_quality(
                        response,
                        required_components=self.config['quality']['required_components']
                    )
                    
                    if passes_quality and quality_score >= self.config['quality']['min_score']:
                        example = {
                            'timestamp': data_point['block_timestamp'],
                            'network': data_point['network'],
                            'market_data': data_point,
                            'reasoning': response,
                            'quality_score': quality_score,
                            'generation_metadata': {
                                'timestamp': datetime.now().isoformat(),
                                'model': "o3-mini",
                                'temperature': self.config['generation']['temperature']
                            }
                        }
                        examples.append(example)
                        successful_samples += 1
                        logger.info(f"Generated quality example (score: {quality_score:.2f})")
                    else:
                        logger.warning(f"Example failed quality check (score: {quality_score:.2f})")
                        retries += 1
                    
                    # Save progress periodically
                    if len(examples) % batch_size == 0:
                        self._save_examples(examples, output_path)
                        
                except Exception as e:
                    logger.error(f"Error generating example: {str(e)}")
                    retries += 1
                    continue
            
            if successful_samples < samples_per_day:
                logger.warning(f"Could not generate {samples_per_day} quality samples after {max_retries} retries")
        
        # Save final results
        self._save_examples(examples, output_path)
        logger.info(f"Generated {len(examples)} total examples")

    def _save_examples(self, examples: List[Dict], output_path: str) -> None:
        """Save examples to JSONL file with proper formatting."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            for example in examples:
                f.write(json.dumps(example, indent=None, ensure_ascii=False) + '\n') 