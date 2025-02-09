"""
Simplified data generation pipeline for crypto market analysis.
Combines Flipside market data with o3-mini reasoning generation.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import pandas as pd
from openai import OpenAI
from flipside import Flipside
import json
from dotenv import load_dotenv
from pathlib import Path
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleDataGenerator:
    def __init__(self, flipside_key: Optional[str] = None, openai_key: Optional[str] = None):
        """Initialize with API clients.
        
        Args:
            flipside_key: Optional Flipside API key. If not provided, will look in environment.
            openai_key: Optional OpenAI API key. If not provided, will look in environment.
        """
        # Load environment variables from project root
        project_root = Path(__file__).resolve().parent.parent
        env_path = project_root / '.env'
        logger.info(f"Loading environment from: {env_path}")
        load_dotenv(env_path, override=True)
        
        # Initialize Flipside client
        self.flipside_key = flipside_key or os.getenv("FLIPSIDE_API_KEY")
        if not self.flipside_key:
            raise ValueError("Flipside API key not found")
        logger.info(f"Loaded Flipside API Key: {self.flipside_key}")
        self.flipside = Flipside(self.flipside_key)
        
        # Initialize OpenAI client
        self.openai_key = openai_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_key:
            raise ValueError("OpenAI API key not found")
        logger.info(f"Loaded OpenAI API Key: {self.openai_key[:8]}... (length: {len(self.openai_key)})")
        self.openai = OpenAI(api_key=self.openai_key)
        
    def get_market_data(self, chain: str = 'ethereum', days: int = 1) -> List[Dict]:
        """Fetch market data for the specified chain."""
        logging.info(f"Fetching {days} days of {chain} market data...")
        
        if chain.lower() == 'ethereum':
            query = f"""
            WITH daily_metrics AS (
                SELECT 
                    DATE_TRUNC('day', block_timestamp) as date,
                    COUNT(DISTINCT tx_hash) as daily_txns,
                    COUNT(DISTINCT from_address) as unique_users,
                    COUNT(DISTINCT to_address) as unique_receivers,
                    COUNT(CASE WHEN status = 'SUCCESS' THEN 1 END)::float / NULLIF(COUNT(*), 0) as success_rate,
                    AVG(value) as avg_tx_value,
                    AVG(gas_used) as avg_gas,
                    AVG(gas_price) as avg_gas_price,
                    COUNT(DISTINCT CASE WHEN to_address IN (SELECT address FROM ethereum.core.dim_contracts) THEN tx_hash END) as contract_calls,
                    SUM(value) as total_volume,
                    'ethereum' as chain
                FROM ethereum.core.fact_transactions
                WHERE block_timestamp >= DATEADD('day', -{days}, CURRENT_DATE())
                GROUP BY 1
                ORDER BY 1 DESC
            )
            SELECT * FROM daily_metrics
            """
        elif chain.lower() == 'near':
            query = """
            WITH daily_metrics AS (
                SELECT
                    DATE_TRUNC('day', block_timestamp) as date,
                    COUNT(DISTINCT tx_hash) as daily_txns,
                    COUNT(DISTINCT tx_signer) as unique_users,
                    COUNT(DISTINCT tx_receiver) as unique_receivers,
                    COUNT(CASE WHEN tx_succeeded = TRUE THEN 1 END)::float / NULLIF(COUNT(*), 0) as success_rate,
                    AVG(COALESCE(transaction_fee / POW(10, 24), 0)) as avg_tx_value,
                    AVG(gas_used / POW(10, 12)) as avg_gas,
                    AVG(attached_gas / POW(10, 12)) as avg_gas_price,
                    COUNT(DISTINCT CASE 
                        WHEN tx_receiver LIKE '%.near' 
                        OR tx_receiver LIKE '%.factory.near'
                        OR tx_receiver LIKE '%.testnet' 
                        THEN tx_receiver 
                    END) as contract_calls,
                    SUM(transaction_fee / POW(10, 24)) as total_volume
                FROM near.core.fact_transactions
                WHERE block_timestamp >= DATEADD('day', -{days}, CURRENT_DATE())
                GROUP BY 1
                ORDER BY 1 DESC
            )
            SELECT 
                date,
                daily_txns,
                unique_users,
                unique_receivers,
                success_rate,
                avg_tx_value,
                avg_gas,
                avg_gas_price,
                contract_calls,
                total_volume,
                'near' as chain
            FROM daily_metrics
            """.format(days=days)
        
        try:
            result = self.flipside.query(query)
            if not result.records:
                raise ValueError(f"No data returned from Flipside for {chain}")
            
            df = pd.DataFrame(result.records)
            logger.info(f"Retrieved {len(df)} days of {chain} market data")
            return df.to_dict(orient='records')
            
        except Exception as e:
            logger.error(f"Error fetching market data: {str(e)}")
            raise
    
    def verify_quality(self, reasoning: str) -> Tuple[bool, Dict[str, float]]:
        """
        Verify the quality of generated reasoning using Sky-T1 inspired metrics.
        
        Args:
            reasoning: Generated reasoning text
            
        Returns:
            Tuple of (passes_quality, quality_metrics)
        """
        try:
            metrics = {
                "calculation_steps": 0,
                "numerical_citations": 0,
                "insight_count": 0,
                "section_completeness": 0
            }
            
            # Count calculation steps (looking for mathematical operations)
            calc_pattern = r'=[\s]*[\d\.\+\-\*\/\(\)]+[\d]'
            calculations = re.findall(calc_pattern, reasoning)
            metrics["calculation_steps"] = len(calculations)
            
            # Count numerical citations
            num_pattern = r'\[[\w_]+\]'
            citations = re.findall(num_pattern, reasoning)
            metrics["numerical_citations"] = len(citations)
            
            # Count actionable insights
            insight_markers = ["actionable insight", "recommendation", "suggests that", "indicates that"]
            metrics["insight_count"] = sum(reasoning.lower().count(marker) for marker in insight_markers)
            
            # Check section completeness
            required_sections = [
                "Network Activity Analysis",
                "Economic Metrics Analysis",
                "Market Health Indicators"
            ]
            metrics["section_completeness"] = sum(1 for section in required_sections if section in reasoning) / len(required_sections)
            
            # Quality thresholds based on Sky-T1's approach
            quality_thresholds = {
                "calculation_steps": 5,      # Minimum number of explicit calculations
                "numerical_citations": 8,     # Minimum number of data citations
                "insight_count": 3,          # Minimum number of actionable insights
                "section_completeness": 0.9   # Minimum section coverage
            }
            
            # Calculate normalized scores (0-1 range)
            normalized_metrics = {
                "calculation_steps": min(metrics["calculation_steps"] / quality_thresholds["calculation_steps"], 1.0),
                "numerical_citations": min(metrics["numerical_citations"] / quality_thresholds["numerical_citations"], 1.0),
                "insight_count": min(metrics["insight_count"] / quality_thresholds["insight_count"], 1.0),
                "section_completeness": metrics["section_completeness"]
            }
            
            # Overall quality check
            passes_quality = all(score >= 0.8 for score in normalized_metrics.values())
            
            if passes_quality:
                logger.info("Quality check passed with scores: " + 
                          ", ".join(f"{k}: {v:.2f}" for k, v in normalized_metrics.items()))
            else:
                logger.warning("Quality check failed with scores: " + 
                             ", ".join(f"{k}: {v:.2f}" for k, v in normalized_metrics.items()))
            
            return passes_quality, normalized_metrics
            
        except Exception as e:
            logger.error(f"Error in quality verification: {str(e)}")
            return False, {}

    def generate_reasoning(self, market_data: Dict) -> str:
        """Generate synthetic reasoning using o3-mini with parameters optimized for high-quality reasoning."""
        chain = market_data.get('chain', 'ethereum')
        unit = 'NEAR' if chain == 'near' else 'ETH'
        
        prompt = f"""Analyze this {chain.upper()} blockchain market data and explain the key trends:

Date: {market_data['date']}
Daily Transactions: {market_data['daily_txns']:,}
Unique Users: {market_data['unique_users']:,}
Unique Receivers: {market_data['unique_receivers']:,}
Total Volume: {market_data['total_volume']:,.2f} {unit}
Success Rate: {market_data['success_rate']*100:.1f}%
Contract Interactions: {market_data['contract_calls']:,}
Average Transaction Value: {market_data['avg_tx_value']:,.6f} {unit}
Average Gas Used: {market_data['avg_gas']:,.0f}
Average Gas Price: {market_data['avg_gas_price']:,.0f}

Please provide a detailed step-by-step analysis covering:

1. Network Activity Analysis:
   a) Calculate and explain the user engagement ratio (daily_txns/unique_users)
   b) Analyze the success rate's impact on effective transaction throughput
   c) Evaluate contract interaction density relative to total transactions
   d) Compare unique receivers to unique users ratio for network distribution

2. Economic Metrics Analysis:
   a) Calculate the total economic value moved (volume * success_rate)
   b) Determine the gas efficiency ratio (gas_used per successful transaction)
   c) Evaluate the cost structure (gas_price * gas_used / transaction_value)
   d) Project the daily network costs for average users

3. Market Health Indicators:
   a) Calculate the network utilization rate
   b) Analyze the distribution of value (gini coefficient approximation)
   c) Evaluate the smart contract adoption rate
   d) Determine the network's economic efficiency score

For each calculation:
1. Show your work step-by-step
2. Explain the significance of each result
3. Compare to typical healthy network metrics
4. Provide actionable insights for users

Use specific numbers and maintain high numerical precision in your analysis."""

        try:
            response = self.openai.chat.completions.create(
                model="o3-mini",
                messages=[
                    {"role": "system", "content": f"""You are an expert blockchain data analyst specializing in {chain.upper()} network metrics.
Your analysis must be:
- Mathematically precise with explicit calculations
- Logically structured with clear sections
- Supported by numerical citations using [metric_name] format
- Focused on actionable insights
- Grounded in blockchain economics

Always show your calculations and explain your reasoning step by step."""},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=20000,
                reasoning_effort="high"
            )
            
            if not response.choices or not response.choices[0].message.content:
                raise ValueError("No content generated by the model")
            
            reasoning = response.choices[0].message.content
            
            # Verify quality before returning
            passes_quality, quality_metrics = self.verify_quality(reasoning)
            if not passes_quality:
                logger.warning("Generated reasoning did not meet quality standards")
                return self.generate_reasoning(market_data)  # Retry once
                
            return reasoning
            
        except Exception as e:
            logger.error(f"Error generating reasoning: {str(e)}")
            raise

    def run(self, chains: List[str] = ['ethereum', 'near'], output_dir: str = "data/market_analysis"):
        """Run the simplified pipeline for specified chains."""
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate timestamp for this run
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for chain in chains:
                output_file = os.path.join(output_dir, f"{chain}_market_reasoning_{timestamp}.jsonl")
                logger.info(f"Processing {chain} chain...")
                
                # Get market data
                market_data = self.get_market_data(chain=chain)
                
                # Generate reasoning for each day
                results = []
                for data_point in market_data:
                    logger.info(f"Analyzing {chain} data for {data_point['date']}")
                    
                    reasoning = self.generate_reasoning(data_point)
                    
                    results.append({
                        "timestamp": datetime.now().isoformat(),
                        "chain": chain,
                        "date": data_point['date'],
                        "market_data": data_point,
                        "reasoning": reasoning
                    })
                
                # Save results
                logger.info(f"Saving {len(results)} examples to {output_file}")
                with open(output_file, 'w') as f:
                    for result in results:
                        f.write(json.dumps(result) + '\n')
            
            logger.info("Pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

if __name__ == "__main__":
    generator = SimpleDataGenerator()
    generator.run() 