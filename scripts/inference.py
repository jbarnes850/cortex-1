#!/usr/bin/env python
"""
Inference script for generating crypto market analysis using the trained model.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

from src.data.flipside_client import FlipsideClient
from src.model.grpo_trainer import CryptoGRPOTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate crypto market analysis")
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing the trained model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to model configuration file"
    )
    parser.add_argument(
        "--chain",
        type=str,
        default="ethereum",
        help="Blockchain to analyze"
    )
    parser.add_argument(
        "--protocol",
        type=str,
        help="Optional DeFi protocol to analyze"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Optional file to save the analysis"
    )
    parser.add_argument(
        "--num-analyses",
        type=int,
        default=3,
        help="Number of analyses to generate"
    )
    return parser.parse_args()

def get_market_context(
    flipside_client: FlipsideClient,
    chain: str,
    protocol: Optional[str] = None
) -> dict:
    """Get current market context from Flipside.
    
    Args:
        flipside_client: Initialized Flipside client
        chain: Blockchain to analyze
        protocol: Optional DeFi protocol
        
    Returns:
        Dictionary of market metrics
    """
    # Get latest market data
    market_data = flipside_client.get_market_data(
        chain=chain,
        start_date="1 day ago",
        end_date="now",
        limit=1000
    )
    
    context = {
        "chain": chain,
        "volume_24h": market_data["amount_usd"].sum(),
        "total_txns": len(market_data),
        "unique_addresses": market_data[["from_address", "to_address"]].nunique().sum(),
        "avg_gas_price": market_data["gas_price"].mean()
    }
    
    # Add protocol metrics if specified
    if protocol:
        protocol_data = flipside_client.get_defi_metrics(
            protocol=protocol,
            start_date="1 day ago",
            end_date="now"
        )
        
        context.update({
            "protocol": protocol,
            "protocol_volume": protocol_data["volume_usd"].sum(),
            "protocol_fees": protocol_data["fees_usd"].sum(),
            "protocol_users": protocol_data["unique_users"].sum()
        })
    
    return context

def create_analysis_prompt(context: dict) -> str:
    """Create a prompt for market analysis.
    
    Args:
        context: Market context dictionary
        
    Returns:
        Formatted prompt string
    """
    prompt = f"""Analyze the following crypto market data and provide a detailed explanation of the market dynamics:

Chain: {context['chain']}
24h Volume: ${context['volume_24h']:,.2f}
Total Transactions: {context['total_txns']:,}
Unique Addresses: {context['unique_addresses']:,}
Average Gas Price: {context['avg_gas_price']:.2f} GWEI
"""

    if "protocol" in context:
        prompt += f"""
Protocol: {context['protocol']}
Protocol Volume: ${context['protocol_volume']:,.2f}
Protocol Fees: ${context['protocol_fees']:,.2f}
Active Users: {context['protocol_users']:,}
"""

    prompt += """
Provide a detailed analysis following this structure:
1. Market Overview
2. On-Chain Activity Assessment
3. User Behavior Analysis
4. Risk Factors
5. Trading Opportunities
6. Final Recommendation

Explain your reasoning for each conclusion and support it with the provided metrics."""

    return prompt

def main():
    args = parse_args()
    
    # Initialize clients
    flipside_client = FlipsideClient()
    model = CryptoGRPOTrainer(config_path=args.config)
    
    # Get market context
    logger.info("Fetching market data from Flipside...")
    context = get_market_context(
        flipside_client=flipside_client,
        chain=args.chain,
        protocol=args.protocol
    )
    
    # Create analysis prompt
    prompt = create_analysis_prompt(context)
    
    # Generate analyses
    logger.info(f"Generating {args.num_analyses} market analyses...")
    analyses = model.generate_analysis(
        prompt=prompt,
        num_return_sequences=args.num_analyses
    )
    
    # Print or save results
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                "context": context,
                "analyses": analyses
            }, f, indent=2)
            
        logger.info(f"Saved analyses to {output_path}")
    else:
        print("\nMarket Analyses:")
        print("=" * 80)
        for i, analysis in enumerate(analyses, 1):
            print(f"\nAnalysis {i}:")
            print("-" * 40)
            print(analysis)
            print("=" * 80)

if __name__ == "__main__":
    main() 