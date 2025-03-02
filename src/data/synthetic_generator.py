"""
Synthetic data generator for creating chain-of-thought reasoning examples.
Optimized for DeepSeek R1's reasoning capabilities to generate high-quality financial analysis.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from collections import defaultdict
import time

# Remove unnecessary imports
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.data.flipside_client import FlipsideClient
from src.model.openrouter_client import OpenRouterClient
from src.data.market_conditions import MarketConditions
from src.rewards import get_default_financial_reward

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SyntheticDataGenerator:
    """
    Generate synthetic chain-of-thought reasoning data using market data and DeepSeek R1.
    This class handles the integration between real Flipside market data and DeepSeek R1's
    reasoning capabilities to produce high-quality financial analysis examples.
    """
    
    def __init__(self, verify_quality: bool = True, target_model: str = "llama-3.3-70b"):
        """
        Initialize the synthetic data generator.
        
        Args:
            verify_quality: Whether to verify the quality of generated examples
            target_model: The target model for fine-tuning (affects prompt engineering)
        """
        self.openrouter_client = OpenRouterClient()
        self.flipside_client = FlipsideClient()
        self.market_conditions = MarketConditions()
        self.verify_quality = verify_quality
        self.target_model = target_model
        
        # Initialize the reward function for quality assessment
        self.reward_fn = get_default_financial_reward()
        
        # Check if we're using DeepSeek R1
        is_r1 = self.openrouter_client._is_deepseek_r1_model(self.openrouter_client.model)
        if is_r1:
            logger.info(f"Using DeepSeek R1 model ({self.openrouter_client.model}) for reasoning generation")
        else:
            logger.warning(f"Not using DeepSeek R1 model. Current model: {self.openrouter_client.model}")
    
    def _create_reasoning_prompt(self, market_data: Dict[str, Any]) -> str:
        """
        Create a focused prompt for market reasoning optimized for DeepSeek R1's capabilities.
        
        Args:
            market_data: Dictionary containing market metrics
            
        Returns:
            A prompt string designed to elicit detailed reasoning
        """
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
            'success_rate': float(market_data.get('success_rate', 0)),
            'date': market_data.get('date', 'Unknown date')
        }
        
        market_conditions = market_data.get('market_conditions', {})
        
        # Create a structured prompt for DeepSeek R1 with improved guidance for citations and structure
        return f"""You are analyzing {metrics['network']} blockchain data for {metrics['date']}. Provide comprehensive financial analysis with detailed step-by-step reasoning.

### KEY METRICS (IMPORTANT: ALWAYS cite these with [metric_name] format)

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

### MARKET CONTEXT
{market_conditions}

### REQUIRED ANALYSIS STRUCTURE (Follow this exactly)

1. Network Activity Analysis: <section>
   - Calculate the user engagement ratio: [daily_txns] / [unique_users] with clear step-by-step work
   - Calculate: [total_volume] * (1 - [success_rate]/100) to find volume affected by failures
   - Project 30-day transaction growth using the 7-day rate ([txn_growth]) with 90% confidence intervals
   - IMPORTANT: Show all calculations with formulas, intermediate steps, and final results
</section>

2. Value Flow Analysis: <section>
   - Calculate the value density: [total_volume] / [daily_txns]
   - Calculate gas efficiency: [gas_used] / [daily_txns]
   - Compare these to industry benchmarks with numerical references
   - IMPORTANT: Cite each metric as [metric_name] every time you reference it
</section>

3. Growth Pattern Analysis: <section>
   - Analyze correlation between [txn_growth] and [user_growth]
   - Project 60-day user acquisition with 90% confidence intervals (show calculation)
   - Analyze how [volatility] affects these patterns with mathematical reasoning
   - IMPORTANT: Include confidence intervals in format: "X ± Y (90% CI)"
</section>

4. Investment Implications: <section>
   - Provide 3 specific, actionable investment recommendations based on the data
   - Calculate 3 different ROI scenarios with projected returns
   - Include risk-adjusted projections with probabilities (e.g., "70% probability of X% return")
   - IMPORTANT: Describe concrete actions investors should take with specific timeframes
</section>

5. Conclusion: <section>
   - Summarize key insights with specific metrics cited as [metric_name]
   - Provide final recommendations with confidence levels
   - List top risks and opportunities with quantified impact assessments
</section>

IMPORTANT REQUIREMENTS:
1. Use [metric_name] format EVERY time you cite a metric - never refer to metrics without brackets
2. Show detailed calculations for EVERY numerical conclusion
3. Include 90% confidence intervals for ALL projections
4. Present concrete investment recommendations with expected ROI
5. Follow the exact section structure provided above

Your analysis will be evaluated on calculation accuracy, proper metric citations, confidence intervals, investment insights, and structured presentation."""

    def _create_r1_system_prompt(self) -> str:
        """
        Create a system prompt optimized for DeepSeek R1 reasoning.
        
        Returns:
            A system prompt string for DeepSeek R1
        """
        return """You are an expert crypto quantitative analyst specializing in blockchain data analysis with a focus on financial reasoning.

Your task is to provide DETAILED, STRUCTURED financial analysis with meticulous attention to the following:

CRITICAL REQUIREMENTS (These will significantly affect your evaluation):

1. CITATIONS:
   - Use [metric_name] format EVERY TIME you reference ANY metric (e.g., "The [daily_txns] of 5.2M shows...")
   - This bracket citation format is MANDATORY for EVERY metric reference
   - Never refer to a metric without using the [metric_name] format

2. CALCULATIONS:
   - Show ALL calculation steps completely (formulas, intermediate values, and final results)
   - For complex calculations, use step numbering and clear math notation
   - ALL numerical conclusions must be backed by explicit calculations

3. CONFIDENCE INTERVALS:
   - Express ALL projections with 90% confidence intervals in format: "X ± Y (90% CI)"
   - Explain the statistical basis for each confidence interval calculation
   - Always show mathematical derivation of confidence interval bounds

4. INVESTMENT INSIGHTS:
   - Provide specific, actionable investment recommendations with:
     * Expected ROI percentages with timeframes
     * Risk levels with quantified probability assessments
     * Clear action steps for different investor types

5. STRUCTURE:
   - Use the EXACT section structure provided in the prompt
   - Clearly mark section boundaries with <section> and </section> tags
   - Maintain consistent formatting with clear subsections and bullet points

Your response must demonstrate advanced financial expertise, rigorous mathematical reasoning, and practical investment advice. Think systematically and show your complete reasoning process step by step.

REMEMBER: Always cite metrics in [brackets], show ALL calculations, include 90% confidence intervals, provide specific investment recommendations, and follow the exact structure requested."""

    def _verify_example_quality(self, response: str, input_data: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Verify the quality of a generated example using the reward function.
        
        Args:
            response: The generated response
            input_data: The input data used to generate the response
            
        Returns:
            A tuple of (passes_quality, quality_score)
        """
        # Calculate reward score
        quality_score = self.reward_fn(response, input_data)
        
        # Get component scores for detailed logging
        component_scores = self.reward_fn.get_component_scores()
        
        # Log detailed quality assessment
        logger.info(f"Quality assessment - Overall score: {quality_score:.4f}")
        for component, score in component_scores.items():
            logger.info(f"  - {component}: {score:.4f}")
        
        # Check if quality passes threshold
        QUALITY_THRESHOLD = 0.45  # Lower threshold to allow more examples to pass initially
        passes_quality = quality_score >= QUALITY_THRESHOLD
        
        return passes_quality, quality_score

    def generate_dataset(self, n_examples: int = 100, chains: List[str] = ["market"]) -> List[Dict]:
        """
        Generate a synthetic dataset with reasoning from DeepSeek R1.
        
        Args:
            n_examples: Number of examples to generate
            chains: Types of chains to generate (market, token, protocol)
            
        Returns:
            A list of generated examples
        """
        logger.info(f"Generating {n_examples} examples for chains: {chains}")
        examples = []
        failed_examples = 0
        quality_scores = []
        generation_times = []
        
        # Fetch recent market data from Flipside
        market_data = []
        for chain in chains:
            try:
                logger.info(f"Fetching market data for {chain}")
                chain_data = self.flipside_client.get_recent_market_data(
                    chain=chain,
                    days=30,
                    limit=min(n_examples * 2, 1000)  # Get more than needed to allow for filtering
                )
                market_data.extend(chain_data)
                logger.info(f"Retrieved {len(chain_data)} data points for {chain}")
            except Exception as e:
                logger.error(f"Error fetching {chain} data: {str(e)}")
                continue
        
        if not market_data:
            logger.error("No market data available. Cannot generate examples.")
            return []
        
        # Sample from available data points
        if len(market_data) > n_examples:
            selected_data = np.random.choice(market_data, n_examples, replace=False)
        else:
            selected_data = market_data
            logger.warning(f"Only {len(market_data)} data points available, less than requested {n_examples}")
        
        # Generate examples
        total_data_points = len(selected_data)
        logger.info(f"Starting generation of {total_data_points} examples")
        
        for i, data_point in enumerate(selected_data):
            # Calculate and display progress
            progress_pct = (i / total_data_points) * 100
            elapsed_time = sum(generation_times) if generation_times else 0
            avg_time_per_example = elapsed_time / max(1, len(examples))
            estimated_remaining = avg_time_per_example * (total_data_points - i - 1)
            
            logger.info(f"Generating example {i+1}/{total_data_points} ({progress_pct:.1f}%)")
            if i > 0:
                logger.info(f"Avg time per example: {avg_time_per_example:.2f}s, Est. remaining time: {estimated_remaining:.2f}s")
                logger.info(f"Success rate so far: {len(examples)/(i+1-failed_examples):.2f}, Failed examples: {failed_examples}")
            
            # Track timing for this example
            example_start_time = time.time()
            
            # Analyze market conditions
            conditions = self.market_conditions.analyze(data_point)
            data_point.update({"market_conditions": conditions})
            
            # Log market condition
            network = data_point.get("network", "").upper()
            txn_growth = data_point.get("txn_growth_pct_7d", 0)
            user_growth = data_point.get("user_growth_pct_7d", 0)
            logger.info(f"Market context: {network} network with {txn_growth:.1f}% txn growth, {user_growth:.1f}% user growth")
            
            # Create prompts
            user_prompt = self._create_reasoning_prompt(data_point)
            system_prompt = self._create_r1_system_prompt()
            
            try:
                # Generate reasoning using DeepSeek R1
                logger.info("Generating chain-of-thought reasoning via DeepSeek R1...")
                reasoning_start = time.time()
                reasoning = self.openrouter_client.generate_completion(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    temperature=0.7,
                    max_tokens=2000  # Increased for more detailed reasoning
                )
                reasoning_time = time.time() - reasoning_start
                logger.info(f"Reasoning generation completed in {reasoning_time:.2f}s ({len(reasoning)} chars)")
                
                # Verify quality if enabled
                passes_quality = True
                quality_score = 1.0
                
                if self.verify_quality:
                    logger.info("Verifying example quality...")
                    quality_start = time.time()
                    passes_quality, quality_score = self._verify_example_quality(
                        reasoning, 
                        {"query": user_prompt, "metrics": data_point}
                    )
                    quality_time = time.time() - quality_start
                    logger.info(f"Quality verification completed in {quality_time:.2f}s")
                    quality_scores.append(quality_score)
                
                # If quality passes or verification is disabled, add to examples
                if passes_quality:
                    logger.info(f"Example passes quality check with score {quality_score:.2f}")
                    
                    # Extract summary output (would be used for final model training)
                    output = self._extract_summary(reasoning)
                    logger.info(f"Extracted summary of {len(output)} chars")
                    
                    example = {
                        'date': data_point.get('date', datetime.now().strftime("%Y-%m-%d")),
                        'chain': data_point.get('network', chain),
                        'input': user_prompt,
                        'reasoning': reasoning,
                        'output': output,
                        'market_data': data_point,
                        'quality_score': quality_score,
                        'model': self.openrouter_client.model,
                        'timestamp': datetime.now().isoformat()
                    }
                    examples.append(example)
                else:
                    logger.warning(f"Example failed quality check with score {quality_score:.2f}")
                    failed_examples += 1
                    
            except Exception as e:
                logger.error(f"Error generating example: {str(e)}")
                failed_examples += 1
                continue
            
            # Track time for this example
            example_time = time.time() - example_start_time
            generation_times.append(example_time)
            logger.info(f"Example {i+1} completed in {example_time:.2f}s")
            
            # Display progress metrics
            if quality_scores:
                avg_quality = sum(quality_scores) / len(quality_scores)
                logger.info(f"Average quality score so far: {avg_quality:.4f}")
        
        # Final statistics
        success_rate = len(examples) / total_data_points if total_data_points > 0 else 0
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        avg_time = sum(generation_times) / len(generation_times) if generation_times else 0
        
        logger.info(f"Successfully generated {len(examples)} examples")
        logger.info(f"Success rate: {success_rate:.2f}")
        logger.info(f"Failed examples: {failed_examples}")
        logger.info(f"Average quality score: {avg_quality:.4f}")
        logger.info(f"Average time per example: {avg_time:.2f}s")
        
        return examples
        
    def _extract_summary(self, reasoning: str) -> str:
        """
        Extract a concise summary from the detailed reasoning.
        This would be used to create the final output for model training.
        
        Args:
            reasoning: The detailed reasoning
            
        Returns:
            A concise summary
        """
        # In a real implementation, this would extract key insights
        # For simplicity, we'll just return the first paragraph or a subset
        lines = reasoning.split('\n')
        summary_lines = []
        
        # Look for Investment Implications or Summary sections
        in_summary = False
        for line in lines:
            if "Investment Implications" in line or "Summary" in line or "Conclusion" in line:
                in_summary = True
                summary_lines.append(line)
                continue
            
            if in_summary and line.strip() and not line.startswith('#'):
                summary_lines.append(line)
            
            # Stop after collecting enough summary content
            if in_summary and len(summary_lines) > 10:
                break
        
        # If no summary section found, use the first few lines
        if not summary_lines:
            summary_lines = [line for line in lines[:10] if line.strip()]
        
        return '\n'.join(summary_lines) 