"""
Synthetic data generator for creating chain-of-thought reasoning examples.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import time

from src.data.flipside_client import FlipsideClient
from src.model.openai_client import OpenAIClient
from src.model.reward_function import RewardFunction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketCondition:
    """Enum-like class for market conditions."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    RECOVERY = "recovery"
    CORRECTION = "correction"

class SyntheticDataGenerator:
    """Generate synthetic chain-of-thought reasoning data using o3-mini."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "o3-mini"):
        """Initialize the synthetic data generator.
        
        Args:
            api_key: OpenAI API key. If not provided, will look for OPENAI_API_KEY in environment.
            model: OpenAI model to use for generation. Defaults to o3-mini for structured reasoning.
        """
        self.openai_client = OpenAIClient(api_key)
        self.reward_function = RewardFunction()
        self.model = model
        
        # Define prompt templates for diverse reasoning tasks
        self.prompt_templates = {
            'prediction': self._create_prediction_prompt,
            'correlation': self._create_correlation_prompt,
            'protocol': self._create_protocol_prompt,
            'risk': self._create_risk_prompt,
            'opportunity': self._create_opportunity_prompt,
            'market_qa': self._create_market_qa_prompt,
            'analytical': self._create_analytical_prompt,
            'financial': self._create_financial_prompt
        }
        
    def _label_market_condition(self, 
                              current_data: Dict[str, Any], 
                              historical_data: List[Dict[str, Any]]) -> str:
        """Label market condition based on metrics.
        
        Args:
            current_data: Current market metrics
            historical_data: Previous market data points
            
        Returns:
            String label of market condition
        """
        # Calculate key indicators
        txn_growth = float(current_data.get('txn_growth_pct_7d', 0))
        user_growth = float(current_data.get('user_growth_pct_7d', 0))
        volatility = float(current_data.get('tx_volatility_7d', 0))
        
        # Get historical averages
        if historical_data:
            hist_volatility = np.mean([float(d.get('tx_volatility_7d', 0)) 
                                     for d in historical_data])
            hist_txn_growth = np.mean([float(d.get('txn_growth_pct_7d', 0)) 
                                     for d in historical_data])
        else:
            hist_volatility = volatility
            hist_txn_growth = txn_growth
        
        # Label conditions (check volatility first)
        if volatility > 150.0:  # Absolute threshold for high volatility
            return MarketCondition.VOLATILE
        elif txn_growth > 10 and user_growth > 5:
            return MarketCondition.BULLISH
        elif txn_growth < -10 and user_growth < -5:
            return MarketCondition.BEARISH
        elif abs(txn_growth) <= 5 and abs(user_growth) <= 3:
            return MarketCondition.SIDEWAYS
        elif txn_growth > hist_txn_growth + 5:
            return MarketCondition.RECOVERY
        elif txn_growth < hist_txn_growth - 5:
            return MarketCondition.CORRECTION
        else:
            return MarketCondition.SIDEWAYS

    def _balance_market_conditions(self, 
                                 market_data: List[Dict[str, Any]], 
                                 window_size: int = 7) -> List[Dict[str, Any]]:
        """Balance dataset across different market conditions with metric normalization.
        
        Args:
            market_data: List of market data points
            window_size: Historical window size for condition labeling
            
        Returns:
            Balanced list of market data points
        """
        # Define reasonable bounds for metrics
        METRIC_BOUNDS = {
            'avg_tx_value': (0.0, 1000.0),  # Reasonable transaction value range
            'avg_gas_used': (21000, 1000000),  # Standard gas usage range
            'avg_gas_price': (1, 500),  # Reasonable gas price range in GWEI
            'txn_growth_pct_7d': (-100, 100),  # Percent change bounds
            'user_growth_pct_7d': (-50, 50),  # Percent change bounds
            'tx_volatility_7d': (0, 100000)  # Reasonable volatility range
        }
        
        def normalize_metrics(data_point: Dict[str, Any]) -> Dict[str, Any]:
            """Normalize metrics to reasonable ranges."""
            normalized = data_point.copy()
            
            for metric, (min_val, max_val) in METRIC_BOUNDS.items():
                if metric in normalized:
                    try:
                        value = float(normalized[metric])
                        # Clip to reasonable bounds
                        normalized[metric] = max(min_val, min(value, max_val))
                    except (ValueError, TypeError):
                        # If conversion fails, use median value
                        normalized[metric] = (min_val + max_val) / 2
                        
            return normalized
        
        # Label market conditions
        labeled_data = []
        for i in range(window_size, len(market_data)):
            current = market_data[i]
            historical = market_data[i-window_size:i]
            
            # Normalize metrics before labeling
            normalized_current = normalize_metrics(current)
            normalized_historical = [normalize_metrics(h) for h in historical]
            
            condition = self._label_market_condition(normalized_current, normalized_historical)
            labeled_data.append((normalized_current, condition))
        
        # Group by condition
        condition_groups = defaultdict(list)
        for data, condition in labeled_data:
            condition_groups[condition].append(data)
        
        # Calculate target size (use mean of non-outlier sizes)
        sizes = [len(group) for group in condition_groups.values()]
        q1, q3 = np.percentile(sizes, [25, 75])
        iqr = q3 - q1
        valid_sizes = [s for s in sizes if q1 - 1.5*iqr <= s <= q3 + 1.5*iqr]
        target_size = int(np.mean(valid_sizes)) if valid_sizes else int(np.mean(sizes))
        
        # Balance each group to the target size
        balanced_data = []
        for condition, group in condition_groups.items():
            if len(group) > target_size:
                # Downsample using systematic sampling
                indices = np.linspace(0, len(group)-1, target_size, dtype=int)
                balanced_group = [group[i] for i in indices]
            else:
                # Upsample using SMOTE-like approach with normalized interpolation
                balanced_group = []
                while len(balanced_group) < target_size:
                    if len(group) == 1:
                        # If only one sample, duplicate it with controlled noise
                        sample = group[0].copy()
                        for metric in METRIC_BOUNDS.keys():
                            if metric in sample and isinstance(sample[metric], (int, float)):
                                min_val, max_val = METRIC_BOUNDS[metric]
                                current_val = float(sample[metric])
                                # Add noise within 10% of the valid range
                                noise_range = (max_val - min_val) * 0.1
                                noise = np.random.uniform(-noise_range, noise_range)
                                sample[metric] = max(min_val, min(current_val + noise, max_val))
                        balanced_group.append(sample)
                    else:
                        # Pick two random samples and interpolate
                        idx1, idx2 = np.random.choice(len(group), 2, replace=False)
                        sample1, sample2 = group[idx1], group[idx2]
                        alpha = np.random.random()
                        
                        interpolated = {}
                        # First copy non-numeric fields from sample1
                        for key in sample1.keys():
                            if not isinstance(sample1[key], (int, float)):
                                interpolated[key] = sample1[key]
                        
                        # Then interpolate numeric fields with bounds checking
                        for key in sample1.keys():
                            if key in sample2 and isinstance(sample1[key], (int, float)) and isinstance(sample2[key], (int, float)):
                                try:
                                    val1, val2 = float(sample1[key]), float(sample2[key])
                                    interpolated_val = alpha * val1 + (1-alpha) * val2
                                    
                                    # Apply bounds if metric has defined bounds
                                    if key in METRIC_BOUNDS:
                                        min_val, max_val = METRIC_BOUNDS[key]
                                        interpolated_val = max(min_val, min(interpolated_val, max_val))
                                    
                                    interpolated[key] = interpolated_val
                                except (ValueError, TypeError):
                                    interpolated[key] = sample1[key]
                            elif isinstance(sample1[key], (int, float)):
                                interpolated[key] = sample1[key]
                        
                        balanced_group.append(interpolated)
            
            balanced_data.extend(balanced_group)
        
        # Final normalization pass
        normalized_data = [normalize_metrics(d) for d in balanced_data]
        
        # Shuffle the final dataset
        np.random.shuffle(normalized_data)
        
        return normalized_data

    def _create_prediction_prompt(self, market_data: Dict[str, Any], outcome_data: Dict[str, Any]) -> str:
        """Create a prompt for market prediction task."""
        # Format metrics with proper error handling
        metrics = {
            'success_rate': float(market_data.get('success_rate', 0)) * 100,
            'txn_growth': float(market_data.get('txn_growth_pct_7d', 0)),
            'user_growth': float(market_data.get('user_growth_pct_7d', 0)),
            'tx_volatility': float(market_data.get('tx_volatility_7d', 0)),
            'avg_tx_value': float(market_data.get('avg_tx_value', 0)),
            'avg_gas_price': float(market_data.get('avg_gas_price', 0))
        }
        
        prompt = f"""<reasoning>
Given the following market data from {market_data.get('block_timestamp', 'N/A')}, predict and explain the likely market behavior over the next 7 days. Support all analysis with specific data citations and compare with historical patterns.

Current Market State:
1. Network Activity [cite as 'Network Metrics']
- Daily Transactions: {market_data.get('num_txs', 'N/A')}
- Success Rate: {metrics['success_rate']:.2f}%
- Gas Price: {metrics['avg_gas_price']:.2f} GWEI

2. User Metrics [cite as 'User Data']
- Unique Users: {market_data.get('unique_senders', 'N/A')}
- Transaction Growth: {metrics['txn_growth']:.1f}%
- User Growth: {metrics['user_growth']:.1f}%

3. Transaction Patterns [cite as 'Transaction Data']
- Average Value: {metrics['avg_tx_value']:.4f}
- Volatility: {metrics['tx_volatility']:.2f}
- Smart Contract Calls: {market_data.get('smart_contract_calls', 'N/A')}

4. Cross-Chain Context [cite as 'Cross-Chain Data']
- Network: {market_data.get('network', 'N/A')}
- Active Contracts: {market_data.get('active_contracts', 'N/A')}
- Bridge Volume: {market_data.get('bridge_volume', 'N/A')}

Historical Outcome Data (for pattern analysis):
- Transaction Change: {float(outcome_data.get('txn_growth_pct_7d', 0)):.1f}%
- User Growth Change: {float(outcome_data.get('user_growth_pct_7d', 0)):.1f}%
- Volatility Change: {float(outcome_data.get('tx_volatility_7d', 0)):.2f}

Required Analysis:
1. Data-Driven Predictions (cite specific metrics for each):
   a) Transaction Growth: Project exact percentage change based on historical patterns
   b) Gas Price Range: Specify min-max range in GWEI with supporting data
   c) User Growth: Project percentage change with confidence interval
   d) Cross-Chain Impact: 
      - List top 3 affected chains
      - Quantify expected volume changes
      - Specify correlation coefficients
   e) Smart Contract Activity: Project growth with supporting metrics

2. Technical Analysis (cite data points):
   a) Support/Resistance:
      - Calculate levels using transaction volumes
      - Identify key price points
      - Project breakout scenarios
   b) Volume Analysis:
      - Compare to 7-day average
      - Identify volume trends
      - Project future volume ranges

3. Risk Assessment (quantify each):
   a) Technical Risks:
      - Specify probability based on historical data
      - Quantify potential impact
      - List leading indicators
   b) Market Risks:
      - Calculate exposure metrics
      - Identify correlation risks
      - Project potential losses
   c) Cross-Chain Risks:
      - Analyze bridge volumes
      - Calculate contagion probability
      - Specify monitoring thresholds

4. Opportunity Analysis (with data support):
   a) Market Inefficiencies:
      - Calculate arbitrage spreads
      - Identify volume imbalances
      - Project potential returns
   b) Entry/Exit Points:
      - Specify exact trigger levels
      - Calculate position sizes
      - Define risk/reward ratios

Requirements:
1. Cite specific metrics for each prediction using [cite as 'X'] format
2. Compare all projections with historical outcome data
3. Provide exact numbers and confidence intervals
4. Keep analysis concise and data-focused
5. Include cross-chain correlation analysis
6. Support all conclusions with quantitative evidence

Format your response in clear sections with minimal prose. Focus on data-driven insights rather than general market commentary.</reasoning>"""

        return prompt
    
    def _create_correlation_prompt(self, chain_data: List[Dict]) -> str:
        """Create a prompt for cross-chain correlation analysis."""
        prompt = f"""<reasoning>
Analyze the correlations and relationships between the following blockchain networks:

{self._format_chain_metrics(chain_data)}

Required Analysis:
1. Correlation Analysis (provide specific metrics):
   a) Metric Correlations:
      - Transaction volume correlation coefficients
      - User growth correlation coefficients
      - Fee market correlation coefficients
   b) Temporal Patterns:
      - Lead/lag relationships (hours)
      - Seasonal patterns
      - Trend synchronization
   c) Divergence Analysis:
      - Key divergence points
      - Magnitude of divergences
      - Duration of divergences

2. Network Effects:
   a) Volume Impact:
      - Cross-chain volume elasticity
      - Volume spillover coefficients
      - Time to impact (minutes/hours)
   b) Fee Market Impact:
      - Gas price correlations
      - Fee market efficiency ratios
      - Arbitrage thresholds
   c) User Behavior:
      - Cross-chain user overlap (%)
      - Migration patterns
      - Activity synchronization

3. Market Implications:
   a) Arbitrage Opportunities:
      - Minimum profitable spreads
      - Required execution speed
      - Capital efficiency ratios
   b) Risk Factors:
      - Contagion risk coefficients
      - Systemic risk exposure
      - Correlation breakdown scenarios
   c) Trading Strategies:
      - Entry/exit spread levels
      - Volume requirements
      - Risk/reward ratios

4. Predictive Value:
   a) Leading Indicators:
      - Identify predictive metrics
      - Confidence intervals
      - Time horizons
   b) Signal Strength:
      - Statistical significance
      - False positive rates
      - Signal persistence
   c) Implementation:
      - Monitoring thresholds
      - Action triggers
      - Position sizing rules

Provide specific numbers, coefficients, and thresholds in your analysis. Support all conclusions with quantitative evidence.</reasoning>"""

        return prompt
    
    def _create_protocol_prompt(self, protocol_data: Dict, chain_data: Dict) -> str:
        """Create a prompt for protocol analysis across chains."""
        prompt = f"""<reasoning>
Analyze the following protocol's performance across different chains:

Protocol Metrics:
{self._format_protocol_metrics(protocol_data)}

Chain Context:
{self._format_chain_context(chain_data)}

Required Analysis:
1. Performance Analysis (provide specific metrics):
   a) Volume Metrics:
      - Daily volume (USD)
      - Market share (%)
      - Volume growth rate (%)
   b) User Metrics:
      - Daily active users
      - User retention rate (%)
      - Average user value (USD)
   c) Efficiency Metrics:
      - Transaction success rate (%)
      - Average gas cost (GWEI)
      - Cost per transaction (USD)

2. Cross-Chain Comparison:
   a) Relative Performance:
      - Volume rank by chain
      - Market share by chain
      - Growth rate differentials
   b) Chain-Specific Factors:
      - Gas efficiency ratio
      - User acquisition cost
      - Competition intensity
   c) Integration Quality:
      - Bridge efficiency (%)
      - Cross-chain latency (seconds)
      - Message success rate (%)

3. Growth Analysis:
   a) Growth Drivers:
      - User growth contribution (%)
      - Volume growth contribution (%)
      - Feature adoption rates (%)
   b) Growth Sustainability:
      - User economics (LTV/CAC)
      - Protocol revenue growth
      - Market penetration rate
   c) Growth Projections:
      - Short-term forecast (7d)
      - Medium-term forecast (30d)
      - Long-term forecast (90d)

4. Risk Assessment:
   a) Technical Risks:
      - Smart contract exposure
      - Oracle dependency
      - Integration complexity
   b) Economic Risks:
      - TVL concentration (%)
      - User concentration (%)
      - Revenue source diversity
   c) Competitive Risks:
      - Market share trends
      - Feature gap analysis
      - Cost competitiveness

5. Optimization Opportunities:
   a) Technical Optimization:
      - Gas optimization potential
      - Latency reduction targets
      - Integration improvements
   b) Economic Optimization:
      - Fee structure efficiency
      - Liquidity utilization
      - Yield optimization
   c) Strategic Optimization:
      - Cross-chain expansion
      - Feature prioritization
      - Partnership leverage

Provide specific numbers, percentages, and ratios in your analysis. Support all conclusions with quantitative evidence.</reasoning>"""

        return prompt

    def _create_risk_prompt(self, market_data: Dict[str, Any]) -> str:
        """Create a prompt focused on risk analysis."""
        prompt = f"""<reasoning>
Analyze the risk factors and potential vulnerabilities in the current market conditions:

Market Context:
{self._format_metrics(market_data)}

Required Analysis:
1. Risk Identification (provide specific metrics for each):
   a) Technical Risks:
      - Smart contract vulnerabilities (probability %)
      - Network congestion thresholds
      - Oracle failure scenarios
   b) Market Risks:
      - Volatility exposure (quantify)
      - Liquidity risks (measure)
      - Correlation risks (coefficients)
   c) Systemic Risks:
      - Cross-chain contagion paths
      - Protocol dependencies
      - Market concentration metrics

2. Impact Assessment:
   a) Quantitative Measures:
      - Value at Risk (VaR)
      - Expected shortfall
      - Maximum drawdown
   b) Probability Estimates:
      - Event likelihood (%)
      - Time horizon
      - Confidence intervals
   c) Scenario Analysis:
      - Best case metrics
      - Base case projections
      - Worst case thresholds

3. Mitigation Strategies:
   a) Risk Controls:
      - Specific thresholds
      - Monitoring frequency
      - Action triggers
   b) Hedging Strategies:
      - Instrument selection
      - Position sizing
      - Rebalancing rules
   c) Monitoring Framework:
      - Key risk indicators
      - Reporting frequency
      - Escalation criteria

Provide specific numbers, percentages, and thresholds in your analysis.</reasoning>"""

        return prompt

    def _create_opportunity_prompt(self, market_data: Dict[str, Any]) -> str:
        """Create a prompt focused on opportunity analysis."""
        prompt = f"""<reasoning>
Identify and analyze potential opportunities in the current market conditions:

Market Context:
{self._format_metrics(market_data)}

Required Analysis:
1. Opportunity Identification (quantify each):
   a) Market Inefficiencies:
      - Price discrepancies (%)
      - Volume imbalances
      - Timing advantages
   b) Yield Opportunities:
      - APY projections
      - Risk-adjusted returns
      - Lock-up requirements
   c) Arbitrage Potential:
      - Cross-chain spreads
      - Protocol differentials
      - Execution costs

2. Strategy Development:
   a) Entry/Exit Rules:
      - Specific price levels
      - Volume thresholds
      - Timing conditions
   b) Position Sizing:
      - Initial allocation (%)
      - Scaling rules
      - Maximum exposure
   c) Execution Requirements:
      - Technical requirements
      - Capital requirements
      - Operational needs

3. Risk/Reward Analysis:
   a) Return Projections:
      - Expected return (%)
      - Time horizon
      - Success probability
   b) Risk Assessment:
      - Maximum drawdown
      - Volatility exposure
      - Correlation factors
   c) Position Management:
      - Stop-loss levels
      - Take-profit targets
      - Rebalancing rules

Provide specific numbers, ratios, and thresholds in your analysis.</reasoning>"""

        return prompt

    def _create_market_qa_prompt(self, market_data: Dict[str, Any], outcome_data: Dict[str, Any]) -> str:
        """Create a historical market Q&A prompt."""
        metrics = self._format_metrics(market_data)
        outcome_metrics = self._format_metrics(outcome_data)
        
        prompt = f"""<reasoning>
Given the following historical market situation from {market_data.get('block_timestamp', 'N/A')}, analyze the conditions and predict the outcome. Support your prediction with detailed reasoning.

Market Context:
{metrics}

Question:
Based on these market conditions, what will be the likely trend in transaction volume and user growth over the next 7 days? Consider network effects, market sentiment, and cross-chain dynamics in your analysis.

Required Response Format:
1. Initial Observations
   - Key metrics analysis
   - Notable patterns
   - Market sentiment indicators

2. Chain of Reasoning
   - Step 1: [Your first logical step]
   - Step 2: [Your second logical step]
   - Step 3: [Your third logical step]
   ...

3. Prediction
   - Transaction volume trend (exact % change)
   - User growth projection (exact % change)
   - Confidence level (%)

4. Supporting Evidence
   - Technical indicators
   - On-chain metrics
   - Cross-chain factors

5. Risk Factors
   - Potential challenges
   - Market uncertainties
   - External influences

Historical Outcome (for training):
{outcome_metrics}

Explain your reasoning step by step, and support your conclusions with specific metrics from the data.</reasoning>"""
        
        return prompt
        
    def _create_analytical_prompt(self, market_data: Dict[str, Any]) -> str:
        """Create an analytical reasoning prompt."""
        metrics = self._format_metrics(market_data)
        
        prompt = f"""<reasoning>
Analyze the following complex market scenario and explain the implications through step-by-step reasoning:

Market Context:
{metrics}

Analytical Problem:
Given that the network shows increasing transaction volume (+{market_data.get('txn_growth_pct_7d', 0):.1f}%) 
but declining average transaction value (-{abs(float(market_data.get('avg_tx_value_change_pct', 0))):.1f}%), 
what does this imply about:
a) User behavior changes
b) Market microstructure
c) Potential arbitrage opportunities

Required Analysis Structure:
1. Data Pattern Recognition
   - Identify key metrics
   - Note correlations
   - Spot anomalies

2. Behavioral Analysis
   - User segments
   - Activity patterns
   - Motivation factors

3. Market Structure Implications
   - Liquidity dynamics
   - Price formation
   - Market efficiency

4. Opportunity Assessment
   - Arbitrage potential
   - Risk factors
   - Implementation considerations

Support each step with quantitative evidence and logical reasoning.</reasoning>"""
        
        return prompt
        
    def _create_financial_prompt(self, market_data: Dict[str, Any]) -> str:
        """Create a general financial reasoning prompt."""
        metrics = self._format_metrics(market_data)
        
        prompt = f"""<reasoning>
Apply fundamental financial reasoning to analyze the following market scenario:

Market Context:
{metrics}

Financial Analysis Problem:
The market shows the following characteristics:
- Gas prices: {market_data.get('avg_gas_price', 0):.2f} GWEI
- Success rate: {float(market_data.get('success_rate', 0)) * 100:.1f}%
- Smart contract calls: {market_data.get('smart_contract_calls', 'N/A')}

Calculate and analyze:
1. Transaction cost efficiency
2. Market depth implications
3. Yield optimization strategies

Required Analysis Framework:
1. Quantitative Analysis
   - Cost calculations
   - Efficiency metrics
   - Risk-adjusted returns

2. Economic Reasoning
   - Supply/demand dynamics
   - Price discovery process
   - Market equilibrium

3. Strategic Implications
   - Optimal transaction sizing
   - Timing considerations
   - Risk management

4. Recommendations
   - Action items
   - Implementation steps
   - Monitoring metrics

Show all calculations and explain your economic reasoning at each step.</reasoning>"""
        
        return prompt

    def generate_dataset(self,
                        market_data: List[Dict],
                        protocol_data: List[Dict],
                        output_path: str,
                        samples_per_prompt: int = 3) -> None:
        """Generate a synthetic dataset of reasoning examples.
        
        Args:
            market_data: Historical market data
            protocol_data: Protocol performance data
            output_path: Path to save generated data
            samples_per_prompt: Number of responses to generate per prompt
        """
        # Balance market conditions
        balanced_data = self._balance_market_conditions(market_data)
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Initialize progress tracking
        total_examples = len(balanced_data) * samples_per_prompt
        examples_generated = 0
        last_save_time = time.time()
        save_interval = 300  # Save every 5 minutes
        
        examples = []
        prompt_types = list(self.prompt_templates.keys())
        
        # Group prompt types by category
        prediction_prompts = ['prediction', 'market_qa']
        analytical_prompts = ['analytical', 'financial']
        market_prompts = ['correlation', 'protocol', 'risk', 'opportunity']
        
        try:
            # Generate examples for each market condition
            for i in range(0, len(balanced_data)-1):
                current_data = balanced_data[i]
                outcome_data = balanced_data[i+1]
                
                # Select prompt type based on position to ensure balanced representation
                if i % 3 == 0:
                    prompt_type = prediction_prompts[i // 3 % len(prediction_prompts)]
                elif i % 3 == 1:
                    prompt_type = analytical_prompts[i // 3 % len(analytical_prompts)]
                else:
                    prompt_type = market_prompts[i // 3 % len(market_prompts)]
                
                prompt_func = self.prompt_templates[prompt_type]
                
                # Generate prompt based on type
                try:
                    if prompt_type in prediction_prompts:
                        prompt = prompt_func(current_data, outcome_data)
                    elif prompt_type == 'protocol':
                        prompt = prompt_func(protocol_data, current_data)
                    else:
                        prompt = prompt_func(current_data)
                    
                    # Generate multiple responses per prompt
                    for _ in range(samples_per_prompt):
                        try:
                            result = self.openai_client.generate_completion(
                                prompt=prompt,
                                system_prompt=self._get_system_prompt(prompt_type),
                                model=self.model
                            )
                            
                            # Calculate rewards including group comparison
                            reward = self.reward_function.calculate_reward(
                                response=result,
                                prompt={
                                    'context_data': current_data,
                                    'outcome_data': outcome_data,
                                    'prompt_type': prompt_type
                                },
                                group_responses=[]  # We'll calculate group metrics later
                            )
                            
                            example = {
                                'type': prompt_type,
                                'prompt': prompt,
                                'response': result,
                                'context_data': current_data,
                                'outcome_data': outcome_data,
                                'market_condition': self._label_market_condition(
                                    current_data,
                                    market_data[max(0, i-7):i]
                                ),
                                'reward': reward
                            }
                            
                            examples.append(example)
                            examples_generated += 1
                            
                            # Save progress periodically
                            current_time = time.time()
                            if current_time - last_save_time > save_interval:
                                self._save_examples(examples, output_path)
                                last_save_time = current_time
                            
                            # Log progress
                            progress = (examples_generated / total_examples) * 100
                            logger.info(f"Progress: {progress:.1f}% ({examples_generated}/{total_examples} examples)")
                            
                        except Exception as e:
                            logger.error(f"Error generating example: {str(e)}")
                            continue
                        
                except Exception as e:
                    logger.error(f"Error creating prompt: {str(e)}")
                    continue
                
        except Exception as e:
            logger.error(f"Error in dataset generation: {str(e)}")
        finally:
            # Save any remaining examples
            if examples:
                self._save_examples(examples, output_path)
            
            # Log final dataset statistics
            self._log_dataset_stats(examples)

    def _format_chain_metrics(self, chain_data: Union[List[Dict], Dict]) -> str:
        """Format metrics for multiple chains.
        
        Args:
            chain_data: Chain metrics data (list of dicts or single dict)
            
        Returns:
            Formatted string of chain metrics
        """
        # Convert single dict to list
        if isinstance(chain_data, dict):
            chain_data = [chain_data]
            
        formatted = []
        for data in chain_data:
            try:
                metrics = {
                    'success_rate': float(data.get('success_rate', 0)) * 100,
                    'txn_growth': float(data.get('txn_growth_pct_7d', 0)),
                    'user_growth': float(data.get('user_growth_pct_7d', 0))
                }
                
                chain_str = f"""Chain: {data.get('network', 'N/A')}
- Daily Transactions: {data.get('num_txs', 'N/A')}
- Success Rate: {metrics['success_rate']:.2f}%
- Transaction Growth: {metrics['txn_growth']:.1f}%
- User Growth: {metrics['user_growth']:.1f}%
- Bridge Volume: {data.get('bridge_volume', 'N/A')}"""
                
                formatted.append(chain_str)
            except (TypeError, ValueError) as e:
                logger.warning(f"Error formatting chain metrics: {str(e)}")
                continue
            
        return "\n\n".join(formatted) if formatted else "No chain metrics available"
    
    def _format_protocol_metrics(self, protocol_data: Union[List[Dict], Dict]) -> str:
        """Format protocol metrics across chains.
        
        Args:
            protocol_data: Protocol metrics data (list of dicts or single dict)
            
        Returns:
            Formatted string of protocol metrics
        """
        # Convert list to dict by grouping by network
        if isinstance(protocol_data, list):
            grouped_data = {}
            for item in protocol_data:
                network = item.get('network', 'ethereum')
                if network not in grouped_data:
                    grouped_data[network] = {
                        'volume_usd': 0,
                        'unique_users': 0,
                        'volume_growth_pct': 0,
                        'volume_share': 0,
                        'success_rate': 0
                    }
                # Aggregate metrics
                grouped_data[network]['volume_usd'] += float(item.get('volume_usd', 0))
                grouped_data[network]['unique_users'] = max(
                    grouped_data[network]['unique_users'],
                    int(item.get('unique_users', 0))
                )
                grouped_data[network]['volume_growth_pct'] = float(item.get('volume_growth_pct', 0))
                grouped_data[network]['volume_share'] = float(item.get('volume_share', 0))
                grouped_data[network]['success_rate'] = float(item.get('success_rate', 0))
            
            protocol_data = grouped_data
        
        metrics = []
        for chain, data in protocol_data.items():
            try:
                chain_metrics = f"""Chain: {chain}
- Volume: ${data.get('volume_usd', 0):,.2f}
- Users: {data.get('unique_users', 0):,}
- Growth: {data.get('volume_growth_pct', 0):.1f}%
- Market Share: {data.get('volume_share', 0):.1f}%
- Success Rate: {data.get('success_rate', 0) * 100:.1f}%"""
                metrics.append(chain_metrics)
            except (TypeError, ValueError) as e:
                logger.warning(f"Error formatting protocol metrics for {chain}: {str(e)}")
                continue
            
        return "\n\n".join(metrics) if metrics else "No protocol metrics available"
    
    def _format_chain_context(self, chain_data: Dict) -> str:
        """Format chain context data."""
        metrics = {
            'success_rate': float(chain_data.get('success_rate', 0)) * 100,
            'txn_growth': float(chain_data.get('txn_growth_pct_7d', 0)),
            'user_growth': float(chain_data.get('user_growth_pct_7d', 0))
        }
        
        return f"""Network: {chain_data.get('network', 'N/A')}
- Daily Transactions: {chain_data.get('num_txs', 'N/A')}
- Success Rate: {metrics['success_rate']:.2f}%
- Transaction Growth: {metrics['txn_growth']:.1f}%
- User Growth: {metrics['user_growth']:.1f}%
- Smart Contract Activity: {chain_data.get('smart_contract_calls', 'N/A')}"""
    
    def _log_dataset_stats(self, examples: List[Dict]) -> None:
        """Log statistics about the generated dataset."""
        stats = {
            'total_examples': len(examples),
            'avg_reward': np.mean([ex['reward']['final_total'] for ex in examples]),
            'market_conditions': defaultdict(int),
            'prompt_types': defaultdict(int),
            'avg_prediction_accuracy': np.mean([
                ex['reward'].get('prediction_accuracy', 0) 
                for ex in examples
            ])
        }
        
        for ex in examples:
            stats['market_conditions'][ex['market_condition']] += 1
            stats['prompt_types'][ex['type']] += 1
        
        logger.info(f"Dataset Statistics:\n{json.dumps(stats, indent=2)}")
    
    def _get_system_prompt(self, prompt_type: str) -> str:
        """Get appropriate system prompt based on task type."""
        prompts = {
            'prediction': "You are an expert crypto analyst focused on prediction and cross-chain analysis.",
            'correlation': "You are an expert in cross-chain analysis and market correlations.",
            'protocol': "You are an expert in DeFi protocol analysis and optimization.",
            'risk': "You are an expert risk analyst specializing in crypto markets.",
            'opportunity': "You are an expert in identifying and analyzing market opportunities.",
            'market_qa': "You are a financial analyst with expertise in market analysis and forecasting.",
            'analytical': "You are a skilled analyst with expertise in complex market scenarios.",
            'financial': "You are a financial analyst with expertise in economic analysis and investment strategies."
        }
        return prompts.get(prompt_type, prompts['prediction'])

    def _format_metrics(self, market_data: Dict[str, Any]) -> str:
        """Format market metrics for prompts.
        
        Args:
            market_data: Dictionary of market metrics
            
        Returns:
            Formatted string of metrics
        """
        try:
            metrics = {
                'success_rate': float(market_data.get('success_rate', 0)) * 100,
                'txn_growth': float(market_data.get('txn_growth_pct_7d', 0)),
                'user_growth': float(market_data.get('user_growth_pct_7d', 0)),
                'volatility': float(market_data.get('tx_volatility_7d', 0)),
                'avg_value': float(market_data.get('avg_tx_value', 0)),
                'gas_price': float(market_data.get('avg_gas_price', 0))
            }
            
            return f"""Network: {market_data.get('network', 'N/A')}

1. Transaction Metrics
- Daily Volume: {market_data.get('num_txs', 'N/A')}
- Success Rate: {metrics['success_rate']:.2f}%
- Average Value: {metrics['avg_value']:.4f}
- Gas Price: {metrics['gas_price']:.2f} GWEI

2. Growth Metrics
- Transaction Growth: {metrics['txn_growth']:.1f}%
- User Growth: {metrics['user_growth']:.1f}%
- Volatility: {metrics['volatility']:.2f}

3. Activity Metrics
- Smart Contract Calls: {market_data.get('smart_contract_calls', 'N/A')}
- Bridge Volume: {market_data.get('bridge_volume', 'N/A')}
- Unique Users: {market_data.get('unique_users', 'N/A')}"""
            
        except (TypeError, ValueError) as e:
            logger.warning(f"Error formatting metrics: {str(e)}")
            return "No market metrics available"

    def _save_examples(self, examples: List[Dict], output_path: str) -> None:
        """Save examples to JSONL file with append mode.
        
        Args:
            examples: List of examples to save
            output_path: Path to save the examples
        """
        with open(output_path, 'a') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
        
        # Clear the examples list after saving
        examples.clear() 