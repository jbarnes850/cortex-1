"""
Historical Context Generator for NEAR Cortex-1

This module provides utilities for creating specialized prompts
that incorporate historical cryptocurrency market data and patterns
for synthetic data generation.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
import random
from typing import Dict, List, Any, Optional, Union, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HistoricalContextGenerator:
    """Generates historical contexts for synthetic data prompts."""
    
    def __init__(self, processed_dir: str = "data/processed", 
                 time_periods: List[str] = ["historical", "recent", "current"]):
        """
        Initialize the historical context generator.
        
        Args:
            processed_dir: Directory with processed data
            time_periods: List of time periods to include
        """
        self.processed_dir = Path(processed_dir)
        self.time_periods = time_periods
        self.historical_data = {}
        self.coins = set()
        
        # Load processed data
        self._load_processed_data()
    
    def _load_processed_data(self):
        """Load processed historical data from merged files."""
        merged_dir = self.processed_dir / "merged"
        
        if not merged_dir.exists():
            logger.warning(f"Merged data directory not found: {merged_dir}")
            return
        
        # Find the most recent merged file
        merged_files = list(merged_dir.glob("merged_*.json"))
        if not merged_files:
            logger.warning(f"No merged data files found in {merged_dir}")
            return
        
        # Sort by modification time (newest first)
        merged_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest_file = merged_files[0]
        
        # Load the data
        try:
            with open(latest_file, 'r') as f:
                data = json.load(f)
            
            # Group by chain and time period
            for record in data:
                chain = record.get("chain", "unknown")
                time_period = record.get("metadata", {}).get("time_period", "unknown")
                
                if chain not in self.historical_data:
                    self.historical_data[chain] = {}
                
                if time_period not in self.historical_data[chain]:
                    self.historical_data[chain][time_period] = []
                
                self.historical_data[chain][time_period].append(record)
                self.coins.add(chain)
            
            logger.info(f"Loaded {len(data)} records from {latest_file}")
            logger.info(f"Found data for {len(self.coins)} coins: {', '.join(self.coins)}")
            
            # Sort records by timestamp
            for chain in self.historical_data:
                for time_period in self.historical_data[chain]:
                    self.historical_data[chain][time_period].sort(
                        key=lambda x: x.get("timestamp", "")
                    )
            
        except Exception as e:
            logger.error(f"Error loading historical data from {latest_file}: {e}")
    
    def get_available_coins(self) -> List[str]:
        """Get the list of available coins in the historical data."""
        return list(self.coins)
    
    def get_available_time_periods(self, coin: str) -> List[str]:
        """Get the list of available time periods for a specific coin."""
        if coin not in self.historical_data:
            return []
        
        return list(self.historical_data[coin].keys())
    
    def get_data_summary(self, coin: str, time_period: Optional[str] = None) -> Dict:
        """Get a summary of the available data for a coin."""
        if coin not in self.historical_data:
            return {}
        
        summary = {}
        
        periods = [time_period] if time_period else self.historical_data[coin].keys()
        
        for period in periods:
            if period in self.historical_data[coin]:
                records = self.historical_data[coin][period]
                if records:
                    # Get first and last timestamps
                    first = records[0].get("timestamp", "")
                    last = records[-1].get("timestamp", "")
                    
                    # Calculate min, max, mean price
                    prices = [r.get("market_data", {}).get("price_usd", 0) for r in records]
                    prices = [p for p in prices if p > 0]
                    
                    min_price = min(prices) if prices else 0
                    max_price = max(prices) if prices else 0
                    avg_price = sum(prices) / len(prices) if prices else 0
                    
                    # Add to summary
                    summary[period] = {
                        "count": len(records),
                        "first_timestamp": first,
                        "last_timestamp": last,
                        "min_price": min_price,
                        "max_price": max_price,
                        "avg_price": avg_price
                    }
        
        return summary
    
    def get_historical_window(self, 
                             coin: str, 
                             window_size: int = 90, 
                             end_date: Optional[str] = None,
                             time_period: Optional[str] = None) -> List[Dict]:
        """
        Get a historical window of data for a specific coin.
        
        Args:
            coin: Coin symbol
            window_size: Number of days to include in the window
            end_date: End date of the window (ISO format), if None, use latest date
            time_period: Specific time period to use, if None, use all
            
        Returns:
            List of records in the window
        """
        if coin not in self.historical_data:
            logger.warning(f"Coin {coin} not found in historical data")
            return []
        
        # Collect all records for the coin across specified time periods
        all_records = []
        periods = [time_period] if time_period else self.historical_data[coin].keys()
        
        for period in periods:
            if period in self.historical_data[coin]:
                all_records.extend(self.historical_data[coin][period])
        
        # Sort by timestamp
        all_records.sort(key=lambda x: x.get("timestamp", ""))
        
        if not all_records:
            logger.warning(f"No records found for {coin}")
            return []
        
        # Determine end date
        if end_date:
            try:
                end_datetime = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            except ValueError:
                logger.warning(f"Invalid end date format: {end_date}")
                end_datetime = datetime.fromisoformat(all_records[-1].get("timestamp", "").replace('Z', '+00:00'))
        else:
            end_datetime = datetime.fromisoformat(all_records[-1].get("timestamp", "").replace('Z', '+00:00'))
        
        # Calculate start date
        start_datetime = end_datetime - timedelta(days=window_size)
        
        # Filter records by date range
        window_records = []
        for record in all_records:
            try:
                record_datetime = datetime.fromisoformat(record.get("timestamp", "").replace('Z', '+00:00'))
                if start_datetime <= record_datetime <= end_datetime:
                    window_records.append(record)
            except ValueError:
                continue
        
        return window_records
    
    def generate_context_prompt(self, 
                               coin: str, 
                               context_window: List[Dict], 
                               analysis_type: str = "price_trend",
                               include_metrics: bool = True) -> str:
        """
        Generate a prompt with historical context for synthetic data generation.
        
        Args:
            coin: Coin symbol
            context_window: List of historical records to include in the context
            analysis_type: Type of analysis to perform
            include_metrics: Whether to include derived metrics
            
        Returns:
            Prompt string with historical context
        """
        if not context_window:
            return f"Analyze recent price trends for {coin}."
        
        # Sort by timestamp
        context_window.sort(key=lambda x: x.get("timestamp", ""))
        
        # Get date range
        start_date = context_window[0].get("timestamp", "").split("T")[0]
        end_date = context_window[-1].get("timestamp", "").split("T")[0]
        
        # Prepare data points for the prompt
        data_points = []
        for record in context_window:
            timestamp = record.get("timestamp", "").split("T")[0]
            price = record.get("market_data", {}).get("price_usd", 0)
            volume = record.get("market_data", {}).get("daily_volume", 0)
            
            point = f"Date: {timestamp}, Price: ${price:.2f}"
            
            if volume > 0:
                point += f", Volume: ${volume:.2f}"
            
            if include_metrics:
                growth = record.get("derived_metrics", {}).get("growth_rate", 0)
                vol_rel = record.get("derived_metrics", {}).get("relative_volume", 0)
                percentile = record.get("derived_metrics", {}).get("historical_percentile", 0)
                
                metrics = []
                if growth != 0:
                    metrics.append(f"Growth: {growth:.2%}")
                if vol_rel > 0:
                    metrics.append(f"Rel Volume: {vol_rel:.2f}x")
                if percentile > 0:
                    metrics.append(f"Historical Percentile: {percentile:.2%}")
                
                if metrics:
                    point += f" ({', '.join(metrics)})"
            
            data_points.append(point)
        
        # Construct the prompt based on analysis type
        prompt = ""
        
        if analysis_type == "price_trend":
            prompt = f"""
You are analyzing historical price data for {coin.upper()} from {start_date} to {end_date}.
Based on the following historical data points:

{chr(10).join(data_points)}

Provide a detailed analysis of the price trends, including:
1. Overall price movement direction and magnitude
2. Key support and resistance levels
3. Significant price reversal points
4. Correlation with market events
5. Volume analysis and its relationship to price changes

Your analysis should focus on identifying patterns and explaining why they occurred.
"""
        
        elif analysis_type == "pattern_recognition":
            prompt = f"""
You are analyzing historical price data for {coin.upper()} from {start_date} to {end_date}.
Based on the following historical data points:

{chr(10).join(data_points)}

Identify and explain the following price patterns if present:
1. Head and shoulders
2. Double tops/bottoms
3. Cup and handle
4. Flags and pennants
5. Triangles (ascending, descending, symmetrical)
6. Support/resistance breakouts
7. Fibonacci retracement levels

For each pattern identified, explain its formation, significance, and potential future implications.
"""
        
        elif analysis_type == "volatility_analysis":
            prompt = f"""
You are analyzing historical price data for {coin.upper()} from {start_date} to {end_date}.
Based on the following historical data points:

{chr(10).join(data_points)}

Provide a detailed volatility analysis, including:
1. Periods of high vs. low volatility
2. Volatility clustering
3. Relationship between volatility and price direction
4. Comparison to historical volatility norms
5. Impact of market events on volatility

Your analysis should identify patterns in volatility and explain potential causes and effects.
"""
        
        elif analysis_type == "market_cycle":
            prompt = f"""
You are analyzing historical price data for {coin.upper()} from {start_date} to {end_date}.
Based on the following historical data points:

{chr(10).join(data_points)}

Analyze the market cycle phase, including:
1. Identification of current market phase (accumulation, uptrend, distribution, downtrend)
2. Key indicators supporting your phase identification
3. Estimated progress within the identified phase
4. Comparison to previous market cycles
5. Potential upcoming phase transitions

Your analysis should consider both technical indicators and market sentiment in identifying cycle phases.
"""
        
        else:
            # Default general analysis
            prompt = f"""
You are analyzing historical price data for {coin.upper()} from {start_date} to {end_date}.
Based on the following historical data points:

{chr(10).join(data_points)}

Provide a comprehensive market analysis, including:
1. Price trend analysis
2. Trading volume analysis
3. Market pattern recognition
4. Volatility assessment
5. Key support and resistance levels
6. Potential future outlook

Your analysis should be detailed and supported by the provided data points.
"""
        
        return prompt.strip()
    
    def generate_temporal_mixing_prompt(self,
                                       coin: str,
                                       historical_window: List[Dict],
                                       current_window: List[Dict],
                                       question_type: str = "comparison") -> str:
        """
        Generate a prompt that requires reasoning across different time periods.
        
        Args:
            coin: Coin symbol
            historical_window: List of historical records
            current_window: List of current records
            question_type: Type of question to generate
            
        Returns:
            Prompt string requiring cross-temporal reasoning
        """
        if not historical_window or not current_window:
            return f"Compare historical and current price trends for {coin}."
        
        # Sort by timestamp
        historical_window.sort(key=lambda x: x.get("timestamp", ""))
        current_window.sort(key=lambda x: x.get("timestamp", ""))
        
        # Get date ranges
        historical_start = historical_window[0].get("timestamp", "").split("T")[0]
        historical_end = historical_window[-1].get("timestamp", "").split("T")[0]
        current_start = current_window[0].get("timestamp", "").split("T")[0]
        current_end = current_window[-1].get("timestamp", "").split("T")[0]
        
        # Extract key metrics
        def extract_metrics(window):
            prices = [r.get("market_data", {}).get("price_usd", 0) for r in window]
            volumes = [r.get("market_data", {}).get("daily_volume", 0) for r in window]
            volatilities = [r.get("market_data", {}).get("volatility", 0) for r in window]
            
            return {
                "start_price": prices[0] if prices else 0,
                "end_price": prices[-1] if prices else 0,
                "min_price": min(prices) if prices else 0,
                "max_price": max(prices) if prices else 0,
                "avg_price": sum(prices) / len(prices) if prices else 0,
                "avg_volume": sum(volumes) / len(volumes) if volumes and any(volumes) else 0,
                "avg_volatility": sum(volatilities) / len(volatilities) if volatilities and any(volatilities) else 0
            }
        
        historical_metrics = extract_metrics(historical_window)
        current_metrics = extract_metrics(current_window)
        
        # Calculate price change percentages
        historical_change = ((historical_metrics["end_price"] - historical_metrics["start_price"]) / 
                            historical_metrics["start_price"]) if historical_metrics["start_price"] > 0 else 0
        
        current_change = ((current_metrics["end_price"] - current_metrics["start_price"]) / 
                         current_metrics["start_price"]) if current_metrics["start_price"] > 0 else 0
        
        # Construct the prompt based on question type
        prompt = ""
        
        if question_type == "comparison":
            prompt = f"""
You are analyzing {coin.upper()} price data across two time periods:
- Historical Period: {historical_start} to {historical_end}
- Current Period: {current_start} to {current_end}

Historical period summary:
- Starting price: ${historical_metrics["start_price"]:.2f}
- Ending price: ${historical_metrics["end_price"]:.2f}
- Price change: {historical_change:.2%}
- Average daily volume: ${historical_metrics["avg_volume"]:.2f}
- Average volatility: {historical_metrics["avg_volatility"]:.4f}

Current period summary:
- Starting price: ${current_metrics["start_price"]:.2f}
- Ending price: ${current_metrics["end_price"]:.2f}
- Price change: {current_change:.2%}
- Average daily volume: ${current_metrics["avg_volume"]:.2f}
- Average volatility: {current_metrics["avg_volatility"]:.4f}

Compare these two periods and provide insights on:
1. Similarities and differences in price action
2. Volume and volatility comparison
3. Market structure comparison (supports, resistances, patterns)
4. What historical patterns might be repeating in the current period
5. What lessons from the historical period can be applied to the current market

Your analysis should highlight meaningful connections between these two periods.
"""
        
        elif question_type == "prediction":
            prompt = f"""
You are analyzing {coin.upper()} price data to predict future movements:
- Historical Reference Period: {historical_start} to {historical_end}
- Current Period: {current_start} to {current_end}

Historical reference data:
- Starting price: ${historical_metrics["start_price"]:.2f}
- Ending price: ${historical_metrics["end_price"]:.2f}
- Price change: {historical_change:.2%}
- Price range: ${historical_metrics["min_price"]:.2f} to ${historical_metrics["max_price"]:.2f}
- Average volatility: {historical_metrics["avg_volatility"]:.4f}

Current market data:
- Starting price: ${current_metrics["start_price"]:.2f}
- Current price: ${current_metrics["end_price"]:.2f}
- Price change so far: {current_change:.2%}
- Price range: ${current_metrics["min_price"]:.2f} to ${current_metrics["max_price"]:.2f}
- Average volatility: {current_metrics["avg_volatility"]:.4f}

Based on similarities between these periods, predict:
1. The likely short-term price direction (next 7-14 days)
2. Key price levels to watch (support, resistance, targets)
3. Expected volatility changes
4. Potential catalysts that could change the projected path
5. The confidence level of your prediction and why

Your prediction should explain which historical patterns inform your forecast and how the current context might differ.
"""
        
        elif question_type == "cycle_analysis":
            prompt = f"""
You are analyzing {coin.upper()} market cycles by comparing:
- Historical Period: {historical_start} to {historical_end}
- Current Period: {current_start} to {current_end}

Historical cycle data:
- Price range: ${historical_metrics["min_price"]:.2f} to ${historical_metrics["max_price"]:.2f} (amplitude: {(historical_metrics["max_price"]/historical_metrics["min_price"] - 1):.2%})
- Duration: {(datetime.fromisoformat(historical_end) - datetime.fromisoformat(historical_start)).days} days
- Average price: ${historical_metrics["avg_price"]:.2f}
- Net change: {historical_change:.2%}

Current market data:
- Price range: ${current_metrics["min_price"]:.2f} to ${current_metrics["max_price"]:.2f} (amplitude: {(current_metrics["max_price"]/current_metrics["min_price"] - 1):.2%})
- Duration so far: {(datetime.fromisoformat(current_end) - datetime.fromisoformat(current_start)).days} days
- Average price: ${current_metrics["avg_price"]:.2f}
- Net change so far: {current_change:.2%}

Perform a detailed market cycle analysis:
1. Identify the current cycle phase compared to the historical reference
2. Estimate the current position within the overall market cycle
3. Project the potential remaining duration of the current cycle phase
4. Identify key cycle indicators and oscillators that support your analysis
5. Discuss similarities and differences between the two cycle periods

Your analysis should consider both technical cycle analysis and fundamental factors affecting market cycles.
"""
        
        else:
            # Default pattern recognition
            prompt = f"""
You are analyzing {coin.upper()} price patterns across two time periods:
- Historical Period: {historical_start} to {historical_end}
- Current Period: {current_start} to {current_end}

Historical market summary:
- Price moved from ${historical_metrics["start_price"]:.2f} to ${historical_metrics["end_price"]:.2f} ({historical_change:.2%})
- Price range: ${historical_metrics["min_price"]:.2f} to ${historical_metrics["max_price"]:.2f}
- Average volatility: {historical_metrics["avg_volatility"]:.4f}

Current market summary:
- Price moved from ${current_metrics["start_price"]:.2f} to ${current_metrics["end_price"]:.2f} ({current_change:.2%})
- Price range: ${current_metrics["min_price"]:.2f} to ${current_metrics["max_price"]:.2f}
- Average volatility: {current_metrics["avg_volatility"]:.4f}

Identify and analyze recurring patterns:
1. Technical patterns present in both periods
2. Volume profiles and their similarities/differences
3. Key support/resistance levels and their significance
4. Volatility patterns and how they've evolved
5. Market sentiment indicators and their correlation

Your analysis should identify which patterns from the historical period are most relevant to understanding the current market situation.
"""
        
        return prompt.strip()
    
    def sample_historical_data(self, 
                              coin: str, 
                              time_period: Optional[str] = None,
                              sample_size: int = 10,
                              random_seed: Optional[int] = None) -> List[Dict]:
        """
        Sample random data points from the historical data.
        
        Args:
            coin: Coin symbol
            time_period: Specific time period to sample from
            sample_size: Number of data points to sample
            random_seed: Random seed for reproducibility
            
        Returns:
            List of sampled data points
        """
        if coin not in self.historical_data:
            logger.warning(f"Coin {coin} not found in historical data")
            return []
        
        # Set random seed if provided
        if random_seed is not None:
            random.seed(random_seed)
        
        # Collect all records for the coin across specified time periods
        all_records = []
        periods = [time_period] if time_period else self.historical_data[coin].keys()
        
        for period in periods:
            if period in self.historical_data[coin]:
                all_records.extend(self.historical_data[coin][period])
        
        # Sample randomly
        if len(all_records) <= sample_size:
            return all_records
        
        return random.sample(all_records, sample_size)
    
    def get_weighted_samples(self,
                           coin: str,
                           sample_size: int = 30,
                           recency_weight: float = 0.7,
                           random_seed: Optional[int] = None) -> List[Dict]:
        """
        Get weighted samples from historical data, favoring more recent data.
        
        Args:
            coin: Coin symbol
            sample_size: Number of data points to sample
            recency_weight: Weight to give to recency (0-1)
            random_seed: Random seed for reproducibility
            
        Returns:
            List of weighted sampled data points
        """
        if coin not in self.historical_data:
            logger.warning(f"Coin {coin} not found in historical data")
            return []
        
        # Set random seed if provided
        if random_seed is not None:
            random.seed(random_seed)
        
        # Collect all records for the coin
        all_records = []
        for period in self.historical_data[coin].keys():
            all_records.extend(self.historical_data[coin][period])
        
        # Sort by timestamp
        all_records.sort(key=lambda x: x.get("timestamp", ""))
        
        if not all_records:
            return []
        
        # Calculate weights based on position in the list
        weights = []
        n = len(all_records)
        for i in range(n):
            # Linear weighting from oldest to newest
            # Adjust the formula to give more weight to recent data
            weight = (1 - recency_weight) + recency_weight * (i / (n - 1)) if n > 1 else 1.0
            weights.append(weight)
        
        # Sample with weights
        if n <= sample_size:
            return all_records
        
        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]
        
        indices = random.choices(range(n), weights=weights, k=sample_size)
        return [all_records[i] for i in indices]


if __name__ == "__main__":
    # Example usage
    generator = HistoricalContextGenerator()
    
    # Get available coins
    coins = generator.get_available_coins()
    print(f"Available coins: {coins}")
    
    # If coins are available
    if coins:
        # Get a sample coin
        coin = coins[0]
        
        # Get data summary
        summary = generator.get_data_summary(coin)
        print(f"\nData summary for {coin}:")
        for period, data in summary.items():
            print(f"  {period}: {data['count']} records from {data['first_timestamp']} to {data['last_timestamp']}")
            print(f"    Price range: ${data['min_price']:.2f} - ${data['max_price']:.2f} (avg: ${data['avg_price']:.2f})")
        
        # Generate a sample prompt
        window = generator.get_historical_window(coin, window_size=60)
        print(f"\nGenerated {len(window)} records for the historical window")
        
        prompt = generator.generate_context_prompt(coin, window, analysis_type="price_trend")
        print("\nSample Prompt:")
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
        
        # Generate a temporal mixing prompt if we have both historical and recent data
        historical_window = generator.get_historical_window(coin, window_size=60, time_period="historical")
        current_window = generator.get_historical_window(coin, window_size=30, time_period="recent")
        
        if historical_window and current_window:
            print(f"\nGenerated {len(historical_window)} historical records and {len(current_window)} current records")
            
            prompt = generator.generate_temporal_mixing_prompt(coin, historical_window, current_window)
            print("\nSample Temporal Mixing Prompt:")
            print(prompt[:500] + "..." if len(prompt) > 500 else prompt) 