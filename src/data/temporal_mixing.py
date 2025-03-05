"""
Temporal Mixing Strategy for NEAR Cortex-1

This module provides utilities for creating mixed datasets
with weighted sampling based on temporal relevance.
"""

import os
import json
import math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
import random
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

from src.data.historical_context import HistoricalContextGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TemporalMixingStrategy:
    """Implements temporal mixing strategies for dataset creation."""
    
    def __init__(self, historical_context: Optional[HistoricalContextGenerator] = None,
                 processed_dir: str = "data/processed"):
        """
        Initialize the temporal mixing strategy.
        
        Args:
            historical_context: Historical context generator
            processed_dir: Directory with processed data
        """
        self.processed_dir = Path(processed_dir)
        
        # Initialize historical context generator if not provided
        if historical_context is None:
            self.historical_context = HistoricalContextGenerator(processed_dir=processed_dir)
        else:
            self.historical_context = historical_context
    
    def calculate_temporal_weight(self, 
                                 timestamp: str, 
                                 current_time: Optional[datetime] = None, 
                                 alpha: float = 0.7,
                                 decay_days: int = 365) -> float:
        """
        Calculate weight based on temporal distance from current time.
        
        Args:
            timestamp: The timestamp of the data point
            current_time: Current reference time (defaults to now)
            alpha: Decay factor (higher values prioritize recent data more)
            decay_days: Number of days over which to decay the weight
            
        Returns:
            float: Weight between 0 and 1
        """
        try:
            # Parse timestamp
            if isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                dt = timestamp
            
            # Use current time if not provided
            if current_time is None:
                current_time = datetime.now()
            
            # Calculate days difference
            time_diff = (current_time - dt).total_seconds() / (3600 * 24)  # days
            
            # Calculate weight using exponential decay
            weight = math.exp(-alpha * time_diff / decay_days)
            
            return weight
            
        except Exception as e:
            logger.warning(f"Error calculating temporal weight for {timestamp}: {e}")
            return 0.0
    
    def create_weighted_dataset(self,
                               coins: List[str],
                               historical_ratio: float = 0.5,
                               current_ratio: float = 0.3,
                               recent_ratio: float = 0.2,
                               sample_size: int = 100,
                               random_seed: Optional[int] = None) -> List[Dict]:
        """
        Create a weighted dataset with specified ratios of historical, recent, and current data.
        
        Args:
            coins: List of coins to include
            historical_ratio: Proportion of historical data
            current_ratio: Proportion of current data
            recent_ratio: Proportion of recent data
            sample_size: Total size of the dataset
            random_seed: Random seed for reproducibility
            
        Returns:
            List of sampled data points
        """
        if random_seed is not None:
            random.seed(random_seed)
        
        # Calculate counts for each period
        historical_count = int(sample_size * historical_ratio)
        current_count = int(sample_size * current_ratio)
        recent_count = sample_size - historical_count - current_count
        
        # Collect all data
        all_data = []
        
        for coin in coins:
            if coin not in self.historical_context.historical_data:
                logger.warning(f"Coin {coin} not found in historical data")
                continue
                
            # Sample from each time period
            for period, count in [("historical", historical_count), 
                                 ("current", current_count), 
                                 ("recent", recent_count)]:
                
                if period not in self.historical_context.historical_data[coin]:
                    logger.warning(f"Time period {period} not found for {coin}")
                    continue
                
                records = self.historical_context.historical_data[coin][period]
                
                # Calculate per-coin count
                per_coin_count = max(1, count // len(coins))
                
                # Sample records
                if len(records) <= per_coin_count:
                    sampled = records
                else:
                    sampled = random.sample(records, per_coin_count)
                
                all_data.extend(sampled)
        
        # If we didn't get enough data, use random sampling to fill the gap
        if len(all_data) < sample_size:
            # Get all available records
            all_records = []
            for coin in coins:
                if coin in self.historical_context.historical_data:
                    for period in self.historical_context.historical_data[coin]:
                        all_records.extend(self.historical_context.historical_data[coin][period])
            
            # Sample randomly to make up the difference
            remaining = sample_size - len(all_data)
            if len(all_records) > remaining:
                additional = random.sample(all_records, remaining)
                all_data.extend(additional)
            else:
                all_data.extend(all_records)
        
        # Shuffle the data
        random.shuffle(all_data)
        
        # Trim to sample size
        return all_data[:sample_size]
    
    def create_exponential_decay_dataset(self,
                                        coins: List[str],
                                        alpha: float = 0.7,
                                        sample_size: int = 100,
                                        random_seed: Optional[int] = None) -> List[Dict]:
        """
        Create a dataset with exponential decay weighting based on recency.
        
        Args:
            coins: List of coins to include
            alpha: Decay factor (higher values prioritize recent data more)
            sample_size: Total size of the dataset
            random_seed: Random seed for reproducibility
            
        Returns:
            List of sampled data points
        """
        if random_seed is not None:
            random.seed(random_seed)
        
        current_time = datetime.now()
        all_records = []
        all_weights = []
        
        # Collect all records and calculate weights
        for coin in coins:
            if coin not in self.historical_context.historical_data:
                logger.warning(f"Coin {coin} not found in historical data")
                continue
            
            for period in self.historical_context.historical_data[coin]:
                records = self.historical_context.historical_data[coin][period]
                
                for record in records:
                    timestamp = record.get("timestamp", "")
                    weight = self.calculate_temporal_weight(timestamp, current_time, alpha)
                    
                    all_records.append(record)
                    all_weights.append(weight)
        
        # Normalize weights
        if not all_weights:
            logger.warning("No records found for specified coins")
            return []
            
        total_weight = sum(all_weights)
        normalized_weights = [w / total_weight for w in all_weights]
        
        # Sample with replacement using weights
        if len(all_records) <= sample_size:
            return all_records
        
        indices = random.choices(range(len(all_records)), weights=normalized_weights, k=sample_size)
        sampled_records = [all_records[i] for i in indices]
        
        return sampled_records
    
    def create_stratified_temporal_dataset(self,
                                          coins: List[str],
                                          strata_configs: List[Dict],
                                          sample_size: int = 100,
                                          random_seed: Optional[int] = None) -> List[Dict]:
        """
        Create a dataset with stratified sampling based on temporal periods.
        
        Args:
            coins: List of coins to include
            strata_configs: List of strata configurations, each with 'period', 'ratio', and optional 'condition'
            sample_size: Total size of the dataset
            random_seed: Random seed for reproducibility
            
        Returns:
            List of sampled data points
        """
        if random_seed is not None:
            random.seed(random_seed)
        
        # Calculate counts for each stratum
        total_ratio = sum(config.get("ratio", 0) for config in strata_configs)
        if total_ratio <= 0:
            logger.warning("Invalid strata ratios, defaulting to equal distribution")
            for config in strata_configs:
                config["ratio"] = 1 / len(strata_configs)
            total_ratio = 1.0
        
        # Normalize ratios
        for config in strata_configs:
            config["ratio"] = config.get("ratio", 0) / total_ratio
            config["count"] = int(sample_size * config["ratio"])
        
        # Adjust last stratum to ensure total is exactly sample_size
        total_assigned = sum(config["count"] for config in strata_configs)
        if total_assigned < sample_size:
            strata_configs[-1]["count"] += (sample_size - total_assigned)
        
        # Collect data for each stratum
        all_data = []
        
        for config in strata_configs:
            period = config.get("period", "all")
            count = config.get("count", 0)
            condition_func = config.get("condition", None)
            
            # Skip if count is zero
            if count <= 0:
                continue
            
            # Collect eligible records
            eligible_records = []
            
            for coin in coins:
                if coin not in self.historical_context.historical_data:
                    continue
                
                if period == "all":
                    # Use all periods
                    for p in self.historical_context.historical_data[coin]:
                        records = self.historical_context.historical_data[coin][p]
                        
                        # Apply condition if provided
                        if condition_func is not None:
                            records = [r for r in records if condition_func(r)]
                        
                        eligible_records.extend(records)
                else:
                    # Use specific period
                    if period in self.historical_context.historical_data[coin]:
                        records = self.historical_context.historical_data[coin][period]
                        
                        # Apply condition if provided
                        if condition_func is not None:
                            records = [r for r in records if condition_func(r)]
                        
                        eligible_records.extend(records)
            
            # Sample records
            if len(eligible_records) <= count:
                sampled = eligible_records
            else:
                sampled = random.sample(eligible_records, count)
            
            all_data.extend(sampled)
        
        # Shuffle the data
        random.shuffle(all_data)
        
        return all_data
    
    def create_time_sliced_dataset(self,
                                  coins: List[str],
                                  slice_configs: List[Dict],
                                  sample_size: int = 100,
                                  random_seed: Optional[int] = None) -> List[Dict]:
        """
        Create a dataset with specific time slices.
        
        Args:
            coins: List of coins to include
            slice_configs: List of time slice configurations, each with 'start_date', 'end_date', and 'ratio'
            sample_size: Total size of the dataset
            random_seed: Random seed for reproducibility
            
        Returns:
            List of sampled data points
        """
        if random_seed is not None:
            random.seed(random_seed)
        
        # Calculate counts for each slice
        total_ratio = sum(config.get("ratio", 0) for config in slice_configs)
        if total_ratio <= 0:
            logger.warning("Invalid slice ratios, defaulting to equal distribution")
            for config in slice_configs:
                config["ratio"] = 1 / len(slice_configs)
            total_ratio = 1.0
        
        # Normalize ratios
        for config in slice_configs:
            config["ratio"] = config.get("ratio", 0) / total_ratio
            config["count"] = int(sample_size * config["ratio"])
        
        # Adjust last slice to ensure total is exactly sample_size
        total_assigned = sum(config["count"] for config in slice_configs)
        if total_assigned < sample_size:
            slice_configs[-1]["count"] += (sample_size - total_assigned)
        
        # Collect data for each slice
        all_data = []
        
        for config in slice_configs:
            start_date = config.get("start_date", None)
            end_date = config.get("end_date", None)
            count = config.get("count", 0)
            
            # Skip if count is zero
            if count <= 0:
                continue
            
            # Collect eligible records
            eligible_records = []
            
            for coin in coins:
                # Get historical window for this time slice
                window = self.historical_context.get_historical_window(
                    coin, 
                    end_date=end_date, 
                    window_size=1000000  # Large window to get all data
                )
                
                # Filter by start date if provided
                if start_date:
                    try:
                        start_datetime = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                        window = [
                            r for r in window 
                            if datetime.fromisoformat(r.get("timestamp", "").replace('Z', '+00:00')) >= start_datetime
                        ]
                    except ValueError:
                        logger.warning(f"Invalid start date format: {start_date}")
                
                eligible_records.extend(window)
            
            # Sample records
            if len(eligible_records) <= count:
                sampled = eligible_records
            else:
                sampled = random.sample(eligible_records, count)
            
            all_data.extend(sampled)
        
        # Shuffle the data
        random.shuffle(all_data)
        
        return all_data
    
    def save_mixed_dataset(self,
                          mixed_data: List[Dict],
                          output_dir: str = "data/synthetic/hybrid",
                          file_prefix: str = "temporal_mixed") -> Path:
        """
        Save a mixed dataset to a file.
        
        Args:
            mixed_data: List of data points
            output_dir: Directory to save the file
            file_prefix: Prefix for the output file name
            
        Returns:
            Path to the saved file
        """
        # Create timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        # Create target directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create output file
        output_file = output_path / f"{file_prefix}_{timestamp}.json"
        
        # Save to JSON
        with open(output_file, 'w') as f:
            json.dump(mixed_data, f, indent=2)
        
        logger.info(f"Saved {len(mixed_data)} records to {output_file}")
        
        return output_file


if __name__ == "__main__":
    # Example usage
    context_generator = HistoricalContextGenerator()
    mixing_strategy = TemporalMixingStrategy(historical_context=context_generator)
    
    # Get available coins
    coins = context_generator.get_available_coins()
    print(f"Available coins: {coins}")
    
    if coins:
        # Create a weighted dataset
        weighted_data = mixing_strategy.create_weighted_dataset(
            coins=coins,
            historical_ratio=0.6,
            current_ratio=0.2,
            recent_ratio=0.2,
            sample_size=50
        )
        
        print(f"\nCreated weighted dataset with {len(weighted_data)} records")
        
        # Create an exponential decay dataset
        decay_data = mixing_strategy.create_exponential_decay_dataset(
            coins=coins,
            alpha=0.7,
            sample_size=50
        )
        
        print(f"Created exponential decay dataset with {len(decay_data)} records")
        
        # Create a stratified dataset
        strata_configs = [
            {"period": "historical", "ratio": 0.5},
            {"period": "recent", "ratio": 0.3},
            {"period": "current", "ratio": 0.2}
        ]
        
        stratified_data = mixing_strategy.create_stratified_temporal_dataset(
            coins=coins,
            strata_configs=strata_configs,
            sample_size=50
        )
        
        print(f"Created stratified dataset with {len(stratified_data)} records")
        
        # Save the mixed dataset as an example
        saved_file = mixing_strategy.save_mixed_dataset(
            mixed_data=weighted_data,
            file_prefix="example_weighted"
        )
        
        print(f"Saved mixed dataset to {saved_file}") 