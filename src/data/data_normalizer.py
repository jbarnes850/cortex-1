"""
Data Normalization Module for NEAR Cortex-1

This module handles standardization and normalization of data
from different sources (Kaggle historical data and Flipside current data)
into a unified schema.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Optional, Union, Tuple, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Standard schema for market data
STANDARD_SCHEMA = {
    "timestamp": "ISO-8601 format",
    "chain": "string",
    "market_data": {
        "price_usd": "float",
        "daily_volume": "float",
        "market_cap": "float",
        "volatility": "float",
        "transaction_count": "int",
        "unique_addresses": "int",
        "average_transaction_value": "float",
        "gas_price": "float"
    },
    "derived_metrics": {
        "growth_rate": "float",
        "relative_volume": "float",
        "historical_percentile": "float"
    },
    "temporal_context": {
        "period_type": "string",
        "relative_to_ath": "float",
        "days_since_major_event": "int"
    },
    "metadata": {
        "data_source": "string",
        "time_period": "string",
        "confidence_level": "float"
    }
}

class DataNormalizer:
    """Handles normalization of data from different sources into a unified schema."""
    
    def __init__(self, processed_dir: str = "data/processed"):
        """
        Initialize the data normalizer.
        
        Args:
            processed_dir: Directory to store processed data
        """
        self.processed_dir = Path(processed_dir)
        
        # Create directories if they don't exist
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        (self.processed_dir / "historical").mkdir(exist_ok=True)
        (self.processed_dir / "current").mkdir(exist_ok=True)
        (self.processed_dir / "merged").mkdir(exist_ok=True)
    
    def normalize_kaggle_price_analysis(self, df: pd.DataFrame, coin_name: str) -> List[Dict[str, Any]]:
        """
        Normalize data from the Cryptocurrency Price Analysis dataset.
        
        Args:
            df: DataFrame with cryptocurrency data
            coin_name: Name of the cryptocurrency
            
        Returns:
            List of normalized data records
        """
        normalized_records = []
        
        for _, row in df.iterrows():
            try:
                # Extract date and convert to ISO format
                date_obj = row['Date']
                timestamp = date_obj.isoformat()
                
                # Calculate volatility (if not already present)
                volatility = row.get('Volatility', 0.0)
                if volatility == 0.0 and 'High' in row and 'Low' in row:
                    volatility = (row['High'] - row['Low']) / row['Low'] if row['Low'] > 0 else 0.0
                
                # Calculate growth rate
                growth_rate = 0.0
                if 'Close' in row and 'Open' in row and row['Open'] > 0:
                    growth_rate = (row['Close'] - row['Open']) / row['Open']
                
                # Map to standard schema
                record = {
                    "timestamp": timestamp,
                    "chain": coin_name.lower(),
                    "market_data": {
                        "price_usd": row.get('Close', 0.0),
                        "daily_volume": row.get('Volume', 0.0),
                        "market_cap": row.get('Market Cap', 0.0),
                        "volatility": volatility,
                        "transaction_count": 0,  # Not available in this dataset
                        "unique_addresses": 0,   # Not available in this dataset
                        "average_transaction_value": 0.0,  # Not available
                        "gas_price": 0.0,        # Not available in this dataset
                        "source": "kaggle_price_analysis",
                        "confidence": 0.9  # High confidence in price data
                    },
                    "derived_metrics": {
                        "growth_rate": growth_rate,
                        "relative_volume": 0.0,  # Requires historical context
                        "historical_percentile": 0.0  # Requires full dataset processing
                    },
                    "temporal_context": {
                        "period_type": "unknown",  # Needs market analysis
                        "relative_to_ath": 0.0,    # Requires full dataset processing
                        "days_since_major_event": 0  # Requires event database
                    },
                    "metadata": {
                        "data_source": "kaggle_price_analysis",
                        "time_period": "historical",
                        "confidence_level": 0.9  # High confidence in price data
                    }
                }
                
                normalized_records.append(record)
                
            except Exception as e:
                logger.warning(f"Error normalizing row for {coin_name}: {e}")
                continue
        
        return normalized_records
    
    def normalize_kaggle_crypto_currency(self, df: pd.DataFrame, coin_name: str) -> List[Dict[str, Any]]:
        """
        Normalize data from the Crypto Currency Datasets.
        
        Args:
            df: DataFrame with cryptocurrency data
            coin_name: Name of the cryptocurrency
            
        Returns:
            List of normalized data records
        """
        normalized_records = []
        
        for _, row in df.iterrows():
            try:
                # Extract date and convert to ISO format
                date_obj = row['Date']
                timestamp = date_obj.isoformat()
                
                # Extract price and volume, handling potential formatting issues
                price = row.get('Close', row.get(' Price ', 0.0))
                if isinstance(price, str):
                    price = float(price.replace('$', '').replace(',', '').strip())
                
                volume = row.get('Volume', row.get(' 24h Volume ', 0.0))
                if isinstance(volume, str):
                    volume = float(volume.replace('$', '').replace(',', '').strip())
                
                market_cap = row.get('Market Cap', 0.0)
                if isinstance(market_cap, str):
                    market_cap = float(market_cap.replace('$', '').replace(',', '').strip())
                
                # Calculate volatility based on available data
                volatility = 0.0
                # For crypto_currency dataset, use 24h change as volatility proxy if no High/Low
                if 'Change_24h' in row:
                    change_24h = row['Change_24h']
                    if isinstance(change_24h, str):
                        change_24h = float(change_24h.replace('%', '').strip()) / 100
                    volatility = abs(change_24h)
                elif 'High' in row and 'Low' in row:
                    volatility = (row['High'] - row['Low']) / row['Low'] if row['Low'] > 0 else 0.0
                
                # Calculate growth rate
                growth_rate = 0.0
                if 'Change_24h' in row:
                    growth_rate = row['Change_24h']
                    if isinstance(growth_rate, str):
                        growth_rate = float(growth_rate.replace('%', '').strip()) / 100
                elif 'Close' in row and 'Open' in row and row['Open'] > 0:
                    growth_rate = (row['Close'] - row['Open']) / row['Open']
                
                # Map to standard schema
                record = {
                    "timestamp": timestamp,
                    "chain": coin_name.lower(),
                    "market_data": {
                        "price_usd": price,
                        "daily_volume": volume,
                        "market_cap": market_cap,
                        "volatility": volatility,
                        "transaction_count": 0,  # Not available in this dataset
                        "unique_addresses": 0,   # Not available in this dataset
                        "average_transaction_value": 0.0,  # Not available
                        "gas_price": 0.0,        # Not available in this dataset
                        "source": "kaggle_crypto_currency",
                        "confidence": 0.85  # Slightly lower confidence
                    },
                    "derived_metrics": {
                        "growth_rate": growth_rate,
                        "relative_volume": 0.0,  # Requires historical context
                        "historical_percentile": 0.0  # Requires full dataset processing
                    },
                    "temporal_context": {
                        "period_type": "unknown",  # Needs market analysis
                        "relative_to_ath": 0.0,    # Requires full dataset processing
                        "days_since_major_event": 0  # Requires event database
                    },
                    "metadata": {
                        "data_source": "kaggle_crypto_currency",
                        "time_period": "historical",
                        "confidence_level": 0.85  # Slightly lower confidence
                    }
                }
                
                normalized_records.append(record)
                
            except Exception as e:
                logger.warning(f"Error normalizing row for {coin_name}: {e}")
                continue
        
        return normalized_records
    
    def normalize_kaggle_crypto_ranking(self, df: pd.DataFrame, coin_name: str) -> List[Dict[str, Any]]:
        """
        Normalize data from the Crypto Currency Ranking dataset format.
        
        Args:
            df: DataFrame with cryptocurrency ranking data
            coin_name: Name of the cryptocurrency (symbol)
            
        Returns:
            List of normalized data records
        """
        normalized_records = []
        
        for _, row in df.iterrows():
            try:
                # Extract date and convert to ISO format
                date_obj = row['Date']
                timestamp = date_obj.isoformat()
                
                # Get full name if available
                full_name = row.get('Coin Name', coin_name)
                
                # Extract price with proper handling
                price = 0.0
                price_str = row.get(' Price ', '0')
                if isinstance(price_str, str):
                    price_str = price_str.replace('$', '').replace(',', '').strip()
                    try:
                        price = float(price_str)
                    except ValueError:
                        price = 0.0
                elif isinstance(price_str, (int, float)):
                    price = float(price_str)
                
                # Extract volume with proper handling
                volume = 0.0
                volume_str = row.get(' 24h Volume ', '0')
                if isinstance(volume_str, str):
                    volume_str = volume_str.replace('$', '').replace(',', '').strip()
                    try:
                        volume = float(volume_str)
                    except ValueError:
                        volume = 0.0
                elif isinstance(volume_str, (int, float)):
                    volume = float(volume_str)
                
                # Extract market cap with proper handling
                market_cap = 0.0
                market_cap_str = row.get(' Market Cap ', '0')
                if isinstance(market_cap_str, str):
                    market_cap_str = market_cap_str.replace('$', '').replace(',', '').strip()
                    try:
                        market_cap = float(market_cap_str)
                    except ValueError:
                        market_cap = 0.0
                elif isinstance(market_cap_str, (int, float)):
                    market_cap = float(market_cap_str)
                
                # Get 24h, 7d, and 30d changes as strings, then convert to float
                change_24h = row.get('24h', '0%')
                if isinstance(change_24h, str):
                    change_24h = float(change_24h.replace('%', '').strip()) / 100
                
                change_7d = row.get('7d', '0%')
                if isinstance(change_7d, str):
                    change_7d = float(change_7d.replace('%', '').strip()) / 100
                
                change_30d = row.get('30d', '0%')
                if isinstance(change_30d, str):
                    change_30d = float(change_30d.replace('%', '').strip()) / 100
                
                # Derive volatility from 24h change
                volatility = abs(change_24h) if change_24h else 0.0
                
                # Map to standard schema
                record = {
                    "timestamp": timestamp,
                    "chain": coin_name.lower(),
                    "market_data": {
                        "price_usd": price,
                        "daily_volume": volume,
                        "market_cap": market_cap,
                        "volatility": volatility,
                        "transaction_count": 0,  # Not available
                        "unique_addresses": 0,   # Not available
                        "average_transaction_value": 0.0,  # Not available
                        "gas_price": 0.0,        # Not available
                        "source": "kaggle_crypto_currency",
                        "confidence": 0.85  # Slightly lower confidence
                    },
                    "derived_metrics": {
                        "growth_rate": change_24h,
                        "relative_volume": 0.0,  # Requires historical context
                        "historical_percentile": 0.0  # Requires full dataset processing
                    },
                    "temporal_context": {
                        "period_type": "unknown",  # Needs market analysis
                        "relative_to_ath": 0.0,    # Requires full dataset processing
                        "days_since_major_event": 0  # Requires event database
                    },
                    "metadata": {
                        "data_source": "kaggle_crypto_currency",
                        "time_period": "historical",
                        "confidence_level": 0.85,  # Slightly lower confidence
                        "full_name": full_name,    # Store the full name for reference
                        "change_7d": change_7d,    # Store additional metrics
                        "change_30d": change_30d
                    }
                }
                
                normalized_records.append(record)
                
            except Exception as e:
                logger.warning(f"Error normalizing row for {coin_name}: {e}")
                continue
        
        return normalized_records
    
    def normalize_flipside_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize data from Flipside API.
        
        Args:
            data: List of data records from Flipside API
            
        Returns:
            List of normalized data records
        """
        normalized_records = []
        
        for record in data:
            try:
                # Map Flipside data to standard schema
                # (This is placeholder logic - actual mapping depends on Flipside schema)
                
                normalized_record = {
                    "timestamp": record.get("date", datetime.now().isoformat()),
                    "chain": record.get("chain", "unknown").lower(),
                    "market_data": {
                        "price_usd": record.get("price", 0.0),
                        "daily_volume": record.get("volume", 0.0),
                        "market_cap": record.get("marketCap", 0.0),
                        "volatility": record.get("volatility", 0.0),
                        "transaction_count": record.get("txCount", 0),
                        "unique_addresses": record.get("uniqueAddresses", 0),
                        "average_transaction_value": record.get("avgTxValue", 0.0),
                        "gas_price": record.get("gasPrice", 0.0),
                        "source": "flipside",
                        "confidence": 0.95  # High confidence in Flipside data
                    },
                    "derived_metrics": {
                        "growth_rate": record.get("growthRate", 0.0),
                        "relative_volume": record.get("relativeVolume", 0.0),
                        "historical_percentile": record.get("historicalPercentile", 0.0)
                    },
                    "temporal_context": {
                        "period_type": record.get("marketCondition", "unknown"),
                        "relative_to_ath": record.get("relativeToATH", 0.0),
                        "days_since_major_event": record.get("daysSinceEvent", 0)
                    },
                    "metadata": {
                        "data_source": "flipside",
                        "time_period": "current",
                        "confidence_level": 0.95  # High confidence in Flipside data
                    }
                }
                
                normalized_records.append(normalized_record)
                
            except Exception as e:
                logger.warning(f"Error normalizing Flipside record: {e}")
                continue
        
        return normalized_records
    
    def save_normalized_data(self, records: List[Dict[str, Any]], 
                            source_type: str, 
                            file_prefix: str) -> Path:
        """
        Save normalized data to a JSON file.
        
        Args:
            records: List of normalized data records
            source_type: Type of data source ('historical' or 'current')
            file_prefix: Prefix for the output file name
            
        Returns:
            Path to the saved file
        """
        if source_type not in ['historical', 'current']:
            raise ValueError("source_type must be 'historical' or 'current'")
        
        # Create timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        # Create target directory
        target_dir = self.processed_dir / source_type
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output file
        output_file = target_dir / f"{file_prefix}_{timestamp}.json"
        
        # Save to JSON
        with open(output_file, 'w') as f:
            json.dump(records, f, indent=2)
        
        logger.info(f"Saved {len(records)} records to {output_file}")
        
        return output_file
    
    def merge_datasets(self, historical_files: List[Path], 
                       current_files: List[Path], 
                       output_prefix: str = "merged") -> Path:
        """
        Merge historical and current datasets.
        
        Args:
            historical_files: List of paths to historical data files
            current_files: List of paths to current data files
            output_prefix: Prefix for the output file name
            
        Returns:
            Path to the merged file
        """
        all_records = []
        
        # Load historical data
        for file_path in historical_files:
            try:
                with open(file_path, 'r') as f:
                    records = json.load(f)
                    all_records.extend(records)
                    logger.info(f"Loaded {len(records)} records from {file_path}")
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
        
        # Load current data
        for file_path in current_files:
            try:
                with open(file_path, 'r') as f:
                    records = json.load(f)
                    all_records.extend(records)
                    logger.info(f"Loaded {len(records)} records from {file_path}")
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
        
        # Sort by timestamp
        all_records.sort(key=lambda x: x.get("timestamp", ""))
        
        # Create timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        # Create target directory
        target_dir = self.processed_dir / "merged"
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output file
        output_file = target_dir / f"{output_prefix}_{timestamp}.json"
        
        # Save to JSON
        with open(output_file, 'w') as f:
            json.dump(all_records, f, indent=2)
        
        logger.info(f"Merged dataset saved to {output_file} with {len(all_records)} records")
        
        return output_file
    
    def calculate_derived_metrics(self, file_path: Path) -> Path:
        """
        Calculate derived metrics for a dataset.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Path to the updated file
        """
        try:
            # Load data
            with open(file_path, 'r') as f:
                records = json.load(f)
            
            # Group by chain
            chains = {}
            for record in records:
                chain = record.get("chain", "unknown")
                if chain not in chains:
                    chains[chain] = []
                chains[chain].append(record)
            
            # Process each chain
            for chain, chain_records in chains.items():
                # Sort by timestamp
                chain_records.sort(key=lambda x: x.get("timestamp", ""))
                
                # Calculate all-time high
                ath = max((r.get("market_data", {}).get("price_usd", 0.0) for r in chain_records), default=0.0)
                
                # Calculate metrics for each record
                for i, record in enumerate(chain_records):
                    # Get price
                    price = record.get("market_data", {}).get("price_usd", 0.0)
                    
                    # Calculate relative to ATH
                    if ath > 0:
                        record["temporal_context"]["relative_to_ath"] = price / ath
                    
                    # Calculate historical percentile (simple implementation)
                    if i > 0 and len(chain_records) > 1:
                        prices = [r.get("market_data", {}).get("price_usd", 0.0) for r in chain_records[:i+1]]
                        prices = [p for p in prices if p > 0]
                        if prices:
                            percentile = sum(1 for p in prices if p <= price) / len(prices)
                            record["derived_metrics"]["historical_percentile"] = percentile
                    
                    # Calculate relative volume (30-day moving average)
                    if i > 0:
                        window = chain_records[max(0, i-30):i]
                        volumes = [r.get("market_data", {}).get("daily_volume", 0.0) for r in window]
                        avg_volume = sum(volumes) / len(volumes) if volumes else 0.0
                        current_volume = record.get("market_data", {}).get("daily_volume", 0.0)
                        if avg_volume > 0:
                            record["derived_metrics"]["relative_volume"] = current_volume / avg_volume
            
            # Save updated data
            with open(file_path, 'w') as f:
                json.dump(records, f, indent=2)
            
            logger.info(f"Calculated derived metrics for {len(records)} records in {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error calculating derived metrics for {file_path}: {e}")
            raise


if __name__ == "__main__":
    # Example usage
    from kaggle_loader import KaggleDatasetLoader
    
    # Create instances
    kaggle_loader = KaggleDatasetLoader()
    normalizer = DataNormalizer()
    
    # Load price analysis data
    price_data = kaggle_loader.load_price_analysis_data()
    
    # Normalize and save first coin as an example
    if price_data:
        first_coin = list(price_data.keys())[0]
        normalized_records = normalizer.normalize_kaggle_price_analysis(
            price_data[first_coin], first_coin
        )
        
        # Save to file
        historical_file = normalizer.save_normalized_data(
            normalized_records, 'historical', f'price_analysis_{first_coin}'
        )
        
        # Calculate derived metrics
        normalizer.calculate_derived_metrics(historical_file)
        
        print(f"Processed {first_coin} data with {len(normalized_records)} records") 