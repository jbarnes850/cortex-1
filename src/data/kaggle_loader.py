"""
Kaggle Dataset Loader for NEAR Cortex-1

This module handles downloading, processing, and managing Kaggle datasets
for historical cryptocurrency analysis.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Optional, Union, Tuple
import kaggle

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
KAGGLE_DATASETS = {
    "price_analysis": {
        "dataset": "adityamhaske/cryptocurrency-price-analysis-dataset",
        "files": ["*"]
    },
    "crypto_currency": {
        "dataset": "mmohaiminulislam/crypto-currency-datasets",
        "files": ["*"]
    }
}

class KaggleDatasetLoader:
    """Handles downloading and processing Kaggle cryptocurrency datasets."""
    
    def __init__(self, data_dir: str = "data", 
                 download_dir: str = "data/raw/kaggle",
                 kaggle_username: Optional[str] = None,
                 kaggle_key: Optional[str] = None):
        """
        Initialize the Kaggle dataset loader.
        
        Args:
            data_dir: Base data directory
            download_dir: Directory to download datasets to
            kaggle_username: Kaggle username (if not set in environment)
            kaggle_key: Kaggle API key (if not set in environment)
        """
        self.data_dir = Path(data_dir)
        self.download_dir = Path(download_dir)
        
        # Create directories if they don't exist
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure Kaggle API credentials
        if kaggle_username and kaggle_key:
            os.environ['KAGGLE_USERNAME'] = kaggle_username
            os.environ['KAGGLE_KEY'] = kaggle_key
        else:
            logger.info("Using Kaggle credentials from environment variables")
            
        # Check if Kaggle credentials are available
        if not os.environ.get('KAGGLE_USERNAME') or not os.environ.get('KAGGLE_KEY'):
            logger.warning("Kaggle credentials not found in environment variables. "
                          "Please set KAGGLE_USERNAME and KAGGLE_KEY environment variables "
                          "or provide them when initializing the loader.")
    
    def download_dataset(self, dataset_key: str) -> Path:
        """
        Download a specific Kaggle dataset.
        
        Args:
            dataset_key: Key for the dataset in KAGGLE_DATASETS
            
        Returns:
            Path to the downloaded dataset
        """
        if dataset_key not in KAGGLE_DATASETS:
            raise ValueError(f"Unknown dataset key: {dataset_key}. "
                             f"Available keys: {list(KAGGLE_DATASETS.keys())}")
        
        dataset_info = KAGGLE_DATASETS[dataset_key]
        dataset_name = dataset_info["dataset"]
        target_dir = self.download_dir / dataset_key
        
        logger.info(f"Downloading dataset {dataset_name} to {target_dir}")
        
        try:
            # Create target directory
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Download dataset
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                dataset=dataset_name,
                path=target_dir,
                unzip=True
            )
            
            logger.info(f"Successfully downloaded {dataset_key} dataset")
            return target_dir
            
        except Exception as e:
            logger.error(f"Error downloading dataset {dataset_key}: {e}")
            raise
    
    def load_price_analysis_data(self, download_if_missing: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Load the Cryptocurrency Price Analysis dataset.
        
        Args:
            download_if_missing: Whether to download the dataset if it's not found
            
        Returns:
            Dictionary of DataFrames with cryptocurrency data
        """
        dataset_key = "price_analysis"
        dataset_dir = self.download_dir / dataset_key
        
        if not dataset_dir.exists() or not any(dataset_dir.iterdir()):
            if download_if_missing:
                self.download_dataset(dataset_key)
            else:
                raise FileNotFoundError(f"Dataset directory {dataset_dir} not found or empty")
        
        # Load all CSV files in the directory
        dataframes = {}
        for file_path in dataset_dir.glob("*.csv"):
            try:
                coin_name = file_path.stem
                df = pd.read_csv(file_path, parse_dates=["Date"])
                dataframes[coin_name] = df
                logger.info(f"Loaded {coin_name} data with {len(df)} rows")
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
        
        return dataframes
    
    def load_crypto_currency_data(self, download_if_missing: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Load the Crypto Currency Datasets.
        
        Args:
            download_if_missing: Whether to download the dataset if it's not found
            
        Returns:
            Dictionary of DataFrames with cryptocurrency data
        """
        dataset_key = "crypto_currency"
        dataset_dir = self.download_dir / dataset_key
        
        if not dataset_dir.exists() or not any(dataset_dir.iterdir()):
            if download_if_missing:
                self.download_dataset(dataset_key)
            else:
                raise FileNotFoundError(f"Dataset directory {dataset_dir} not found or empty")
        
        # For this dataset, we have a single file with multiple coins
        dataframes = {}
        file_path = dataset_dir / "CryptocurrencyData.csv"
        
        try:
            if file_path.exists():
                # Load the file, which contains multiple coins
                df = pd.read_csv(file_path)
                
                # Add a timestamp column since this dataset doesn't have one
                # Using current date as a fallback
                df['Date'] = datetime.now().strftime('%Y-%m-%d')
                
                # Group by Symbol to create separate dataframes for each coin
                grouped = df.groupby('Symbol')
                
                for symbol, group_df in grouped:
                    # Clean the symbol
                    symbol = symbol.strip()
                    
                    # Clean numeric columns by removing commas and $ signs
                    for col in [' Price ', ' 24h Volume ', 'Circulating Supply', ' Market Cap ']:
                        if col in group_df.columns:
                            group_df[col] = group_df[col].astype(str).str.replace(',', '').str.replace('$', '').str.replace(' ', '')
                            
                            # Convert to numeric, coerce errors to NaN
                            group_df[col] = pd.to_numeric(group_df[col], errors='coerce')
                    
                    # Rename columns to match expected format
                    renamed_df = group_df.rename(columns={
                        ' Price ': 'Close',
                        '24h': 'Change_24h',
                        '7d': 'Change_7d',
                        '30d': 'Change_30d',
                        ' 24h Volume ': 'Volume',
                        ' Market Cap ': 'Market Cap'
                    })
                    
                    dataframes[symbol] = renamed_df
                    logger.info(f"Loaded {symbol} data with {len(renamed_df)} rows from combined file")
            else:
                logger.warning(f"CryptocurrencyData.csv not found in {dataset_dir}")
                
        except Exception as e:
            logger.warning(f"Error loading crypto_currency dataset: {e}")
        
        return dataframes
    
    def get_available_coins(self, dataset_key: str) -> List[str]:
        """
        Get a list of available coins in a dataset.
        
        Args:
            dataset_key: Key for the dataset in KAGGLE_DATASETS
            
        Returns:
            List of coin names
        """
        if dataset_key == "price_analysis":
            data = self.load_price_analysis_data(download_if_missing=False)
        elif dataset_key == "crypto_currency":
            data = self.load_crypto_currency_data(download_if_missing=False)
        else:
            raise ValueError(f"Unknown dataset key: {dataset_key}")
        
        return list(data.keys())
    
    def get_date_range(self, dataset_key: str) -> Dict[str, Tuple[datetime, datetime]]:
        """
        Get the date range for each coin in a dataset.
        
        Args:
            dataset_key: Key for the dataset in KAGGLE_DATASETS
            
        Returns:
            Dictionary mapping coin names to (start_date, end_date) tuples
        """
        if dataset_key == "price_analysis":
            data = self.load_price_analysis_data(download_if_missing=False)
        elif dataset_key == "crypto_currency":
            data = self.load_crypto_currency_data(download_if_missing=False)
        else:
            raise ValueError(f"Unknown dataset key: {dataset_key}")
        
        date_ranges = {}
        for coin, df in data.items():
            if "Date" in df.columns:
                start_date = df["Date"].min()
                end_date = df["Date"].max()
                date_ranges[coin] = (start_date, end_date)
        
        return date_ranges


if __name__ == "__main__":
    # Example usage
    loader = KaggleDatasetLoader()
    
    # Download datasets
    loader.download_dataset("price_analysis")
    loader.download_dataset("crypto_currency")
    
    # Load price analysis data
    price_data = loader.load_price_analysis_data()
    print(f"Loaded {len(price_data)} coins from price analysis dataset")
    
    # Load crypto currency data
    crypto_data = loader.load_crypto_currency_data()
    print(f"Loaded {len(crypto_data)} coins from crypto currency dataset")
    
    # Get available coins
    price_coins = loader.get_available_coins("price_analysis")
    print(f"Available coins in price analysis: {price_coins}")
    
    # Get date ranges
    date_ranges = loader.get_date_range("price_analysis")
    for coin, (start, end) in list(date_ranges.items())[:5]:  # Show first 5
        print(f"{coin}: {start} to {end}") 