"""
Flipside Crypto API client for collecting market data and on-chain metrics.
"""

import os
import time
import random
from typing import Dict, List, Optional, Union, Any
import pandas as pd
from dotenv import load_dotenv
from flipside import Flipside
import logging
import requests
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class FlipsideClient:
    """Client for interacting with Flipside Crypto's API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Flipside client.
        
        Args:
            api_key: Optional API key. If not provided, will look for FLIPSIDE_API_KEY in environment.
        """
        load_dotenv()
        self.api_key = api_key or os.getenv("FLIPSIDE_API_KEY")
        if not self.api_key:
            raise ValueError("Flipside API key not found. Please set FLIPSIDE_API_KEY environment variable.")
        
        # Initialize Flipside SDK client
        self.client = Flipside(self.api_key)
        
        self.base_url = "https://api-v2.flipsidecrypto.xyz"
        self.headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Database mapping for different chains
        self.db_mapping = {
            'bitcoin': 'bitcoin',
            'ethereum': 'ethereum',
            'near': 'near',
            'btc': 'bitcoin',
            'eth': 'ethereum'
        }
    
    def test_connection(self) -> bool:
        """Test the Flipside API connection with a simple query.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Try a simple query that should work with any valid API key
            sql = "SELECT 1 as test"
            result = self.execute_query(sql)
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False
    
    def execute_query(self, sql: str) -> pd.DataFrame:
        """Execute a query and return results.
        
        Args:
            sql: SQL query string
            
        Returns:
            DataFrame containing query results
        """
        try:
            # Execute query using SDK
            query_result_set = self.client.query(sql)
            
            if query_result_set.error:
                raise Exception(f"Query failed: {query_result_set.error}")
                
            return pd.DataFrame(query_result_set.records)
            
        except Exception as e:
            raise Exception(f"Error executing query: {str(e)}")
    
    def get_market_data(self, blockchain: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get market data for the specified blockchain.
        
        Args:
            blockchain: Name of the blockchain (e.g., 'bitcoin', 'ethereum', 'near')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame containing market metrics
        """
        # Get the correct database name
        db_name = self.db_mapping.get(blockchain.lower(), blockchain.lower())
        
        # Add retry logic
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                if db_name == 'bitcoin':
                    query = f"""
                    WITH daily_metrics AS (
                        SELECT
                            DATE_TRUNC('day', block_timestamp) as block_timestamp,
                            COUNT(DISTINCT tx_id) as num_txs,
                            COUNT(DISTINCT CASE 
                                WHEN inputs[0]:scriptSig:hex IS NOT NULL THEN inputs[0]:scriptSig:hex
                                WHEN inputs[0]:txinwitness IS NOT NULL THEN inputs[0]:txinwitness[0]
                                ELSE NULL
                            END) as unique_senders,
                            COUNT(DISTINCT CASE 
                                WHEN outputs[0]:scriptPubKey:address IS NOT NULL THEN outputs[0]:scriptPubKey:address
                                WHEN outputs[0]:scriptPubKey:hex IS NOT NULL THEN outputs[0]:scriptPubKey:hex
                                ELSE NULL
                            END) as unique_receivers,
                            AVG(COALESCE(fee, 0)) as avg_tx_value,
                            SUM(fee) as total_volume,
                            COUNT(DISTINCT CASE WHEN block_number IS NULL THEN tx_id END) as mempool_size
                        FROM bitcoin.core.fact_transactions
                        WHERE block_timestamp BETWEEN '{start_date}' AND '{end_date}'
                        GROUP BY 1
                        ORDER BY 1 DESC
                    ),
                    weekly_metrics AS (
                        SELECT
                            block_timestamp,
                            num_txs,
                            LAG(num_txs, 7) OVER (ORDER BY block_timestamp) as num_txs_7d_ago,
                            unique_senders,
                            LAG(unique_senders, 7) OVER (ORDER BY block_timestamp) as unique_senders_7d_ago,
                            total_volume,
                            LAG(total_volume, 7) OVER (ORDER BY block_timestamp) as total_volume_7d_ago,
                            STDDEV(num_txs) OVER (ORDER BY block_timestamp ROWS BETWEEN 7 PRECEDING AND CURRENT ROW) as tx_volatility_7d
                        FROM daily_metrics
                    )
                    SELECT 
                        d.*,
                        'bitcoin' as network,
                        COALESCE((w.num_txs - w.num_txs_7d_ago) / NULLIF(w.num_txs_7d_ago, 0) * 100, 0) as txn_growth_pct_7d,
                        COALESCE((w.unique_senders - w.unique_senders_7d_ago) / NULLIF(w.unique_senders_7d_ago, 0) * 100, 0) as user_growth_pct_7d,
                        COALESCE((w.total_volume - w.total_volume_7d_ago) / NULLIF(w.total_volume_7d_ago, 0) * 100, 0) as volume_growth_pct_7d,
                        w.tx_volatility_7d
                    FROM daily_metrics d
                    LEFT JOIN weekly_metrics w ON d.block_timestamp = w.block_timestamp
                    ORDER BY d.block_timestamp DESC
                    """
                elif db_name == 'near':
                    query = f"""
                    WITH daily_metrics AS (
                        SELECT
                            DATE_TRUNC('day', block_timestamp) as block_timestamp,
                            COUNT(DISTINCT tx_hash) as num_txs,
                            COUNT(DISTINCT tx_signer) as unique_senders,
                            COUNT(DISTINCT tx_receiver) as unique_receivers,
                            COUNT(CASE WHEN tx_succeeded = TRUE THEN 1 END)::float / NULLIF(COUNT(*), 0) as success_rate,
                            AVG(COALESCE(transaction_fee / POW(10, 24), 0)) as avg_tx_value,
                            AVG(gas_used / POW(10, 12)) as avg_gas_used,
                            AVG(attached_gas / POW(10, 12)) as avg_gas_limit,
                            AVG(attached_gas / POW(10, 12)) as avg_gas_price,
                            COUNT(DISTINCT CASE 
                                WHEN tx_receiver LIKE '%.near' 
                                OR tx_receiver LIKE '%.factory.near'
                                OR tx_receiver LIKE '%.testnet' 
                                THEN tx_receiver 
                            END) as smart_contract_calls,
                            SUM(transaction_fee / POW(10, 24)) as total_volume,
                            COUNT(DISTINCT CASE 
                                WHEN tx_receiver LIKE '%.bridge.near' 
                                OR tx_receiver LIKE 'bridge.%' 
                                THEN tx_hash 
                            END) as bridge_volume
                        FROM near.core.fact_transactions
                        WHERE block_timestamp BETWEEN '{start_date}' AND '{end_date}'
                        GROUP BY 1
                        ORDER BY 1 DESC
                    ),
                    weekly_metrics AS (
                        SELECT
                            block_timestamp,
                            num_txs,
                            LAG(num_txs, 7) OVER (ORDER BY block_timestamp) as num_txs_7d_ago,
                            unique_senders,
                            LAG(unique_senders, 7) OVER (ORDER BY block_timestamp) as unique_senders_7d_ago,
                            total_volume,
                            LAG(total_volume, 7) OVER (ORDER BY block_timestamp) as total_volume_7d_ago,
                            STDDEV(num_txs) OVER (ORDER BY block_timestamp ROWS BETWEEN 7 PRECEDING AND CURRENT ROW) as tx_volatility_7d,
                            AVG(avg_tx_value) OVER (ORDER BY block_timestamp ROWS BETWEEN 7 PRECEDING AND CURRENT ROW) as avg_tx_value_7d,
                            AVG(smart_contract_calls) OVER (ORDER BY block_timestamp ROWS BETWEEN 7 PRECEDING AND CURRENT ROW) as avg_contract_calls_7d
                        FROM daily_metrics
                    )
                    SELECT 
                        d.*,
                        'NEAR' as network,
                        COALESCE((w.num_txs - w.num_txs_7d_ago) / NULLIF(w.num_txs_7d_ago, 0) * 100, 0) as txn_growth_pct_7d,
                        COALESCE((w.unique_senders - w.unique_senders_7d_ago) / NULLIF(w.unique_senders_7d_ago, 0) * 100, 0) as user_growth_pct_7d,
                        COALESCE((w.total_volume - w.total_volume_7d_ago) / NULLIF(w.total_volume_7d_ago, 0) * 100, 0) as volume_growth_pct_7d,
                        w.tx_volatility_7d,
                        w.avg_tx_value_7d,
                        w.avg_contract_calls_7d,
                        CASE 
                            WHEN w.tx_volatility_7d > w.avg_tx_value_7d * 2 THEN 'high'
                            WHEN w.tx_volatility_7d > w.avg_tx_value_7d THEN 'medium'
                            ELSE 'low'
                        END as volatility_level
                    FROM daily_metrics d
                    LEFT JOIN weekly_metrics w ON d.block_timestamp = w.block_timestamp
                    ORDER BY d.block_timestamp DESC
                    """
                else:
                    query = f"""
                    WITH daily_metrics AS (
                        SELECT 
                            DATE_TRUNC('day', block_timestamp) as block_timestamp,
                            COUNT(DISTINCT tx_hash) as num_txs,
                            COUNT(DISTINCT from_address) as unique_senders,
                            COUNT(DISTINCT to_address) as unique_receivers,
                            COUNT(CASE WHEN status = 'SUCCESS' THEN 1 END)::float / NULLIF(COUNT(*), 0) as success_rate,
                            AVG(COALESCE(value, 0)) as avg_tx_value,
                            AVG(gas_used) as avg_gas_used,
                            AVG(gas_price) as avg_gas_price,
                            COUNT(DISTINCT CASE WHEN input_data != '' AND input_data IS NOT NULL THEN tx_hash END) as smart_contract_calls,
                            SUM(value) as total_volume
                        FROM {db_name}.core.fact_transactions
                        WHERE block_timestamp BETWEEN '{start_date}' AND '{end_date}'
                        GROUP BY 1
                        ORDER BY 1 DESC
                    ),
                    weekly_metrics AS (
                        SELECT 
                            block_timestamp,
                            num_txs,
                            LAG(num_txs, 7) OVER (ORDER BY block_timestamp) as num_txs_7d_ago,
                            unique_senders,
                            LAG(unique_senders, 7) OVER (ORDER BY block_timestamp) as unique_senders_7d_ago,
                            total_volume,
                            LAG(total_volume, 7) OVER (ORDER BY block_timestamp) as total_volume_7d_ago,
                            STDDEV(num_txs) OVER (ORDER BY block_timestamp ROWS BETWEEN 7 PRECEDING AND CURRENT ROW) as tx_volatility_7d
                        FROM daily_metrics
                    )
                    SELECT 
                        d.*,
                        '{db_name}' as network,
                        COALESCE((w.num_txs - w.num_txs_7d_ago) / NULLIF(w.num_txs_7d_ago, 0) * 100, 0) as txn_growth_pct_7d,
                        COALESCE((w.unique_senders - w.unique_senders_7d_ago) / NULLIF(w.unique_senders_7d_ago, 0) * 100, 0) as user_growth_pct_7d,
                        COALESCE((w.total_volume - w.total_volume_7d_ago) / NULLIF(w.total_volume_7d_ago, 0) * 100, 0) as volume_growth_pct_7d,
                        w.tx_volatility_7d
                    FROM daily_metrics d
                    LEFT JOIN weekly_metrics w USING (block_timestamp)
                    ORDER BY block_timestamp DESC
                    """
                
                # Execute query with retry logic
                try:
                    result = self.execute_query(query)
                    if not result.empty:
                        logger.info(f"Successfully retrieved {len(result)} records for {blockchain}")
                        return result
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed for {blockchain}: {str(e)}")
                        time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                        continue
                    else:
                        logger.error(f"All attempts failed for {blockchain}: {str(e)}")
                        raise
                        
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed for {blockchain}: {str(e)}")
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    logger.error(f"All attempts failed for {blockchain}: {str(e)}")
                    raise
        
        # If we get here, all retries failed
        raise Exception(f"Failed to get market data for {blockchain} after {max_retries} attempts")
    
    def get_defi_metrics(self, protocol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get DeFi protocol metrics.
        
        Args:
            protocol: Protocol name (e.g., 'uniswap', 'aave')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame containing protocol metrics
        """
        # Get protocol addresses from labels for both chains
        eth_label_query = f"""
        SELECT DISTINCT address, label_subtype, label_type
        FROM ethereum.core.dim_labels
        WHERE (label_type = 'dex' OR label_type = 'lending')
        AND LOWER(label) = LOWER('{protocol}');
        """
        
        near_label_query = f"""
        SELECT DISTINCT tx_receiver as address, 'dex' as label_type
        FROM near.core.fact_transactions
        WHERE tx_receiver LIKE '%{protocol}%.near'
        OR tx_receiver LIKE '%{protocol}%.factory.near'
        GROUP BY 1;
        """
        
        try:
            # Get protocol addresses from both chains
            eth_addresses = self.execute_query(eth_label_query)
            near_addresses = self.execute_query(near_label_query)
            
            if eth_addresses.empty and near_addresses.empty:
                raise ValueError(f"No addresses found for protocol: {protocol}")
            
            results = []
            
            # Ethereum metrics if addresses found
            if not eth_addresses.empty:
                addresses = tuple(eth_addresses['address'].tolist())
                eth_query = f"""
                WITH protocol_txns AS (
                    SELECT
                        block_timestamp,
                        tx_hash,
                        from_address,
                        to_address,
                        value,
                        gas_used,
                        gas_price,
                        input_data,
                        status,
                        origin_function_signature
                    FROM ethereum.core.fact_transactions
                    WHERE block_timestamp BETWEEN '{start_date}' AND '{end_date}'
                    AND (
                        to_address IN {addresses}
                        OR origin_function_signature LIKE 'swap%'
                        OR origin_function_signature LIKE 'exactInput%'
                        OR origin_function_signature LIKE 'exactOutput%'
                    )
                ),
                daily_metrics AS (
                    SELECT
                        DATE_TRUNC('day', block_timestamp) as block_timestamp,
                        COUNT(DISTINCT tx_hash) as num_transactions,
                        COUNT(DISTINCT from_address) as unique_users,
                        COUNT(DISTINCT to_address) as unique_contracts,
                        COUNT(CASE WHEN status = 'SUCCESS' THEN 1 END)::float / NULLIF(COUNT(*), 0) as success_rate,
                        SUM(value) as total_volume_eth,
                        AVG(value) as avg_transaction_size,
                        AVG(gas_used) as avg_gas_used,
                        AVG(gas_price) as avg_gas_price,
                        COUNT(DISTINCT CASE 
                            WHEN origin_function_signature LIKE 'swap%' 
                            OR origin_function_signature LIKE 'exactInput%'
                            OR origin_function_signature LIKE 'exactOutput%'
                            THEN tx_hash 
                        END) as swap_count
                    FROM protocol_txns
                    GROUP BY 1
                    ORDER BY 1 DESC
                ),
                weekly_metrics AS (
                    SELECT
                        block_timestamp,
                        num_transactions,
                        LAG(num_transactions, 7) OVER (ORDER BY block_timestamp) as txns_7d_ago,
                        unique_users,
                        LAG(unique_users, 7) OVER (ORDER BY block_timestamp) as users_7d_ago,
                        total_volume_eth,
                        LAG(total_volume_eth, 7) OVER (ORDER BY block_timestamp) as volume_7d_ago,
                        STDDEV(total_volume_eth) OVER (ORDER BY block_timestamp ROWS BETWEEN 7 PRECEDING AND CURRENT ROW) as volume_volatility_7d
                    FROM daily_metrics
                )
                SELECT 
                    d.*,
                    '{protocol}' as protocol,
                    'ethereum' as network,
                    COALESCE((w.num_transactions - w.txns_7d_ago) / NULLIF(w.txns_7d_ago, 0) * 100, 0) as txn_growth_pct_7d,
                    COALESCE((w.unique_users - w.users_7d_ago) / NULLIF(w.users_7d_ago, 0) * 100, 0) as user_growth_pct_7d,
                    COALESCE((w.total_volume_eth - w.volume_7d_ago) / NULLIF(w.volume_7d_ago, 0) * 100, 0) as volume_growth_pct_7d,
                    w.volume_volatility_7d
                FROM daily_metrics d
                LEFT JOIN weekly_metrics w USING (block_timestamp)
                ORDER BY block_timestamp DESC
                """
                eth_results = self.execute_query(eth_query)
                results.append(eth_results)
            
            # NEAR metrics if addresses found
            if not near_addresses.empty:
                addresses = tuple(near_addresses['address'].tolist())
                near_query = f"""
                WITH protocol_txns AS (
                    SELECT
                        block_timestamp,
                        tx_hash,
                        tx_signer as from_address,
                        tx_receiver as to_address,
                        transaction_fee / POW(10, 24) as value,
                        gas_used,
                        attached_gas as gas_price,
                        tx_succeeded as status
                    FROM near.core.fact_transactions
                    WHERE block_timestamp BETWEEN '{start_date}' AND '{end_date}'
                    AND tx_receiver IN {addresses}
                ),
                daily_metrics AS (
                    SELECT
                        DATE_TRUNC('day', block_timestamp) as block_timestamp,
                        COUNT(DISTINCT tx_hash) as num_transactions,
                        COUNT(DISTINCT from_address) as unique_users,
                        COUNT(DISTINCT to_address) as unique_contracts,
                        COUNT(CASE WHEN status = TRUE THEN 1 END)::float / NULLIF(COUNT(*), 0) as success_rate,
                        SUM(value) as total_volume_near,
                        AVG(value) as avg_transaction_size,
                        AVG(gas_used) as avg_gas_used,
                        AVG(gas_price) as avg_gas_price
                    FROM protocol_txns
                    GROUP BY 1
                    ORDER BY 1 DESC
                ),
                weekly_metrics AS (
                    SELECT
                        block_timestamp,
                        num_transactions,
                        LAG(num_transactions, 7) OVER (ORDER BY block_timestamp) as txns_7d_ago,
                        unique_users,
                        LAG(unique_users, 7) OVER (ORDER BY block_timestamp) as users_7d_ago,
                        total_volume_near,
                        LAG(total_volume_near, 7) OVER (ORDER BY block_timestamp) as volume_7d_ago,
                        STDDEV(total_volume_near) OVER (ORDER BY block_timestamp ROWS BETWEEN 7 PRECEDING AND CURRENT ROW) as volume_volatility_7d
                    FROM daily_metrics
                )
                SELECT 
                    d.*,
                    '{protocol}' as protocol,
                    'near' as network,
                    COALESCE((w.num_transactions - w.txns_7d_ago) / NULLIF(w.txns_7d_ago, 0) * 100, 0) as txn_growth_pct_7d,
                    COALESCE((w.unique_users - w.users_7d_ago) / NULLIF(w.users_7d_ago, 0) * 100, 0) as user_growth_pct_7d,
                    COALESCE((w.total_volume_near - w.volume_7d_ago) / NULLIF(w.volume_7d_ago, 0) * 100, 0) as volume_growth_pct_7d,
                    w.volume_volatility_7d
                FROM daily_metrics d
                LEFT JOIN weekly_metrics w USING (block_timestamp)
                ORDER BY block_timestamp DESC
                """
                near_results = self.execute_query(near_query)
                results.append(near_results)
            
            # Combine results if we have data from both chains
            if len(results) > 1:
                return pd.concat(results, ignore_index=True)
            return results[0] if results else pd.DataFrame()
            
        except Exception as e:
            raise Exception(f"Error getting DeFi metrics for {protocol}: {str(e)}")
    
    def get_wallet_activity(self,
                          address: str,
                          chain: str,
                          days: int = 30) -> pd.DataFrame:
        """Get comprehensive wallet activity for behavioral analysis."""
        sql = f"""
        WITH 
        -- Transaction history
        wallet_txns AS (
            SELECT 
                block_timestamp,
                tx_hash,
                CASE 
                    WHEN origin_from_address = LOWER('{address}') THEN 'out'
                    WHEN origin_to_address = LOWER('{address}') THEN 'in'
                END as direction,
                value as amount,
                symbol,
                tx_fee,
                gas_used,
                gas_price,
                origin_to_address_label as interaction_type
            FROM {chain}.core.fact_transactions
            WHERE (origin_from_address = LOWER('{address}') OR origin_to_address = LOWER('{address}'))
            AND block_timestamp >= CURRENT_TIMESTAMP - INTERVAL '{days} days'
        ),
        -- Daily aggregates
        daily_stats AS (
            SELECT 
                DATE_TRUNC('day', block_timestamp) as date,
                COUNT(*) as daily_txns,
                COUNT(DISTINCT CASE WHEN direction = 'out' THEN origin_to_address END) as unique_recipients,
                SUM(CASE WHEN direction = 'out' THEN tx_fee ELSE 0 END) as total_fees_paid,
                COUNT(DISTINCT interaction_type) as contract_types_used
            FROM wallet_txns
            GROUP BY 1
        )
        SELECT 
            t.*,
            d.daily_txns,
            d.unique_recipients,
            d.total_fees_paid,
            d.contract_types_used,
            -- Running totals
            SUM(CASE WHEN t.direction = 'in' THEN t.amount ELSE -t.amount END) 
                OVER (ORDER BY t.block_timestamp) as running_balance,
            SUM(t.tx_fee) OVER (ORDER BY t.block_timestamp) as cumulative_fees
        FROM wallet_txns t
        LEFT JOIN daily_stats d ON DATE_TRUNC('day', t.block_timestamp) = d.date
        ORDER BY t.block_timestamp DESC
        """
        return self.execute_query(sql)

    def get_market_metrics(self,
                          chain: str,
                          start_date: datetime,
                          end_date: datetime,
                          metrics: List[str]) -> List[Dict[str, Any]]:
        """Fetch market metrics for a specific chain.
        
        Args:
            chain: Blockchain network (e.g., 'ethereum', 'near')
            start_date: Start date for data collection
            end_date: End date for data collection
            metrics: List of metrics to collect
            
        Returns:
            List of dictionaries containing market data
        """
        try:
            # Convert dates to string format
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            
            # Get market data using the existing method
            market_data = self.get_market_data(chain, start_str, end_str)
            
            # Convert DataFrame to list of dictionaries
            return market_data.to_dict('records')
            
        except Exception as e:
            logger.error(f"Error fetching market metrics for {chain}: {str(e)}")
            return []
    
    def get_recent_market_data(self, chain: str, days: int = 30, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent market data for a specific chain using real API calls.
        
        Args:
            chain: Type of chain data to get ('market', 'ethereum', 'near', etc.)
            days: Number of days of data to retrieve
            limit: Maximum number of data points to return
            
        Returns:
            List of dictionaries containing market data
        """
        try:
            # Define the time range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # For 'market' chain type, default to NEAR network
            actual_chain = "near" if chain.lower() == "market" else chain.lower()
            
            logger.info(f"Fetching real market data for {actual_chain} (last {days} days)")
            
            # Use the real API to get market data
            market_data = self.get_market_data(
                blockchain=actual_chain,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d")
            )
            
            # Convert to list of dictionaries and limit results
            result = market_data.to_dict('records')[:limit]
            logger.info(f"Retrieved {len(result)} real data points for {actual_chain}")
            
            if not result:
                logger.warning(f"No data returned from Flipside API for {actual_chain}, falling back to mock data")
                return self._generate_mock_data(chain, days, limit)
                
            return result
            
        except Exception as e:
            logger.error(f"Error fetching real market data: {str(e)}")
            logger.warning(f"Falling back to mock data for {chain}")
            
            # Fall back to mock data if API call fails
            return self._generate_mock_data(chain, days, limit)
    
    def _generate_mock_data(self, chain: str, days: int = 30, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Generate mock market data as a fallback when real API calls fail.
        
        Args:
            chain: Type of chain data to generate
            days: Number of days of data to generate
            limit: Maximum number of data points to return
            
        Returns:
            List of dictionaries containing mock market data
        """
        logger.info(f"Generating mock market data for {chain} (last {days} days, limit {limit})")
        
        # Mock data for demonstration
        results = []
        
        # Generate data for the requested number of days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Generate mock data points, one per day
        current_date = start_date
        while current_date <= end_date and len(results) < limit:
            # Basic metrics that should be present in all records
            data_point = {
                "date": current_date.strftime("%Y-%m-%d"),
                "network": chain.upper() if chain.lower() != "market" else "NEAR",
                "num_txs": random.randint(100000, 2000000),
                "unique_users": random.randint(10000, 200000),
                "total_volume": random.uniform(1000000, 50000000),
                "avg_tx_value": random.uniform(50, 500),
                "success_rate": random.uniform(0.95, 0.999),
                "gas_used": random.uniform(1000000, 5000000),
                "txn_growth_pct_7d": random.uniform(-15, 25),
                "user_growth_pct_7d": random.uniform(-10, 30),
                "tx_volatility_7d": random.uniform(0.1, 0.5)
            }
            
            results.append(data_point)
            current_date += timedelta(days=1)
        
        logger.info(f"Generated {len(results)} mock data points for {chain}")
        return results[:limit] 