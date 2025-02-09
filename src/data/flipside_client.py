"""
Flipside Crypto API client for collecting market data and on-chain metrics.
"""

import os
import time
from typing import Dict, List, Optional, Union
import pandas as pd
from dotenv import load_dotenv
from flipside import Flipside

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
            print(f"Connection test failed: {str(e)}")
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
            blockchain: Name of the blockchain (e.g., 'ethereum', 'near')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame containing market metrics
        """
        if blockchain.lower() == 'near':
            query = f"""
            WITH daily_metrics AS (
                SELECT
                    DATE_TRUNC('day', block_timestamp) as block_timestamp,
                    COUNT(DISTINCT tx_hash) as num_txs,
                    COUNT(DISTINCT tx_signer) as unique_senders,
                    COUNT(DISTINCT tx_receiver) as unique_receivers,
                    -- Success Rate
                    COUNT(CASE WHEN tx_succeeded = TRUE THEN 1 END)::float / NULLIF(COUNT(*), 0) as success_rate,
                    -- Value Metrics
                    AVG(COALESCE(transaction_fee, 0)) as avg_tx_value,
                    -- Gas Metrics
                    AVG(gas_used) as avg_gas_used,
                    AVG(attached_gas) as avg_gas_limit,
                    SUM(transaction_fee) as total_gas_cost,
                    -- Contract Activity
                    COUNT(DISTINCT tx_receiver) as active_contracts
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
                    STDDEV(num_txs) OVER (ORDER BY block_timestamp ROWS BETWEEN 7 PRECEDING AND CURRENT ROW) as tx_volatility_7d
                FROM daily_metrics
            )
            SELECT 
                d.*,
                'NEAR' as network,
                'NEAR' as blockchain,
                COALESCE((w.num_txs - w.num_txs_7d_ago) / NULLIF(w.num_txs_7d_ago, 0) * 100, 0) as txn_growth_pct_7d,
                COALESCE((w.unique_senders - w.unique_senders_7d_ago) / NULLIF(w.unique_senders_7d_ago, 0) * 100, 0) as user_growth_pct_7d,
                w.tx_volatility_7d
            FROM daily_metrics d
            LEFT JOIN weekly_metrics w USING (block_timestamp)
            ORDER BY block_timestamp DESC
            """
        else:
            query = f"""
            WITH daily_metrics AS (
                SELECT 
                    DATE_TRUNC('day', block_timestamp) as block_timestamp,
                    COUNT(DISTINCT tx_hash) as num_txs,
                    COUNT(DISTINCT from_address) as unique_senders,
                    COUNT(DISTINCT to_address) as unique_receivers,
                    -- Network Health Metrics
                    COUNT(CASE WHEN status = 'SUCCESS' THEN 1 END)::float / NULLIF(COUNT(*), 0) as success_rate,
                    -- Transaction Value Metrics
                    AVG(COALESCE(value, 0)) as avg_tx_value,
                    -- Gas Metrics
                    AVG(gas_used) as avg_gas_used,
                    AVG(gas_price) as avg_gas_price,
                    -- Contract Activity
                    COUNT(DISTINCT CASE WHEN input_data != '' THEN tx_hash END) as smart_contract_calls
                FROM {blockchain}.core.fact_transactions
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
                    STDDEV(num_txs) OVER (ORDER BY block_timestamp ROWS BETWEEN 7 PRECEDING AND CURRENT ROW) as tx_volatility_7d
                FROM daily_metrics
            )
            SELECT 
                d.*,
                '{blockchain}' as network,
                '{blockchain}' as blockchain,
                COALESCE((w.num_txs - w.num_txs_7d_ago) / NULLIF(w.num_txs_7d_ago, 0) * 100, 0) as txn_growth_pct_7d,
                COALESCE((w.unique_senders - w.unique_senders_7d_ago) / NULLIF(w.unique_senders_7d_ago, 0) * 100, 0) as user_growth_pct_7d,
                w.tx_volatility_7d
            FROM daily_metrics d
            LEFT JOIN weekly_metrics w USING (block_timestamp)
            ORDER BY block_timestamp DESC
            """
            
        return self.execute_query(query)
    
    def get_defi_metrics(self, protocol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get DeFi protocol metrics.
        
        Args:
            protocol: Protocol name (e.g., 'uniswap', 'aave')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame containing protocol metrics
        """
        if protocol.lower() == 'uniswap':
            query = f"""
            WITH daily_metrics AS (
                SELECT
                    DATE_TRUNC('day', block_timestamp) as block_timestamp,
                    COUNT(*) as num_swaps,
                    COUNT(DISTINCT from_address) as unique_traders,
                    COUNT(DISTINCT to_address) as unique_pools,
                    SUM(CASE WHEN origin_function_signature LIKE 'swap%' THEN value ELSE 0 END) as volume_eth,
                    AVG(CASE WHEN origin_function_signature LIKE 'swap%' THEN value ELSE NULL END) as avg_swap_size,
                    AVG(gas_price) as avg_gas_price,
                    AVG(gas_used) as avg_gas_used
                FROM ethereum.core.fact_transactions
                WHERE block_timestamp BETWEEN '{start_date}' AND '{end_date}'
                AND (origin_function_signature LIKE 'swap%' OR origin_function_signature LIKE 'exactInput%' OR origin_function_signature LIKE 'exactOutput%')
                GROUP BY 1
                ORDER BY 1 DESC
            ),
            weekly_metrics AS (
                SELECT
                    block_timestamp,
                    num_swaps,
                    LAG(num_swaps, 7) OVER (ORDER BY block_timestamp) as num_swaps_7d_ago,
                    unique_traders,
                    LAG(unique_traders, 7) OVER (ORDER BY block_timestamp) as unique_traders_7d_ago,
                    volume_eth,
                    LAG(volume_eth, 7) OVER (ORDER BY block_timestamp) as volume_eth_7d_ago
                FROM daily_metrics
            )
            SELECT 
                d.*,
                'Uniswap' as protocol,
                'Ethereum' as blockchain,
                COALESCE((w.num_swaps - w.num_swaps_7d_ago) / NULLIF(w.num_swaps_7d_ago, 0) * 100, 0) as volume_growth_pct,
                COALESCE((w.unique_traders - w.unique_traders_7d_ago) / NULLIF(w.unique_traders_7d_ago, 0) * 100, 0) as user_growth_pct
            FROM daily_metrics d
            LEFT JOIN weekly_metrics w USING (block_timestamp)
            ORDER BY block_timestamp DESC
            """
        elif protocol.lower() == 'aave':
            query = f"""
            WITH daily_metrics AS (
                SELECT
                    DATE_TRUNC('day', block_timestamp) as block_timestamp,
                    COUNT(*) as num_actions,
                    COUNT(DISTINCT from_address) as unique_users,
                    COUNT(DISTINCT to_address) as unique_markets,
                    SUM(CASE WHEN origin_function_signature IN ('supply(address,uint256,address,uint16)', 'deposit(address,uint256,address,uint16)') THEN value ELSE 0 END) as supply_volume,
                    SUM(CASE WHEN origin_function_signature IN ('borrow(address,uint256,uint256,uint16,address)', 'borrow(uint256,uint256,uint16,address)') THEN value ELSE 0 END) as borrow_volume,
                    AVG(gas_price) as avg_gas_price,
                    AVG(gas_used) as avg_gas_used
                FROM ethereum.core.fact_transactions
                WHERE block_timestamp BETWEEN '{start_date}' AND '{end_date}'
                AND (
                    origin_function_signature IN (
                        'supply(address,uint256,address,uint16)',
                        'deposit(address,uint256,address,uint16)',
                        'borrow(address,uint256,uint256,uint16,address)',
                        'borrow(uint256,uint256,uint16,address)',
                        'repay(address,uint256,uint256,address)',
                        'withdraw(address,uint256,address)'
                    )
                )
                GROUP BY 1
                ORDER BY 1 DESC
            ),
            weekly_metrics AS (
                SELECT
                    block_timestamp,
                    num_actions,
                    LAG(num_actions, 7) OVER (ORDER BY block_timestamp) as num_actions_7d_ago,
                    unique_users,
                    LAG(unique_users, 7) OVER (ORDER BY block_timestamp) as unique_users_7d_ago,
                    supply_volume,
                    LAG(supply_volume, 7) OVER (ORDER BY block_timestamp) as supply_volume_7d_ago,
                    borrow_volume,
                    LAG(borrow_volume, 7) OVER (ORDER BY block_timestamp) as borrow_volume_7d_ago
                FROM daily_metrics
            )
            SELECT 
                d.*,
                'Aave' as protocol,
                'Ethereum' as blockchain,
                COALESCE((w.num_actions - w.num_actions_7d_ago) / NULLIF(w.num_actions_7d_ago, 0) * 100, 0) as volume_growth_pct,
                COALESCE((w.unique_users - w.unique_users_7d_ago) / NULLIF(w.unique_users_7d_ago, 0) * 100, 0) as user_growth_pct
            FROM daily_metrics d
            LEFT JOIN weekly_metrics w USING (block_timestamp)
            ORDER BY block_timestamp DESC
            """
        else:
            raise ValueError(f"Unsupported protocol: {protocol}")
            
        return self.execute_query(query)
    
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