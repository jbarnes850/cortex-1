#!/usr/bin/env python
"""Test script for Flipside API connection."""

from src.data.flipside_client import FlipsideClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("Testing Flipside API connection...")
    client = FlipsideClient()
    
    # Test basic connection
    if client.test_connection():
        print("✅ Basic connection test passed")
    else:
        print("❌ Basic connection test failed")
        return
    
    # Check Ethereum transaction schema
    try:
        logger.info("\nTesting Ethereum transaction schema...")
        sql = "DESCRIBE TABLE ethereum.core.fact_transactions;"
        result = client.execute_query(sql)
        print("\nAvailable columns in ethereum.core.fact_transactions:")
        for row in result.to_dict('records'):
            print(f"- {row}")
    except Exception as e:
        logger.error(f"❌ Failed to get Ethereum transaction schema: {str(e)}")
        
    # Check NEAR transaction schema
    try:
        logger.info("\nTesting NEAR transaction schema...")
        sql = "DESCRIBE TABLE near.core.fact_transactions;"
        result = client.execute_query(sql)
        print("\nAvailable columns in near.core.fact_transactions:")
        for row in result.to_dict('records'):
            print(f"- {row}")
    except Exception as e:
        logger.error(f"❌ Failed to get NEAR transaction schema: {str(e)}")
        
    # Check DeFi tables and labels
    try:
        logger.info("\nChecking Ethereum DeFi tables...")
        
        # Check DEX events
        sql = """
        SELECT DISTINCT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'ethereum'
        AND table_name LIKE '%dex%events%';
        """
        result = client.execute_query(sql)
        print("\nEthereum DEX Event tables:")
        for row in result.to_dict('records'):
            print(f"- {row['table_name']}")
            
        # Check DeFi labels
        sql = """
        SELECT DISTINCT label_type, label_subtype, label, address
        FROM ethereum.core.dim_labels
        WHERE label_type = 'dex'
        OR label_type = 'lending'
        LIMIT 5;
        """
        result = client.execute_query(sql)
        print("\nSample DeFi labels:")
        for row in result.to_dict('records'):
            print(f"- {row}")
            
        # Check Uniswap specific tables
        logger.info("\nChecking Uniswap tables...")
        sql = """
        SELECT DISTINCT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'ethereum'
        AND LOWER(table_name) LIKE '%uniswap%';
        """
        result = client.execute_query(sql)
        print("\nUniswap-related tables:")
        for row in result.to_dict('records'):
            print(f"- {row['table_name']}")
            
    except Exception as e:
        logger.error(f"❌ Failed to check DeFi tables: {str(e)}")
        logger.error(f"Error details: {str(e)}")

    # Test DeFi metric query
    print("Testing DeFi metric query...")
    try:
        query = """
        WITH dex_labels AS (
            SELECT address 
            FROM ethereum.core.dim_labels 
            WHERE label_type = 'dex'
            AND label = 'uniswap'
        )
        SELECT 
            DATE_TRUNC('day', block_timestamp) as date,
            COUNT(DISTINCT tx_hash) as swap_count,
            COUNT(DISTINCT from_address) as unique_traders
        FROM ethereum.core.fact_transactions
        WHERE to_address IN (SELECT address FROM dex_labels)
        AND block_timestamp >= DATEADD('day', -1, CURRENT_TIMESTAMP())
        GROUP BY 1
        ORDER BY 1 DESC
        LIMIT 1
        """
        result = client.execute_query(query)
        print(f"\nLast day's DeFi activity:\n{result}")
    except Exception as e:
        print(f"❌ Failed to execute DeFi metric query: {e}")

if __name__ == "__main__":
    main() 