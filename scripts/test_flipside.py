#!/usr/bin/env python
"""Test script for Flipside API connection."""

from src.data.flipside_client import FlipsideClient

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
        sql = "DESCRIBE TABLE ethereum.core.fact_transactions;"
        result = client.execute_query(sql)
        print("\nAvailable columns in ethereum.core.fact_transactions:")
        for row in result.to_dict('records'):
            print(f"- {row}")
    except Exception as e:
        print(f"❌ Failed to get transaction schema: {str(e)}")
        
    # Check NEAR transaction schema
    try:
        sql = "DESCRIBE TABLE near.core.fact_transactions;"
        result = client.execute_query(sql)
        print("\nAvailable columns in near.core.fact_transactions:")
        for row in result.to_dict('records'):
            print(f"- {row}")
    except Exception as e:
        print(f"❌ Failed to get NEAR transaction schema: {str(e)}")
        
    # Check DeFi tables
    try:
        sql = "SHOW TABLES IN ethereum.core LIKE '%dex%';"
        result = client.execute_query(sql)
        print("\nAvailable DEX tables in ethereum.core:")
        for row in result.to_dict('records'):
            print(f"- {row}")
            
        sql = "SHOW TABLES IN ethereum.core LIKE '%lending%';"
        result = client.execute_query(sql)
        print("\nAvailable lending tables in ethereum.core:")
        for row in result.to_dict('records'):
            print(f"- {row}")
    except Exception as e:
        print(f"❌ Failed to list DeFi tables: {str(e)}")

if __name__ == "__main__":
    main() 