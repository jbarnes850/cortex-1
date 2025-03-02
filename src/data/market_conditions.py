"""
Market conditions analyzer for determining market context.
"""

import logging
from typing import Dict, Any, List, Optional

from src.utils.logger import setup_logger
logger = setup_logger(__name__)

class MarketConditions:
    """
    Analyzes market data to determine market conditions and context.
    This provides additional context for the synthetic data generation.
    """
    
    def __init__(self):
        """Initialize the market conditions analyzer."""
        self.market_conditions = {
            "bull": "The market is in a strong uptrend with increasing prices, volumes, and user activity.",
            "bear": "The market is in a downtrend with decreasing prices, volumes, and user activity.",
            "sideways": "The market is trading in a range without a clear directional trend.",
            "volatile": "The market is experiencing high volatility with rapid price and volume changes.",
            "recovery": "The market is showing signs of recovery after a period of decline.",
            "correction": "The market is experiencing a temporary reversal within a larger trend.",
            "accumulation": "The market is in a phase of accumulation with increasing holdings by long-term investors.",
            "distribution": "The market is in a distribution phase with large holders reducing their positions."
        }
    
    def analyze(self, market_data: Dict[str, Any]) -> str:
        """
        Analyze market data to determine market conditions.
        
        Args:
            market_data: Dictionary of market metrics
            
        Returns:
            String description of market conditions
        """
        try:
            # Extract key metrics
            txn_growth = market_data.get("txn_growth_pct_7d", 0)
            user_growth = market_data.get("user_growth_pct_7d", 0)
            volatility = market_data.get("tx_volatility_7d", 0)
            
            # Determine market condition based on metrics
            if txn_growth > 15 and user_growth > 10:
                condition = "bull"
            elif txn_growth < -10 and user_growth < -5:
                condition = "bear"
            elif abs(txn_growth) < 5 and abs(user_growth) < 5:
                condition = "sideways"
            elif volatility > 0.3:
                condition = "volatile"
            elif -10 < txn_growth < 0 and user_growth > 0:
                condition = "accumulation"
            elif 0 < txn_growth < 10 and user_growth < 0:
                condition = "distribution"
            elif txn_growth > 0 and user_growth > 0 and txn_growth < 15:
                condition = "recovery"
            elif txn_growth < 0 and user_growth < 0 and txn_growth > -10:
                condition = "correction"
            else:
                condition = "sideways"  # Default
            
            # Get the description
            description = self.market_conditions.get(condition, "")
            
            # Add specific metrics to the description
            custom_description = f"{description} Transaction growth is {txn_growth:.1f}% over 7 days, with user growth at {user_growth:.1f}%. Volatility measures at {volatility:.2f}."
            
            # Add network-specific context if available
            network = market_data.get("network", "").upper()
            if network:
                custom_description += f" This analysis is specific to the {network} network."
            
            return custom_description
            
        except Exception as e:
            logger.error(f"Error analyzing market conditions: {str(e)}")
            return "Current market conditions are mixed with no clear directional trend."
    
    def get_condition_list(self) -> List[str]:
        """
        Get a list of possible market conditions.
        
        Returns:
            List of market condition names
        """
        return list(self.market_conditions.keys())
    
    def get_condition_description(self, condition: str) -> Optional[str]:
        """
        Get the description for a specific market condition.
        
        Args:
            condition: Name of the market condition
            
        Returns:
            Description of the market condition
        """
        return self.market_conditions.get(condition) 