"""
Market conditions analyzer for blockchain data.
"""

import logging
from typing import Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

class MarketConditions:
    """Analyzes market conditions from blockchain data."""
    
    def __init__(self):
        """Initialize the market conditions analyzer."""
        pass
        
    def analyze(self, data_point: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market conditions from a single data point.
        
        Args:
            data_point: Dictionary containing market metrics
            
        Returns:
            Dictionary containing market condition analysis
        """
        try:
            conditions = {
                'market_state': self._determine_market_state(data_point),
                'growth_metrics': self._analyze_growth(data_point),
                'risk_metrics': self._analyze_risk(data_point),
                'efficiency_metrics': self._analyze_efficiency(data_point)
            }
            return conditions
            
        except Exception as e:
            logger.error(f"Error analyzing market conditions: {str(e)}")
            return {}
            
    def _determine_market_state(self, data: Dict[str, Any]) -> str:
        """Determine the overall market state."""
        try:
            txn_growth = float(data.get('txn_growth_pct_7d', 0))
            user_growth = float(data.get('user_growth_pct_7d', 0))
            volatility = float(data.get('tx_volatility_7d', 0))
            
            # Simple state determination based on growth and volatility
            if txn_growth > 10 and user_growth > 5:
                return 'expanding'
            elif txn_growth < -10 or user_growth < -5:
                return 'contracting'
            elif volatility > 2.0:
                return 'volatile'
            else:
                return 'stable'
                
        except Exception as e:
            logger.warning(f"Error determining market state: {str(e)}")
            return 'unknown'
            
    def _analyze_growth(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze growth metrics."""
        try:
            txn_growth = float(data.get('txn_growth_pct_7d', 0))
            user_growth = float(data.get('user_growth_pct_7d', 0))
            
            return {
                'growth_rate': (txn_growth + user_growth) / 2,
                'growth_stability': abs(txn_growth - user_growth) < 5,
                'growth_trend': 'positive' if txn_growth > 0 and user_growth > 0 else 'negative'
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing growth: {str(e)}")
            return {}
            
    def _analyze_risk(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze risk metrics."""
        try:
            volatility = float(data.get('tx_volatility_7d', 0))
            success_rate = float(data.get('success_rate', 100))
            
            return {
                'risk_level': 'high' if volatility > 2.0 or success_rate < 95 else 'moderate' if volatility > 1.0 or success_rate < 98 else 'low',
                'volatility_score': min(volatility / 2.0, 1.0),
                'reliability_score': success_rate / 100
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing risk: {str(e)}")
            return {}
            
    def _analyze_efficiency(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze network efficiency metrics."""
        try:
            gas_used = float(data.get('gas_used', 0))
            num_txs = float(data.get('num_txs', 1))
            success_rate = float(data.get('success_rate', 100))
            
            return {
                'gas_efficiency': gas_used / max(num_txs, 1),
                'transaction_efficiency': success_rate / 100,
                'network_load': 'high' if num_txs > 1000000 else 'moderate' if num_txs > 100000 else 'low'
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing efficiency: {str(e)}")
            return {}

def classify_market_condition(market_data: dict) -> str:
    """
    Classify market condition based on comprehensive metrics analysis.
    Returns one of: 'bullish', 'bearish', 'sideways', 'volatile'
    """
    # Extract key metrics
    tx_growth = market_data.get('txn_growth_pct_7d', 0)
    user_growth = market_data.get('user_growth_pct_7d', 0)
    volume_growth = market_data.get('volume_growth_pct_7d', 0)
    volatility = market_data.get('tx_volatility_7d', 0)
    avg_tx_value_change = market_data.get('avg_tx_value_change_pct', 0)
    smart_contract_activity = market_data.get('smart_contract_calls', 0)
    bridge_activity = market_data.get('bridge_volume', 0)
    
    # Define thresholds
    GROWTH_THRESHOLD = 5.0  # 5% growth
    HIGH_VOLATILITY = 20.0  # 20% volatility
    SIGNIFICANT_CONTRACT_ACTIVITY = 1000  # Number of contract calls
    SIGNIFICANT_BRIDGE_VOLUME = 10000  # Bridge volume threshold
    
    # Calculate composite scores
    growth_score = (tx_growth + user_growth + volume_growth) / 3
    activity_score = 1 if (smart_contract_activity > SIGNIFICANT_CONTRACT_ACTIVITY or 
                          bridge_activity > SIGNIFICANT_BRIDGE_VOLUME) else 0
    
    # Classify market condition
    if volatility > HIGH_VOLATILITY:
        return 'volatile'
    elif growth_score > GROWTH_THRESHOLD and avg_tx_value_change > 0:
        return 'bullish'
    elif growth_score < -GROWTH_THRESHOLD and avg_tx_value_change < 0:
        return 'bearish'
    else:
        return 'sideways'

def analyze_market_trends(market_data_list: list[dict]) -> dict:
    """
    Analyze market trends over time using multiple data points.
    Returns trend analysis with key metrics and patterns.
    """
    if not market_data_list:
        return {
            'dominant_condition': 'unknown',
            'trend_strength': 0,
            'volatility_level': 'low',
            'growth_trajectory': 'stable'
        }
        
    conditions = [classify_market_condition(data) for data in market_data_list]
    
    # Calculate trend metrics
    condition_counts = Counter(conditions)
    dominant_condition = max(condition_counts.items(), key=lambda x: x[1])[0]
    
    # Calculate trend strength (0-1)
    max_count = max(condition_counts.values())
    trend_strength = max_count / len(conditions)
    
    # Analyze volatility
    volatility_scores = [data.get('tx_volatility_7d', 0) for data in market_data_list]
    avg_volatility = sum(volatility_scores) / len(volatility_scores) if volatility_scores else 0
    
    if avg_volatility > 30:
        volatility_level = 'high'
    elif avg_volatility > 15:
        volatility_level = 'medium'
    else:
        volatility_level = 'low'
        
    # Analyze growth trajectory
    growth_rates = [data.get('txn_growth_pct_7d', 0) for data in market_data_list]
    avg_growth = sum(growth_rates) / len(growth_rates) if growth_rates else 0
    
    if avg_growth > 10:
        growth_trajectory = 'accelerating'
    elif avg_growth > 0:
        growth_trajectory = 'growing'
    elif avg_growth < -10:
        growth_trajectory = 'declining'
    else:
        growth_trajectory = 'stable'
        
    return {
        'dominant_condition': dominant_condition,
        'trend_strength': trend_strength,
        'volatility_level': volatility_level,
        'growth_trajectory': growth_trajectory
    } 