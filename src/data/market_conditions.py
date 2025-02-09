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