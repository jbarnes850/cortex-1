"""
NEAR Cortex-1 Rewards Module
This package contains specialized reward functions for financial analysis.
"""

from src.rewards.finance_rewards import (
    CalculationAccuracyReward,
    ConfidenceIntervalReward,
    InvestmentInsightReward
)

from src.rewards.format_rewards import (
    CitationFormatReward,
    StructureReward,
    CompletenessReward
)

from src.rewards.citation_rewards import (
    MetricCitationReward,
    HistoricalReferenceReward
)

from src.rewards.composite_reward import (
    CompositeReward,
    create_financial_reward
)

# Convenience function to create standard financial reward
def get_default_financial_reward():
    """Return the default composite financial reward function."""
    return create_financial_reward() 