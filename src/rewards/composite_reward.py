"""
Composite reward function that combines individual rewards for financial analysis evaluation.
"""

from typing import Dict, List, Optional, Union, Any, Callable
from src.rewards.finance_rewards import (
    BaseReward,
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


class CompositeReward(BaseReward):
    """Composite reward function that combines multiple individual rewards."""
    
    def __init__(self, 
                 rewards: List[BaseReward],
                 name: str = "CompositeReward"):
        """Initialize the composite reward.
        
        Args:
            rewards: List of reward functions to combine
            name: Name for this composite reward
        """
        super().__init__(weight=1.0, name=name)
        self.rewards = rewards
    
    def calculate(self, response: str, context: Optional[Dict[str, Any]] = None) -> float:
        """Calculate the composite reward score.
        
        Args:
            response: The model's generated text
            context: Optional context information
            
        Returns:
            The weighted sum of all reward components
        """
        total_weight = sum(reward.weight for reward in self.rewards)
        
        if total_weight == 0:
            return 0.0
        
        # Calculate and normalize individual reward scores
        reward_scores = {}
        weighted_sum = 0.0
        
        for reward in self.rewards:
            score = reward(response, context)
            reward_scores[reward.name] = score
            weighted_sum += score  # Each reward function applies its own weight
        
        # Store the individual scores for debugging
        self.last_scores = reward_scores
        
        # Return the combined score
        return weighted_sum / total_weight
    
    def get_component_scores(self) -> Dict[str, float]:
        """Get the scores of the individual reward components from the last calculation.
        
        Returns:
            A dictionary mapping reward names to scores
        """
        if hasattr(self, 'last_scores'):
            return self.last_scores
        return {}


def create_financial_reward(weights: Optional[Dict[str, float]] = None) -> CompositeReward:
    """Create a standard composite reward for financial analysis.
    
    Args:
        weights: Optional dictionary of reward component weights
        
    Returns:
        A composite reward function
    """
    # Default weights if none provided
    default_weights = {
        "calculation_accuracy": 1.0,
        "confidence_interval": 0.8,
        "investment_insight": 1.0,
        "citation_format": 0.7,
        "structure": 0.6,
        "completeness": 0.8,
        "metric_citation": 0.9,
        "historical_reference": 0.7
    }
    
    # Use provided weights or defaults
    weights = weights or default_weights
    
    # Create individual reward functions with appropriate weights
    rewards = [
        CalculationAccuracyReward(weight=weights.get("calculation_accuracy", default_weights["calculation_accuracy"])),
        ConfidenceIntervalReward(weight=weights.get("confidence_interval", default_weights["confidence_interval"])),
        InvestmentInsightReward(weight=weights.get("investment_insight", default_weights["investment_insight"])),
        CitationFormatReward(weight=weights.get("citation_format", default_weights["citation_format"])),
        StructureReward(weight=weights.get("structure", default_weights["structure"])),
        CompletenessReward(weight=weights.get("completeness", default_weights["completeness"])),
        MetricCitationReward(weight=weights.get("metric_citation", default_weights["metric_citation"])),
        HistoricalReferenceReward(weight=weights.get("historical_reference", default_weights["historical_reference"]))
    ]
    
    return CompositeReward(rewards, name="FinancialAnalysisReward") 