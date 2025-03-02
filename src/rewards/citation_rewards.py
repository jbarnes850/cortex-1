"""
Citation-specific reward functions for evaluating LLM responses in financial analysis.
These functions implement specialized rewards that focus on proper citation of metrics
and historical references in market analysis.
"""

import re
from typing import Dict, List, Optional, Union, Any, Set
from src.rewards.finance_rewards import BaseReward


class MetricCitationReward(BaseReward):
    """Reward function that evaluates proper citation of specific metrics."""
    
    def __init__(self, weight: float = 1.0, required_metrics: Optional[List[str]] = None):
        """Initialize the metric citation reward.
        
        Args:
            weight: The weight of this reward
            required_metrics: List of metrics that should be cited
        """
        super().__init__(weight=weight)
        
        # Default required metrics if none provided
        self.required_metrics = required_metrics or [
            "daily_txns", 
            "unique_users", 
            "total_volume", 
            "avg_tx_value", 
            "txn_growth", 
            "user_growth", 
            "volatility", 
            "gas_used", 
            "success_rate"
        ]
    
    def calculate(self, response: str, context: Optional[Dict[str, Any]] = None) -> float:
        """Calculate the reward based on citation of required metrics.
        
        Args:
            response: The model's generated text
            context: Optional context with additional metrics
            
        Returns:
            A score between 0.0 and 1.0
        """
        # Get metrics that should be present in the context
        expected_metrics = set(self.required_metrics)
        
        # If context provides additional metrics, add them
        if context and "metrics" in context:
            expected_metrics.update(context["metrics"])
        
        # Extract citations from the response
        citations = self._extract_citations(response)
        cited_metrics = set(citations)
        
        if not cited_metrics:
            return 0.0
        
        # Calculate how many required metrics were cited
        cited_required = cited_metrics.intersection(expected_metrics)
        citation_ratio = len(cited_required) / len(expected_metrics)
        
        # Apply a penalty if very few citations in general
        if len(cited_metrics) < 5:
            citation_ratio *= 0.8
        
        # Apply a bonus for citing metrics in calculations
        if self._has_citations_in_calculations(response, cited_metrics):
            citation_ratio = min(1.0, citation_ratio * 1.2)
        
        return min(1.0, citation_ratio)
    
    def _extract_citations(self, response: str) -> List[str]:
        """Extract metric citations from the response.
        
        Args:
            response: The model's generated text
            
        Returns:
            A list of cited metric names
        """
        # Pattern for [metric_name] citations
        citation_pattern = r'\[([a-zA-Z_][a-zA-Z0-9_]*)\]'
        return re.findall(citation_pattern, response)
    
    def _has_citations_in_calculations(self, response: str, cited_metrics: Set[str]) -> bool:
        """Check if citations are used in calculations.
        
        Args:
            response: The model's generated text
            cited_metrics: Set of metrics cited in the response
            
        Returns:
            True if citations are used in calculations
        """
        # Pattern for calculations involving cited metrics
        for metric in cited_metrics:
            # Look for the metric in a calculation context
            pattern = r'\[' + re.escape(metric) + r'\]\s*(?:[-+*/]|is|equals|was|calculated as)'
            if re.search(pattern, response, re.IGNORECASE):
                return True
            
            # Look for calculations using the metric
            pattern = r'(?:[-+*/]|calculate|computing|using)\s+\[' + re.escape(metric) + r'\]'
            if re.search(pattern, response, re.IGNORECASE):
                return True
        
        return False


class HistoricalReferenceReward(BaseReward):
    """Reward function that evaluates proper reference to historical data/patterns."""
    
    def __init__(self, weight: float = 1.0):
        """Initialize the historical reference reward.
        
        Args:
            weight: The weight of this reward
        """
        super().__init__(weight=weight)
        
        # Historical reference keywords
        self.historical_terms = [
            "historical", "previous", "prior", "past", "earlier",
            "last week", "last month", "last year", "trend", "pattern",
            "compared to", "relative to", "baseline", "benchmark"
        ]
        
        # Time period references
        self.time_periods = [
            "day", "week", "month", "quarter", "year", "period"
        ]
    
    def calculate(self, response: str, context: Optional[Dict[str, Any]] = None) -> float:
        """Calculate the reward based on historical references.
        
        Args:
            response: The model's generated text
            context: Optional context (not used)
            
        Returns:
            A score between 0.0 and 1.0
        """
        # Extract historical references from the response
        references = self._extract_historical_references(response)
        
        if not references:
            return 0.0
        
        # Score the quality of references
        quality_scores = []
        for reference in references:
            score = self._score_reference_quality(reference)
            quality_scores.append(score)
        
        # Calculate the average quality score
        avg_quality = sum(quality_scores) / len(quality_scores)
        
        # Apply a bonus for the number of references (up to 5)
        reference_count_factor = min(len(references) / 5, 1.0)
        
        # Final score combines quality and quantity
        return avg_quality * (0.7 + 0.3 * reference_count_factor)
    
    def _extract_historical_references(self, response: str) -> List[str]:
        """Extract historical references from the response.
        
        Args:
            response: The model's generated text
            
        Returns:
            A list of historical reference strings
        """
        # Build patterns for historical references
        patterns = []
        
        # Pattern for phrases with historical terms
        for term in self.historical_terms:
            patterns.append(r'(?:[^.!?\n]+\b' + re.escape(term) + r'\b[^.!?\n]+[.!?])')
        
        # Pattern for comparisons with time periods
        for period in self.time_periods:
            patterns.append(r'(?:[^.!?\n]+\b(?:previous|last|prior)\s+' + re.escape(period) + r'\b[^.!?\n]+[.!?])')
            patterns.append(r'(?:[^.!?\n]+\b' + re.escape(period) + r'\s+(?:ago|prior|before)\b[^.!?\n]+[.!?])')
        
        # Find all matches
        references = []
        for pattern in patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            references.extend(match.group(0) for match in matches)
        
        return list(set(references))  # Remove duplicates
    
    def _score_reference_quality(self, reference: str) -> float:
        """Score the quality of a historical reference.
        
        Args:
            reference: A historical reference string
            
        Returns:
            A quality score between 0.0 and 1.0
        """
        score = 0.5  # Base score for having a reference
        
        # Bonus for specific time periods
        if re.search(r'\b\d+\s+(?:day|week|month|year)s?\b', reference, re.IGNORECASE):
            score += 0.2
        
        # Bonus for numerical comparisons
        if re.search(r'\b(?:increased|decreased|grew|declined)\s+by\s+\d+(?:\.\d+)?%', reference, re.IGNORECASE):
            score += 0.2
        
        # Bonus for explicit trends
        if re.search(r'\b(?:trend|pattern|cycle|historically)\b', reference, re.IGNORECASE):
            score += 0.1
        
        # Bonus for comparative analysis
        if re.search(r'\b(?:compared to|relative to|against|versus)\b', reference, re.IGNORECASE):
            score += 0.1
        
        return min(1.0, score) 