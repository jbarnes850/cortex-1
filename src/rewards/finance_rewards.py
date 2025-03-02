"""
Finance-specific reward functions for evaluating LLM responses in financial analysis.
These functions implement specialized rewards that focus on calculation accuracy,
confidence intervals, and the quality of investment insights.
"""

import re
import math
from typing import Dict, List, Optional, Union, Any, Callable
import numpy as np
from abc import ABC, abstractmethod


class BaseReward(ABC):
    """Abstract base class for all reward functions."""
    
    def __init__(self, weight: float = 1.0, name: Optional[str] = None):
        """Initialize the reward function.
        
        Args:
            weight: The weight of this reward in a composite reward
            name: Optional name for the reward function
        """
        self.weight = weight
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def calculate(self, response: str, context: Optional[Dict[str, Any]] = None) -> float:
        """Calculate the reward score for the response.
        
        Args:
            response: The model's generated text
            context: Optional context information (e.g., metrics, expected values)
            
        Returns:
            A float representing the reward score
        """
        pass
    
    def __call__(self, response: str, context: Optional[Dict[str, Any]] = None) -> float:
        """Make the reward function callable.
        
        Args:
            response: The model's generated text
            context: Optional context information
            
        Returns:
            The weighted reward score
        """
        score = self.calculate(response, context)
        return score * self.weight


class CalculationAccuracyReward(BaseReward):
    """Reward function that evaluates the accuracy of financial calculations."""
    
    def __init__(self, weight: float = 1.0, tolerance: float = 0.05):
        """Initialize the calculation accuracy reward.
        
        Args:
            weight: The weight of this reward
            tolerance: The error tolerance for numerical calculations (as a percentage)
        """
        super().__init__(weight=weight)
        self.tolerance = tolerance
    
    def calculate(self, response: str, context: Optional[Dict[str, Any]] = None) -> float:
        """Calculate the reward based on calculation accuracy.
        
        Args:
            response: The model's generated text
            context: Optional context with expected calculations
            
        Returns:
            A score between 0.0 and 1.0
        """
        # Extract calculations from the response
        calculations = self._extract_calculations(response)
        
        if not calculations:
            return 0.0
        
        # If we have expected calculations in the context, verify them
        if context and "expected_calculations" in context:
            expected = context["expected_calculations"]
            verified_count = 0
            
            for calc in calculations:
                result = self._extract_calculation_result(calc)
                if result and any(self._is_close_match(result, exp_val, self.tolerance) 
                                 for exp_val in expected.values()):
                    verified_count += 1
            
            return min(1.0, verified_count / len(expected))
        
        # Without expected values, we reward for showing calculation steps
        completeness_scores = []
        for calc in calculations:
            # Check if calculation shows steps (equation followed by result)
            if "=" in calc and self._has_numerical_result(calc):
                completeness_scores.append(1.0)
            # Partial credit for showing just a formula or just a result
            elif self._has_formula(calc) or self._has_numerical_result(calc):
                completeness_scores.append(0.5)
            else:
                completeness_scores.append(0.0)
        
        if not completeness_scores:
            return 0.0
            
        return sum(completeness_scores) / len(completeness_scores)
    
    def _extract_calculations(self, response: str) -> List[str]:
        """Extract calculation snippets from the response.
        
        Args:
            response: The model's generated text
            
        Returns:
            A list of calculation strings
        """
        # Pattern to find calculation sections
        patterns = [
            # Explicit calculation sections
            r'(?:Calculation|Computing|Formula):\s*(.+?)(?=\n\n|\n[A-Z]|\Z)',
            # Equations with equals sign
            r'([^.\n]+?\s*=\s*[-+]?[0-9]*\.?[0-9]+[^.\n]*)',
            # Named calculations
            r'([a-zA-Z_]+\s+(?:is|was|equals|calculated as)\s+[-+]?[0-9]*\.?[0-9]+(?:\s*%)?)',
        ]
        
        calculations = []
        for pattern in patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE | re.DOTALL)
            calculations.extend(match.group(1).strip() for match in matches)
        
        return list(set(calculations))  # Remove duplicates
    
    def _extract_calculation_result(self, calculation: str) -> Optional[float]:
        """Extract the numerical result from a calculation.
        
        Args:
            calculation: A calculation string
            
        Returns:
            The extracted numerical result or None
        """
        # Pattern to find the result after equals sign
        result_match = re.search(r'=\s*([-+]?[0-9]*\.?[0-9]+)', calculation)
        if result_match:
            try:
                return float(result_match.group(1))
            except ValueError:
                return None
        
        # Pattern to find result stated as "X is Y"
        is_match = re.search(r'(?:is|was|equals|calculated as)\s+([-+]?[0-9]*\.?[0-9]+)', calculation, re.IGNORECASE)
        if is_match:
            try:
                return float(is_match.group(1))
            except ValueError:
                return None
        
        return None
    
    def _is_close_match(self, value: float, expected: float, tolerance: float) -> bool:
        """Check if a calculated value is close to the expected value.
        
        Args:
            value: The calculated value
            expected: The expected value
            tolerance: The allowed tolerance as a percentage
            
        Returns:
            True if the values match within tolerance
        """
        if expected == 0:
            return abs(value) < tolerance
        
        rel_error = abs((value - expected) / expected)
        return rel_error <= tolerance
    
    def _has_numerical_result(self, text: str) -> bool:
        """Check if the text contains a numerical result.
        
        Args:
            text: The text to check
            
        Returns:
            True if a numerical result is found
        """
        return bool(re.search(r'[-+]?[0-9]*\.?[0-9]+', text))
    
    def _has_formula(self, text: str) -> bool:
        """Check if the text contains a mathematical formula.
        
        Args:
            text: The text to check
            
        Returns:
            True if a formula is found
        """
        return bool(re.search(r'[-+*/()]|divide|multiply|subtract|add', text, re.IGNORECASE))


class ConfidenceIntervalReward(BaseReward):
    """Reward function that evaluates the proper use of confidence intervals."""
    
    def __init__(self, weight: float = 1.0):
        """Initialize the confidence interval reward.
        
        Args:
            weight: The weight of this reward
        """
        super().__init__(weight=weight)
    
    def calculate(self, response: str, context: Optional[Dict[str, Any]] = None) -> float:
        """Calculate the reward based on confidence interval usage.
        
        Args:
            response: The model's generated text
            context: Optional context (not used)
            
        Returns:
            A score between 0.0 and 1.0
        """
        # Check for confidence intervals in the response
        intervals = self._extract_confidence_intervals(response)
        
        if not intervals:
            return 0.0
        
        # Score each interval based on its quality
        quality_scores = []
        for interval in intervals:
            score = 0.0
            
            # Does it specify a confidence level?
            if self._has_confidence_level(interval):
                score += 0.5
            
            # Does it have proper bounds?
            if self._has_proper_bounds(interval):
                score += 0.5
            
            quality_scores.append(score)
        
        # Return the average quality score
        return sum(quality_scores) / len(quality_scores)
    
    def _extract_confidence_intervals(self, response: str) -> List[str]:
        """Extract confidence interval statements from the response.
        
        Args:
            response: The model's generated text
            
        Returns:
            A list of confidence interval strings
        """
        # Patterns to find confidence intervals
        patterns = [
            r'(?:[0-9]{1,3}%)\s*confidence\s*interval\s*(?:is|of|:)?\s*\[?([^]\n]+?)\]?(?=\.|,|\n)',
            r'(?:with|at)\s+([0-9]{1,3}%)\s+confidence\s*[:,]?\s*([^.\n]+)',
            r'(?:range|interval|bounds)\s+of\s+([^.\n]+)\s+with\s+([0-9]{1,3}%)\s+confidence',
        ]
        
        intervals = []
        for pattern in patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            intervals.extend(match.group(0) for match in matches)
        
        return intervals
    
    def _has_confidence_level(self, interval: str) -> bool:
        """Check if the interval specifies a confidence level.
        
        Args:
            interval: A confidence interval string
            
        Returns:
            True if a confidence level is specified
        """
        return bool(re.search(r'([0-9]{1,3})%', interval))
    
    def _has_proper_bounds(self, interval: str) -> bool:
        """Check if the interval has proper lower and upper bounds.
        
        Args:
            interval: A confidence interval string
            
        Returns:
            True if proper bounds are found
        """
        # Check for bracket notation [lower, upper]
        bracket_match = re.search(r'\[([-+]?[0-9]*\.?[0-9]+)\s*,\s*([-+]?[0-9]*\.?[0-9]+)\]', interval)
        if bracket_match:
            try:
                lower = float(bracket_match.group(1))
                upper = float(bracket_match.group(2))
                return lower < upper
            except ValueError:
                return False
        
        # Check for range description "between X and Y"
        between_match = re.search(r'between\s+([-+]?[0-9]*\.?[0-9]+)\s+and\s+([-+]?[0-9]*\.?[0-9]+)', interval, re.IGNORECASE)
        if between_match:
            try:
                lower = float(between_match.group(1))
                upper = float(between_match.group(2))
                return lower < upper
            except ValueError:
                return False
        
        return False


class InvestmentInsightReward(BaseReward):
    """Reward function that evaluates the quality of investment insights."""
    
    def __init__(self, weight: float = 1.0):
        """Initialize the investment insight reward.
        
        Args:
            weight: The weight of this reward
        """
        super().__init__(weight=weight)
        
        # Keywords that indicate investment insights
        self.insight_keywords = [
            "investment", "opportunity", "strategy", "portfolio", 
            "allocate", "risk", "return", "profit", "upside", "downside",
            "position", "investor", "market timing", "entry point", "exit point",
            "hodl", "accumulate", "divest", "rebalance"
        ]
        
        # Phrases that indicate quantitative reasoning
        self.quantitative_phrases = [
            "percent", "ratio", "correlation", "expected return", 
            "risk-adjusted", "probability", "likelihood", "ROI",
            "basis points", "yield", "volume", "growth rate"
        ]
    
    def calculate(self, response: str, context: Optional[Dict[str, Any]] = None) -> float:
        """Calculate the reward based on investment insight quality.
        
        Args:
            response: The model's generated text
            context: Optional context (not used)
            
        Returns:
            A score between 0.0 and 1.0
        """
        # Extract investment insights
        insights = self._extract_investment_insights(response)
        
        if not insights:
            return 0.0
        
        # Score each insight based on quality criteria
        insight_scores = []
        for insight in insights:
            score = 0.0
            
            # Check for specific recommendation
            if self._has_specific_recommendation(insight):
                score += 0.3
            
            # Check for quantitative support
            if self._has_quantitative_support(insight):
                score += 0.4
            
            # Check for risk assessment
            if self._has_risk_assessment(insight):
                score += 0.3
            
            insight_scores.append(score)
        
        # Calculate the average quality score
        avg_quality = sum(insight_scores) / len(insight_scores)
        
        # Apply a bonus for the number of insights (up to 3)
        insight_count_factor = min(len(insights) / 3, 1.0)
        
        # Final score combines quality and quantity
        return avg_quality * (0.7 + 0.3 * insight_count_factor)
    
    def _extract_investment_insights(self, response: str) -> List[str]:
        """Extract investment insight sections from the response.
        
        Args:
            response: The model's generated text
            
        Returns:
            A list of investment insight strings
        """
        # First try to find dedicated investment sections
        section_patterns = [
            r'(?:Investment|Trading)\s+(?:Insights?|Implications?|Recommendations?|Strategy):\s*(.+?)(?=\n\n|\n[A-Z]|\Z)',
            r'(?:Implications|Opportunities|Strategies)\s+for\s+(?:Investors|Traders):\s*(.+?)(?=\n\n|\n[A-Z]|\Z)',
        ]
        
        sections = []
        for pattern in section_patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE | re.DOTALL)
            sections.extend(match.group(1).strip() for match in matches)
        
        if sections:
            # If we found sections, split them into individual insights
            insights = []
            for section in sections:
                # Split by numbered points or bullet points
                point_splits = re.split(r'\n\s*(?:[0-9]+\.|•|\*|\-)\s+', '\n' + section)
                points = [p.strip() for p in point_splits if p.strip()]
                insights.extend(points)
            return insights
        
        # If no dedicated sections, look for sentences with insight keywords
        insight_sentences = []
        sentences = re.split(r'(?<=[.!?])\s+', response)
        
        for sentence in sentences:
            lower_sent = sentence.lower()
            # Check if the sentence contains insight keywords
            if any(keyword in lower_sent for keyword in self.insight_keywords):
                insight_sentences.append(sentence)
        
        return insight_sentences
    
    def _has_specific_recommendation(self, insight: str) -> bool:
        """Check if the insight contains a specific actionable recommendation.
        
        Args:
            insight: An investment insight string
            
        Returns:
            True if a specific recommendation is found
        """
        recommendation_patterns = [
            r'(?:should|could|recommend|advised to|consider)\s+(?:buy|sell|hold|accumulate|divest|rebalance)',
            r'(?:buying|selling|holding|accumulating)\s+(?:opportunity|position)',
            r'(?:enter|exit|increase|decrease|maintain)\s+(?:position|exposure|allocation)',
            r'(?:bullish|bearish)\s+(?:signal|indicator|outlook)',
        ]
        
        return any(re.search(pattern, insight, re.IGNORECASE) for pattern in recommendation_patterns)
    
    def _has_quantitative_support(self, insight: str) -> bool:
        """Check if the insight is supported by quantitative analysis.
        
        Args:
            insight: An investment insight string
            
        Returns:
            True if quantitative support is found
        """
        # Check for numbers
        has_numbers = bool(re.search(r'[0-9]+(?:\.[0-9]+)?(?:\s*%)?', insight))
        
        # Check for quantitative phrases
        has_quant_phrases = any(phrase in insight.lower() for phrase in self.quantitative_phrases)
        
        return has_numbers or has_quant_phrases
    
    def _has_risk_assessment(self, insight: str) -> bool:
        """Check if the insight includes risk assessment.
        
        Args:
            insight: An investment insight string
            
        Returns:
            True if risk assessment is found
        """
        risk_patterns = [
            r'(?:risk|downside|volatility|uncertainty|exposure)',
            r'(?:potential|possible)\s+(?:loss|decline|decrease|drop)',
            r'if\s+.+\s+(?:fails|decreases|drops)',
            r'stop[\s-]loss',
        ]
        
        return any(re.search(pattern, insight, re.IGNORECASE) for pattern in risk_patterns) 