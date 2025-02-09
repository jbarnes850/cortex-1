"""
Reward function implementation for NEAR Cortex-1 training.
"""

import re
from typing import Dict, List, Optional, Union, Any
import numpy as np
from difflib import SequenceMatcher

class RewardFunction:
    """Implements the reward function design from the training plan."""
    
    def __init__(self):
        """Initialize reward function components."""
        self.technical_indicators = [
            "moving average", "volatility", "momentum", "trend line",
            "support level", "resistance level", "volume profile",
            "correlation coefficient", "regression analysis"
        ]
        
        self.market_concepts = [
            "liquidity", "market depth", "order book", "slippage",
            "market maker", "arbitrage", "yield farming", "impermanent loss"
        ]
        
        self.risk_factors = [
            "volatility risk", "smart contract risk", "regulatory risk",
            "liquidity risk", "counterparty risk", "oracle risk",
            "governance risk", "economic risk"
        ]

        self.cross_chain_concepts = [
            "bridge volume", "cross-chain liquidity", "token bridge",
            "interoperability", "atomic swaps", "wrapped tokens",
            "multi-chain yield", "cross-chain arbitrage"
        ]

        self.required_sections = [
            "Initial Observations",
            "Analysis",
            "Technical Assessment",
            "Risk Evaluation",
            "Opportunities",
            "Conclusion"
        ]

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using SequenceMatcher."""
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

    def _extract_numerical_predictions(self, text: str) -> Dict[str, float]:
        """Extract numerical predictions from text with improved pattern matching."""
        predictions = {}
        
        # Pattern for percentage predictions
        pct_patterns = [
            (r'(?:transaction|tx|volume)\s*growth\s*(?:of|:)?\s*([-+]?\d+(?:\.\d+)?)\s*%', 'transaction_growth'),
            (r'user\s*growth\s*(?:of|:)?\s*([-+]?\d+(?:\.\d+)?)\s*%', 'user_growth'),
            (r'gas\s*(?:price|fee)\s*(?:increase|decrease|change)?\s*(?:of|:)?\s*([-+]?\d+(?:\.\d+)?)\s*%', 'gas_price_change')
        ]
        
        # Pattern for absolute value predictions
        abs_patterns = [
            (r'gas\s*price\s*(?:of|:)?\s*(\d+(?:\.\d+)?)\s*(?:gwei|GWEI)', 'gas_price'),
            (r'daily\s*transactions\s*(?:of|:)?\s*(\d+(?:,\d+)*)', 'daily_transactions'),
            (r'average\s*value\s*(?:of|:)?\s*(\d+(?:\.\d+)?)', 'avg_value')
        ]
        
        # Extract percentages
        for pattern, key in pct_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    predictions[key] = float(match.group(1))
                except (ValueError, IndexError):
                    continue
        
        # Extract absolute values
        for pattern, key in abs_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    # Remove commas from numbers
                    value = match.group(1).replace(',', '')
                    predictions[key] = float(value)
                except (ValueError, IndexError):
                    continue
        
        return predictions

    def _calculate_prediction_accuracy(self, response: str, expected: Dict[str, Any]) -> float:
        """Calculate accuracy of market predictions with trend-based scoring."""
        score = 0.0
        predictions = self._extract_numerical_predictions(response)
        
        for metric, expected_value in expected.items():
            if metric in predictions:
                if isinstance(expected_value, (int, float)):
                    # Calculate trend direction
                    pred_value = predictions[metric]
                    
                    # Trend scoring (40% weight)
                    pred_trend = np.sign(pred_value - expected_value)
                    actual_trend = np.sign(expected_value - pred_value)
                    trend_score = 1.0 if pred_trend == actual_trend else 0.0
                    
                    # Magnitude scoring (60% weight)
                    error = abs(pred_value - expected_value)
                    max_val = max(abs(expected_value), 1e-6)
                    magnitude_score = max(0.0, 1.0 - (error / max_val))
                    
                    # Combined score with weights
                    metric_score = (0.4 * trend_score) + (0.6 * magnitude_score)
                    score += metric_score
                    
                elif isinstance(expected_value, str):
                    # Fuzzy matching for categorical predictions
                    similarity = self._calculate_similarity(
                        str(predictions[metric]), 
                        expected_value
                    )
                    score += similarity
                
                elif isinstance(expected_value, bool):
                    score += 1.0 if predictions[metric] == expected_value else 0.0
        
        return min(1.0, score / max(1, len(expected)))

    def _evaluate_reasoning_depth(self, response: str) -> float:
        """Evaluate depth and quality of chain-of-thought reasoning."""
        score = 0.0
        
        # Check for reasoning steps
        reasoning_markers = ["first", "second", "third", "next", "finally", "therefore"]
        step_count = sum(1 for marker in reasoning_markers if marker.lower() in response.lower())
        score += min(0.2, step_count * 0.05)
        
        # Check for causal relationships
        causal_markers = ["because", "due to", "as a result", "consequently", "leads to"]
        causal_count = sum(1 for marker in causal_markers if marker.lower() in response.lower())
        score += min(0.2, causal_count * 0.05)
        
        # Check for uncertainty handling
        uncertainty_markers = ["likely", "probability", "could", "might", "uncertain"]
        uncertainty_count = sum(1 for marker in uncertainty_markers if marker.lower() in response.lower())
        score += min(0.1, uncertainty_count * 0.02)
        
        return min(0.5, score)
    
    def _evaluate_technical_analysis(self, response: str) -> float:
        """Evaluate quality of technical analysis."""
        score = 0.0
        
        # Check for technical indicators
        indicator_count = sum(1 for indicator in self.technical_indicators 
                            if indicator.lower() in response.lower())
        score += min(0.2, indicator_count * 0.05)
        
        # Check for quantitative analysis
        if re.search(r'\d+%|\d+\s*day|\d+\s*week', response):
            score += 0.1
            
        # Check for multi-timeframe analysis
        timeframes = ["hourly", "daily", "weekly", "monthly"]
        timeframe_count = sum(1 for tf in timeframes if tf in response.lower())
        score += min(0.1, timeframe_count * 0.025)
        
        return min(0.4, score)
    
    def _evaluate_market_understanding(self, response: str) -> float:
        """Evaluate understanding of market dynamics."""
        score = 0.0
        
        # Check for market concepts
        concept_count = sum(1 for concept in self.market_concepts 
                          if concept.lower() in response.lower())
        score += min(0.2, concept_count * 0.04)
        
        # Check for market participant analysis
        participants = ["traders", "investors", "market makers", "arbitrageurs"]
        participant_count = sum(1 for p in participants if p in response.lower())
        score += min(0.1, participant_count * 0.025)
        
        return min(0.3, score)
    
    def _evaluate_risk_assessment(self, response: str) -> float:
        """Evaluate quality of risk assessment."""
        score = 0.0
        
        # Check for risk factors
        risk_count = sum(1 for risk in self.risk_factors 
                        if risk.lower() in response.lower())
        score += min(0.2, risk_count * 0.04)
        
        # Check for mitigation strategies
        if re.search(r'hedge|diversify|monitor|stop[ -]loss', response.lower()):
            score += 0.1
            
        return min(0.3, score)
    
    def _evaluate_cross_chain_analysis(self, response: str) -> float:
        """Evaluate quality of cross-chain analysis with enhanced scoring."""
        score = 0.0
        
        # Check for cross-chain concepts (40% weight)
        concept_count = sum(1 for concept in self.cross_chain_concepts 
                          if concept.lower() in response.lower())
        score += min(0.08, concept_count * 0.02)
        
        # Check for bridge analysis (20% weight)
        bridge_patterns = [
            r'bridge\s*volume\s*(?:of|:)?\s*\d+',
            r'cross-chain\s*liquidity\s*(?:of|:)?\s*\d+',
            r'interoperability\s*metrics?'
        ]
        bridge_matches = sum(1 for pattern in bridge_patterns 
                           if re.search(pattern, response, re.IGNORECASE))
        score += min(0.04, bridge_matches * 0.02)
        
        # Check for correlation analysis (20% weight)
        correlation_patterns = [
            r'correlation\s*(?:of|between|among)',
            r'spillover\s*effects?',
            r'contagion\s*risk',
            r'cross-chain\s*impact'
        ]
        correlation_matches = sum(1 for pattern in correlation_patterns 
                                if re.search(pattern, response, re.IGNORECASE))
        score += min(0.04, correlation_matches * 0.02)
        
        # Check for multi-chain strategy (20% weight)
        strategy_patterns = [
            r'arbitrage\s*opportunity',
            r'cross-chain\s*yield',
            r'multi-chain\s*portfolio',
            r'bridge\s*strategy'
        ]
        strategy_matches = sum(1 for pattern in strategy_patterns 
                             if re.search(pattern, response, re.IGNORECASE))
        score += min(0.04, strategy_matches * 0.02)
        
        return min(0.2, score)
    
    def _calculate_group_bonus(self, 
                             base_score: float,
                             response: str,
                             group_responses: List[str]) -> float:
        """Calculate bonus based on relative performance within group."""
        if not group_responses:
            return 0.0
            
        # Calculate uniqueness score
        similarity_scores = []
        for other in group_responses:
            if other != response:
                similarity = self._calculate_similarity(response, other)
                similarity_scores.append(similarity)
                
        avg_similarity = np.mean(similarity_scores) if similarity_scores else 1.0
        uniqueness_bonus = max(0.0, 0.2 * (1.0 - avg_similarity))
        
        # Calculate relative quality bonus
        other_scores = [self._quick_quality_score(r) for r in group_responses if r != response]
        avg_score = np.mean(other_scores) if other_scores else base_score
        quality_bonus = max(0.0, 0.2 * (base_score - avg_score))
        
        return min(0.3, uniqueness_bonus + quality_bonus)
    
    def _calculate_penalties(self, response: str) -> float:
        """Calculate quality penalties."""
        penalties = 0.0
        
        # Check for contradictions
        contradictions = [
            (r"bullish.*bearish", "same paragraph"),
            (r"increase.*decrease", "same sentence"),
            (r"positive.*negative", "same context")
        ]
        for pattern, context in contradictions:
            if re.search(pattern, response.lower()):
                penalties += 0.1
                
        # Check for vague language
        vague_terms = ["very", "really", "quite", "somewhat", "kind of"]
        vague_count = sum(1 for term in vague_terms if term in response.lower())
        penalties += min(0.2, vague_count * 0.04)
        
        # Check for unsupported claims
        if re.search(r"always|never|guaranteed|definitely", response.lower()):
            penalties += 0.1
            
        return min(0.5, penalties)
    
    def _evaluate_data_usage(self, response: str, context_data: Union[Dict, List]) -> float:
        """Evaluate how effectively the response uses provided data.
        
        Args:
            response: The model's generated response
            context_data: Dictionary or list of market data
            
        Returns:
            Float score between 0.0 and 0.3
        """
        score = 0.0
        
        # Convert list to dictionary if needed
        if isinstance(context_data, list):
            # Handle list of dictionaries
            metrics = {}
            for item in context_data:
                if isinstance(item, dict):
                    metrics.update(item)
                else:
                    # Handle primitive list items
                    metrics[str(item)] = item
        else:
            metrics = context_data
            
        # Check for metric citations
        metrics_cited = 0
        total_metrics = 0
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                total_metrics += 1
                # Check for the value in the response with some flexibility
                value_str = f"{value:.2f}" if isinstance(value, float) else str(value)
                if value_str in response or key in response.lower():
                    metrics_cited += 1
                    
        # Calculate citation score
        if total_metrics > 0:
            citation_ratio = metrics_cited / total_metrics
            score += min(0.2, citation_ratio * 0.3)
        
        # Check for comparative analysis
        comparisons = len(re.findall(r'increased by|decreased by|grew|declined|compared to', response))
        score += min(0.1, comparisons * 0.02)
        
        # Check for data interpretation
        interpretation_markers = [
            r'\d+%\s*increase', r'\d+%\s*decrease',
            r'growth rate of\s*\d+', r'decline of\s*\d+',
            r'trending\s*upward', r'trending\s*downward'
        ]
        
        interpretations = sum(1 for marker in interpretation_markers 
                            if re.search(marker, response.lower()))
        score += min(0.1, interpretations * 0.02)
        
        return min(0.3, score)
    
    def _quick_quality_score(self, response: str) -> float:
        """Quick quality assessment for group comparison."""
        score = 0.0
        
        # Check reasoning markers
        reasoning_markers = ["because", "therefore", "as a result", "due to"]
        score += 0.3 * sum(1 for marker in reasoning_markers if marker in response.lower()) / len(reasoning_markers)
        
        # Check technical content
        score += 0.4 * sum(1 for indicator in self.technical_indicators if indicator in response.lower()) / len(self.technical_indicators)
        
        # Check data usage
        score += 0.3 * len(re.findall(r'\d+\.?\d*%|\$\d+\.?\d*[KMB]?', response)) / 10
        
        return min(1.0, score)

    def calculate_reward(self, 
                        response: str,
                        prompt: Dict[str, Any],
                        group_responses: Optional[List[str]] = None) -> Dict[str, float]:
        """Calculate the composite reward score for a model response."""
        rewards = {}
        
        # Base requirements check
        if len(response.split()) < 100:  # Minimum length requirement
            return {'final_total': 0.0, 'error': 'Response too short'}
        
        # 1. Section Coverage (0.0 - 1.0)
        section_count = sum(1 for section in self.required_sections 
                          if section.lower() in response.lower())
        section_score = section_count / len(self.required_sections)
        if section_score < 0.8:  # Must have at least 80% of required sections
            return {'final_total': 0.0, 'error': 'Insufficient section coverage'}
        
        # 2. Prediction Accuracy (0.0 - 1.0)
        if 'expected_outcome' in prompt or 'outcome_data' in prompt:
            rewards['prediction_accuracy'] = self._calculate_prediction_accuracy(
                response,
                prompt.get('expected_outcome', prompt.get('outcome_data', {}))
            )
        
        # 3. Chain-of-Thought Depth (0.0 - 0.5)
        rewards['reasoning_depth'] = self._evaluate_reasoning_depth(response)
        
        # 4. Technical Analysis Quality (0.0 - 0.4)
        rewards['technical_quality'] = self._evaluate_technical_analysis(response)
        
        # 5. Market Understanding (0.0 - 0.3)
        rewards['market_understanding'] = self._evaluate_market_understanding(response)
        
        # 6. Risk Assessment (0.0 - 0.3)
        rewards['risk_assessment'] = self._evaluate_risk_assessment(response)
        
        # 7. Data Usage (0.0 - 0.3)
        if 'context_data' in prompt:
            rewards['data_usage'] = self._evaluate_data_usage(
                response,
                prompt['context_data']
            )
        
        # 8. Cross-Chain Analysis (0.0 - 0.2)
        rewards['cross_chain'] = self._evaluate_cross_chain_analysis(response)
        
        # Calculate base reward
        rewards['base_total'] = sum(rewards.values())
        
        # Apply group policy adjustment if group responses provided
        if group_responses:
            rewards['group_bonus'] = self._calculate_group_bonus(
                rewards['base_total'],
                response,
                group_responses
            )
            rewards['final_total'] = rewards['base_total'] + rewards['group_bonus']
        else:
            rewards['final_total'] = rewards['base_total']
        
        # Apply quality penalties
        penalties = self._calculate_penalties(response)
        rewards['penalties'] = penalties
        rewards['final_total'] = max(0.0, rewards['final_total'] - penalties)
        
        return rewards 