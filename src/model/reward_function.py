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

    def calculate_reward(self, response: str, context_data: Dict[str, Any], outcome_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate reward score for model response."""
        scores = {
            'prediction_accuracy': 0.0,
            'reasoning_depth': 0.0,
            'technical_quality': 0.0,
            'market_understanding': 0.0,
            'risk_assessment': 0.0,
            'data_usage': 0.0,
            'cross_chain': 0.0,
            'penalties': 0.0
        }
        
        # 1. Prediction Accuracy (0.0 - 1.0)
        predictions = self._extract_predictions(response)
        if predictions:
            for pred in predictions:
                if self._has_confidence_interval(pred) and self._has_exact_value(pred):
                    scores['prediction_accuracy'] += 0.25
                if self._validate_prediction_against_outcome(pred, outcome_data):
                    scores['prediction_accuracy'] += 0.25
        
        # 2. Reasoning Depth (0.0 - 1.0)
        sections = self._extract_analysis_sections(response)
        if len(sections) >= 4:  # All required sections present
            scores['reasoning_depth'] = min(1.0, len(sections) * 0.25)
        
        # 3. Technical Quality (0.0 - 1.0)
        calculations = self._extract_calculations(response)
        if calculations:
            scores['technical_quality'] = min(1.0, len(calculations) * 0.2)
            if all(self._has_error_bounds(calc) for calc in calculations):
                scores['technical_quality'] += 0.2
        
        # 4. Market Understanding (0.0 - 1.0)
        citations = self._extract_citations(response)
        if citations:
            scores['market_understanding'] = min(1.0, len(citations) * 0.1)
            if all(self._validate_citation(cite, context_data) for cite in citations):
                scores['market_understanding'] += 0.5
        
        # 5. Risk Assessment (0.0 - 1.0)
        risks = self._extract_risk_factors(response)
        if risks:
            for risk in risks:
                if self._has_probability(risk) and self._has_impact_metric(risk):
                    scores['risk_assessment'] += 0.2
        
        # 6. Data Usage (0.0 - 1.0)
        metrics_used = self._extract_metrics_used(response)
        total_metrics = len(context_data.keys())
        scores['data_usage'] = min(1.0, len(metrics_used) / total_metrics)
        
        # 7. Cross Chain Analysis (0.0 - 1.0)
        correlations = self._extract_correlations(response)
        if correlations:
            scores['cross_chain'] = min(1.0, len(correlations) * 0.33)
            if all(self._validate_correlation(corr) for corr in correlations):
                scores['cross_chain'] += 0.4
        
        # Penalties
        if self._has_inconsistencies(response):
            scores['penalties'] += 0.2
        if self._has_missing_citations(response):
            scores['penalties'] += 0.2
        if self._has_vague_statements(response):
            scores['penalties'] += 0.1
        
        # Calculate base total
        base_total = sum(v for k, v in scores.items() if k != 'penalties')
        
        # Apply confidence bonus (0.1 for each well-supported prediction)
        confidence_bonus = self._calculate_confidence_bonus(response)
        
        # Calculate final total
        final_total = max(0.0, base_total + confidence_bonus - scores['penalties'])
        
        return {
            **scores,
            'base_total': base_total,
            'final_total': final_total
        }

    def _calculate_confidence_bonus(self, response: str) -> float:
        """Calculate confidence bonus based on well-supported predictions."""
        bonus = 0.0
        predictions = self._extract_predictions(response)
        
        for pred in predictions:
            if (self._has_confidence_interval(pred) and 
                self._has_supporting_calculation(pred) and 
                self._has_historical_validation(pred)):
                bonus += 0.1
        
        return min(0.3, bonus)  # Cap bonus at 0.3

    def _has_confidence_interval(self, prediction: str) -> bool:
        """Check if prediction includes a valid confidence interval."""
        confidence_patterns = [
            r'\d{2}%\s*CI\s*:\s*\[\s*-?\d+\.?\d*\s*,\s*-?\d+\.?\d*\s*\]',
            r'with\s+\d{2}%\s+confidence\s+interval'
        ]
        return any(re.search(pattern, prediction, re.IGNORECASE) for pattern in confidence_patterns)

    def _has_supporting_calculation(self, prediction: str) -> bool:
        """Check if prediction includes supporting calculations."""
        calculation_patterns = [
            r'calculated\s+(?:using|by|from)',
            r'=\s*[-+]?\d*\.?\d+',
            r'[-+]?\d*\.?\d+\s*[+\-*/]\s*[-+]?\d*\.?\d+'
        ]
        return any(re.search(pattern, prediction, re.IGNORECASE) for pattern in calculation_patterns)

    def _has_historical_validation(self, prediction: str) -> bool:
        """Check if prediction references historical data for validation."""
        historical_patterns = [
            r'historical\s+(?:data|pattern|trend)',
            r'previous\s+(?:period|week|month)',
            r'compared\s+to\s+[-+]?\d*\.?\d+',
            r'based\s+on\s+past\s+performance'
        ]
        return any(re.search(pattern, prediction, re.IGNORECASE) for pattern in historical_patterns)

    def _extract_predictions(self, response: str) -> List[str]:
        """Extract numerical predictions from response."""
        prediction_patterns = [
            r'predict\s+(?:a|an)?\s*(?:increase|decrease|change)\s+of\s+[-+]?\d+\.?\d*%',
            r'projection:\s*[-+]?\d+\.?\d*%',
            r'forecast:\s*[-+]?\d+\.?\d*%',
            r'expected\s+(?:value|change):\s*[-+]?\d+\.?\d*'
        ]
        predictions = []
        for pattern in prediction_patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            predictions.extend(match.group(0) for match in matches)
        return predictions

    def _extract_analysis_sections(self, response: str) -> List[str]:
        """Extract analysis sections from response."""
        section_patterns = [
            r'(?:Initial|Market)\s+Assessment:?.*?(?=\n\n|\Z)',
            r'Technical\s+Analysis:?.*?(?=\n\n|\Z)',
            r'Risk\s+(?:Analysis|Assessment):?.*?(?=\n\n|\Z)',
            r'(?:Cross-Chain|Correlation)\s+Analysis:?.*?(?=\n\n|\Z)'
        ]
        sections = []
        for pattern in section_patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE | re.DOTALL)
            sections.extend(match.group(0) for match in matches)
        return sections

    def _extract_calculations(self, response: str) -> List[str]:
        """Extract calculations from response."""
        calculation_patterns = [
            r'[-+]?\d+\.?\d*\s*[+\-*/]\s*[-+]?\d+\.?\d*\s*=\s*[-+]?\d+\.?\d*',
            r'calculated\s+as:?\s*[-+]?\d+\.?\d*',
            r'computed\s+value:?\s*[-+]?\d+\.?\d*'
        ]
        calculations = []
        for pattern in calculation_patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            calculations.extend(match.group(0) for match in matches)
        return calculations

    def _extract_citations(self, response: str) -> List[str]:
        """Extract data citations from response."""
        citation_pattern = r'\[cite as \'[^\']+\'\]'
        return re.findall(citation_pattern, response)

    def _extract_risk_factors(self, response: str) -> List[str]:
        """Extract risk factor analysis from response."""
        risk_patterns = [
            r'risk:?\s*[-+]?\d+\.?\d*%',
            r'probability:?\s*[-+]?\d+\.?\d*%',
            r'likelihood:?\s*[-+]?\d+\.?\d*%'
        ]
        risks = []
        for pattern in risk_patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            risks.extend(match.group(0) for match in matches)
        return risks

    def _extract_metrics_used(self, response: str) -> List[str]:
        """Extract metrics referenced in response."""
        metric_patterns = [
            r'(?:transaction|tx|volume)\s*growth',
            r'user\s*growth',
            r'volatility',
            r'gas\s*price',
            r'success\s*rate'
        ]
        metrics = []
        for pattern in metric_patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            metrics.extend(match.group(0) for match in matches)
        return list(set(metrics))

    def _extract_correlations(self, response: str) -> List[str]:
        """Extract correlation analysis from response."""
        correlation_patterns = [
            r'correlation\s*coefficient:?\s*[-+]?\d+\.?\d*',
            r'correlation:?\s*[-+]?\d+\.?\d*%',
            r'correlated\s*at\s*[-+]?\d+\.?\d*'
        ]
        correlations = []
        for pattern in correlation_patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            correlations.extend(match.group(0) for match in matches)
        return correlations

    def _has_inconsistencies(self, response: str) -> bool:
        """Check for logical inconsistencies in response."""
        contradiction_patterns = [
            (r'increase.*decrease', 'same sentence'),
            (r'positive.*negative', 'same context'),
            (r'bullish.*bearish', 'same paragraph')
        ]
        return any(re.search(f'{pattern[0]}.*{pattern[1]}', response, re.IGNORECASE)
                  for pattern in contradiction_patterns)

    def _has_missing_citations(self, response: str) -> bool:
        """Check for numerical claims without citations."""
        uncited_pattern = r'(?<!cite as)[^[]*\d+\.?\d*%'
        return bool(re.search(uncited_pattern, response))

    def _has_vague_statements(self, response: str) -> bool:
        """Check for vague or unquantified statements."""
        vague_terms = [
            r'\bmight\b', r'\bmaybe\b', r'\bpossibly\b',
            r'\bsomewhat\b', r'\bkind of\b', r'\bsort of\b'
        ]
        return any(re.search(term, response, re.IGNORECASE) for term in vague_terms)

    def _has_exact_value(self, prediction: str) -> bool:
        """Check if prediction includes exact numerical values."""
        return bool(re.search(r'[-+]?\d+\.?\d*%?', prediction))

    def _has_error_bounds(self, calculation: str) -> bool:
        """Check if calculation includes error bounds or confidence intervals."""
        return bool(re.search(r'±\s*\d+\.?\d*|error\s*(?:of|:)\s*\d+\.?\d*', calculation))

    def _validate_citation(self, citation: str, context_data: Dict[str, Any]) -> bool:
        """Validate that citation references actual context data."""
        metric = re.search(r'\'([^\']+)\'', citation)
        if metric and metric.group(1) in context_data:
            return True
        return False

    def _validate_correlation(self, correlation: str) -> bool:
        """Validate correlation coefficient is in valid range."""
        coef = re.search(r'[-+]?\d+\.?\d*', correlation)
        if coef:
            value = float(coef.group(0))
            return -1 <= value <= 1
        return False

    def _validate_prediction_against_outcome(self, prediction: str, outcome_data: Dict[str, Any]) -> bool:
        """Validate prediction against actual outcome."""
        # Extract predicted value and metric type
        value_match = re.search(r'[-+]?\d+\.?\d*', prediction)
        if not value_match:
            return False
            
        predicted_value = float(value_match.group(0))
        
        # Determine metric type from prediction text
        metric_type = None
        if 'transaction' in prediction.lower() or 'volume' in prediction.lower():
            metric_type = 'txn_growth_pct_7d'
        elif 'user' in prediction.lower():
            metric_type = 'user_growth_pct_7d'
        elif 'volatility' in prediction.lower():
            metric_type = 'tx_volatility_7d'
            
        if not metric_type or metric_type not in outcome_data:
            return False
            
        # Compare prediction with outcome
        actual_value = float(outcome_data[metric_type])
        error_margin = abs(actual_value * 0.1)  # 10% error margin
        
        return abs(predicted_value - actual_value) <= error_margin 

    def _has_probability(self, risk: str) -> bool:
        """Check if risk assessment includes probability estimate."""
        probability_patterns = [
            r'probability:\s*\d+\.?\d*%',
            r'likelihood:\s*\d+\.?\d*%',
            r'chance:\s*\d+\.?\d*%'
        ]
        return any(re.search(pattern, risk, re.IGNORECASE) for pattern in probability_patterns)

    def _has_impact_metric(self, risk: str) -> bool:
        """Check if risk assessment includes impact quantification."""
        impact_patterns = [
            r'impact:\s*(?:high|medium|low)\s*\(\d+\.?\d*%?\)',
            r'severity:\s*\d+\.?\d*(?:\s*\/\s*10)?',
            r'potential\s+loss:\s*\d+\.?\d*%',
            r'exposure:\s*\d+\.?\d*[KMB]?'
        ]
        return any(re.search(pattern, risk, re.IGNORECASE) for pattern in impact_patterns) 