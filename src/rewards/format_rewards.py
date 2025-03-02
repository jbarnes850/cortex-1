"""
Format-specific reward functions for evaluating LLM responses in financial analysis.
These functions implement specialized rewards that focus on proper formatting,
structure, and completeness of market analysis responses.
"""

import re
from typing import Dict, List, Optional, Union, Any
from src.rewards.finance_rewards import BaseReward


class CitationFormatReward(BaseReward):
    """Reward function that evaluates proper citation of metrics in [metric_name] format."""
    
    def __init__(self, weight: float = 1.0):
        """Initialize the citation format reward.
        
        Args:
            weight: The weight of this reward
        """
        super().__init__(weight=weight)
    
    def calculate(self, response: str, context: Optional[Dict[str, Any]] = None) -> float:
        """Calculate the reward based on proper metric citations.
        
        Args:
            response: The model's generated text
            context: Optional context with expected metrics
            
        Returns:
            A score between 0.0 and 1.0
        """
        # Extract citations from the response
        citations = self._extract_citations(response)
        
        if not citations:
            return 0.0
        
        # Score based on the number of properly formatted citations
        proper_count = 0
        total_metrics_mentioned = 0
        
        # Extract all metric mentions (both properly and improperly cited)
        metric_mentions = self._extract_metric_mentions(response)
        total_metrics_mentioned = len(metric_mentions)
        
        # Count properly formatted citations
        proper_count = len(citations)
        
        if total_metrics_mentioned == 0:
            return 0.0
        
        # Calculate the proportion of properly cited metrics
        citation_ratio = proper_count / total_metrics_mentioned
        
        # Apply a penalty if there are very few citations overall
        if proper_count < 3:
            citation_ratio *= 0.7
        
        return min(1.0, citation_ratio)
    
    def _extract_citations(self, response: str) -> List[str]:
        """Extract properly formatted citations from the response.
        
        Args:
            response: The model's generated text
            
        Returns:
            A list of citation strings
        """
        # Pattern for [metric_name] citations
        citation_pattern = r'\[([a-zA-Z_][a-zA-Z0-9_]*)\]'
        return re.findall(citation_pattern, response)
    
    def _extract_metric_mentions(self, response: str) -> List[str]:
        """Extract all metric mentions from the response.
        
        Args:
            response: The model's generated text
            
        Returns:
            A list of metric mention strings
        """
        # Common market metrics that should be cited
        metric_patterns = [
            r'(?:daily|transaction|tx|total)\s+(?:volume|transactions)',
            r'user\s+(?:growth|count|activity)',
            r'(?:transaction|tx)\s+(?:growth|increase|decrease)',
            r'volatility',
            r'gas\s+(?:price|used|cost)',
            r'success\s+rate',
            r'avg\s+(?:transaction|tx)\s+value',
            r'unique\s+users',
            r'market\s+(?:depth|liquidity)',
            r'correlation\s+(?:coefficient|value)',
        ]
        
        mentions = []
        for pattern in metric_patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            mentions.extend(match.group(0) for match in matches)
        
        return list(set(mentions))  # Remove duplicates


class StructureReward(BaseReward):
    """Reward function that evaluates the structure of the financial analysis."""
    
    def __init__(self, weight: float = 1.0):
        """Initialize the structure reward.
        
        Args:
            weight: The weight of this reward
        """
        super().__init__(weight=weight)
        
        # Required sections for a complete analysis
        self.required_sections = [
            "Initial Observations",
            "Analysis",
            "Technical Assessment",
            "Risk Evaluation",
            "Opportunities",
            "Conclusion"
        ]
    
    def calculate(self, response: str, context: Optional[Dict[str, Any]] = None) -> float:
        """Calculate the reward based on the response structure.
        
        Args:
            response: The model's generated text
            context: Optional context (not used)
            
        Returns:
            A score between 0.0 and 1.0
        """
        # Extract section headings from the response
        sections = self._extract_sections(response)
        
        if not sections:
            return 0.0
        
        # Check how many required sections are present
        section_scores = []
        for required in self.required_sections:
            # Check if a similar section is present
            if any(self._is_similar_section(section, required) for section in sections):
                section_scores.append(1.0)
            else:
                section_scores.append(0.0)
        
        # Calculate the section coverage
        section_coverage = sum(section_scores) / len(self.required_sections)
        
        # Check if sections are in a logical order
        order_score = self._calculate_order_score(sections)
        
        # Combine scores with more weight on section coverage
        return 0.7 * section_coverage + 0.3 * order_score
    
    def _extract_sections(self, response: str) -> List[str]:
        """Extract section headings from the response.
        
        Args:
            response: The model's generated text
            
        Returns:
            A list of section heading strings
        """
        # Pattern for section headings
        section_patterns = [
            r'^([A-Z][A-Za-z0-9\s]+):',  # Capitalized section with colon
            r'^([A-Z][A-Za-z0-9\s]+)\n',  # Capitalized section with newline
            r'\n([A-Z][A-Za-z0-9\s]+):'   # Newline + capitalized section with colon
        ]
        
        sections = []
        for pattern in section_patterns:
            matches = re.finditer(pattern, response, re.MULTILINE)
            sections.extend(match.group(1).strip() for match in matches)
        
        return list(set(sections))  # Remove duplicates
    
    def _is_similar_section(self, section: str, required: str) -> bool:
        """Check if a section is similar to a required section.
        
        Args:
            section: A detected section heading
            required: A required section heading
            
        Returns:
            True if the sections are similar
        """
        # Convert to lowercase for comparison
        section_lower = section.lower()
        required_lower = required.lower()
        
        # Check for exact match or substring
        if required_lower in section_lower:
            return True
        
        # Check for word overlap
        section_words = set(section_lower.split())
        required_words = set(required_lower.split())
        
        # If more than 50% of words overlap, consider it similar
        overlap = section_words.intersection(required_words)
        if len(overlap) >= len(required_words) / 2:
            return True
        
        return False
    
    def _calculate_order_score(self, sections: List[str]) -> float:
        """Calculate a score based on the logical ordering of sections.
        
        Args:
            sections: A list of detected section headings
            
        Returns:
            A score between 0.0 and 1.0
        """
        # Map detected sections to required sections
        mapped_sections = []
        for section in sections:
            for i, required in enumerate(self.required_sections):
                if self._is_similar_section(section, required):
                    mapped_sections.append((section, i))
                    break
        
        if len(mapped_sections) <= 1:
            return 0.0
        
        # Check if the sections are in ascending order of their required index
        is_ordered = True
        for i in range(1, len(mapped_sections)):
            if mapped_sections[i][1] < mapped_sections[i-1][1]:
                is_ordered = False
                break
        
        return 1.0 if is_ordered else 0.5  # Partial credit for having sections, even if out of order


class CompletenessReward(BaseReward):
    """Reward function that evaluates the completeness of the financial analysis."""
    
    def __init__(self, weight: float = 1.0):
        """Initialize the completeness reward.
        
        Args:
            weight: The weight of this reward
        """
        super().__init__(weight=weight)
        
        # Components that should be present in a complete analysis
        self.components = {
            "transaction_analysis": [
                "daily transactions", "transaction volume", "transaction growth"
            ],
            "user_analysis": [
                "unique users", "user growth", "user engagement"
            ],
            "technical_indicators": [
                "volatility", "correlation", "growth rate", "trend"
            ],
            "risk_assessment": [
                "risk", "probability", "likelihood", "uncertainty", "exposure"
            ],
            "investment_implications": [
                "opportunity", "strategy", "recommendation", "action", "position"
            ]
        }
    
    def calculate(self, response: str, context: Optional[Dict[str, Any]] = None) -> float:
        """Calculate the reward based on the completeness of the analysis.
        
        Args:
            response: The model's generated text
            context: Optional context (not used)
            
        Returns:
            A score between 0.0 and 1.0
        """
        # Score each component category
        component_scores = {}
        for category, terms in self.components.items():
            component_scores[category] = self._score_component(response, terms)
        
        # Calculate the average component score
        avg_component_score = sum(component_scores.values()) / len(component_scores)
        
        # Check for minimum length
        length_score = min(1.0, len(response) / 1000)  # Reward responses up to 1000 characters
        
        # Combine scores
        return 0.8 * avg_component_score + 0.2 * length_score
    
    def _score_component(self, response: str, terms: List[str]) -> float:
        """Score a component category based on term presence.
        
        Args:
            response: The model's generated text
            terms: Terms that should be present for this component
            
        Returns:
            A score between 0.0 and 1.0
        """
        # Count how many terms are present
        term_count = 0
        for term in terms:
            if re.search(r'\b' + re.escape(term) + r'\b', response, re.IGNORECASE):
                term_count += 1
        
        # Score based on the proportion of terms present
        return min(1.0, term_count / len(terms)) 