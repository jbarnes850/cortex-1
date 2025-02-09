"""
OpenAI client for generating synthetic market analysis.
"""

import os
from typing import Dict, List, Optional, Tuple
import openai
from dotenv import load_dotenv
import logging
import re
import json

logger = logging.getLogger(__name__)

class OpenAIClient:
    """Client for interacting with OpenAI's API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the OpenAI client.
        
        Args:
            api_key: Optional API key. If not provided, will look for OPENAI_API_KEY in environment.
        """
        load_dotenv()
        openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
            
    def generate_completion(self, 
                          prompt: str,
                          system_prompt: str,
                          model: str = "o3-mini",
                          temperature: float = 0.7,
                          max_tokens: int = 1000) -> str:
        """Generate text completion using OpenAI's API.
        
        Args:
            prompt: The user prompt to generate from
            system_prompt: System message to guide the model's behavior
            model: Model to use for generation
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            
        Returns:
            str: Generated completion text
            
        Raises:
            Exception: If there's an error during generation
        """
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "text"}
            )
            return response.choices[0].message.content
            
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error generating completion: {str(e)}")
            raise
            
    def verify_quality(self, text: str, required_components: Optional[List[str]] = None) -> Tuple[bool, float]:
        """Verify the quality of generated text.
        
        Args:
            text: The text to verify
            required_components: List of required components to check for
            
        Returns:
            Tuple[bool, float]: (passes_quality, quality_score)
        """
        try:
            # Check for required components first
            if required_components:
                missing_components = self._check_required_components(text, required_components)
                if missing_components:
                    logger.warning(f"Missing required components: {missing_components}")
                    return False, 0.0
            
            eval_prompt = f"""Score this crypto market analysis on data quality (0-10):

Analysis:
{text}

Scoring Criteria:

1. Data Citations (0-10):
- Uses [metric_name] format consistently
- References specific numerical values
- Connects data points to conclusions

2. Calculations & Predictions (0-10):
- Shows clear mathematical steps
- Includes confidence intervals
- Explains calculation logic

3. Analysis Depth (0-10):
- Provides meaningful insights
- Considers multiple factors
- Explains significance of findings

4. Technical Accuracy (0-10):
- Calculations are correct
- Uses appropriate methods
- Avoids logical errors

Score each criterion (0-10) and explain why.
Overall score will be the average of all criteria."""

            response = openai.chat.completions.create(
                model="o3-mini",
                messages=[
                    {"role": "system", "content": "You are a strict evaluator focusing on data quality, calculations, and analysis depth."},
                    {"role": "user", "content": eval_prompt}
                ],
                temperature=0.3,
                max_tokens=500,
                response_format={"type": "text"}
            )
            
            evaluation = response.choices[0].message.content
            
            # Extract scores
            scores = []
            for line in evaluation.split('\n'):
                if ':' in line and any(c.isdigit() for c in line):
                    try:
                        score = float([s for s in line.split() if s.replace('.', '').isdigit()][0])
                        scores.append(score)
                    except (ValueError, IndexError):
                        continue
            
            if not scores:
                logger.warning("No scores found in evaluation")
                return False, 0.0
                
            avg_score = sum(scores) / len(scores)
            normalized_score = avg_score / 10.0
            
            # Additional quality checks
            quality_checks = {
                'citations': self._check_citations(text),
                'calculations': self._check_calculations(text),
                'confidence_intervals': self._check_confidence_intervals(text)
            }
            
            # Log quality check results
            logger.info(f"Quality check results: {json.dumps(quality_checks, indent=2)}")
            
            # Require all quality checks to pass
            passes = all(quality_checks.values()) and normalized_score >= 0.7
            
            return passes, normalized_score
            
        except Exception as e:
            logger.error(f"Error verifying quality: {str(e)}")
            return False, 0.0
            
    def _check_required_components(self, text: str, required_components: List[str]) -> List[str]:
        """Check for required components in the text."""
        missing = []
        
        component_patterns = {
            'data_citations': r'\[[\w_]+\]',
            'calculations': r'=\s*[\d\.\+\-\*\/\(\)\s]+',
            'confidence_intervals': r'(?i)confidence interval|CI|±',
            'market_analysis': r'(?i)market|trend|pattern|growth'
        }
        
        for component in required_components:
            if component in component_patterns:
                pattern = component_patterns[component]
                if not re.search(pattern, text):
                    missing.append(component)
                    
        return missing
        
    def _check_citations(self, text: str) -> bool:
        """Check for proper data citations."""
        citation_pattern = r'\[[\w_]+\]'
        citations = re.findall(citation_pattern, text)
        return len(citations) >= 5  # Require at least 5 citations
        
    def _check_calculations(self, text: str) -> bool:
        """Check for mathematical calculations."""
        calculation_pattern = r'=\s*[\d\.\+\-\*\/\(\)\s]+'
        calculations = re.findall(calculation_pattern, text)
        return len(calculations) >= 3  # Require at least 3 calculations
        
    def _check_confidence_intervals(self, text: str) -> bool:
        """Check for confidence intervals."""
        ci_pattern = r'(?i)(?:confidence interval|CI|±)'
        return bool(re.search(ci_pattern, text)) 