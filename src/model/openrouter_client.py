"""
OpenRouter client for generating synthetic market analysis.
"""

import os
from typing import Dict, List, Optional, Tuple
from openai import OpenAI
from dotenv import load_dotenv
import logging
import re
import json

logger = logging.getLogger(__name__)

class OpenRouterClient:
    """Client for interacting with OpenRouter's API using the OpenAI-compatible interface."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """Initialize the OpenRouter client.
        
        Args:
            api_key: Optional API key. If not provided, will look for OPENROUTER_API_KEY in environment.
            base_url: Optional base URL. If not provided, will look for OPENROUTER_BASE_URL in environment.
        """
        load_dotenv()
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key not found. Please set OPENROUTER_API_KEY environment variable.")
            
        self.base_url = base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.model = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-r1:free")
        
        # Initialize OpenAI client with OpenRouter configurations
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        
        logger.info(f"Initialized OpenRouter client with model: {self.model}")
            
    def _is_deepseek_r1_model(self, model: str) -> bool:
        """Check if the model is a DeepSeek R1 variant."""
        return model and "deepseek" in model.lower() and "r1" in model.lower()
            
    def generate_completion(self, 
                          prompt: str,
                          system_prompt: str,
                          model: Optional[str] = None,
                          temperature: float = 0.7,
                          max_tokens: int = 1000) -> str:
        """Generate text completion using OpenRouter's API.
        
        Args:
            prompt: The user prompt to generate from
            system_prompt: System message to guide the model's behavior
            model: Model to use for generation (defaults to OPENROUTER_MODEL env var)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            
        Returns:
            str: Generated completion text
            
        Raises:
            Exception: If there's an error during generation
        """
        try:
            model_to_use = model or self.model
            
            response = self.client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            # Check if we need to extract reasoning for DeepSeek R1 models
            if self._is_deepseek_r1_model(model_to_use):
                # Extract the reasoning field if it exists
                response_dict = response.model_dump()
                reasoning = None
                
                if response_dict and 'choices' in response_dict and len(response_dict['choices']) > 0:
                    choice = response_dict['choices'][0]
                    if 'message' in choice and 'reasoning' in choice['message']:
                        reasoning = choice['message']['reasoning']
                
                # If reasoning is available, use it; otherwise fall back to content
                if reasoning:
                    logger.info(f"Using reasoning field from DeepSeek R1 model (length: {len(reasoning)})")
                    return reasoning
            
            # For all other models or if no reasoning field, use the standard content
            content = response.choices[0].message.content
            
            # Log warning if content is empty
            if not content:
                logger.warning("Received empty response content")
                
            return content or ""
            
        except Exception as e:
            logger.error(f"OpenRouter API error: {str(e)}")
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
- Shows step-by-step reasoning
- Provides clear numerical predictions
- Explains methodology

3. Confidence Intervals (0-10):
- Provides probability estimates
- Quantifies uncertainty
- Gives ranges instead of point estimates

4. Technical Quality (0-10):
- Free of factual errors
- Logical consistency
- Properly formatted and readable

Provide a total score (0-10) and brief justification."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a crypto market analysis expert who evaluates the quality of market analysis reports."},
                    {"role": "user", "content": eval_prompt}
                ],
                temperature=0.3,
                max_tokens=300,
            )
            
            # Get content or reasoning based on model
            evaluation = response.choices[0].message.content
            
            # Check for DeepSeek R1 reasoning
            if not evaluation and self._is_deepseek_r1_model(self.model):
                # Extract reasoning if available
                response_dict = response.model_dump()
                if response_dict and 'choices' in response_dict and len(response_dict['choices']) > 0:
                    choice = response_dict['choices'][0]
                    if 'message' in choice and 'reasoning' in choice['message']:
                        evaluation = choice['message']['reasoning']
                        logger.info("Using reasoning field for quality evaluation")
            
            # If still no evaluation content, return default values
            if not evaluation:
                logger.warning("Received empty evaluation response")
                return False, 0.0
                
            # Extract score using regex
            score_match = re.search(r'(\d+(\.\d+)?)/10', evaluation)
            score_match2 = re.search(r'score.*?(\d+(\.\d+)?)', evaluation.lower())
            if score_match:
                score = float(score_match.group(1))
            elif score_match2:
                score = float(score_match2.group(1))
            else:
                # Default to moderate score if we can't extract
                score = 5.0
                
            # Normalize to 0-1 range
            quality_score = score / 10.0
            passes_quality = quality_score >= 0.7
            
            return passes_quality, quality_score
            
        except Exception as e:
            logger.error(f"Error verifying quality: {str(e)}")
            # Default to fail if there's an error
            return False, 0.0
    
    def _check_required_components(self, text: str, required_components: List[str]) -> List[str]:
        """Check if the text contains all required components.
        
        Args:
            text: The text to check
            required_components: List of required component patterns
            
        Returns:
            List[str]: List of missing components
        """
        missing = []
        for component in required_components:
            if not re.search(component, text, re.IGNORECASE):
                missing.append(component)
        return missing
        
    def _check_citations(self, text: str) -> bool:
        """Check if the text contains proper data citations."""
        citation_pattern = r'\[([\w\s_]+)\]'
        return len(re.findall(citation_pattern, text)) >= 3
        
    def _check_calculations(self, text: str) -> bool:
        """Check if the text contains calculations or numerical reasoning."""
        calculation_pattern = r'(calculated|computed|estimated|increased by|decreased by|\d+(\.\d+)?%)'
        return bool(re.search(calculation_pattern, text, re.IGNORECASE))
        
    def _check_confidence_intervals(self, text: str) -> bool:
        """Check if the text contains confidence intervals or probability estimates."""
        confidence_pattern = r'(confidence|probability|likely|unlikely|certainly|possibly|range of|between \d+ and \d+)'
        return bool(re.search(confidence_pattern, text, re.IGNORECASE)) 