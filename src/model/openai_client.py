"""
OpenAI client for generating synthetic market analysis.
"""

import os
from typing import Dict, List, Optional
import openai
from dotenv import load_dotenv

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
                          max_completion_tokens: int = 20000) -> str:
        """Generate text completion using OpenAI's API.
        
        Args:
            prompt: Input prompt
            system_prompt: System prompt for setting context
            model: Model to use for generation
            max_completion_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        try:
            # Wrap prompt with reasoning tags for structured output
            wrapped_prompt = f"""<reasoning>
Given the following market data, let's analyze it systematically:

{prompt}

Please provide a thorough analysis following this structure:

1. Initial Observations:
<observe>
What are the key metrics and patterns in the data?
Consider transaction volumes, user growth, and market dynamics.
</observe>

2. Analysis:
<analyze>
How do these metrics relate to each other?
What trends or anomalies are present?
What might be causing these patterns?
</analyze>

3. Technical Assessment:
<technical>
Evaluate the technical indicators and market structure.
Consider support/resistance levels, volume profiles, and network metrics.
</technical>

4. Risk Evaluation:
<risks>
What are the potential risks and challenges?
Consider market, technical, and systemic risks.
</risks>

5. Opportunities:
<opportunities>
What opportunities arise from this analysis?
What conditions would need to be met?
</opportunities>

6. Conclusion:
<conclude>
Synthesize the analysis into actionable insights.
Provide specific recommendations based on the evidence.
</conclude>

Support each point with data and explain your reasoning process.</reasoning>"""

            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": f"{system_prompt}\nProvide thorough analysis with clear reasoning for each point."},
                    {"role": "user", "content": wrapped_prompt}
                ],
                max_completion_tokens=max_completion_tokens,
                response_format={"type": "text"}
            )
            return response.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"Error generating completion: {str(e)}")
            
    def verify_quality(self, text: str) -> tuple[bool, float]:
        """Verify the quality of generated text using o3-mini.
        
        Args:
            text: Generated analysis text
            
        Returns:
            Tuple of (passes_check, quality_score)
        """
        try:
            # Create a prompt to evaluate the analysis with reasoning tags
            eval_prompt = f"""<reasoning>
Evaluate the following crypto market analysis for quality and accuracy:

Analysis to evaluate:
{text}

Please evaluate systematically:

1. Reasoning Quality:
<evaluate>
- Is the reasoning process clear and logical?
- Are conclusions supported by evidence?
- Is the chain of thought well-structured?
Score (0-10): 
</evaluate>

2. Data Usage:
<evaluate>
- How effectively are data points incorporated?
- Are metrics interpreted correctly?
- Is quantitative evidence provided?
Score (0-10):
</evaluate>

3. Technical Depth:
<evaluate>
- Is the technical analysis thorough?
- Are market mechanics well-understood?
- Are complex concepts explained clearly?
Score (0-10):
</evaluate>

4. Risk Assessment:
<evaluate>
- Are risks properly identified?
- Is the risk analysis balanced?
- Are mitigation strategies proposed?
Score (0-10):
</evaluate>

5. Actionability:
<evaluate>
- Are recommendations specific and practical?
- Is timing and context considered?
- Are conditions for action clear?
Score (0-10):
</evaluate>

Overall Assessment:
<conclusion>
Synthesize the evaluation scores and provide a final assessment.
Include specific strengths and areas for improvement.
Final Score (0-10):
</conclusion>
</reasoning>"""
            
            response = openai.chat.completions.create(
                model="o3-mini",
                messages=[
                    {"role": "system", "content": "You are an expert evaluator of market analysis. Be thorough and critical in your assessment."},
                    {"role": "user", "content": eval_prompt}
                ],
                max_completion_tokens=500,
                response_format={"type": "text"}
            )
            
            evaluation = response.choices[0].message.content
            
            # Extract scores (simple heuristic)
            scores = [float(line.split(":")[-1].strip().split("/")[0])
                     for line in evaluation.split("\n")
                     if ":" in line and "/10" in line]
            
            if scores:
                avg_score = sum(scores) / len(scores)
                passes_check = avg_score >= 7.0  # Quality threshold
                return passes_check, avg_score / 10.0
            
            return False, 0.0
            
        except Exception as e:
            print(f"Error verifying quality: {str(e)}")
            return False, 0.0 