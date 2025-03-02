#!/usr/bin/env python
"""
Test script for verifying OpenRouter integration works with reasoning tokens.
"""

import os
import logging
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import json

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Test OpenRouter integration with DeepSeek R1 model."""
    # Load environment variables from project root
    env_path = Path(project_root) / '.env'
    logger.info(f"Loading environment from: {env_path}")
    load_dotenv(env_path, override=True)
    
    # Get OpenRouter configuration from environment
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        logger.error("OPENROUTER_API_KEY not found in environment")
        sys.exit(1)
    
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-r1:free")
    
    logger.info(f"Testing OpenRouter with model: {model}")
    logger.info(f"Base URL: {base_url}")
    
    # Initialize OpenAI client with OpenRouter configuration
    client = OpenAI(
        base_url=base_url,
        api_key=openrouter_api_key,
        timeout=120.0,  # Increase timeout to 2 minutes
    )
    
    # Test with prompt that requests reasoning
    try:
        logger.info("Sending test request to OpenRouter with reasoning prompt...")
        start_time = time.time()
        
        # Prompt specifically requesting reasoning steps
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that shows step-by-step reasoning. Always think through problems carefully."},
                {"role": "user", "content": "Calculate 17 × 24 step by step, showing your work."}
            ],
            temperature=0.2,
            max_tokens=500,
            top_p=1,
            presence_penalty=0,
            frequency_penalty=0,
        )
        
        # Calculate response time
        response_time = time.time() - start_time
        logger.info(f"Response received in {response_time:.2f} seconds!")
        
        # Get standard content
        content = response.choices[0].message.content
        
        # Extract the reasoning field using the response dictionary
        response_dict = response.model_dump()
        reasoning = None
        
        if response_dict and 'choices' in response_dict and len(response_dict['choices']) > 0:
            choice = response_dict['choices'][0]
            if 'message' in choice and 'reasoning' in choice['message']:
                reasoning = choice['message']['reasoning']
        
        # Print raw response content
        if content:
            logger.info(f"Content field: {repr(content)}")
        else:
            logger.warning("Content field is empty")
            
        # Print reasoning if available
        if reasoning:
            logger.info(f"Reasoning field found with length: {len(reasoning)} characters")
        else:
            logger.warning("No reasoning field found in response")
        
        # Print the content and reasoning
        print("\n" + "="*50)
        print("STANDARD CONTENT:")
        print("="*50)
        print(content or "[Empty content field]")
        print("="*50)
        
        print("\n" + "="*50)
        print("REASONING OUTPUT:")
        print("="*50)
        print(reasoning or "[No reasoning found]")
        print("="*50 + "\n")
        
        # Print full details about the model and response
        logger.info(f"Model used: {response.model}")
        logger.info(f"Total tokens: {response.usage.total_tokens}")
        logger.info(f"Prompt tokens: {response.usage.prompt_tokens}")
        logger.info(f"Completion tokens: {response.usage.completion_tokens}")
        
        logger.info("OpenRouter integration test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error testing OpenRouter: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 