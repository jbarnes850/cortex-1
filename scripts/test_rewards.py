#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for evaluating reward functions on mock financial analysis responses.
This script generates mock LLM responses with varying quality and evaluates them
using the reward functions from the rewards module.
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Any, Tuple
import random

# Add the src directory to the path so we can import modules from there
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rewards import get_default_financial_reward
from src.rewards.finance_rewards import CalculationAccuracyReward, ConfidenceIntervalReward, InvestmentInsightReward
from src.rewards.format_rewards import CitationFormatReward, StructureReward, CompletenessReward
from src.rewards.citation_rewards import MetricCitationReward, HistoricalReferenceReward
from src.rewards.composite_reward import CompositeReward, create_financial_reward


def generate_mock_response(quality: str = "high") -> str:
    """
    Generate a mock financial analysis response of the specified quality.
    
    Args:
        quality: The quality of the response to generate ("high", "medium", or "low")
        
    Returns:
        A mock financial analysis response as a string
    """
    if quality == "high":
        return """
# Financial Analysis: NEAR Protocol Market Conditions

## Executive Summary
Based on the data analyzed, NEAR Protocol shows strong growth potential with a 15.3% increase in TVL over the past 30 days. Transaction volume has increased by 27.8% (from 1.45M to 1.85M daily transactions), indicating growing adoption. The price stability coefficient is 0.78, suggesting moderate volatility compared to the broader market.

## Key Metrics Analysis
- **Total Value Locked (TVL)**: $342.5M, up 15.3% MoM
- **Daily Active Addresses**: 125,750, representing a 12.4% increase from the previous month
- **Average Transaction Fee**: 0.00025 NEAR ($0.0012), down 5% from previous month
- **Protocol Revenue**: $1.85M in the past 30 days, a 18.2% increase
- **Staking Ratio**: 42.5% of circulating supply, up 2.1% from previous month

## Detailed Calculations
The 30-day price stability coefficient is calculated as:
1. Standard deviation of daily returns = 0.032
2. Market average standard deviation = 0.041
3. Stability coefficient = 1 - (0.032/0.041) = 0.78

The projected TVL growth rate can be estimated using the compound monthly growth rate:
- Current TVL growth rate: 15.3% monthly
- Projected 90-day TVL (with 95% confidence interval): $524.8M ± $42.5M

## Market Context
Historical data shows that NEAR has maintained a correlation coefficient of 0.68 with ETH and 0.52 with BTC over the past 90 days. During previous market cycles in Q2 2023, similar growth patterns resulted in sustained price appreciation over a 4-5 month period.

## Investment Implications
Based on the analysis, I recommend increasing allocation to NEAR with a 90-day time horizon. The strong fundamentals, including growing TVL and active addresses, suggest continued adoption. The relatively stable price action provides a favorable risk-reward ratio of approximately 3.2:1 based on technical support and resistance levels.

Risk assessment: Medium. While metrics are positive, broader market conditions and potential regulatory changes could impact performance. It's advisable to implement a 15% stop-loss to manage downside risk.
"""

    elif quality == "medium":
        return """
# NEAR Protocol Analysis

## Summary
NEAR Protocol shows growth with TVL increasing and more transactions happening. The price has been relatively stable compared to other cryptocurrencies.

## Some Metrics
- TVL: $342.5M (up 15.3%)
- Daily Active Addresses: 125,750
- Transaction Fee: 0.00025 NEAR
- Revenue: $1.85M last month

## Calculations
The stability coefficient is 0.78.
Projected TVL growth: around $525M in 90 days.

## Market Information
NEAR correlates with ETH and BTC, with coefficients of 0.68 and 0.52 respectively. The market has been in a similar pattern before.

## Investment Thoughts
NEAR looks like a good investment for the next 90 days. The metrics are strong and suggest the price might go up. There are some risks to consider though.
"""

    else:  # low quality
        return """
NEAR Protocol looks good. The price has been going up lately and might continue to rise. Some people are using it more based on transaction numbers.

The total value locked is $342.5M and there are about 125,000 active addresses. Fees are low.

I think NEAR is a good buy right now because the trend is positive. But be careful because crypto is risky.
"""


def generate_test_cases() -> List[Dict[str, Any]]:
    """
    Generate a list of test cases with different quality responses and expected scores.
    
    Returns:
        A list of dictionaries with test cases
    """
    return [
        {
            "name": "High Quality Response",
            "response": generate_mock_response("high"),
            "expected_scores": {
                "calculation_accuracy": (0.8, 1.0),
                "confidence_interval": (0.8, 1.0), 
                "investment_insight": (0.8, 1.0),
                "citation_format": (0.7, 1.0),
                "structure": (0.8, 1.0),
                "completeness": (0.8, 1.0),
                "metric_citation": (0.8, 1.0),
                "historical_reference": (0.7, 1.0),
                "composite": (0.75, 1.0)
            }
        },
        {
            "name": "Medium Quality Response",
            "response": generate_mock_response("medium"),
            "expected_scores": {
                "calculation_accuracy": (0.4, 0.7),
                "confidence_interval": (0.3, 0.6), 
                "investment_insight": (0.4, 0.7),
                "citation_format": (0.3, 0.6),
                "structure": (0.5, 0.8),
                "completeness": (0.4, 0.7),
                "metric_citation": (0.4, 0.7),
                "historical_reference": (0.3, 0.6),
                "composite": (0.4, 0.7)
            }
        },
        {
            "name": "Low Quality Response",
            "response": generate_mock_response("low"),
            "expected_scores": {
                "calculation_accuracy": (0.0, 0.3),
                "confidence_interval": (0.0, 0.2), 
                "investment_insight": (0.1, 0.4),
                "citation_format": (0.0, 0.3),
                "structure": (0.1, 0.4),
                "completeness": (0.1, 0.4),
                "metric_citation": (0.0, 0.3),
                "historical_reference": (0.0, 0.2),
                "composite": (0.05, 0.35)
            }
        }
    ]


def evaluate_response(response: str) -> Dict[str, float]:
    """
    Evaluate a mock response using all reward functions.
    
    Args:
        response: The mock response to evaluate
        
    Returns:
        A dictionary with the scores from each reward function
    """
    # Create individual reward functions
    calculation_reward = CalculationAccuracyReward(weight=1.0)
    confidence_reward = ConfidenceIntervalReward(weight=1.0)
    investment_reward = InvestmentInsightReward(weight=1.0)
    citation_format_reward = CitationFormatReward(weight=1.0)
    structure_reward = StructureReward(weight=1.0)
    completeness_reward = CompletenessReward(weight=1.0)
    metric_citation_reward = MetricCitationReward(weight=1.0)
    historical_reward = HistoricalReferenceReward(weight=1.0)
    
    # Get the default composite reward
    composite_reward = get_default_financial_reward()
    
    # Mock input data (normally would be extracted from a real context)
    mock_input = {
        "query": "Analyze the current market conditions for NEAR Protocol and provide investment recommendations.",
        "metrics": {
            "tvl": 342500000,
            "daily_active_addresses": 125750,
            "transaction_fee": 0.00025,
            "protocol_revenue": 1850000,
            "staking_ratio": 0.425
        }
    }
    
    # Calculate scores for each reward function
    scores = {
        "calculation_accuracy": calculation_reward(response, mock_input),
        "confidence_interval": confidence_reward(response, mock_input),
        "investment_insight": investment_reward(response, mock_input),
        "citation_format": citation_format_reward(response, mock_input),
        "structure": structure_reward(response, mock_input),
        "completeness": completeness_reward(response, mock_input),
        "metric_citation": metric_citation_reward(response, mock_input),
        "historical_reference": historical_reward(response, mock_input),
        "composite": composite_reward(response, mock_input)
    }
    
    return scores


def check_score_in_range(score: float, expected_range: Tuple[float, float]) -> bool:
    """
    Check if a score is within the expected range.
    
    Args:
        score: The calculated score
        expected_range: A tuple with the minimum and maximum expected score
        
    Returns:
        True if the score is within the expected range, False otherwise
    """
    min_score, max_score = expected_range
    return min_score <= score <= max_score


def main():
    parser = argparse.ArgumentParser(description="Test reward functions with mock financial analysis responses")
    parser.add_argument("--output", type=str, default="reward_test_results.json", help="Path to save test results")
    parser.add_argument("--verbose", action="store_true", help="Print detailed test results")
    args = parser.parse_args()
    
    # Generate test cases
    test_cases = generate_test_cases()
    
    # Run tests and collect results
    test_results = []
    
    print(f"\n{'=' * 60}")
    print(f"REWARD FUNCTION TEST RESULTS")
    print(f"{'=' * 60}")
    
    for test_case in test_cases:
        print(f"\n{'-' * 60}")
        print(f"Test: {test_case['name']}")
        print(f"{'-' * 60}")
        
        # Evaluate the response
        scores = evaluate_response(test_case["response"])
        
        # Check if scores are within expected ranges
        passed = True
        result = {
            "name": test_case["name"],
            "scores": scores,
            "expected_ranges": test_case["expected_scores"],
            "component_results": {}
        }
        
        for component, score in scores.items():
            expected_range = test_case["expected_scores"].get(component)
            if expected_range:
                in_range = check_score_in_range(score, expected_range)
                result["component_results"][component] = {
                    "score": score,
                    "expected_range": expected_range,
                    "passed": in_range
                }
                if not in_range:
                    passed = False
            
            if args.verbose:
                range_str = f"({expected_range[0]:.2f}-{expected_range[1]:.2f})"
                status = "✅" if in_range else "❌"
                print(f"{component:25} Score: {score:.4f} {range_str:15} {status}")
        
        result["passed"] = passed
        test_results.append(result)
        
        # Print summary for this test case
        if passed:
            print(f"\nResult: PASSED ✅")
        else:
            print(f"\nResult: FAILED ❌")
        
        # Print composite score
        print(f"Composite Score: {scores['composite']:.4f}")
    
    # Save results to file
    with open(args.output, "w") as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"Summary: {sum(1 for r in test_results if r['passed'])}/{len(test_results)} tests passed")
    print(f"Detailed results saved to: {args.output}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main() 