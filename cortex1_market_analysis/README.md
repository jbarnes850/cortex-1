---
language:
- en
license: mit
task_categories:
- text-generation
- financial-analysis
pretty_name: NEAR Cortex-1 Market Analysis Dataset
size_categories:
- n<1K
tags:
- defi
- blockchain
- market-analysis
- chain-of-thought
- near-protocol
- ethereum
---

# NEAR Cortex-1 Market Analysis Dataset (Preview)

## Dataset Description

- **Homepage:** [GitHub Repository](https://github.com/near/cortex-1)
- **Repository:** https://huggingface.co/datasets/Jarrodbarnes/cortex-1-market-analysis
- **Paper:** N/A
- **Point of Contact:** NEAR Foundation

### Dataset Summary

This is a preliminary dataset of blockchain market analyses combining real-world data with chain-of-thought reasoning. The dataset contains examples from Ethereum and NEAR chains, demonstrating high-quality market analysis with explicit calculations, numerical citations, and actionable insights.

### Supported Tasks and Leaderboards

- **Tasks:**
  - `text-generation`: The dataset can be used to train models for generating detailed market analysis from blockchain metrics
  - `financial-analysis`: Models can learn to perform quantitative analysis of blockchain market data

### Languages

The dataset is in English (en).

### Dataset Structure

#### Data Instances

Each example contains:
```python
{
    'timestamp': '2025-02-09T11:38:41',
    'chain': 'ethereum',  # or 'near'
    'date': '2025-02-09',
    'market_data': {
        'daily_txns': 1234567,
        'unique_users': 98765,
        'success_rate': 0.998,
        'avg_tx_value': 0.05,
        'total_volume': 12345.67
    },
    'reasoning': 'Detailed market analysis with calculations...'
}
```

#### Data Fields

- `timestamp`: Timestamp of data generation
- `chain`: Blockchain network (ethereum or near)
- `date`: Date of market data
- `market_data`: Dictionary containing market metrics
  - `daily_txns`: Number of daily transactions
  - `unique_users`: Number of unique users
  - `success_rate`: Transaction success rate
  - `avg_tx_value`: Average transaction value
  - `total_volume`: Total transaction volume
- `reasoning`: Detailed market analysis with calculations and insights

#### Data Splits

The preview dataset contains 4 examples in the training split:
- `train`: 4 examples (2 Ethereum, 2 NEAR)

### Dataset Creation

#### Curation Rationale

The dataset is designed to demonstrate high-quality market analysis combining:
1. Real-world blockchain metrics from Flipside Crypto
2. Chain-of-thought reasoning with explicit calculations
3. Numerical citations and data-driven insights
4. Cross-chain analysis and comparisons

#### Source Data

- **Initial Data Collection and Normalization:**
  - Blockchain data collected via Flipside Crypto API
  - Metrics normalized across Ethereum and NEAR chains
  - Time range: February 2025

#### Annotations

The reasoning annotations follow strict quality criteria:
- Calculation Steps: Must show explicit mathematical operations
- Numerical Citations: Must cite specific metrics in [metric_name] format
- Insight Count: Must provide multiple actionable insights
- Section Completeness: Must cover all required analysis sections

**Quality Thresholds:**
- All examples achieve perfect scores (1.00) across quality metrics
- Failed generations are automatically filtered out
- Each example undergoes automated quality verification

### Considerations for Using the Data

#### Social Impact of Dataset

The dataset aims to:
1. Democratize access to high-quality blockchain market analysis
2. Promote data-driven decision making in DeFi
3. Enable development of AI models for financial analysis
4. Support cross-chain market understanding

#### Discussion of Biases

Potential biases to consider:
1. Time period limitations (February 2025 only)
2. Limited chain coverage (Ethereum and NEAR only)
3. Focus on successful transactions and active users
4. Bias towards quantitative over qualitative analysis

#### Other Known Limitations

1. Preview dataset size (only 4 examples)
2. Limited market conditions covered
3. Focus on daily aggregated metrics
4. Chain-specific metric differences

### Additional Information

#### Dataset Curators

This dataset is curated by the NEAR Foundation as part of the Cortex-1 project.

#### Licensing Information

The dataset is released under the MIT License.

#### Citation Information

```bibtex
@misc{near2025cortex,
    title={NEAR Cortex-1 Market Analysis Dataset},
    author={NEAR Foundation},
    year={2025},
    publisher={HuggingFace},
    url={https://huggingface.co/datasets/near/cortex-1-market-analysis}
}
```

#### Contributions

Thanks to:

- NEAR Foundation for dataset curation
- Flipside Crypto for market data access
