# NEAR Cortex-1 Training Plan

## Overview

NEAR Cortex-1 is a specialized AI model that can reason, understand, and predict crypto market movements by learning from cross-chain data, then automatically routes capital to the most promising opportunities. Think of it as building an AI that can reason about market dynamics the way experienced traders do, but at a massive scale and with perfect recall of historical patterns.

## 1. Hardware Strategy

### Local Development (Mac Studio M2 Ultra, 192 GB RAM)

- Use for data preprocessing and debugging
- Prototype on reduced capacity models (e.g., 7–13B) to design the pipeline
- Orchestrate jobs and run CPU-bound tasks (tokenization, lightweight inference)

### Cloud GPU Instances

- Rent high-end GPUs (A100/H100 or similar) for full fine-tuning a 70B+ model
- Consider using multi-GPU setups (e.g., 4–8 × A100 80GB) for distributed training using frameworks like DeepSpeed ZeRO-3 or FSDP
- Optionally leverage QLoRA (4-bit quantization) to reduce VRAM needs, should full fine-tuning on many GPUs prove challenging

### Hybrid Offloading & Cost Management

- Use the Mac AS an orchestration controller alongside cloud GPUs
- Experiment locally with smaller batches and use gradient accumulation
- Leverage spot instances/short-term rentals along with frequent checkpointing

## 2. Model Selection

### Base Model Recommendation

- Use **Llama 3.3 70B Instruct** for its excellent reasoning capability and flexibility
- This model size is expected to capture complex patterns (especially crucial for DeFi market predictions) while being compatible with Unsloth's GRPO fine-tuning

### Synthetic Data Strategy for Labeling

- Combine high-fidelity historical data from Flipside with synthetic chain-of-thought (CoT) data
- Use an LLM to create CoT explanations when manual annotation is prohibitive
- For example, for each training sample fetched from Flipside, prompt an LLM to generate a detailed reasoning trace that the 70B model can learn from

## 3. Reward Function Design

### Primary Objectives

- **Correctness:** Reward predictions that match the actual market outcomes
- **Quality of Reasoning:** Reward detailed and well-structured chain-of-thought explanations
- **Penalties:** Deduct points for errors such as spelling mistakes or logical inconsistencies

### Reward Components

- **Market Prediction Accuracy:** +1.0 for perfect predictions; use a scaled error metric for numeric predictions
- **Chain-of-Thought Depth:** +0.5 for explanations over a certain token threshold; lesser bonus if short
- **Quality Checks:** Add rewards for clear logical progression while subtracting for incoherent sentences

### Group Policy in GRPO

- Model generates multiple responses for a given prompt
- Every response is scored individually
- **Relative scoring** is applied, where responses above the group-average receive positive reinforcement

## 4. Fine-Tuning Strategy

### Dataset Preparation

- **Historical Data:** Collect and clean high-quality DeFi and crypto data via Flipside
- **Synthetic Data Integration:**
  - Use an LLM to generate chain-of-thought reasoning traces for each sample
  - Validate synthetic labels with automated checks or a human review loop

### Training Phases

1. **Supervised Fine-Tuning (SFT)**
   - Initially fine-tune Llama 3.3 70B Instruct on the combined real & synthetic dataset
   - Use a very low learning rate (e.g., 1e-5) and monitor validation loss closely

2. **GRPO Reinforcement Fine-Tuning**
   - Load the SFT checkpoint into Unsloth's GRPO pipeline
   - Generate multiple candidate responses and apply the custom reward function
   - Use policy gradient updates to reinforce high-performing responses

### Hyperparameter Tuning

- Explore learning rate (1e-6 to 5e-6 during RL phase)
- Optimize batch sizes and reward scaling methods
- Validate with both automated metrics and human review

## 5. Training Execution

### Environment Setup

- Use cloud infrastructure with high-speed networking
- Install required dependencies: Unsloth, PyTorch, Hugging Face Transformers, diffusers, and vLLM

### Distributed Training

- Employ model-parallelism (DeepSpeed ZeRO-3 or FSDP) to shard the 70B model
- Utilize gradient accumulation for large batch simulation

### Monitoring & Checkpointing

- Save checkpoints regularly (every 1000 steps or few hours)
- Log training metrics for monitoring and debugging
- Plan for potential failures with robust resumption strategies

## 6. Inference & Deployment

### Model Export

- Export the fine-tuned model (merge LoRA weights if used)
- Optimize with 4-bit or 8-bit quantization using GPTQ
- Save in Hugging Face format for vLLM integration

### Serving Architecture

- Deploy using cloud GPU instance with vLLM
- Wrap model in FastAPI service
- Containerize solution for easy scaling

## 7. Benchmarking Metrics

### Prediction Accuracy

- Evaluate on custom benchmark set with known outcomes
- Compute accuracy and regression error metrics

### Reasoning Quality

- Use automated heuristics and human judgment
- Consider external benchmarks (MMLU, BBH)

### Computational Efficiency

- Measure inference throughput and latency
- Track memory footprint and scaling performance

### Real-World Testing

- Conduct strategy backtesting
- Monitor consistency and safety
- Track live prediction accuracy
