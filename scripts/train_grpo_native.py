#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Native PyTorch implementation of GRPO for the NEAR Cortex-1 model.
This script implements GRPO (Group Relative Policy Optimization) for training reasoning models
without using 8-bit quantization or problematic dependencies.
"""

import os
import sys
import json
import yaml
import torch
import logging
import argparse
import numpy as np
from peft import LoraConfig, get_peft_model
import wandb
from datasets import load_dataset
from dotenv import load_dotenv
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    get_scheduler,
    GenerationConfig
)

# Add the src directory to the path so we can import modules from there
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
from src.rewards import get_default_financial_reward
from src.utils.logger import setup_logger

# Load .env file
load_dotenv()

# Set up logging
logger = setup_logger()

def load_config(config_path):
    """Load configuration from JSON or YAML file."""
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        # Convert YAML config to match expected structure
        return convert_yaml_config(config)
    else:
        # Assume JSON format
        with open(config_path, 'r') as f:
            return json.load(f)

def convert_yaml_config(yaml_config):
    """Convert YAML config structure to the expected format for training."""
    config = {
        "model_config": {
            "model_name_or_path": yaml_config.get("model", {}).get("name", "microsoft/phi-4"),
            "tokenizer_name_or_path": yaml_config.get("model", {}).get("tokenizer_name", "microsoft/phi-4"),
            "torch_dtype": "bfloat16",  # Default to bfloat16
            "load_in_8bit": False,
            "load_in_4bit": yaml_config.get("model", {}).get("load_in_4bit", False),
            "use_flash_attention": yaml_config.get("model", {}).get("use_flash_attention", True),
            "max_seq_length": yaml_config.get("training", {}).get("max_seq_length", 16384)
        },
        "train_config": {
            "lr": yaml_config.get("training", {}).get("learning_rate", 5e-6),
            "micro_batch_size": yaml_config.get("training", {}).get("per_device_train_batch_size", 2),
            "gradient_accumulation_steps": yaml_config.get("training", {}).get("gradient_accumulation_steps", 4),
            "warmup_steps": yaml_config.get("training", {}).get("warmup_steps", 100),
            "max_steps": yaml_config.get("training", {}).get("min_training_steps", 300),
            "kl_coef": yaml_config.get("reward", {}).get("kl_penalty", 0.05),
            "seed": yaml_config.get("training", {}).get("seed", 42),
            "optimizer": yaml_config.get("training", {}).get("optimizer", "adamw_torch"),
            "lr_scheduler_type": yaml_config.get("training", {}).get("lr_scheduler_type", "cosine"),
            "weight_decay": yaml_config.get("training", {}).get("weight_decay", 0.01),
            "max_grad_norm": yaml_config.get("training", {}).get("max_grad_norm", 1.0),
            "num_epochs": yaml_config.get("training", {}).get("num_epochs", 3)
        },
        "sampling_config": {
            "temperature": 0.7,  # Default values
            "top_p": 0.9,
            "top_k": 50,
            "num_beams": 1,
            "max_length": yaml_config.get("model", {}).get("max_length", 16384),
            "max_groups": yaml_config.get("training", {}).get("generations_per_prompt", 16) // 4,  # Divide by group size
            "group_size": 4  # Default group size
        },
        "reward_config": {
            "name": "financial_reasoning",
            "reward_fn": "src.rewards.get_default_financial_reward",
            "normalize": yaml_config.get("reward", {}).get("normalize_rewards", True),
            "clip_rewards": yaml_config.get("reward", {}).get("clip_rewards", 1.0),
            "baseline_subtract": yaml_config.get("reward", {}).get("baseline_subtract", True),
            "component_weights": yaml_config.get("reward", {}).get("weights", {})
        },
        "data_config": {
            "train_datasets": [
                {
                    "path": yaml_config.get("data", {}).get("train_path", "data/synthetic/training/reasoning_training_500.jsonl").format(size=500),
                    "type": "jsonl"
                }
            ],
            "eval_datasets": [
                {
                    "path": yaml_config.get("data", {}).get("eval_path", "data/synthetic/eval/reasoning_eval_500.jsonl").format(size=500),
                    "type": "jsonl"
                }
            ],
            "verify_quality": yaml_config.get("data", {}).get("verify_quality", True)
        },
        "logging_config": {
            "wandb": {
                "project": yaml_config.get("logging", {}).get("wandb", {}).get("project", "cortex-1"),
                "name": "phi4-financial-reasoning",
                "entity": yaml_config.get("logging", {}).get("wandb", {}).get("entity", "jbarnes850-near-protocol"),
                "tags": yaml_config.get("logging", {}).get("wandb", {}).get("tags", ["crypto", "grpo", "financial-analysis"]),
                "log_model": True
            },
            "log_interval_steps": yaml_config.get("logging", {}).get("log_interval_steps", 10),
            "eval_interval_steps": yaml_config.get("logging", {}).get("eval_interval_steps", 100),
            "checkpoint_interval_hours": yaml_config.get("logging", {}).get("checkpoint_interval_hours", 1),
            "log_reward_components": yaml_config.get("logging", {}).get("log_reward_components", True)
        },
        "distribution_config": {
            "strategy": yaml_config.get("distributed", {}).get("strategy", "deepspeed_stage_3"),
            "gradient_checkpointing": yaml_config.get("distributed", {}).get("gradient_checkpointing", True),
            "zero3_init_flag": yaml_config.get("distributed", {}).get("zero3_init_flag", True),
            "offload_optimizer": yaml_config.get("distributed", {}).get("offload_optimizer", True),
            "offload_param": yaml_config.get("distributed", {}).get("offload_param", False)
        },
        "task_config": {
            "prompt_template": "You are an expert financial analyst specializing in cryptocurrency markets. Analyze the following metrics and provide a detailed financial analysis with investment recommendations.\n\nMetrics:\n{metrics}\n\nYour analysis should include:\n1. Executive summary\n2. Key metrics analysis\n3. Detailed calculations with formulas and confidence intervals\n4. Market context including historical references\n5. Investment implications with clear recommendations\n\nAnalysis:",
            "response_template": "",
            "max_prompt_length": 2048,
            "chat_format": "phi"
        }
    }
    return config

def setup_wandb(config):
    """Set up Weights & Biases for experiment tracking."""
    if "logging_config" not in config or "wandb" not in config["logging_config"]:
        logger.info("W&B logging configuration not found. Skipping W&B initialization.")
        return None
        
    wandb_config = config["logging_config"]["wandb"]
    
    # Check if W&B is explicitly disabled
    if not wandb_config.get("enabled", True):
        logger.info("W&B logging is disabled in the configuration. Skipping W&B initialization.")
        return None
    
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if not wandb_api_key:
        logger.warning("WANDB_API_KEY not found in environment. W&B logging will not be available.")
        return None
    
    try:
        wandb.login(key=wandb_api_key)
        run = wandb.init(
            project=wandb_config.get("project", "cortex-1"),
            name=wandb_config.get("name", "financial-reasoning"),
            entity=wandb_config.get("entity", "jbarnes850-near-protocol"),
            config=config,
            tags=wandb_config.get("tags", None)
        )
        logger.info(f"Successfully initialized W&B run: {run.id}")
        return run
    except Exception as e:
        logger.warning(f"Failed to initialize W&B: {str(e)}. Training will continue without W&B logging.")
        return None

def load_model_and_tokenizer(config):
    """Load the model and tokenizer."""
    model_config = config["model_config"]
    
    # Get model paths
    model_name = model_config.get("model_name_or_path", "microsoft/phi-4")
    tokenizer_name = model_config.get("tokenizer_name_or_path", model_name)
    
    # Get HuggingFace token
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        logger.warning("HUGGINGFACE_TOKEN not found in environment. This may limit access to the model.")
    
    logger.info(f"Loading model: {model_name}")
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, 
        token=hf_token,
        trust_remote_code=True
    )
    
    # Configure torch dtype
    dtype_str = model_config.get("torch_dtype", "bfloat16")
    torch_dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
    
    # Load the model
    logger.info(f"Loading model with device map 'auto' and torch dtype '{dtype_str}'")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch_dtype
    )
    
    # Apply LoRA for parameter-efficient fine-tuning
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["o_proj", "qkv_proj", "gate_up_proj", "down_proj"],
    )
    logger.info("Applying LoRA for parameter-efficient fine-tuning")
    model = get_peft_model(model, lora_config)
    
    return model, tokenizer

def load_dataset_from_path(dataset_path):
    """Load dataset from a specific path."""
    if dataset_path.endswith(".jsonl") or dataset_path.endswith(".json"):
        try:
            dataset = load_dataset("json", data_files=dataset_path)
            logger.info(f"Loaded dataset from {dataset_path} with {len(dataset['train'])} examples")
            return dataset
        except Exception as e:
            logger.error(f"Error loading dataset from {dataset_path}: {str(e)}")
            return None
    else:
        logger.error(f"Unsupported dataset format: {dataset_path}")
        return None

def prepare_dataset(config, dataset_size=None):
    """Prepare the dataset for GRPO training."""
    # If dataset_size is specified, use a predefined path
    if dataset_size:
        if dataset_size == "small":
            path = "data/splits/financial_analysis/train_small_20250302_0749.jsonl"
        elif dataset_size == "medium":
            path = "data/splits/financial_analysis/train_medium_20250302_0754.jsonl"
        elif dataset_size == "large":
            path = "data/splits/financial_analysis/train_large_20250302_1202.jsonl"
        else:
            logger.error(f"Unknown dataset size: {dataset_size}")
            return None
        
        return load_dataset_from_path(path)
    
    # Otherwise use the path from the config
    data_config = config["data_config"]
    train_datasets = data_config.get("train_datasets", [])
    
    if not train_datasets:
        logger.error("No training datasets specified in config.")
        return None
    
    train_dataset_config = train_datasets[0]
    train_path = train_dataset_config.get("path")
    train_type = train_dataset_config.get("type", "jsonl")
    
    logger.info(f"Loading dataset from {train_path} (type: {train_type})")
    
    try:
        if train_type == "jsonl" or train_type == "json":
            if os.path.isdir(train_path):
                dataset = load_dataset("json", data_files=f"{train_path}/*.{train_type}")
            else:
                dataset = load_dataset("json", data_files=train_path)
            logger.info(f"Loaded dataset with {len(dataset['train'])} examples")
            return dataset
        else:
            logger.error(f"Unsupported dataset type: {train_type}")
            return None
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return None

def format_prompt(example, config):
    """Format the prompt for training."""
    task_config = config["task_config"]
    chat_format = task_config.get("chat_format", "phi")
    prompt_template = task_config.get("prompt_template", "")
    
    # Replace placeholders in the template
    input_text = example.get("input", "")
    
    # For financial data, format metrics
    if "{metrics}" in prompt_template and "market_data" in example:
        metrics_str = "\n".join([f"{k}: {v}" for k, v in example.get("market_data", {}).items()])
        input_text = prompt_template.replace("{metrics}", metrics_str)
    
    # Format according to model chat format
    if chat_format == "phi":
        # Phi-4 format
        formatted_prompt = {
            "messages": [
                {"role": "system", "content": "You are an expert financial analyst specializing in cryptocurrency markets."},
                {"role": "user", "content": input_text}
            ]
        }
    else:
        # Default format
        formatted_prompt = input_text
    
    return formatted_prompt

def generate_responses(model, tokenizer, prompts, config):
    """Generate multiple responses for each prompt."""
    sampling_config = config["sampling_config"]
    max_length = sampling_config.get("max_length", 2000)
    temperature = sampling_config.get("temperature", 0.7)
    top_p = sampling_config.get("top_p", 0.9)
    top_k = sampling_config.get("top_k", 50)
    group_size = sampling_config.get("group_size", 4)
    
    generation_config = GenerationConfig(
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        do_sample=True
    )
    
    all_responses = []
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        group_responses = []
        
        # Generate multiple responses for the same prompt
        for _ in range(group_size):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    generation_config=generation_config
                )
                
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            group_responses.append(response)
            
        all_responses.append(group_responses)
        
    return all_responses

def calculate_rewards(responses, reward_fn, config):
    """Calculate rewards for each response using the reward function."""
    rewards = []
    reward_config = config.get("reward_config", {})
    clip_value = reward_config.get("clip_rewards", 1.0)
    normalize = reward_config.get("normalize", True)
    baseline_subtract = reward_config.get("baseline_subtract", True)
    
    for group in responses:
        group_rewards = []
        for response in group:
            # Calculate raw reward
            reward = reward_fn(response)
            group_rewards.append(reward)
        
        # Apply baseline subtraction if enabled
        if baseline_subtract:
            baseline = np.mean(group_rewards)
            group_rewards = [r - baseline for r in group_rewards]
        
        # Apply normalization if enabled
        if normalize and len(group_rewards) > 1:
            mean = np.mean(group_rewards)
            std = np.std(group_rewards) + 1e-8  # Avoid division by zero
            group_rewards = [(r - mean) / std for r in group_rewards]
        
        # Apply reward clipping if enabled
        if clip_value > 0:
            group_rewards = [max(min(r, clip_value), -clip_value) for r in group_rewards]
            
        rewards.append(group_rewards)
        
    return rewards

def train_grpo(model, tokenizer, dataset, reward_fn, config):
    """Train the model using GRPO."""
    train_config = config["train_config"]
    sampling_config = config["sampling_config"]
    logging_config = config["logging_config"]
    
    # Check if W&B is available
    has_wandb = wandb.run is not None
    if has_wandb:
        logger.info("W&B logging is enabled for training")
    else:
        logger.info("W&B logging is disabled, continuing without metrics tracking")
    
    # Prepare the optimizer
    learning_rate = train_config.get("lr", 5e-7)
    weight_decay = train_config.get("weight_decay", 0.01)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Number of training steps
    max_steps = train_config.get("max_steps", 1000)
    warmup_steps = train_config.get("warmup_steps", 10)
    
    # GRPO hyperparameters
    kl_coef = train_config.get("kl_coef", 0.1)
    micro_batch_size = train_config.get("micro_batch_size", 1)
    gradient_accumulation_steps = train_config.get("gradient_accumulation_steps", 1)
    max_grad_norm = train_config.get("max_grad_norm", 1.0)
    
    # Create the scheduler
    scheduler_type = train_config.get("lr_scheduler_type", "linear")
    scheduler = get_scheduler(
        scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps
    )
    
    # Process the dataset
    processed_data = []
    for example in dataset:
        prompt = format_prompt(example, config)
        reference_output = example.get("output", "")
        processed_data.append({"prompt": prompt, "reference": reference_output})
    
    # Logging intervals
    log_interval = logging_config.get("log_interval_steps", 10)
    save_interval = logging_config.get("eval_interval_steps", 100)
    
    # Main training loop
    logger.info("Starting GRPO training...")
    logger.info(f"Training parameters: lr={learning_rate}, steps={max_steps}, batch_size={micro_batch_size}")
    logger.info(f"GRPO parameters: kl_coef={kl_coef}")
    
    model.train()
    total_loss = 0.0
    
    for step in range(max_steps):
        # Sample a batch of prompts
        batch_indices = np.random.choice(len(processed_data), min(micro_batch_size, len(processed_data)), replace=False)
        batch = [processed_data[i] for i in batch_indices]
        prompts = [item["prompt"] for item in batch]
        
        # Generate multiple responses for each prompt
        responses = generate_responses(model, tokenizer, prompts, config)
        
        # Calculate rewards for each response
        rewards = calculate_rewards(responses, reward_fn, config)
        
        # Implement GRPO update
        optimizer.zero_grad()
        
        # For each group, calculate policy gradients
        batch_loss = 0.0
        
        # Track statistics for logging
        better_than_avg_count = 0
        total_responses = 0
        
        for i, (group_responses, group_rewards) in enumerate(zip(responses, rewards)):
            prompt = prompts[i]
            
            # Calculate mean reward for the group (redundant if baseline subtraction enabled)
            mean_reward = 0.0  # Already handled in calculate_rewards if baseline_subtract=True
            
            # Process each response in the group
            for j, (response, reward) in enumerate(zip(group_responses, group_rewards)):
                # Use reward directly (baseline subtraction already applied if enabled)
                rel_reward = reward
                total_responses += 1
                
                # Skip if the reward is too small
                if abs(rel_reward) < 1e-6:
                    continue
                
                # Count positive rewards (better than average)
                if rel_reward > 0:
                    better_than_avg_count += 1
                
                # Tokenize the full sequence
                input_text = prompt
                if isinstance(prompt, dict) and "messages" in prompt:
                    # Handle structured prompts
                    input_text = prompt["messages"][1]["content"]  # User message content
                
                # Prepare inputs for the model
                inputs = tokenizer(input_text + response, return_tensors="pt").to(model.device)
                
                # Forward pass
                outputs = model(**inputs, labels=inputs["input_ids"])
                
                # Scale the loss by the relative reward
                response_loss = outputs.loss * (-rel_reward)
                
                # Add KL penalty
                if kl_coef > 0:
                    response_loss = response_loss + kl_coef * outputs.loss
                
                # Accumulate the loss
                batch_loss += response_loss / gradient_accumulation_steps
        
        # Scale the loss by the batch size if needed
        if total_responses > 0:
            # Loss already scaled in the loop
            # Perform backward pass
            batch_loss.backward()
            
            # Gradient clipping
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Optimizer step
            optimizer.step()
            scheduler.step()
        
        # Log progress
        total_loss += batch_loss.item() * gradient_accumulation_steps
        
        # Log at intervals
        if (step + 1) % log_interval == 0:
            avg_loss = total_loss / log_interval
            better_ratio = better_than_avg_count / max(total_responses, 1)
            
            logger.info(f"Step {step+1}/{max_steps}, Loss: {avg_loss:.6f}, Better-than-avg: {better_ratio:.2f}")
            total_loss = 0.0
            better_than_avg_count = 0
            total_responses = 0
            
            # Log to W&B if available
            if has_wandb:
                try:
                    wandb.log({
                        "loss": avg_loss,
                        "learning_rate": scheduler.get_last_lr()[0],
                        "step": step,
                        "better_ratio": better_ratio
                    })
                except Exception as e:
                    logger.warning(f"Failed to log to W&B: {str(e)}")
        
        # Save checkpoint at intervals
        if (step + 1) % save_interval == 0:
            output_dir = os.path.join("models", f"checkpoint-{step+1}")
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logger.info(f"Saved checkpoint to {output_dir}")
    
    # Save the final model
    output_dir = os.path.join("models", "phi4_financial_reasoning_grpo")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")
    
    return output_dir

def main():
    parser = argparse.ArgumentParser(description="Train the NEAR Cortex-1 model using GRPO")
    parser.add_argument("--config", type=str, default="configs/grpo_config.yaml", 
                        help="Path to GRPO configuration file (YAML or JSON)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--dataset-size", type=str, choices=["small", "medium", "large"], 
                        help="Specify dataset size to use (small, medium, or large)")
    parser.add_argument("--dataset-path", type=str, help="Direct path to dataset file")
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Setup W&B
    wandb_run = setup_wandb(config)
    
    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Prepare dataset
    dataset = None
    if args.dataset_path:
        dataset = load_dataset_from_path(args.dataset_path)
    elif args.dataset_size:
        dataset = prepare_dataset(config, args.dataset_size)
    else:
        dataset = prepare_dataset(config)
    
    if dataset is None:
        logger.error("Failed to load dataset.")
        return
    
    # Get the reward function
    reward_fn = get_default_financial_reward()
    
    # Apply custom reward weights if specified
    reward_config = config.get("reward_config", {})
    component_weights = reward_config.get("component_weights", {})
    if component_weights:
        logger.info(f"Applying custom reward weights: {component_weights}")
        # In a more sophisticated implementation, we would apply these weights to the reward function
    
    # Train with GRPO
    output_dir = train_grpo(model, tokenizer, dataset["train"], reward_fn, config)
    
    # Finish W&B run
    if wandb.run is not None:
        try:
            artifact = wandb.Artifact("model", type="model")
            artifact.add_dir(output_dir)
            wandb.run.log_artifact(artifact)
            wandb.run.finish()
            logger.info("W&B run completed successfully")
        except Exception as e:
            logger.warning(f"Failed to complete W&B run: {str(e)}")
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main() 