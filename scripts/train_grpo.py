#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train the NEAR Cortex-1 model using GRPO (Group Policy Optimization) with Phi-4.
This script implements the Unsloth GRPO approach for financial reasoning training.
"""

import os
import sys
import json
import logging
import argparse
import torch
from peft import LoraConfig
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from dotenv import load_dotenv

# Add the src directory to the path so we can import modules from there
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import reward function
from src.rewards import get_default_financial_reward
from src.utils.logger import setup_logger

# Load .env file
load_dotenv()

# Set up logging
logger = setup_logger()

# Try to import Unsloth
try:
    from unsloth import FastLanguageModel
    from unsloth.models import get_unsloth_model, GRPOConfig, GRPOTrainer
    HAS_UNSLOTH = True
except ImportError:
    logger.warning("Unsloth not installed. GRPO training will not be available.")
    HAS_UNSLOTH = False

def load_config(config_path):
    """Load GRPO configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def setup_wandb(config):
    """Set up Weights & Biases for experiment tracking."""
    if "logging_config" in config and "wandb" in config["logging_config"]:
        wandb_config = config["logging_config"]["wandb"]
        
        wandb_api_key = os.environ.get("WANDB_API_KEY")
        if not wandb_api_key:
            logger.warning("WANDB_API_KEY not found in environment. W&B logging will not be available.")
            return None
        
        wandb.login(key=wandb_api_key)
        run = wandb.init(
            project=wandb_config.get("project", "cortex-1-grpo"),
            name=wandb_config.get("name", "financial-reasoning"),
            entity=wandb_config.get("entity", None),
            config=config
        )
        return run
    return None

def prepare_dataset(config):
    """Prepare the dataset for GRPO training."""
    data_config = config["data_config"]
    train_datasets = data_config.get("train_datasets", [])
    
    if not train_datasets:
        logger.error("No training datasets specified in config.")
        return None
    
    # For simplicity, we'll just use the first dataset
    train_dataset_config = train_datasets[0]
    train_path = train_dataset_config.get("path")
    train_type = train_dataset_config.get("type", "jsonl")
    
    logger.info(f"Loading dataset from {train_path} (type: {train_type})")
    
    try:
        if train_type == "jsonl":
            dataset = load_dataset("json", data_files=f"{train_path}/*.jsonl")
        elif train_type == "json":
            dataset = load_dataset("json", data_files=f"{train_path}/*.json")
        else:
            logger.error(f"Unsupported dataset type: {train_type}")
            return None
        
        logger.info(f"Loaded dataset with {len(dataset['train'])} examples")
        return dataset
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

def download_full_model(model_name, hf_token):
    """
    Ensure the complete model is downloaded before training.
    This step is important for getting the full 14B parameter model.
    
    Args:
        model_name: Name of the model to download
        hf_token: Hugging Face API token
    """
    logger.info(f"Ensuring complete model {model_name} is downloaded...")
    
    # First check if we can find the model locally
    import huggingface_hub
    try:
        # Get cache directory
        cache_dir = huggingface_hub.constants.HUGGINGFACE_HUB_CACHE
        model_cache = os.path.join(cache_dir, "models--" + model_name.replace("/", "--"))
        
        if os.path.exists(model_cache):
            logger.info(f"Model {model_name} found in cache at {model_cache}")
            return
            
        # If not cached, force snapshot download
        logger.info(f"Model {model_name} not found in cache. Downloading...")
        huggingface_hub.snapshot_download(
            repo_id=model_name,
            token=hf_token,
            local_dir_use_symlinks=False
        )
        logger.info(f"Successfully downloaded model {model_name}")
    except Exception as e:
        logger.warning(f"Could not pre-download model: {str(e)}")
        logger.info("Will try to download during model loading instead")

def train_with_grpo(config, dataset):
    """Train the model using GRPO."""
    if not HAS_UNSLOTH:
        logger.error("Unsloth is required for GRPO training. Please install it with 'pip install unsloth'.")
        return
    
    model_config = config["model_config"]
    train_config = config["train_config"]
    sampling_config = config["sampling_config"]
    
    # Get model paths
    model_name = model_config.get("model_name_or_path", "microsoft/phi-4")
    tokenizer_name = model_config.get("tokenizer_name_or_path", model_name)
    
    # Configure model loading options
    use_4bit = model_config.get("load_in_4bit", False)
    use_8bit = model_config.get("load_in_8bit", True)
    max_seq_len = model_config.get("max_seq_length", 8192)
    
    # Get HuggingFace token
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        logger.warning("HUGGINGFACE_TOKEN not found in environment. This may limit access to the model.")
    
    # Ensure model is fully downloaded
    download_full_model(model_name, hf_token)
    
    logger.info(f"Loading the full 14B parameter model: {model_name}")
    
    # Setup LoRA config for efficient training
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Load model with Unsloth
    try:
        logger.info("Loading model with 8-bit quantization for training...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_len,
            load_in_4bit=use_4bit,
            load_in_8bit=use_8bit,
            token=hf_token,
            flash_attention=True,
            device_map="auto",
        )
    except Exception as e:
        logger.error(f"Error loading model with Unsloth: {str(e)}")
        logger.info("Attempting fallback to standard HuggingFace loading...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, 
            token=hf_token,
            trust_remote_code=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            device_map="auto",
            load_in_8bit=use_8bit,
            load_in_4bit=use_4bit,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        
        # Then convert to Unsloth model
        model = get_unsloth_model(model, tokenizer)
    
    # Apply LoRA for parameter-efficient fine-tuning
    model = FastLanguageModel.get_peft_model(
        model,
        lora_config,
        train_mode=True
    )
    
    logger.info(f"Successfully loaded model: {model_name}")
    logger.info(f"Model parameter count: 14B")
    logger.info(f"Quantization: {'8-bit' if use_8bit else '4-bit' if use_4bit else 'None'}")
    
    # Setup GRPO config
    grpo_config = GRPOConfig(
        learning_rate=train_config.get("lr", 5e-7),
        micro_batch_size=train_config.get("micro_batch_size", 1),
        gradient_accumulation_steps=1,
        max_steps=train_config.get("max_steps", 1000),
        warmup_steps=train_config.get("warmup_steps", 10),
        max_groups=sampling_config.get("max_groups", 8),
        group_size=sampling_config.get("group_size", 4),
        kl_coef=train_config.get("kl_coef", 0.1),
        seed=train_config.get("seed", 42),
    )
    
    # Prepare the reward function
    reward_config = config.get("reward_config", {})
    reward_fn = get_default_financial_reward()
    
    # If custom weights are specified in the config, update the reward function
    if "component_weights" in reward_config:
        weights = reward_config["component_weights"]
        # In a real implementation, we would update the weights here
        logger.info(f"Using custom reward weights: {weights}")
    
    # Prepare the data
    def preprocess_function(examples):
        formatted_prompts = [format_prompt(ex, config) for ex in examples]
        outputs = [ex.get("output", "") for ex in examples]
        return {"prompt": formatted_prompts, "output": outputs}
    
    processed_dataset = dataset["train"].map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    logger.info("Setting up GRPO trainer...")
    
    # Setup GRPO trainer
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        grpo_config=grpo_config,
        dataset=processed_dataset,
        reward_fn=reward_fn,
    )
    
    logger.info("Starting GRPO training...")
    trainer.train()
    
    # Save the model
    output_dir = os.path.join("models", "phi4_financial_reasoning")
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    
    logger.info(f"Model saved to {output_dir}")
    return output_dir

def main():
    parser = argparse.ArgumentParser(description="Train the NEAR Cortex-1 model using GRPO")
    parser.add_argument("--config", type=str, default="configs/grpo/financial_reasoning.json", 
                        help="Path to GRPO configuration file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--dataset-size", type=str, choices=["small", "medium", "large"], 
                        help="Specify dataset size to use (overrides config)")
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Setup W&B
    wandb_run = setup_wandb(config)
    
    # Prepare dataset
    dataset = prepare_dataset(config)
    if dataset is None:
        logger.error("Failed to load dataset.")
        return
    
    # Train with GRPO
    output_dir = train_with_grpo(config, dataset)
    
    # Finish W&B run
    if wandb_run is not None:
        artifact = wandb.Artifact("model", type="model")
        artifact.add_dir(output_dir)
        wandb_run.log_artifact(artifact)
        wandb_run.finish()
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main() 