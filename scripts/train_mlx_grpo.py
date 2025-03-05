#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MLX-based implementation of GRPO for the NEAR Cortex-1 model on Apple Silicon.
This script implements GRPO (Group Relative Policy Optimization) for training reasoning models
specifically optimized for Apple Silicon using MLX.
"""

import os
import sys
import json
import yaml
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
import argparse
import logging
from tqdm import tqdm
from pathlib import Path
import time
from dotenv import load_dotenv
import random
from datetime import datetime
from datasets import load_dataset
from huggingface_hub import login
import traceback

# Add the src directory to the path so we can import modules from there
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
from src.rewards import get_default_financial_reward
from src.utils.logger import setup_logger

# Local imports for MLX model handling
try:
    # Import MLX-LM modules with proper error handling
    import mlx
    import mlx_lm
    from mlx_lm import load
    from mlx_lm.lora import train_model, linear_to_lora_layers, TrainingArgs
    from mlx_lm.utils import generate, TokenizerWrapper
    from transformers import AutoTokenizer
except ImportError as e:
    print(f"Error importing MLX-LM: {e}")
    print("Please install the required packages: pip install mlx-lm transformers")
    sys.exit(1)

# Load .env file
load_dotenv()

# Set up logging
logger = setup_logger()

# Authenticate with HuggingFace if token is available
def setup_huggingface_auth():
    """Setup Hugging Face authentication using token from environment variable."""
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token:
        try:
            login(token=hf_token)
            logger.info("Successfully authenticated with Hugging Face")
            return True
        except Exception as e:
            logger.error(f"Failed to authenticate with Hugging Face: {e}")
            return False
    else:
        logger.warning("No Hugging Face token found in environment variables")
        return False

def tree_flatten(tree):
    """
    Flatten a nested dictionary of parameters.
    For MLX models, this allows counting parameters.
    """
    leaves = []
    aux = []
    
    if isinstance(tree, dict):
        for k, v in tree.items():
            l, a = tree_flatten(v)
            leaves.extend([(k + "." + k2 if k2 else k, v2) for k2, v2 in l])
            aux.extend([(k + "." + k2 if k2 else k, v2) for k2, v2 in a])
    elif isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            l, a = tree_flatten(v)
            leaves.extend([(str(i) + "." + k if k else str(i), v) for k, v in l])
            aux.extend([(str(i) + "." + k if k else str(i), v) for k, v in a])
    elif hasattr(tree, "parameters") and callable(tree.parameters):
        # MLX modules have a parameters method
        return tree_flatten(tree.parameters())
    elif isinstance(tree, mx.array):
        leaves = [("", tree)]
    else:
        aux = [("", tree)]
    
    return leaves, aux

@dataclass
class MLXGRPOConfig:
    """Configuration for MLX GRPO training."""
    # Model parameters
    model_name_or_path: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    tokenizer_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    max_seq_length: int = 4096  # As per model specs
    model_dtype: str = "bfloat16"  # Model uses BF16 by default
    
    # LoRA parameters
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "attention.wq", "attention.wk", "attention.wv", "attention.wo", 
        "feed_forward.w1", "feed_forward.w2", "feed_forward.w3"
    ])
    
    # Training parameters
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    warmup_steps: int = 50
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    min_training_steps: int = 100
    early_stopping_patience: int = 5
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    optimizer: str = "adamw"
    lr_scheduler_type: str = "cosine"
    seed: int = 42
    
    # SFT specific parameters
    sft_epochs: int = 3
    sft_batch_size: int = 1
    sft_learning_rate: float = 5e-6
    sft_weight_decay: float = 0.01
    sft_gradient_accumulation_steps: int = 4
    sft_warmup_steps: int = 50
    sft_save_interval: int = 50
    sft_log_interval: int = 5
    sft_val_check_interval: int = 50
    
    # GRPO specific parameters
    generations_per_prompt: int = 8
    
    # Sampling parameters (as per DeepSeek recommendations)
    temperature: float = 0.6  # Recommended value
    top_p: float = 0.95  # As per evaluation settings
    top_k: int = 50
    group_size: int = 2
    
    # Reward parameters
    baseline_subtract: bool = True
    normalize_rewards: bool = True
    clip_rewards: float = 1.0
    kl_penalty: float = 0.05
    
    # Reward weights
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        "financial_insight": 0.3,
        "reasoning_depth": 0.3,
        "calculation_accuracy": 0.2,
        "recommendation_clarity": 0.2,
        "confidence_interval": 0.8,
        "investment_insight": 1.0,
        "citation_format": 0.7,
        "structure": 0.6,
        "completeness": 0.8,
        "metric_citation": 0.9,
        "historical_reference": 0.7
    })
    
    # Data paths
    train_path: str = "data/splits/financial_analysis/train_small_20250302_0749.jsonl"
    eval_path: str = "data/splits/financial_analysis/eval_small_20250302_0749.jsonl"
    verify_quality: bool = True
    
    # Logging parameters
    log_interval_steps: int = 5
    eval_interval_steps: int = 50
    save_interval_steps: int = 50
    checkpoint_interval_hours: int = 1
    log_reward_components: bool = True
    
    # Output parameters
    output_dir: str = "models/mlx-grpo-phi4"
    save_format: str = "safetensors"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MLXGRPOConfig":
        """Create a config object from a dictionary."""
        config = cls()
        
        # Handle nested dictionaries
        for key, value in config_dict.get("model", {}).items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Add model_name_or_path if present under different key
        if "name_or_path" in config_dict.get("model", {}):
            config.model_name_or_path = config_dict["model"]["name_or_path"]
            
        for key, value in config_dict.get("training", {}).items():
            if hasattr(config, key):
                setattr(config, key, value)
                
        # Handle SFT parameters
        for key, value in config_dict.get("sft", {}).items():
            if hasattr(config, key):
                setattr(config, key, value)
            elif hasattr(config, f"sft_{key}"):
                setattr(config, f"sft_{key}", value)
                
        for key, value in config_dict.get("sampling_config", {}).items():
            if hasattr(config, key):
                setattr(config, key, value)
                
        for key, value in config_dict.get("reward", {}).items():
            if key == "weights":
                config.reward_weights = value
            elif hasattr(config, key):
                setattr(config, key, value)
                
        for key, value in config_dict.get("data", {}).items():
            if hasattr(config, key):
                setattr(config, key, value)
                
        for key, value in config_dict.get("logging", {}).items():
            if hasattr(config, key):
                setattr(config, key, value)
                
        for key, value in config_dict.get("output", {}).items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    # MLX doesn't have a specific seed setting function, but NumPy seed affects MLX random operations

def load_model_and_tokenizer(config: MLXGRPOConfig):
    """Load the model and tokenizer."""
    logger.info(f"Loading model from {config.model_name_or_path}")
    
    # Set up HuggingFace authentication if needed
    if not setup_huggingface_auth():
        logger.warning("HuggingFace authentication not set up. This may cause issues with model loading.")
    
    try:
        # Check if this is a local path or a HF model id
        model_path = config.model_name_or_path
        if not os.path.exists(model_path):
            # If not a local path, check if it's in the deepseek-mlx directory
            local_path = os.path.join("deepseek-mlx/models", os.path.basename(config.model_name_or_path))
            if os.path.exists(local_path):
                model_path = local_path
                logger.info(f"Using local model path: {model_path}")
            else:
                logger.info(f"Loading model from Hugging Face: {model_path}")
        
        # Set the dtype in model_config
        dtype_map = {
            "float16": mx.float16,
            "bfloat16": mx.bfloat16,
            "float32": mx.float32
        }
        dtype = dtype_map.get(config.model_dtype, mx.bfloat16)
        
        model_config = {
            "dtype": dtype,
            "trust_remote_code": True  # Required for DeepSeek models
        }
        
        logger.info(f"Loading model with dtype: {dtype}")
        
        # Load the model and tokenizer using MLX-LM's load function
        model, tokenizer = load(
            model_path,
            model_config=model_config
        )
        
        # Verify model loading
        if model is None or tokenizer is None:
            raise ValueError("Failed to load model or tokenizer")
            
        # Log model configuration
        logger.info(f"Model loaded successfully:")
        logger.info(f"- Model type: {type(model).__name__}")
        logger.info(f"- Tokenizer type: {type(tokenizer).__name__}")
        
        # Apply LoRA if specified
        if config.use_lora:
            logger.info("Applying LoRA to model layers")
            # Convert linear layers to LoRA layers
            linear_to_lora_layers(
                model, 
                r=config.lora_r,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                target_modules=config.lora_target_modules
            )
            
            # Report parameter counts
            total_params, _ = tree_flatten(model.parameters())
            total_params = sum(p.size for k, p in total_params) / 10**6
            logger.info(f"Total parameters: {total_params:.2f}M")
        
        logger.info(f"Successfully loaded model and tokenizer")
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.error(f"Detailed error: {str(e)}")
        sys.exit(1)

def prepare_dataset(config: MLXGRPOConfig, dataset_size: Optional[str] = None):
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
    else:
        # Use a default path
        path = "data/splits/financial_analysis/train_large_20250302_1202.jsonl"
    
    logger.info(f"Loading dataset from {path}")
    
    try:
        dataset = load_dataset("json", data_files=path)
        logger.info(f"Loaded dataset with {len(dataset['train'])} examples")
        return dataset["train"]
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return None

def tree_update(model, updates):
    """Update model parameters with gradients in a functional way."""
    return mx.tree_map(lambda p, u: p + u, model, updates)

def format_prompt(example, tokenizer):
    """Format a prompt for training or generation."""
    # Check if example has required fields
    if not isinstance(example, dict) or "input" not in example or "output" not in example:
        logger.warning(f"Example missing required fields: {example}")
        return None
    
    # Format the prompt based on example type
    input_text = example["input"].strip()
    output_text = example["output"].strip()
    
    # Special handling for reasoning if available
    reasoning = example.get("reasoning", "")
    
    # Return the formatted example
    return {
        "prompt": input_text,
        "completion": output_text,
        "reasoning": reasoning,
        "formatted_text": input_text + output_text
    }

def generate_responses(model, tokenizer, prompts: List[Dict], config: MLXGRPOConfig) -> List[List[str]]:
    """Generate multiple responses for each prompt using MLX-LM's generate function.
    
    Args:
        model: The model to generate responses with.
        tokenizer: The tokenizer to use for encoding and decoding.
        prompts: A list of formatted prompts.
        config: The configuration for generation.
        
    Returns:
        A list of lists of generated responses.
    """
    all_responses = []
    
    for prompt_data in prompts:
        prompt = prompt_data["prompt"]
        responses = []
        
        for _ in range(config.generations_per_prompt):
            try:
                # Use MLX-LM's generate function
                generation_args = {
                    "prompt": prompt,
                    "temp": config.temperature,
                    "top_p": config.top_p,
                    "top_k": config.top_k,
                    "max_tokens": config.max_seq_length // 2,  # Reasonable limit for responses
                    "verbose": False
                }
                
                # Generate the response
                output = generate(model, tokenizer, **generation_args)
                
                # Extract just the generated text (not the prompt)
                generated_text = output
                if generated_text:
                    # Handle potential overlap between prompt and response
                    if generated_text.startswith(prompt):
                        generated_text = generated_text[len(prompt):]
                    
                    responses.append(generated_text)
                else:
                    # If generation failed, add an empty string
                    responses.append("")
                    logger.warning(f"Empty generation for prompt: {prompt[:50]}...")
            
            except Exception as e:
                logger.error(f"Error during generation: {e}")
                # Add an empty string as a fallback
                responses.append("")
        
        all_responses.append(responses)
    
    return all_responses

def calculate_rewards(responses: List[List[str]], reward_fn, config: MLXGRPOConfig) -> List[List[float]]:
    """Calculate rewards for each response using the reward function."""
    rewards = []
    
    for group in responses:
        group_rewards = []
        for response in group:
            # Calculate raw reward for this response
            # The reward_fn expects a response and returns a score
            reward = reward_fn(response)
            group_rewards.append(reward)
        
        # Apply baseline subtraction if enabled
        if config.baseline_subtract and len(group_rewards) > 1:
            baseline = np.mean(group_rewards)
            group_rewards = [r - baseline for r in group_rewards]
        
        # Apply normalization if enabled
        if config.normalize_rewards and len(group_rewards) > 1:
            mean = np.mean(group_rewards)
            std = np.std(group_rewards) + 1e-8  # Avoid division by zero
            group_rewards = [(r - mean) / std for r in group_rewards]
        
        # Apply reward clipping if enabled
        if config.clip_rewards > 0:
            group_rewards = [max(min(r, config.clip_rewards), -config.clip_rewards) for r in group_rewards]
            
        rewards.append(group_rewards)
        
    return rewards

def train_sft(model, tokenizer, dataset, config: MLXGRPOConfig):
    """Train the model using Supervised Fine-Tuning (SFT)."""
    logger.info("Starting supervised fine-tuning")
    
    # Format the dataset for training
    training_examples = []
    
    for example in dataset:
        formatted = format_prompt(example, tokenizer)
        if formatted:
            training_examples.append(formatted)
    
    logger.info(f"Prepared {len(training_examples)} examples for training")
    
    # Split data into training and validation sets
    if len(training_examples) > 10:
        random.seed(config.seed)
        random.shuffle(training_examples)
        split_idx = int(len(training_examples) * 0.9)
        train_examples = training_examples[:split_idx]
        val_examples = training_examples[split_idx:]
        logger.info(f"Training on {len(train_examples)} examples, validating on {len(val_examples)}")
    else:
        train_examples = training_examples
        val_examples = []
        logger.info(f"Training on all {len(train_examples)} examples, no validation set")
    
    # Configure optimizer
    optimizer = optim.AdamW(
        learning_rate=config.sft_learning_rate,
        weight_decay=config.sft_weight_decay
    )
    
    # Initialize optimizer state
    try:
        opt_state = optimizer.init(model.parameters())
        logger.info("Optimizer state initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing optimizer state: {e}")
        raise
    
    # Define the loss function
    def loss_fn(model_params, batch):
        """Calculate the loss for a batch."""
        input_ids = batch["input_ids"]
        labels = batch.get("labels", input_ids)
        attention_mask = batch.get("attention_mask", mx.ones_like(input_ids))
        
        # Forward pass with parameters
        logits = model(input_ids, params=model_params)
        
        # Shift for next token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        shift_mask = attention_mask[:, 1:]
        
        # Calculate cross entropy loss with masking
        ce_loss = mx.losses.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            reduction="none"
        ).reshape(shift_labels.shape)
        
        masked_loss = ce_loss * shift_mask
        return mx.mean(masked_loss)
    
    # Define the training step
    @mx.compile
    def train_step(model_params, batch, opt_state):
        """Perform a single training step."""
        loss, grads = mx.value_and_grad(loss_fn)(model_params, batch)
        
        if config.max_grad_norm > 0:
            grads = clip_by_global_norm(grads, config.max_grad_norm)
        
        updates, new_opt_state = optimizer.update(grads, opt_state, model_params)
        new_model_params = mx.tree_map(lambda p, u: p + u, model_params, updates)
        
        return new_model_params, new_opt_state, loss
    
    def clip_by_global_norm(grads, max_norm):
        """Clip gradients by global norm."""
        total_norm = mx.sqrt(sum(mx.sum(g * g) for g in mx.tree_leaves(grads)))
        clip_coef = mx.minimum(max_norm / (total_norm + 1e-6), 1.0)
        return mx.tree_map(lambda g: g * clip_coef, grads)
    
    def tokenize_batch(batch_inputs):
        """Tokenize a batch of inputs using the appropriate tokenizer method."""
        if hasattr(tokenizer, "_tokenizer"):
            # Use the underlying HuggingFace tokenizer
            tokenizer_fn = tokenizer._tokenizer
        elif hasattr(tokenizer, "encode"):
            # Use the encode method if available
            tokenizer_fn = tokenizer
        else:
            raise ValueError("Tokenizer has no usable tokenization method")
        
        # Tokenize the inputs
        if hasattr(tokenizer_fn, "__call__"):
            encoded = tokenizer_fn(
                batch_inputs,
                padding=True,
                truncation=True,
                max_length=config.max_seq_length,
                return_tensors="np"
            )
        else:
            # Fallback to encode method
            encoded = tokenizer_fn.encode(
                batch_inputs,
                padding=True,
                truncation=True,
                max_length=config.max_seq_length,
                return_tensors="np"
            )
        
        # Convert to MLX arrays
        return {
            "input_ids": mx.array(encoded["input_ids"]),
            "attention_mask": mx.array(encoded.get("attention_mask", np.ones_like(encoded["input_ids"]))),
            "labels": mx.array(encoded["input_ids"])
        }
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    steps_since_eval = 0
    steps_since_save = 0
    
    logger.info("Starting training loop")
    
    for epoch in range(config.sft_epochs):
        random.shuffle(train_examples)
        
        for i in range(0, len(train_examples), config.sft_batch_size):
            batch_examples = train_examples[i:i + config.sft_batch_size]
            
            try:
                # Tokenize inputs
                batch_inputs = [ex["formatted_text"] for ex in batch_examples]
                batch = tokenize_batch(batch_inputs)
                
                # Training step
                model.parameters = train_step(model.parameters, batch, opt_state)
                
                # Log progress
                if (i // config.sft_batch_size) % config.sft_log_interval == 0:
                    logger.info(f"Epoch {epoch+1}/{config.sft_epochs}, "
                              f"Batch {i//config.sft_batch_size}, Loss: {loss.item():.4f}")
                
                steps_since_eval += 1
                steps_since_save += 1
                
                # Evaluation
                if steps_since_eval >= config.sft_val_check_interval and val_examples:
                    steps_since_eval = 0
                    val_loss = evaluate_model(model, tokenizer, val_examples, config)
                    logger.info(f"Validation Loss: {val_loss:.4f}")
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        save_path = os.path.join(config.output_dir, "best_sft_model")
                        save_model(model, tokenizer, save_path, config)
                        logger.info(f"Saved best model to {save_path}")
                    else:
                        patience_counter += 1
                        if patience_counter >= config.early_stopping_patience:
                            logger.info("Early stopping triggered")
                            return model
                
                # Save checkpoint
                if steps_since_save >= config.sft_save_interval:
                    steps_since_save = 0
                    save_path = os.path.join(
                        config.output_dir,
                        f"checkpoint_epoch{epoch+1}_batch{i//config.sft_batch_size}"
                    )
                    save_model(model, tokenizer, save_path, config)
                    logger.info(f"Saved checkpoint to {save_path}")
            
            except Exception as e:
                logger.error(f"Error in training step: {e}")
                logger.error(traceback.format_exc())
                continue
    
    # Save final model
    save_path = os.path.join(config.output_dir, "final_sft_model")
    save_model(model, tokenizer, save_path, config)
    logger.info(f"Saved final model to {save_path}")
    
    return model

def evaluate_model(model, tokenizer, eval_examples, config):
    """Evaluate the model on a validation set."""
    total_loss = 0.0
    batch_size = config.sft_batch_size
    
    def eval_loss_fn(model, batch):
        """Calculate the loss for a batch during evaluation."""
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        mask = batch["attention_mask"]
        
        logits = model(input_ids)
        loss = mx.mean(
            mx.losses.cross_entropy(logits[:, :-1, :], labels[:, 1:]) * mask[:, 1:]
        )
        return loss
    
    for i in range(0, len(eval_examples), batch_size):
        batch_examples = eval_examples[i:i+batch_size]
        batch_inputs = [ex["formatted_text"] for ex in batch_examples]
        
        # Access the HuggingFace tokenizer inside TokenizerWrapper
        if hasattr(tokenizer, "_tokenizer"):
            huggingface_tokenizer = tokenizer._tokenizer
            encoded = huggingface_tokenizer(batch_inputs,
                                          return_tensors="np",
                                          padding=True,
                                          truncation=True,
                                          max_length=config.max_seq_length)
        else:
            # If it's already a HuggingFace tokenizer, use it directly
            encoded = tokenizer.encode(batch_inputs,
                                     padding=True,
                                     truncation=True,
                                     max_length=config.max_seq_length,
                                     return_tensors="np")
        
        # Convert numpy arrays to MLX arrays
        batch = {
            "input_ids": mx.array(encoded["input_ids"]),
            "attention_mask": mx.array(encoded["attention_mask"]),
            "labels": mx.array(encoded["input_ids"])  # For causal LM, labels are the same as inputs
        }
        
        # Calculate loss
        loss = eval_loss_fn(model, batch)
        total_loss += loss.item() * len(batch_examples)
    
    return total_loss / len(eval_examples)

def save_model(model, tokenizer, path, config):
    """Save the model and tokenizer."""
    os.makedirs(path, exist_ok=True)
    
    try:
        # Save model weights
        logger.info(f"Saving model weights to {path}")
        weights_path = os.path.join(path, "weights.safetensors")
        mx.save(weights_path, model.parameters())
        
        # Save model configuration
        config_path = os.path.join(path, "config.json")
        with open(config_path, "w") as f:
            json.dump({
                "model_name": config.model_name_or_path,
                "max_seq_length": config.max_seq_length,
                "model_dtype": config.model_dtype,
                "training_config": {
                    "learning_rate": config.learning_rate,
                    "weight_decay": config.weight_decay,
                    "warmup_steps": config.warmup_steps,
                    "max_grad_norm": config.max_grad_norm
                }
            }, f, indent=2)
        
        # Save tokenizer if it has a save_pretrained method
        if hasattr(tokenizer, "save_pretrained"):
            logger.info(f"Saving tokenizer to {path}")
            tokenizer.save_pretrained(path)
        elif hasattr(tokenizer, "_tokenizer") and hasattr(tokenizer._tokenizer, "save_pretrained"):
            logger.info(f"Saving wrapped tokenizer to {path}")
            tokenizer._tokenizer.save_pretrained(path)
        else:
            logger.warning("Could not save tokenizer - no save_pretrained method found")
        
        logger.info(f"Successfully saved model and configuration to {path}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        logger.error(traceback.format_exc())
        raise

def train_grpo(model, tokenizer, dataset, reward_fn, config: MLXGRPOConfig):
    """Train the model using Group Policy Reinforcement Optimization (GRPO)."""
    logger.info("Starting GRPO training phase")
    
    # Create the optimizer
    optimizer = optim.AdamW(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Process the dataset
    prompts = []
    for example in dataset:
        formatted = format_prompt(example, tokenizer)
        prompts.append({
            "prompt": formatted["prompt"],
            "example": example  # Keep the full example for reward calculation
        })
    
    # Create directories for saving models
    os.makedirs(config.output_dir, exist_ok=True)
    rewards_log_path = os.path.join(config.output_dir, "rewards.txt")
    
    # Initialize logs
    with open(rewards_log_path, "w") as f:
        f.write("step,avg_reward,kl_divergence\n")
    
    # Main training loop
    best_reward = -float('inf')
    no_improvement_steps = 0
    
    # For MLX models, we access and modify the parameters directly
    model_params = model.parameters()
    
    # Use the MLX-LM generate function - no need to re-implement it
    def sample_responses(batch_prompts, num_samples=config.generations_per_prompt):
        """Sample responses from the model for a batch of prompts."""
        all_responses = []
        
        for prompt_info in batch_prompts:
            prompt = prompt_info["prompt"]
            
            # Tokenize the prompt
            input_ids = tokenizer.encode(prompt)
            
            # Generate multiple responses for this prompt
            batch_responses = []
            for _ in range(num_samples):
                output = generate(
                    model, 
                    tokenizer,
                    prompt=prompt,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    max_tokens=config.max_seq_length,
                    verbose=False
                )
                
                # Add the generated text to our responses
                batch_responses.append(output)
                
            all_responses.append(batch_responses)
        
        return all_responses
    
    def grpo_loss_fn(params, batch, batch_rewards, kl_penalty=config.kl_penalty):
        """
        GRPO loss function that scales the cross-entropy loss by the calculated rewards
        and adds an optional KL penalty to keep the model close to the reference model.
        """
        # Forward pass with the model parameters
        logits = model.forward(batch["input_ids"], params=params)
        
        # Compute regular cross-entropy loss
        shift_logits = logits[..., :-1, :]
        shift_labels = batch["input_ids"][..., 1:]
        
        # Get per-token cross entropy loss
        ce_loss = nn.losses.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            reduction="none"
        ).reshape(shift_labels.shape)
        
        # Scale the loss by the rewards
        if batch_rewards is not None:
            # Expand rewards to match the token dimensions
            token_rewards = batch_rewards[:, None].repeat(1, ce_loss.shape[1])
            
            # Apply rewards scaling to the loss
            scaled_loss = ce_loss * token_rewards
        else:
            scaled_loss = ce_loss
        
        # Average the loss
        loss = mx.mean(scaled_loss)
        
        # TODO: Add optional KL penalty if needed
        
        return loss
    
    # Compile the training step function
    @mx.compile
    def train_step(params, batch, batch_rewards, opt_state):
        """Single training step with gradient update."""
        loss_val, grads = mx.value_and_grad(grpo_loss_fn)(params, batch, batch_rewards)
        
        # Clip gradients if needed
        if config.max_grad_norm > 0:
            norm = mx.sqrt(sum(mx.sum(g ** 2) for g in grads.values()))
            scale = mx.minimum(mx.array(1.0), config.max_grad_norm / (norm + 1e-8))
            grads = mx.tree_map(lambda g: g * scale, grads)
        
        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = mx.tree_map(lambda p, u: p + u, params, updates)
        
        return params, opt_state, loss_val
    
    # Initialize optimizer state
    opt_state = optimizer.init(model_params)
    
    # Training loop
    logger.info(f"Starting GRPO training for {config.min_training_steps} steps")
    
    for step in range(config.min_training_steps):
        # Sample batch of prompts
        batch_indices = np.random.choice(
            len(prompts),
            min(config.per_device_train_batch_size, len(prompts)),
            replace=False
        )
        batch_prompts = [prompts[i] for i in batch_indices]
        
        # Generate multiple responses for each prompt
        all_responses = sample_responses(batch_prompts)
        
        # Calculate rewards for all responses
        all_rewards = []
        for i, responses in enumerate(all_responses):
            example = batch_prompts[i]["example"]
            prompt = batch_prompts[i]["prompt"]
            
            # Get rewards for this group of responses
            example_rewards = calculate_rewards([responses], reward_fn, config)[0]
            all_rewards.append(example_rewards)
            
            # Log detailed rewards if enabled
            if config.log_reward_components and step % config.log_interval_steps == 0 and i == 0:
                detailed_rewards = reward_fn.get_detailed_rewards(example, prompt, responses[0])
                logger.info(f"Reward components: {detailed_rewards}")
        
        # Flatten the responses and rewards for batch processing
        flat_responses = []
        flat_rewards = []
        
        for i, (responses, rewards) in enumerate(zip(all_responses, all_rewards)):
            # For each prompt, use the top-k responses based on rewards
            sorted_indices = np.argsort(rewards)[::-1]  # Sort in descending order
            top_indices = sorted_indices[:config.group_size]
            
            for idx in top_indices:
                flat_responses.append(responses[idx])
                flat_rewards.append(rewards[idx])
        
        # Create input tensors for training
        input_ids = []
        for resp in flat_responses:
            tokens = tokenizer.encode(resp)
            input_ids.append(tokens)
        
        # Create the batch dictionary
        max_len = max(len(ids) for ids in input_ids)
        padded_ids = [ids + [tokenizer.pad_token_id] * (max_len - len(ids)) for ids in input_ids]
        
        batch_dict = {
            "input_ids": mx.array(padded_ids),
            "rewards": mx.array(flat_rewards)
        }
        
        # Training step
        model_params, opt_state, loss = train_step(
            model_params, 
            batch_dict, 
            batch_dict["rewards"], 
            opt_state
        )
        
        # Update model parameters
        model.update_parameters(model_params)
        
        # Log progress
        if (step + 1) % config.log_interval_steps == 0:
            avg_reward = np.mean(flat_rewards)
            logger.info(f"GRPO Step {step+1}/{config.min_training_steps}, " 
                        f"Loss: {loss.item():.6f}, Avg Reward: {avg_reward:.6f}")
            
            # Log rewards
            with open(rewards_log_path, "a") as f:
                f.write(f"{step+1},{avg_reward:.6f},0.0\n")  # Using 0.0 as placeholder for KL
            
            # Check for improvement
            if avg_reward > best_reward:
                best_reward = avg_reward
                no_improvement_steps = 0
                
                # Save best model
                best_model_path = os.path.join(config.output_dir, "best_model")
                os.makedirs(best_model_path, exist_ok=True)
                mx.save(os.path.join(best_model_path, "weights.safetensors"), model_params)
                logger.info(f"New best model with reward {best_reward:.6f} saved to {best_model_path}")
            else:
                no_improvement_steps += 1
        
        # Save checkpoint
        if (step + 1) % config.save_interval_steps == 0:
            checkpoint_dir = os.path.join(config.output_dir, f"checkpoint_{step+1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            mx.save(os.path.join(checkpoint_dir, "weights.safetensors"), model_params)
            logger.info(f"Saved checkpoint to {checkpoint_dir}")
        
        # Early stopping
        if no_improvement_steps >= config.early_stopping_patience:
            logger.info(f"No improvement for {config.early_stopping_patience} steps. Stopping early.")
            break
    
    # Save final model
    final_model_path = os.path.join(config.output_dir, "final_model")
    os.makedirs(final_model_path, exist_ok=True)
    mx.save(os.path.join(final_model_path, "weights.safetensors"), model_params)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Load the best model if it exists
    best_model_path = os.path.join(config.output_dir, "best_model", "weights.safetensors")
    if os.path.exists(best_model_path):
        logger.info(f"Loading best model from {best_model_path}")
        best_params = mx.load(best_model_path)
        model.update_parameters(best_params)
    
    logger.info("GRPO training completed")
    return model

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    logger.info(f"Loading configuration from {config_path}")
    
    if not os.path.exists(config_path):
        logger.error(f"Configuration file {config_path} not found")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    return yaml_config

def main():
    parser = argparse.ArgumentParser(description="Train NEAR Cortex-1 with MLX on Apple Silicon")
    parser.add_argument("--config", type=str, default="configs/mlx_grpo_config.yaml", 
                        help="Path to configuration file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--dataset-size", type=str, choices=["small", "medium", "large"], 
                        default="small", help="Specify dataset size")
    parser.add_argument("--skip-sft", action="store_true", help="Skip the SFT phase")
    parser.add_argument("--hf-token", type=str, help="Hugging Face API token (or set HUGGINGFACE_TOKEN env var)")
    parser.add_argument("--output-dir", type=str, help="Override output directory in config")
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("mlx_lm").setLevel(logging.INFO)
    
    # Log training information
    logger.info(f"Starting training with dataset size: {args.dataset_size}")
    logger.info(f"SFT phase will be {'skipped' if args.skip_sft else 'included'}")
    logger.info(f"Using config file: {args.config}")
    
    # Setup Hugging Face authentication
    if args.hf_token:
        os.environ["HUGGINGFACE_TOKEN"] = args.hf_token
    setup_huggingface_auth()
    
    try:
        # Load configuration
        config_dict = load_config(args.config)
        config = MLXGRPOConfig.from_dict(config_dict)
        
        # Override output directory if specified
        if args.output_dir:
            config.output_dir = args.output_dir
        
        # Ensure output directory exists with a timestamp to prevent overwriting
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.output_dir = os.path.join(config.output_dir, f"run_{timestamp}")
        os.makedirs(config.output_dir, exist_ok=True)
        logger.info(f"Output will be saved to {config.output_dir}")
        
        # Save the configuration for reproducibility
        with open(os.path.join(config.output_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
        
        # Set random seed
        set_seed(config.seed)
        
        # Load the model and tokenizer
        model, tokenizer = load_model_and_tokenizer(config)
        
        # Prepare dataset
        dataset = prepare_dataset(config, args.dataset_size)
        if dataset is None:
            logger.error("Failed to load dataset. Exiting.")
            sys.exit(1)
        
        # Get the reward function
        reward_fn = get_default_financial_reward()
        
        # Training phases
        if not args.skip_sft:
            logger.info("Starting SFT phase...")
            model = train_sft(model, tokenizer, dataset, config)
        
        logger.info("Starting GRPO phase...")
        model = train_grpo(model, tokenizer, dataset, reward_fn, config)
        
        logger.info(f"Training completed successfully! Model saved to {config.output_dir}")
    
    except Exception as e:
        logger.error(f"Error during training: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 