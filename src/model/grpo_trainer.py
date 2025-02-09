"""
GRPO (Group Policy Optimization) trainer for fine-tuning LLMs on crypto analysis.
"""

import os
import json
import torch
import wandb
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from unsloth import FastLanguageModel
from unsloth.grpo import GRPOConfig, GRPOTrainer
from peft import LoraConfig
import yaml

@dataclass
class GRPOTrainingConfig:
    """Configuration for GRPO training."""
    model_name: str
    tokenizer_name: str
    output_dir: str
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    max_grad_norm: float
    weight_decay: float
    warmup_steps: int
    logging_steps: int
    save_steps: int
    eval_steps: int
    num_candidates: int
    kl_penalty: float
    reward_scale: float
    max_length: int
    load_in_4bit: bool
    use_flash_attention: bool

class CryptoGRPOTrainer:
    """Trainer for fine-tuning LLMs using GRPO."""
    
    def __init__(self, config_path: str):
        """Initialize the GRPO trainer.
        
        Args:
            config_path: Path to YAML config file
        """
        with open(config_path) as f:
            config = yaml.safe_load(f)
            
        self.config = GRPOTrainingConfig(**config['training']['grpo'])
        self.model_config = config['model']
        self.distributed_config = config['distributed']
        
        # Initialize wandb
        if 'wandb' in config['output']:
            wandb.init(
                project=config['output']['wandb']['project'],
                entity=config['output']['wandb']['entity'],
                tags=config['output']['wandb']['tags']
            )
            
        # Setup model and tokenizer
        self._setup_model_and_tokenizer()
        
    def _setup_model_and_tokenizer(self):
        """Initialize the model and tokenizer with Unsloth optimizations."""
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_name,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize model with Unsloth optimizations
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_length,
            load_in_4bit=self.config.load_in_4bit,
            use_flash_attention=self.config.use_flash_attention
        )
        
        # Configure LoRA if using parameter-efficient fine-tuning
        if self.distributed_config.get('use_lora', False):
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = FastLanguageModel.get_peft_model(model, lora_config)
            
        self.model = model
        
    def _create_training_args(self) -> TrainingArguments:
        """Create training arguments for the GRPO trainer."""
        return TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            max_grad_norm=self.config.max_grad_norm,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps",
            logging_strategy="steps",
            save_strategy="steps",
            fp16=True,
            gradient_checkpointing=self.distributed_config['gradient_checkpointing'],
            report_to="wandb" if wandb.run is not None else None,
            deepspeed=self.distributed_config['strategy'] if self.distributed_config['strategy'].startswith('deepspeed') else None
        )
        
    def _create_grpo_config(self) -> GRPOConfig:
        """Create configuration for GRPO training."""
        return GRPOConfig(
            num_candidates=self.config.num_candidates,
            kl_penalty=self.config.kl_penalty,
            reward_scale=self.config.reward_scale
        )
        
    def train(self, 
              train_dataset: Union[str, List[Dict]],
              eval_dataset: Optional[Union[str, List[Dict]]] = None):
        """Train the model using GRPO.
        
        Args:
            train_dataset: Path to JSONL dataset or list of examples
            eval_dataset: Optional evaluation dataset
        """
        # Load datasets
        if isinstance(train_dataset, str):
            with open(train_dataset) as f:
                train_data = [json.loads(line) for line in f]
        else:
            train_data = train_dataset
            
        if isinstance(eval_dataset, str):
            with open(eval_dataset) as f:
                eval_data = [json.loads(line) for line in f]
        else:
            eval_data = eval_dataset
            
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Initialize GRPO trainer
        training_args = self._create_training_args()
        grpo_config = self._create_grpo_config()
        
        trainer = GRPOTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            grpo_config=grpo_config
        )
        
        # Train the model
        trainer.train()
        
        # Save the final model
        if trainer.is_world_process_zero():
            trainer.save_model(self.config.output_dir)
            
    def generate_analysis(self, 
                         prompt: str,
                         max_length: Optional[int] = None,
                         num_return_sequences: int = 1,
                         temperature: float = 0.7) -> List[str]:
        """Generate crypto market analysis using the fine-tuned model.
        
        Args:
            prompt: Input prompt for analysis
            max_length: Maximum length of generated text
            num_return_sequences: Number of analyses to generate
            temperature: Sampling temperature
            
        Returns:
            List of generated analyses
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        outputs = self.model.generate(
            **inputs,
            max_length=max_length or self.config.max_length,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        return [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ] 