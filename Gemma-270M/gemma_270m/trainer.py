#!/usr/bin/env python3
"""
Training pipeline for Gemma-270M

This module provides a comprehensive training pipeline with:
- Training loop with gradient accumulation
- Evaluation and validation
- Checkpointing and model saving
- Learning rate scheduling
- Logging and monitoring
- Mixed precision training
- Gradient clipping
"""

import os
import json
import time
import math
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

from .model import GemmaModel
from .data import GemmaDataLoader
from .config import ExperimentConfig, TrainingConfig, ModelConfig, DataConfig


class GemmaTrainer:
    """Training pipeline for Gemma-270M model"""
    
    def __init__(
        self,
        config: ExperimentConfig,
        model: Optional[GemmaModel] = None,
        data_loader: Optional[GemmaDataLoader] = None,
        resume_from: Optional[str] = None
    ):
        """Initialize trainer
        
        Args:
            config: Complete experiment configuration
            model: Pre-initialized model (optional)
            data_loader: Pre-initialized data loader (optional)
            resume_from: Path to checkpoint to resume from (optional)
        """
        self.config = config
        self.device = torch.device(config.training.device)
        
        # Training state
        self.current_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = []
        self.val_history = []
        
        # Initialize model
        if model is None:
            self.model = GemmaModel(config.model)
        else:
            self.model = model
            
        # Move model to device and set to training mode
        self.model = self.model.to(self.device)
        self.model.train()
        
        # Initialize data loader
        if data_loader is None:
            self.data_loader = GemmaDataLoader(config.data, config.training.block_size)
        else:
            self.data_loader = data_loader
            
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize mixed precision scaler if using mixed precision
        self.scaler = GradScaler() if config.training.dtype in ["float16", "bfloat16"] else None
        
        # Setup output directory
        self.output_dir = Path(config.training.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.log_file = self.output_dir / "training.log"
        self.metrics_file = self.output_dir / "metrics.json"
        
        # Resume from checkpoint if specified
        if resume_from:
            self.resume_from_checkpoint(resume_from)
        elif config.training.resume_from:
            self.resume_from_checkpoint(config.training.resume_from)
            
        # Log initialization
        self.log_info("Trainer initialized successfully")
        self.log_info(f"Model: {self.get_model_info()}")
        self.log_info(f"Training config: {self.get_training_info()}")
    
    def _create_optimizer(self) -> AdamW:
        """Create AdamW optimizer with weight decay"""
        # Separate parameters with and without weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Don't apply weight decay to biases and layer norm parameters
                if 'bias' in name or 'norm' in name or 'ln' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        param_groups = [
            {'params': decay_params, 'weight_decay': self.config.training.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        optimizer = AdamW(
            param_groups,
            lr=self.config.training.learning_rate,
            betas=(self.config.training.beta1, self.config.training.beta2),
            eps=self.config.training.eps
        )
        
        return optimizer
    
    def _create_scheduler(self) -> CosineAnnealingLR:
        """Create cosine annealing learning rate scheduler"""
        return CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.training.max_iters,
            eta_min=self.config.training.min_lr
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'layers': self.config.model.n_layers,
            'embedding_dim': self.config.model.emb_dim,
            'hidden_dim': self.config.model.hidden_dim,
            'heads': self.config.model.n_heads,
            'vocab_size': self.config.model.vocab_size
        }
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get training configuration information"""
        return {
            'batch_size': self.config.training.batch_size,
            'block_size': self.config.training.block_size,
            'gradient_accumulation_steps': self.config.training.gradient_accumulation_steps,
            'effective_batch_size': self.config.training.effective_batch_size,
            'max_iters': self.config.training.max_iters,
            'learning_rate': self.config.training.learning_rate,
            'min_lr': self.config.training.min_lr,
            'weight_decay': self.config.training.weight_decay,
            'warmup_steps': self.config.training.warmup_steps,
            'device': str(self.device),
            'mixed_precision': self.config.training.dtype != "float32"
        }
    
    def log_info(self, message: str):
        """Log information message"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        # Write to log file
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
    
    def save_metrics(self):
        """Save training metrics to JSON file"""
        metrics = {
            'training_history': self.training_history,
            'validation_history': self.val_history,
            'current_step': self.current_step,
            'current_epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'model_info': self.get_model_info(),
            'training_info': self.get_training_info()
        }
        
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def save_checkpoint(self, is_best: bool = False, step: Optional[int] = None):
        """Save model checkpoint
        
        Args:
            is_best: Whether this is the best model so far
            step: Step number (uses current_step if None)
        """
        if step is None:
            step = self.current_step
            
        checkpoint = {
            'step': step,
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': asdict(self.config),
            'training_history': self.training_history,
            'val_history': self.val_history
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.log_info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model if this is the best one
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.log_info(f"Saved best model: {best_path}")
        
        # Save latest checkpoint
        latest_path = self.output_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)
        
        # Clean up old checkpoints (keep only last 3)
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Keep only the 3 most recent checkpoints"""
        checkpoint_files = list(self.output_dir.glob("checkpoint_step_*.pt"))
        if len(checkpoint_files) > 3:
            # Sort by step number
            checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
            # Remove oldest checkpoints
            for old_checkpoint in checkpoint_files[:-3]:
                old_checkpoint.unlink()
    
    def resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self.log_info(f"Resuming from checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state if using mixed precision
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load training state
        self.current_step = checkpoint['step']
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint.get('training_history', [])
        self.val_history = checkpoint.get('val_history', [])
        
        self.log_info(f"Resumed training from step {self.current_step}")
    
    def estimate_loss(self, eval_iters: Optional[int] = None) -> Dict[str, float]:
        """Evaluate model on train and validation sets
        
        Args:
            eval_iters: Number of iterations for evaluation
            
        Returns:
            Dictionary with train and validation losses
        """
        if eval_iters is None:
            eval_iters = self.config.training.eval_iters
        
        self.model.eval()
        losses = {}
        
        for split in ['train', 'val']:
            total_loss = 0.0
            num_batches = 0
            
            with torch.no_grad():
                for i in range(eval_iters):
                    try:
                        if split == 'train':
                            batch = self.data_loader.get_batch('train', self.config.training.batch_size)
                        else:
                            batch = self.data_loader.get_batch('val', self.config.training.batch_size)
                        
                        input_ids, targets = batch
                        input_ids = input_ids.to(self.device)
                        targets = targets.to(self.device)
                        
                        # Forward pass
                        with autocast(enabled=self.scaler is not None):
                            logits, loss = self.model(input_ids, targets)
                        
                        total_loss += loss.item()
                        num_batches += 1
                        
                    except Exception as e:
                        self.log_info(f"Error in evaluation batch {i}: {e}")
                        continue
            
            if num_batches > 0:
                losses[split] = total_loss / num_batches
            else:
                losses[split] = float('inf')
        
        self.model.train()
        return losses
    
    def train_step(self) -> float:
        """Perform one training step with gradient accumulation
        
        Returns:
            Loss for this step
        """
        total_loss = 0.0
        self.optimizer.zero_grad()
        
        for micro_step in range(self.config.training.gradient_accumulation_steps):
            try:
                # Get batch
                batch = self.data_loader.get_batch('train', self.config.training.batch_size)
                input_ids, targets = batch
                input_ids = input_ids.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass with mixed precision
                with autocast(enabled=self.scaler is not None):
                    logits, loss = self.model(input_ids, targets)
                    loss = loss / self.config.training.gradient_accumulation_steps  # Scale loss
                
                # Backward pass
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                total_loss += loss.item()
                
            except Exception as e:
                self.log_info(f"Error in training micro-step {micro_step}: {e}")
                continue
        
        # Clip gradients
        if self.config.training.grad_clip > 0:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip)
        
        # Optimizer step
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        # Scheduler step
        self.scheduler.step()
        
        return total_loss * self.config.training.gradient_accumulation_steps
    
    def train(self):
        """Main training loop"""
        self.log_info("Starting training...")
        self.log_info(f"Training for {self.config.training.max_iters} steps")
        
        start_time = time.time()
        
        while self.current_step < self.config.training.max_iters:
            # Training step
            step_start_time = time.time()
            train_loss = self.train_step()
            step_time = time.time() - step_start_time
            
            self.current_step += 1
            
            # Log training progress
            if self.current_step % self.config.training.log_interval == 0:
                lr = self.scheduler.get_last_lr()[0]
                tokens_per_sec = (self.config.training.effective_batch_size * 
                                self.config.training.block_size) / step_time
                
                self.log_info(
                    f"Step {self.current_step:6d} | "
                    f"Loss: {train_loss:.4f} | "
                    f"LR: {lr:.2e} | "
                    f"Tokens/s: {tokens_per_sec:.0f} | "
                    f"Time: {step_time:.2f}s"
                )
                
                # Store training history
                self.training_history.append({
                    'step': self.current_step,
                    'loss': train_loss,
                    'learning_rate': lr,
                    'tokens_per_second': tokens_per_sec,
                    'time': step_time
                })
            
            # Evaluation
            if self.current_step % self.config.training.eval_interval == 0:
                self.log_info("Running evaluation...")
                eval_start_time = time.time()
                
                losses = self.estimate_loss()
                eval_time = time.time() - eval_start_time
                
                train_loss_eval = losses['train']
                val_loss = losses['val']
                
                self.log_info(
                    f"Eval {self.current_step:6d} | "
                    f"Train: {train_loss_eval:.4f} | "
                    f"Val: {val_loss:.4f} | "
                    f"Time: {eval_time:.2f}s"
                )
                
                # Store validation history
                self.val_history.append({
                    'step': self.current_step,
                    'train_loss': train_loss_eval,
                    'val_loss': val_loss,
                    'eval_time': eval_time
                })
                
                # Check if this is the best model
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self.log_info(f"New best validation loss: {val_loss:.4f}")
                
                # Save checkpoint
                if self.config.training.save_best_only:
                    if is_best:
                        self.save_checkpoint(is_best=True)
                else:
                    self.save_checkpoint(is_best=is_best)
                
                # Save metrics
                self.save_metrics()
                
                # Early stopping check
                if (self.config.training.early_stopping_patience and 
                    len(self.val_history) > self.config.training.early_stopping_patience):
                    
                    recent_losses = [h['val_loss'] for h in self.val_history[-self.config.training.early_stopping_patience:]]
                    if all(loss >= self.best_val_loss for loss in recent_losses):
                        self.log_info("Early stopping triggered!")
                        break
            
            # Regular checkpoint saving
            if (self.config.training.save_interval > 0 and 
                self.current_step % self.config.training.save_interval == 0):
                self.save_checkpoint()
                self.save_metrics()
        
        # Final checkpoint and evaluation
        self.log_info("Training completed!")
        total_time = time.time() - start_time
        self.log_info(f"Total training time: {total_time:.2f}s ({total_time/3600:.2f}h)")
        
        # Final evaluation
        self.log_info("Running final evaluation...")
        final_losses = self.estimate_loss()
        self.log_info(f"Final losses - Train: {final_losses['train']:.4f}, Val: {final_losses['val']:.4f}")
        
        # Save final checkpoint
        self.save_checkpoint()
        self.save_metrics()
        
        return {
            'final_train_loss': final_losses['train'],
            'final_val_loss': final_losses['val'],
            'best_val_loss': self.best_val_loss,
            'total_steps': self.current_step,
            'total_time': total_time
        }


def create_trainer_from_config(config_path: str, **kwargs) -> GemmaTrainer:
    """Create trainer from configuration file
    
    Args:
        config_path: Path to configuration file
        **kwargs: Additional arguments to pass to trainer
        
    Returns:
        Initialized GemmaTrainer
    """
    config = ExperimentConfig.load(config_path)
    return GemmaTrainer(config, **kwargs)


if __name__ == "__main__":
    # Example usage
    from .config import get_default_config
    
    # Create default configuration
    config = get_default_config()
    
    # Create trainer
    trainer = GemmaTrainer(config)
    
    # Start training
    results = trainer.train()
    
    print("Training completed!")
    print(f"Final results: {results}")
