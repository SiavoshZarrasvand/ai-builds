#!/usr/bin/env python3
"""
Main training script for Gemma-270M

This script provides a command-line interface for training the Gemma-270M model
with various configuration options and utilities.

Usage:
    python train.py --config configs/optimized_config.yaml
    python train.py --config configs/optimized_config.yaml --resume checkpoints/latest_checkpoint.pt
    python train.py --model.n_layers 12 --training.batch_size 32
"""

import argparse
import sys
import os
from pathlib import Path
import json
import torch
import warnings

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from gemma_270m.config import ExperimentConfig, get_default_config, get_small_config
from gemma_270m.trainer import GemmaTrainer, create_trainer_from_config
from gemma_270m.model import GemmaModel
from gemma_270m.data import GemmaDataLoader


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train Gemma-270M Language Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with default configuration
    python train.py
    
    # Train with custom config file
    python train.py --config configs/optimized_config.yaml
    
    # Resume from checkpoint
    python train.py --config configs/optimized_config.yaml --resume checkpoints/latest_checkpoint.pt
    
    # Override specific parameters
    python train.py --model.n_layers 12 --training.batch_size 32 --training.learning_rate 2e-4
    
    # Use small config for testing
    python train.py --preset small --training.max_iters 1000
    
    # Train on CPU (for testing)
    python train.py --preset small --training.device cpu --training.max_iters 100
        """
    )
    
    # Configuration options
    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file (YAML or JSON)'
    )
    config_group.add_argument(
        '--preset',
        choices=['default', 'small'],
        default='default',
        help='Use preset configuration (default: default)'
    )
    
    # Training options
    training_group = parser.add_argument_group('Training')
    training_group.add_argument(
        '--resume',
        type=str,
        help='Resume training from checkpoint'
    )
    training_group.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for checkpoints and logs'
    )
    training_group.add_argument(
        '--name',
        type=str,
        help='Experiment name'
    )
    
    # Model parameters
    model_group = parser.add_argument_group('Model Parameters')
    model_group.add_argument('--model.n_layers', type=int, help='Number of layers')
    model_group.add_argument('--model.emb_dim', type=int, help='Embedding dimension')
    model_group.add_argument('--model.hidden_dim', type=int, help='Hidden dimension')
    model_group.add_argument('--model.n_heads', type=int, help='Number of attention heads')
    model_group.add_argument('--model.vocab_size', type=int, help='Vocabulary size')
    
    # Training parameters
    training_group.add_argument('--training.batch_size', type=int, help='Batch size')
    training_group.add_argument('--training.block_size', type=int, help='Sequence length')
    training_group.add_argument('--training.learning_rate', type=float, help='Learning rate')
    training_group.add_argument('--training.max_iters', type=int, help='Maximum iterations')
    training_group.add_argument('--training.eval_interval', type=int, help='Evaluation interval')
    training_group.add_argument('--training.save_interval', type=int, help='Save interval')
    training_group.add_argument('--training.device', type=str, help='Device (cuda/cpu)')
    training_group.add_argument('--training.gradient_accumulation_steps', type=int, help='Gradient accumulation steps')
    
    # Data parameters
    data_group = parser.add_argument_group('Data Parameters')
    data_group.add_argument('--data.dataset_name', type=str, help='Dataset name')
    data_group.add_argument('--data.data_dir', type=str, help='Data directory')
    
    # Utility options
    util_group = parser.add_argument_group('Utilities')
    util_group.add_argument(
        '--dry-run',
        action='store_true',
        help='Show configuration without training'
    )
    util_group.add_argument(
        '--validate-config',
        action='store_true',
        help='Validate configuration and exit'
    )
    util_group.add_argument(
        '--print-model',
        action='store_true',
        help='Print model architecture and exit'
    )
    util_group.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with verbose logging'
    )
    
    return parser.parse_args()


def override_config_from_args(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    """Override configuration from command line arguments"""
    
    # Direct overrides
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.name:
        config.name = args.name
    if args.resume:
        config.training.resume_from = args.resume
    
    # Model parameters
    model_overrides = {
        'n_layers': getattr(args, 'model.n_layers'),
        'emb_dim': getattr(args, 'model.emb_dim'),
        'hidden_dim': getattr(args, 'model.hidden_dim'),
        'n_heads': getattr(args, 'model.n_heads'),
        'vocab_size': getattr(args, 'model.vocab_size'),
    }
    
    for key, value in model_overrides.items():
        if value is not None:
            setattr(config.model, key, value)
    
    # Training parameters
    training_overrides = {
        'batch_size': getattr(args, 'training.batch_size'),
        'block_size': getattr(args, 'training.block_size'),
        'learning_rate': getattr(args, 'training.learning_rate'),
        'max_iters': getattr(args, 'training.max_iters'),
        'eval_interval': getattr(args, 'training.eval_interval'),
        'save_interval': getattr(args, 'training.save_interval'),
        'device': getattr(args, 'training.device'),
        'gradient_accumulation_steps': getattr(args, 'training.gradient_accumulation_steps'),
    }
    
    for key, value in training_overrides.items():
        if value is not None:
            setattr(config.training, key, value)
    
    # Data parameters
    data_overrides = {
        'dataset_name': getattr(args, 'data.dataset_name'),
        'data_dir': getattr(args, 'data.data_dir'),
    }
    
    for key, value in data_overrides.items():
        if value is not None:
            setattr(config.data, key, value)
    
    return config


def print_config_summary(config: ExperimentConfig):
    """Print configuration summary"""
    print("\n" + "="*60)
    print("🚀 GEMMA-270M TRAINING CONFIGURATION")
    print("="*60)
    
    print(f"\n📋 Experiment:")
    print(f"   Name: {config.name}")
    print(f"   Description: {config.description}")
    print(f"   Tags: {', '.join(config.tags)}")
    
    print(f"\n🧠 Model Architecture:")
    print(f"   Layers: {config.model.n_layers}")
    print(f"   Embedding Dimension: {config.model.emb_dim}")
    print(f"   Hidden Dimension: {config.model.hidden_dim}")
    print(f"   Attention Heads: {config.model.n_heads}")
    print(f"   Head Dimension: {config.model.head_dim}")
    print(f"   Vocabulary Size: {config.model.vocab_size:,}")
    print(f"   Context Length: {config.model.context_length:,}")
    
    print(f"\n🎯 Training Configuration:")
    print(f"   Batch Size: {config.training.batch_size}")
    print(f"   Sequence Length: {config.training.block_size}")
    print(f"   Gradient Accumulation: {config.training.gradient_accumulation_steps}")
    print(f"   Effective Batch Size: {config.training.effective_batch_size}")
    print(f"   Maximum Iterations: {config.training.max_iters:,}")
    print(f"   Learning Rate: {config.training.learning_rate:.2e}")
    print(f"   Min Learning Rate: {config.training.min_lr:.2e}")
    print(f"   Weight Decay: {config.training.weight_decay}")
    print(f"   Device: {config.training.device}")
    print(f"   Mixed Precision: {config.training.dtype}")
    
    print(f"\n📊 Training Schedule:")
    print(f"   Total Tokens: {config.training.total_tokens:,}")
    print(f"   Evaluation Interval: {config.training.eval_interval}")
    print(f"   Save Interval: {config.training.save_interval}")
    print(f"   Log Interval: {config.training.log_interval}")
    
    print(f"\n💾 Data Configuration:")
    print(f"   Dataset: {config.data.dataset_name}")
    print(f"   Tokenizer: {config.data.tokenizer_name}")
    print(f"   Data Directory: {config.data.data_dir}")
    print(f"   Train Path: {config.data.train_path}")
    print(f"   Val Path: {config.data.val_path}")
    
    print(f"\n📁 Output:")
    print(f"   Output Directory: {config.training.output_dir}")
    if config.training.resume_from:
        print(f"   Resume From: {config.training.resume_from}")
    
    print("="*60)


def print_model_summary(model: GemmaModel):
    """Print model architecture summary"""
    print("\n" + "="*60)
    print("🏗️  MODEL ARCHITECTURE SUMMARY")
    print("="*60)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Parameter Count:")
    print(f"   Total Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"   Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
    print(f"   Model Size: {total_params * 4 / (1024**2):.1f} MB (float32)")
    
    print(f"\n🏗️  Architecture:")
    print(f"   Input: Token Embedding ({model.config.vocab_size} vocab)")
    print(f"   Layers: {len(model.blocks)} Transformer Blocks")
    
    for i, block in enumerate(model.blocks):
        layer_type = model.config.layer_types[i] if i < len(model.config.layer_types) else "unknown"
        print(f"     Block {i+1:2d}: {layer_type}")
    
    print(f"   Output: Final Layer Norm + Linear Head")
    print(f"   Output Shape: (..., {model.config.vocab_size})")
    
    print("="*60)


def validate_config(config: ExperimentConfig) -> bool:
    """Validate configuration and return True if valid"""
    try:
        # Test model creation
        print("🔍 Validating model configuration...")
        model = GemmaModel(config.model)
        print("✅ Model configuration valid")
        
        # Test data loader creation
        print("🔍 Validating data configuration...")
        data_loader = GemmaDataLoader(config.data, config.training.block_size)
        print("✅ Data configuration valid")
        
        # Check device availability
        if config.training.device == "cuda" and not torch.cuda.is_available():
            print("⚠️  Warning: CUDA not available, but device set to 'cuda'")
            return False
        
        # Check memory requirements (basic estimation)
        if config.training.device == "cuda" and torch.cuda.is_available():
            available_memory = torch.cuda.get_device_properties(0).total_memory
            model_params = sum(p.numel() for p in model.parameters())
            estimated_memory = model_params * 4 * 3  # Model + gradients + optimizer states
            
            if estimated_memory > available_memory * 0.8:  # Use 80% as threshold
                print(f"⚠️  Warning: Estimated memory usage ({estimated_memory/1e9:.1f}GB) "
                      f"may exceed available GPU memory ({available_memory/1e9:.1f}GB)")
        
        print("✅ Configuration validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False


def main():
    """Main training function"""
    args = parse_arguments()
    
    print("🤗 Gemma-270M Training Pipeline")
    print("=" * 40)
    
    try:
        # Load configuration
        if args.config:
            print(f"📖 Loading configuration from: {args.config}")
            config = ExperimentConfig.load(args.config)
        elif args.preset == 'small':
            print("📖 Using small preset configuration")
            config = get_small_config()
        else:
            print("📖 Using default configuration")
            config = get_default_config()
        
        # Override with command line arguments
        config = override_config_from_args(config, args)
        
        # Print configuration summary
        if not args.dry_run:
            print_config_summary(config)
        
        # Validate configuration
        if args.validate_config or args.dry_run:
            valid = validate_config(config)
            if args.validate_config:
                sys.exit(0 if valid else 1)
        
        # Print model architecture if requested
        if args.print_model or args.dry_run:
            model = GemmaModel(config.model)
            print_model_summary(model)
            if args.print_model:
                sys.exit(0)
        
        # Exit if dry run
        if args.dry_run:
            print("\n✅ Dry run completed successfully!")
            sys.exit(0)
        
        # Create trainer
        print("\n🏗️  Initializing trainer...")
        trainer = GemmaTrainer(config)
        
        # Start training
        print("\n🚀 Starting training...")
        results = trainer.train()
        
        # Print final results
        print("\n" + "="*60)
        print("🎉 TRAINING COMPLETED!")
        print("="*60)
        print(f"Final Train Loss: {results['final_train_loss']:.4f}")
        print(f"Final Validation Loss: {results['final_val_loss']:.4f}")
        print(f"Best Validation Loss: {results['best_val_loss']:.4f}")
        print(f"Total Steps: {results['total_steps']:,}")
        print(f"Total Time: {results['total_time']:.2f}s ({results['total_time']/3600:.2f}h)")
        print("="*60)
        
        return results
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
