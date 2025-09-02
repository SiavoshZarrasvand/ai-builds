#!/usr/bin/env python3
"""
Configuration management for Gemma-270M

This module provides dataclasses for managing model, training, and data configurations
with validation, serialization, and default parameter handling.
"""

import torch
import json
import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any, Union


@dataclass
class ModelConfig:
    """Configuration for Gemma-270M model architecture"""
    
    # Model architecture
    vocab_size: int = 50257
    context_length: int = 32768  
    emb_dim: int = 896  # Increased for true 270M
    n_heads: int = 8    # Increased number of heads
    n_layers: int = 22  # More layers for 270M
    hidden_dim: int = 3584  # Larger MLP
    head_dim: int = 112  # emb_dim // n_heads
    qk_norm: bool = True
    n_kv_groups: int = 1
    
    # RoPE configuration
    rope_local_base: float = 10000.0
    rope_base: float = 1000000.0
    sliding_window: int = 512
    
    # Layer configuration
    layer_types: List[str] = field(default_factory=lambda: [
        "sliding_attention", "sliding_attention", "sliding_attention",
        "sliding_attention", "sliding_attention", "full_attention",
        "sliding_attention", "sliding_attention", "sliding_attention",
        "sliding_attention", "sliding_attention", "full_attention", 
        "sliding_attention", "sliding_attention", "sliding_attention",
        "sliding_attention", "sliding_attention", "full_attention",
        "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"
    ])
    
    # Data type and attention scaling
    dtype: torch.dtype = torch.float32
    query_pre_attn_scalar: int = 256
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if len(self.layer_types) != self.n_layers:
            raise ValueError(f"layer_types length ({len(self.layer_types)}) must match n_layers ({self.n_layers})")
        
        if self.emb_dim % self.n_heads != 0:
            raise ValueError(f"emb_dim ({self.emb_dim}) must be divisible by n_heads ({self.n_heads})")
        
        if self.n_heads % self.n_kv_groups != 0:
            raise ValueError(f"n_heads ({self.n_heads}) must be divisible by n_kv_groups ({self.n_kv_groups})")
        
        valid_layer_types = {"sliding_attention", "full_attention"}
        invalid_types = set(self.layer_types) - valid_layer_types
        if invalid_types:
            raise ValueError(f"Invalid layer types: {invalid_types}. Must be one of {valid_layer_types}")


@dataclass 
class TrainingConfig:
    """Configuration for training parameters"""
    
    # Learning rate and schedule
    learning_rate: float = 1e-4
    min_lr: float = 5e-5
    warmup_steps: int = 1000
    max_iters: int = 150000
    
    # Batch and sequence parameters (optimized for longer context)
    batch_size: int = 16
    block_size: int = 384
    gradient_accumulation_steps: int = 64
    
    # Evaluation and checkpointing
    eval_iters: int = 500
    eval_interval: int = 500
    save_interval: int = 1000
    
    # Optimization
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-9
    grad_clip: float = 1.0
    
    # Mixed precision and device
    dtype: str = "float32"  # "float32", "bfloat16", "float16"
    device: str = "cuda"
    compile_model: bool = False
    
    # Checkpointing and logging
    output_dir: str = "checkpoints"
    log_interval: int = 10
    save_best_only: bool = True
    early_stopping_patience: Optional[int] = None
    
    # Resume training
    resume_from: Optional[str] = None
    
    def __post_init__(self):
        """Validate training configuration"""
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        
        if self.min_lr < 0 or self.min_lr > self.learning_rate:
            raise ValueError("min_lr must be between 0 and learning_rate")
        
        if self.batch_size <= 0 or self.gradient_accumulation_steps <= 0:
            raise ValueError("batch_size and gradient_accumulation_steps must be positive")
        
        if self.dtype not in ["float32", "bfloat16", "float16"]:
            raise ValueError(f"dtype must be one of ['float32', 'bfloat16', 'float16'], got {self.dtype}")
        
        if self.device not in ["cuda", "cpu"]:
            raise ValueError(f"device must be 'cuda' or 'cpu', got {self.device}")
    
    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size including gradient accumulation"""
        return self.batch_size * self.gradient_accumulation_steps
    
    @property
    def total_tokens(self) -> int:
        """Calculate total tokens processed during training"""
        return self.max_iters * self.effective_batch_size * self.block_size


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing"""
    
    # Dataset parameters
    dataset_name: str = "roneneldan/TinyStories"
    tokenizer_name: str = "gpt2"
    
    # Data processing
    num_proc: int = 8
    force_reprocess: bool = False
    
    # File paths
    train_path: str = "train.bin"
    val_path: str = "validation.bin"
    data_dir: str = "."
    
    # Data loading
    pin_memory: bool = True
    num_workers: int = 0
    
    def __post_init__(self):
        """Validate data configuration"""
        if self.num_proc <= 0:
            raise ValueError("num_proc must be positive")
        
        # Convert relative paths to absolute based on data_dir
        data_dir = Path(self.data_dir)
        self.train_path = str(data_dir / Path(self.train_path).name)
        self.val_path = str(data_dir / Path(self.val_path).name)


@dataclass
class ExperimentConfig:
    """Complete configuration for a training experiment"""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Experiment metadata
    name: str = "gemma-270m-experiment"
    description: str = "Gemma-270M training experiment"
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate and sync configurations"""
        # Ensure consistent block_size between training and model
        if hasattr(self.training, 'block_size') and hasattr(self.model, 'context_length'):
            if self.training.block_size > self.model.context_length:
                raise ValueError(
                    f"training.block_size ({self.training.block_size}) "
                    f"cannot exceed model.context_length ({self.model.context_length})"
                )
    
    def save(self, path: Union[str, Path], format: str = "yaml") -> None:
        """Save configuration to file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict for serialization
        config_dict = self.to_dict()
        
        if format.lower() == "yaml":
            with open(path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif format.lower() == "json":
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'yaml' or 'json'")
    
    @classmethod
    def load(cls, path: Union[str, Path], format: Optional[str] = None) -> "ExperimentConfig":
        """Load configuration from file"""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file {path} not found")
        
        # Auto-detect format if not specified
        if format is None:
            format = path.suffix.lower().lstrip('.')
            if format not in ["yaml", "yml", "json"]:
                raise ValueError(f"Cannot auto-detect format from extension: {path.suffix}")
        
        if format in ["yaml", "yml"]:
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif format == "json":
            with open(path, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        config_dict = asdict(self)
        
        # Handle torch.dtype serialization
        if 'dtype' in config_dict['model']:
            config_dict['model']['dtype'] = str(config_dict['model']['dtype'])
        
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ExperimentConfig":
        """Create configuration from dictionary"""
        # Handle torch.dtype deserialization
        if 'model' in config_dict and 'dtype' in config_dict['model']:
            dtype_str = config_dict['model']['dtype']
            if isinstance(dtype_str, str):
                dtype_map = {
                    'torch.float32': torch.float32,
                    'torch.bfloat16': torch.bfloat16,
                    'torch.float16': torch.float16,
                    'float32': torch.float32,
                    'bfloat16': torch.bfloat16,
                    'float16': torch.float16,
                }
                config_dict['model']['dtype'] = dtype_map.get(dtype_str, torch.bfloat16)
        
        # Create nested configs
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        
        # Create experiment config
        exp_dict = {k: v for k, v in config_dict.items() if k not in ['model', 'training', 'data']}
        
        return cls(
            model=model_config,
            training=training_config,
            data=data_config,
            **exp_dict
        )


def get_default_config() -> ExperimentConfig:
    """Get default experiment configuration"""
    return ExperimentConfig(
        name="gemma-270m-default",
        description="Default Gemma-270M training configuration",
        tags=["gemma-270m", "transformer", "language-model"]
    )


def get_small_config() -> ExperimentConfig:
    """Get small configuration for quick testing"""
    config = get_default_config()
    config.name = "gemma-270m-small"
    config.description = "Small Gemma-270M config for quick testing"
    
    # Smaller model
    config.model.n_layers = 6
    config.model.emb_dim = 256
    config.model.hidden_dim = 1024
    config.model.head_dim = 128
    config.model.layer_types = [
        "sliding_attention", "sliding_attention", "full_attention",
        "sliding_attention", "sliding_attention", "full_attention"
    ]
    
    # Faster training
    config.training.max_iters = 5000
    config.training.batch_size = 16
    config.training.block_size = 64
    config.training.gradient_accumulation_steps = 8
    config.training.eval_interval = 100
    config.training.save_interval = 500
    
    config.tags = ["gemma-270m", "small", "test"]
    return config


if __name__ == "__main__":
    # Test configuration system
    print("Testing configuration system...")
    
    # Test default config
    config = get_default_config()
    print(f"✅ Default config created: {config.name}")
    print(f"   Model: {config.model.n_layers} layers, {config.model.emb_dim} dim")
    print(f"   Training: {config.training.max_iters} iters, batch={config.training.batch_size}")
    print(f"   Effective batch size: {config.training.effective_batch_size}")
    print(f"   Total tokens: {config.training.total_tokens:,}")
    
    # Test small config  
    small_config = get_small_config()
    print(f"✅ Small config created: {small_config.name}")
    
    # Test serialization
    config.save("test_config.yaml", format="yaml")
    config.save("test_config.json", format="json")
    print("✅ Configs saved to test_config.yaml and test_config.json")
    
    # Test loading
    loaded_yaml = ExperimentConfig.load("test_config.yaml")
    loaded_json = ExperimentConfig.load("test_config.json")
    print("✅ Configs loaded successfully")
    
    # Cleanup
    import os
    os.remove("test_config.yaml")
    os.remove("test_config.json")
    print("✅ Test files cleaned up")
    
    print("✅ Configuration system tests passed!")
