# Gemma-270M Language Model

## ✨ Key Features

## 🚀 Quick Start

**Note**: This project demonstrates how to build a complete training pipeline from scratch. For practical applications, consider using pre-trained models (see Practical Usage section).

**Hardware Requirements**: NVIDIA GPU with 8GB+ VRAM recommended for pipeline testing. High-end cloud GPUs (A100/V100) required for full training.

### Installation

```powershell
git clone https://github.com/SiavoshZarrasvand/ai-builds.git
cd ai-builds/Gemma-270M

# Create virtual environment and install dependencies
uv venv --python 3.11
.venv\Scripts\activate

# Install dependencies with CUDA support (automatic)
uv sync
```

### Training Pipeline (Educational)

```bash
# Quick pipeline test (500 steps, ~10 minutes)
python run_pipeline.py --quick

# Full model training (requires A100/V100, ~17 days)
python run_pipeline.py --config configs/optimized_config.yaml

# Individual steps for learning
python run_pipeline.py --steps clean data  # Just data preparation
python run_pipeline.py --steps train       # Just training
python run_pipeline.py --steps test        # Just testing
```

### Text Generation

```bash
# Generate from prompt
python generate.py --checkpoint checkpoints/best_model.pt --prompt "Once upon a time"

# Interactive generation
python generate.py --checkpoint checkpoints/best_model.pt --interactive

# Chat mode
python generate.py --checkpoint checkpoints/best_model.pt --chat

# Benchmark performance
python generate.py --checkpoint checkpoints/best_model.pt --benchmark
```

## 💻 Programming Interface

### Text Generation

```python
from gemma_270m.inference import create_generator_from_checkpoint
from transformers import GPT2Tokenizer

# Load trained model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
generator = create_generator_from_checkpoint(
    checkpoint_path="checkpoints/best_model.pt",
    tokenizer=tokenizer
)

# Generate text
response = generator.generate_text(
    prompt="The future of artificial intelligence is",
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.9
)
print(response)
```

## 🔧 Configuration Options

### Pre-configured Training Presets

**Optimized Config** (`configs/optimized_config.yaml`) - Recommended:

- Batch size: 16, Sequence length: 384
- Memory usage: ~24% of 8GB GPU
- Best for: Story generation, long-form text, conversational AI

**Conservative Config** (`configs/conservative_config.yaml`) - Fallback:

- Batch size: 16, Sequence length: 256
- Memory usage: ~22% of 8GB GPU
- Best for: Stable training, resource-constrained environments

### Model Configuration

```yaml
model:
  n_layers: 22 # Number of transformer layers
  emb_dim: 896 # Embedding dimension
  hidden_dim: 3584 # Feed-forward dimension
  n_heads: 8 # Attention heads
  vocab_size: 50257 # Vocabulary size
  context_length: 32768 # Maximum sequence length
  sliding_window: 512 # Local attention window size
```

### Training Configuration

```yaml
training:
  batch_size: 16 # Optimized batch size (GPU memory efficient)
  block_size: 384 # Optimized sequence length (3x improvement)
  gradient_accumulation_steps: 64 # Effective batch = 1024
  learning_rate: 1e-4 # Peak learning rate
  max_iters: 150000 # Training iterations
  device: cuda # Training device
  dtype: bfloat16 # Mixed precision for efficiency
```

### Generation Configuration

```yaml
generation:
  max_new_tokens: 100 # Tokens to generate
  temperature: 1.0 # Sampling temperature
  top_k: 50 # Top-k sampling
  top_p: 0.9 # Nucleus sampling
  repetition_penalty: 1.0 # Repetition penalty
```
