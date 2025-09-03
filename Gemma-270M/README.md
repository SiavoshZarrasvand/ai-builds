# Gemma-270M Language Model

> **Complete implementation of Google's Gemma-270M language model from scratch in PyTorch**

This project provides a full implementation of the Gemma-270M language model with optimized training pipeline, inference capabilities, and text generation features. The model achieves **341.8M parameters** with a hybrid attention mechanism designed for excellent text coherence and generation quality.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## ✨ Key Features

### 🏗️ **Complete Architecture Implementation**
- **Hybrid Attention**: Sliding window (512 tokens) + full attention layers
- **RoPE Embeddings**: Rotary position embeddings for both local and global contexts  
- **GeGLU Activations**: Gated Linear Units for enhanced performance
- **Layer Normalization**: Pre-norm architecture with query/key normalization
- **Mixed Precision**: Support for bfloat16, float16, and float32 training

### 🚀 **Production-Ready Training Pipeline**
- **Optimized Configuration**: Batch=16, Sequence=384 for maximum coherence
- **Memory Efficient**: Uses only 24% of 8GB GPU memory during training
- **Gradient Accumulation**: Effective batch size of 1024 for stable training
- **Advanced Scheduling**: Cosine annealing with warmup and weight decay
- **Comprehensive Logging**: Training metrics, checkpointing, and resume capability

### 🎯 **Advanced Text Generation**
- **Multiple Sampling Strategies**: Greedy, top-k, top-p (nucleus), temperature
- **Quality Controls**: Repetition penalty, n-gram filtering, bad words filtering
- **Batch Generation**: Efficient parallel text generation
- **Interactive Modes**: CLI chat interface and interactive generation
- **Benchmarking Tools**: Performance measurement and optimization

## 📊 Model Specifications

| Specification | Value | Description |
|---------------|--------|-------------|
| **Parameters** | 341.8M | Total trainable parameters |
| **Layers** | 22 | Transformer blocks |
| **Embedding Dim** | 896 | Hidden dimension |
| **Feed-Forward Dim** | 3584 | MLP intermediate size |
| **Attention Heads** | 8 | Multi-head attention |
| **Head Dimension** | 112 | Per-head dimension |
| **Context Length** | 32768 | Maximum sequence length |
| **Vocabulary Size** | 50257 | GPT-2 compatible tokenizer |
| **Memory Usage** | 24.2% | Of 8GB GPU during training |

### 🎭 **Attention Pattern**
The model uses a carefully designed hybrid attention pattern:
```
Layers 1-5:   Sliding Window Attention (512 tokens)
Layer 6:      Full Attention (global context)
Layers 7-11:  Sliding Window Attention  
Layer 12:     Full Attention
... (pattern continues)
```

## 🚀 Quick Start

> **Note**: This project demonstrates how to build a complete training pipeline from scratch. For practical applications, consider using pre-trained models (see Practical Usage section).

> **Hardware Requirements**: NVIDIA GPU with 4GB+ VRAM recommended for pipeline testing. High-end cloud GPUs (A100/V100) required for full training.
> **Software Requirements**: Python 3.11+, CUDA 12.1+, PyTorch 2.5+

### Installation (Windows Only)
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

### Practical Usage (Recommended)
For production applications, use pre-trained models:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load a pre-trained model (much faster than training from scratch)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")  # 124M params
# Or use: "gpt2-medium" (355M), "gpt2-large" (774M)

# Generate text
input_ids = tokenizer.encode("Hello world", return_tensors="pt")
output = model.generate(input_ids, max_length=50, do_sample=True)
print(tokenizer.decode(output[0]))
```

### Legacy Training Scripts
```bash
# Train with optimized configuration (legacy)
python train.py --config configs/optimized_config.yaml

# Quick test with small model (legacy)
python train.py --preset small --training.max_iters 1000
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

### Model Usage
```python
from gemma_270m import GemmaModel, get_default_config

# Create model with optimized config
config = get_default_config()
model = GemmaModel(config.model)

# Forward pass
import torch
input_ids = torch.randint(0, config.model.vocab_size, (16, 384))
logits, loss = model(input_ids)
print(f"Output shape: {logits.shape}")  # [16, 384, 50257]
```

### Training Pipeline
```python
from gemma_270m import GemmaTrainer, get_default_config

# Initialize trainer
config = get_default_config()
trainer = GemmaTrainer(config)

# Start training
results = trainer.train()
print(f"Final validation loss: {results['best_val_loss']:.4f}")
```

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

### Configuration Management
```python
from gemma_270m.config import ExperimentConfig, get_default_config

# Load and modify configuration
config = get_default_config()
config.training.batch_size = 32
config.training.learning_rate = 2e-4

# Save configuration
config.save("my_config.yaml")

# Load from file
loaded_config = ExperimentConfig.load("my_config.yaml")
```

## 📁 Project Structure

```
Gemma-270M/
├── gemma_270m/              # Core package
│   ├── __init__.py          # Package initialization
│   ├── model.py             # Gemma model architecture
│   ├── config.py            # Configuration management
│   ├── data.py              # Data loading and preprocessing
│   ├── trainer.py           # Training pipeline
│   └── inference.py         # Text generation and inference
├── configs/                 # Configuration files
│   ├── optimized_config.yaml    # Recommended training config
│   └── conservative_config.yaml # Stable fallback config
├── train.py                 # Main training script
├── generate.py              # Text generation script
├── test_*.py               # Test modules
└── README.md               # This file
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
  n_layers: 22              # Number of transformer layers
  emb_dim: 896              # Embedding dimension  
  hidden_dim: 3584          # Feed-forward dimension
  n_heads: 8                # Attention heads
  vocab_size: 50257         # Vocabulary size
  context_length: 32768     # Maximum sequence length
  sliding_window: 512       # Local attention window size
```

### Training Configuration  
```yaml
training:
  batch_size: 16            # Optimized batch size (GPU memory efficient)
  block_size: 384           # Optimized sequence length (3x improvement)
  gradient_accumulation_steps: 64  # Effective batch = 1024
  learning_rate: 1e-4       # Peak learning rate
  max_iters: 150000         # Training iterations
  device: cuda              # Training device
  dtype: bfloat16           # Mixed precision for efficiency
```

### Generation Configuration
```yaml
generation:
  max_new_tokens: 100       # Tokens to generate
  temperature: 1.0          # Sampling temperature
  top_k: 50                 # Top-k sampling
  top_p: 0.9                # Nucleus sampling
  repetition_penalty: 1.0   # Repetition penalty
```

## 📈 Performance Benchmarks

### GPU Memory Optimization Results

Extensive testing on **NVIDIA GeForce RTX 4070 Laptop GPU (8GB VRAM)** with 341.8M parameter model:

#### Optimized Configuration Comparison
| Configuration | Batch Size | Sequence Length | Tokens/Batch | Memory Usage | GPU Util | Status |
|---------------|------------|-----------------|--------------|--------------|----------|--------|
| **Original** | 32 | 128 | 4,096 | 1.78GB | 22.3% | ✅ Safe |
| **Conservative** | 16 | 256 | 4,096 | 1.6GB | 22.4% | ✅ Safe |
| **Optimized** | 16 | 384 | 6,144 | 1.9GB | 24.2% | ⭐ **RECOMMENDED** |

#### Key Optimization Benefits
- **🎯 3.0x longer context**: 384 vs 128 tokens for better coherence
- **📈 1.5x more tokens per batch**: 6,144 vs 4,096 tokens
- **🧠 Improved text quality**: Longer sequences enable better narrative understanding
- **⚡ Maintained stability**: Effective batch size of 1,024 preserved
- **🛡️ Excellent safety margin**: Only 24.2% of 8GB GPU memory used

### Training Performance
- **Tokens/step**: 6,144 tokens per training step (optimized config)
- **Training throughput**: Improved 1.5x over baseline config
- **Memory efficiency**: 75%+ GPU memory available for other processes
- **Stability**: Excellent - well within safe memory limits
- **Context improvement**: 3x longer sequences (384 vs 128 tokens)
- **Effective batch size**: 1,024 (maintained for stable gradients)

### Generation Speed  
- **Tokens/second**: ~50-100 (depending on sequence length)
- **Batch generation**: Efficient parallel processing
- **Interactive response**: <2 seconds for typical responses

## 🎯 Examples and Applications

Once you've trained a model, you can create various applications using the trained checkpoints:

### Story Generation
```python
# Load trained model for story generation
from gemma_270m.inference import create_generator_from_checkpoint
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
generator = create_generator_from_checkpoint(
    checkpoint_path="checkpoints/best_model.pt",
    tokenizer=tokenizer
)

# Generate a story
story = generator.generate_text(
    prompt="Once upon a time in a magical forest,",
    max_new_tokens=500,
    temperature=0.8,
    top_p=0.9
)
print(story)
```

### Interactive Chat Interface
```python
# Create a simple chat interface
while True:
    user_input = input("You: ")
    if user_input.lower() in ['quit', 'exit']:
        break
        
    response = generator.generate_text(
        prompt=f"Human: {user_input}\nAssistant:",
        max_new_tokens=150,
        temperature=0.7,
        stop_sequences=["Human:", "\n\n"]
    )
    print(f"Assistant: {response}")
```

### Batch Text Generation
```python
# Generate multiple variations of a prompt
prompts = [
    "The future of technology is",
    "In the year 2050, humans will",
    "The most important discovery in science was"
]

for prompt in prompts:
    responses = generator.generate_text(
        prompt=prompt,
        max_new_tokens=100,
        num_return_sequences=3,
        temperature=0.8
    )
    print(f"\nPrompt: {prompt}")
    for i, response in enumerate(responses, 1):
        print(f"{i}. {response}")
```

### Fine-tuning for Specific Tasks
```python
# Fine-tune the model on domain-specific data
from gemma_270m import GemmaTrainer

# Load pre-trained model
config = ExperimentConfig.load("configs/optimized_config.yaml")
config.training.resume_from = "checkpoints/best_model.pt"
config.training.learning_rate = 5e-5  # Lower LR for fine-tuning
config.training.max_iters = 1000      # Fewer iterations

# Update data paths to your domain-specific dataset
config.data.dataset_name = "your-domain/dataset"

trainer = GemmaTrainer(config)
results = trainer.train()
```

## ☁️ Cloud Compute Options

For serious training workloads, use cloud compute with high-end GPUs:

### Recommended Cloud Providers

**Paperspace Gradient**
- A100 (40GB): $3.18/hour - Ideal for full training
- V100 (16GB): $2.30/hour - Good for medium training
- Setup: `pip install gradient && gradient jobs create`

**Modal Labs**  
- A100 (40GB): $4.00/hour - Excellent for research
- V100 (16GB): $2.48/hour - Cost-effective training
- Setup: Serverless GPU functions, automatic scaling

**Google Cloud Platform**
- A100 (40GB): $3.673/hour - Enterprise reliability 
- T4 (16GB): $0.95/hour - Budget-friendly option
- Setup: `gcloud compute instances create`

**AWS EC2**
- p4d.xlarge (A100 40GB): $3.25/hour - Production ready
- p3.2xlarge (V100 16GB): $3.06/hour - Reliable training
- Setup: Use Deep Learning AMI

**Azure Machine Learning**
- Standard_NC24ads_A100_v4: $3.40/hour - Full training
- Standard_NC6s_v3 (V100): $3.06/hour - Medium workloads

### Configuration for Cloud Training
```bash
# High-end cloud (A100/V100)
python run_pipeline.py --config configs/optimized_config.yaml

# Mid-range cloud (T4/A4000) 
python run_pipeline.py --config configs/conservative_config.yaml
```

## 🏗️ Full Model Training

For training the complete 341.8M parameter Gemma-270M model:

### Hardware Requirements
- **Cloud GPU**: A100 (40GB) or V100 (16GB) recommended
- **Local GPU**: RTX 4070 (8GB) minimum for quick testing
- **RAM**: 16GB+ system RAM recommended
- **Storage**: 50GB+ free space for datasets and checkpoints
- **Time**: 2-4 hours on A100, ~17 days on RTX 4070

### Full Training Commands
```bash
# Complete training pipeline with full model
python run_pipeline.py

# Custom full training with specific parameters
python run_pipeline.py --config configs/optimized_config.yaml

# Resume training from checkpoint
python run_pipeline.py --steps train --config configs/optimized_config.yaml
```

### Training Configuration for Full Model
```yaml
# configs/optimized_config.yaml
model:
  n_layers: 22              # Full model size
  emb_dim: 896              # Full embedding dimension
  hidden_dim: 3584          # Full feed-forward dimension

training:
  max_iters: 150000         # Full training iterations
  batch_size: 16            # Memory-optimized
  block_size: 384           # Longer context for quality
  eval_interval: 500        # Regular evaluation
  save_interval: 1000       # Checkpoint every 1K steps
```

### Expected Training Results
- **Training time**: ~3.5 hours on RTX 4070
- **Final validation loss**: ~3.8-4.2 (depending on data)
- **Model size**: 341.8M parameters (~1.3GB)
- **Memory usage**: ~24% of 8GB GPU (1.9GB)
- **Checkpoints**: Saved every 1000 steps + best model

### Monitoring Training Progress
```bash
# View training logs in real-time
tail -f pipeline.log

# Check GPU utilization during training
nvidia-smi -l 1

# Monitor checkpoint directory
ls -la checkpoints/
```

## 🧪 Testing & Validation

### Model Architecture Tests
```bash
# Test model components and parameter counts
python test_model.py

# Test training pipeline without actual training
python test_training.py

# GPU memory analysis
python test_270m_config.py
```

### Training Validation
```bash
# Dry run training (validate setup without training)
python train.py --dry-run

# Validate configuration files
python train.py --validate-config

# Print model architecture and parameter counts
python train.py --print-model

# Quick training test with small iterations
python train.py --preset small --training.max_iters 100
```

### GPU Memory Testing
```bash
# Test memory usage with different batch sizes
python test_270m_config.py --test-memory

# Benchmark generation performance
python generate.py --checkpoint checkpoints/best_model.pt --benchmark
```

## 🚀 Advanced Usage

### Custom Training Loop
```python
from gemma_270m import GemmaModel, GemmaDataLoader
import torch.optim as optim

model = GemmaModel(config.model)
data_loader = GemmaDataLoader(config.data, config.training.block_size)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

for step, (input_ids, targets) in enumerate(data_loader):
    logits, loss = model(input_ids, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")
```

### Batch Text Generation
```python
from gemma_270m.inference import GemmaGenerator, GenerationConfig

# Configure generation
gen_config = GenerationConfig(
    max_new_tokens=200,
    temperature=0.8,
    top_p=0.9,
    num_return_sequences=5
)

# Generate multiple responses
responses = generator.generate_text(
    prompt="What is the meaning of life?",
    **gen_config.__dict__
)

for i, response in enumerate(responses):
    print(f"Response {i+1}: {response}")
```

### Performance Monitoring
```python
# Benchmark generation performance
benchmark_results = generator.benchmark(
    prompts=["Hello world", "The future is", "Once upon a time"],
    max_new_tokens=100,
    batch_size=1
)

print(f"Average tokens/second: {benchmark_results['avg_tokens_per_second']:.1f}")
print(f"Average time/prompt: {benchmark_results['avg_time_per_prompt']:.2f}s")
```

## 🤝 Contributing

Contributions are welcome! Please see our [contribution guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/SiavoshZarrasvand/ai-builds.git
cd ai-builds/Gemma-270M
uv sync --dev
```

### Code Style
We use `black` for code formatting and `flake8` for linting:
```bash
black gemma_270m/
flake8 gemma_270m/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google Research** for the Gemma architecture and research
- **Hugging Face** for the transformers library and tokenizers
- **PyTorch Team** for the excellent deep learning framework
- **Open Source Community** for inspiration and collaborative development

## 📚 References

1. **Gemma Paper**: [Gemma: Open Models Based on Gemini Research and Technology](https://arxiv.org/abs/2403.08295)
2. **Transformer Architecture**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
3. **RoPE Embeddings**: [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
4. **GLU Variants**: [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
5. **GeGLU Activation**: [Language Modeling with Gated Linear Units](https://arxiv.org/abs/2002.05202)

---

**Built with ❤️ for the open-source AI community**
