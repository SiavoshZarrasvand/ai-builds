#!/usr/bin/env python3
"""
Test the training loop from the Gemma-270M notebook
Quick validation that training works with our Windows/CUDA setup
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from contextlib import nullcontext
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
import os
import time

print("Testing Gemma-270M Training Loop")
print("=" * 50)

# Test 1: Check environment
print("\n=== Environment Check ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Test 2: Check data files
print("\n=== Data Files Check ===")
# Check in data directory instead of root
train_path = 'data/train.bin' if os.path.exists('data/train.bin') else 'train.bin'
val_path = 'data/validation.bin' if os.path.exists('data/validation.bin') else 'validation.bin'

train_exists = os.path.exists(train_path)
val_exists = os.path.exists(val_path)
print(f"train.bin exists: {train_exists} ({os.path.getsize(train_path) / 1e6:.0f} MB)" if train_exists else "train.bin: Missing!")
print(f"validation.bin exists: {val_exists} ({os.path.getsize(val_path) / 1e6:.0f} MB)" if val_exists else "validation.bin: Missing!")

if not (train_exists and val_exists):
    print("[FAIL] Binary data files missing! Run tokenization first.")
    exit(1)

# Test 3: Training Configuration
print("\n=== Training Configuration ===")
# Reduced config for quick testing
config = {
    'learning_rate': 1e-4,
    'max_iters': 100,  # Very small for testing
    'warmup_steps': 10,
    'min_lr': 5e-5,
    'eval_iters': 20,
    'batch_size': 4,    # Small batch for testing
    'block_size': 64,   # Small context for testing
    'gradient_accumulation_steps': 4,
}

device = "cuda" if torch.cuda.is_available() else "cpu"
device_type = 'cuda' if 'cuda' in device else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

print(f"Device: {device}")
print(f"Data type: {dtype}")
print(f"Batch size: {config['batch_size']}")
print(f"Block size: {config['block_size']}")

# Test 4: Data Loading Function
print("\n=== Data Loading Test ===")
def get_batch(split):
    """Load batch from binary files"""
    if split == 'train':
        data = np.memmap(train_path, dtype=np.uint16, mode='r')
    else:
        data = np.memmap(val_path, dtype=np.uint16, mode='r')
    
    ix = torch.randint(len(data) - config['block_size'], (config['batch_size'],))
    x = torch.stack([torch.from_numpy((data[i:i+config['block_size']]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+config['block_size']]).astype(np.int64)) for i in ix])
    
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

try:
    X, y = get_batch("train")
    print(f"[PASS] Train batch: X.shape={X.shape}, y.shape={y.shape}")
    X_val, y_val = get_batch("val")
    print(f"[PASS] Validation batch: X_val.shape={X_val.shape}, y_val.shape={y_val.shape}")
    print(f"[PASS] Token range: {X.min().item()}-{X.max().item()}")
except Exception as e:
    print(f"[FAIL] Data loading failed: {e}")
    exit(1)

# Test 5: Simple Model for Testing (not full Gemma)
print("\n=== Simple Model Test ===")
class SimpleTestModel(nn.Module):
    def __init__(self, vocab_size=50257, emb_dim=256, n_layers=2, n_heads=4):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.position_embedding = nn.Embedding(config['block_size'], emb_dim)
        
        # Simple transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=emb_dim, 
                nhead=n_heads,
                dim_feedforward=emb_dim*4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Token and position embeddings
        tok_emb = self.token_embedding(idx)  # (B, T, C)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.position_embedding(pos)  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        
        # Transformer layers (simplified)
        # Create causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=idx.device) * float('-inf'), diagonal=1)
        
        for layer in self.layers:
            x = layer(x, x, tgt_mask=causal_mask)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        
        return logits, loss

# Create and test model
try:
    model = SimpleTestModel().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[PASS] Test model created: {total_params:,} parameters")
    
    # Test forward pass
    with ctx:
        logits, loss = model(X, y)
    print(f"[PASS] Forward pass: logits.shape={logits.shape}, loss={loss.item():.4f}")
    
except Exception as e:
    print(f"[FAIL] Model test failed: {e}")
    exit(1)

# Test 6: Optimizer and Training Step
print("\n=== Optimizer Test ===")
try:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], betas=(0.9, 0.95), weight_decay=0.1)
    
    # Training step
    optimizer.zero_grad()
    with ctx:
        logits, loss = model(X, y)
        loss = loss / config['gradient_accumulation_steps']
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    print(f"[PASS] Training step completed: loss={loss.item():.4f}")
    
except Exception as e:
    print(f"[FAIL] Training step failed: {e}")
    exit(1)

# Test 7: Short Training Loop
print("\n=== Mini Training Loop ===")
print("Running 10 training steps...")

model.train()
start_time = time.time()

for step in range(10):
    # Get batch
    X, y = get_batch("train")
    
    # Forward pass
    with ctx:
        logits, loss = model(X, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    if step % 5 == 0:
        print(f"  Step {step}: loss={loss.item():.4f}")

end_time = time.time()
print(f"[PASS] Mini training completed in {end_time - start_time:.2f}s")

# Test 8: GPU Memory Check
if torch.cuda.is_available():
    print(f"\n=== GPU Memory Usage ===")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e6:.0f} MB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1e6:.0f} MB")
    print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1e6:.0f} MB")

print("\n[SUCCESS] Training loop test completed successfully!")
print("\nNext steps:")
print("1. [OK] Training infrastructure works")  
print("2. [TODO] Convert to full Gemma architecture")
print("3. [BUILD]  Build Python project structure")
print("4. [SCALE] Scale up for full training")

if torch.cuda.is_available():
    torch.cuda.empty_cache()  # Clean up
