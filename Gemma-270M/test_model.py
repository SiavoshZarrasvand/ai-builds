#!/usr/bin/env python3
"""
Test the Gemma-270M model architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from contextlib import nullcontext

print("Testing Gemma-270M Model Architecture")
print("=" * 50)

# Test 1: Basic PyTorch setup
print("\n=== Test 1: Basic PyTorch Setup ===")
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"✓ CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Test 2: Rope Positional Encoding functions
print("\n=== Test 2: RoPE Functions ===")
def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
    assert head_dim % 2 == 0, "Embedding dimension must be even"
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))
    positions = torch.arange(context_length, dtype=dtype)
    angles = positions[:, None] * inv_freq[None, :]
    angles = torch.cat([angles, angles], dim=1)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin

def apply_rope(x, cos, sin):
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)
    return x_rotated.to(dtype=x.dtype)

try:
    # Test RoPE with small dimensions
    head_dim = 64
    context_length = 128
    cos, sin = compute_rope_params(head_dim, context_length=context_length)
    print(f"✓ RoPE params computed: cos.shape={cos.shape}, sin.shape={sin.shape}")
    
    # Test apply_rope
    batch_size, num_heads, seq_len = 2, 4, 32
    x = torch.randn(batch_size, num_heads, seq_len, head_dim)
    x_rope = apply_rope(x, cos, sin)
    print(f"✓ RoPE applied: input.shape={x.shape}, output.shape={x_rope.shape}")
    
except Exception as e:
    print(f"✗ RoPE test failed: {e}")

# Test 3: RMSNorm
print("\n=== Test 3: RMSNorm ===")
class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype
        x_f = x.float()
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_f * torch.rsqrt(var + self.eps)
        out = x_norm * (1.0 + self.scale.float())
        if self.shift is not None:
            out = out + self.shift.float()
        return out.to(input_dtype)

try:
    emb_dim = 640
    rms_norm = RMSNorm(emb_dim)
    x = torch.randn(2, 32, emb_dim)
    x_norm = rms_norm(x)
    print(f"✓ RMSNorm: input.shape={x.shape}, output.shape={x_norm.shape}")
    print(f"✓ RMSNorm parameters: {sum(p.numel() for p in rms_norm.parameters())}")
except Exception as e:
    print(f"✗ RMSNorm test failed: {e}")

# Test 4: GroupedQueryAttention (simplified)
print("\n=== Test 4: Grouped Query Attention ===")
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, query_pre_attn_scalar=None, dtype=None):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"
        
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups
        
        if head_dim is None:
            assert d_in % num_heads == 0, "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            head_dim = d_in // num_heads
        
        self.head_dim = head_dim
        self.d_out = num_heads * head_dim
        
        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)
        
        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None
        
        if query_pre_attn_scalar is not None:
            self.scaling = (query_pre_attn_scalar) ** -0.5
        else:
            self.scaling = (head_dim) ** -0.5

    def forward(self, x, mask, cos, sin):
        b, num_tokens, _ = x.shape
        
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys = self.k_norm(keys)
        
        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)
        
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)
        
        queries = queries * self.scaling
        
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(context)

try:
    d_in = 640
    num_heads = 4
    num_kv_groups = 1
    head_dim = 256
    seq_len = 32
    batch_size = 2
    
    attention = GroupedQueryAttention(d_in, num_heads, num_kv_groups, head_dim, qk_norm=True, query_pre_attn_scalar=256)
    
    # Create inputs
    x = torch.randn(batch_size, seq_len, d_in)
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    cos, sin = compute_rope_params(head_dim, context_length=seq_len)
    
    # Forward pass
    output = attention(x, mask, cos, sin)
    print(f"✓ GroupedQueryAttention: input.shape={x.shape}, output.shape={output.shape}")
    print(f"✓ Attention parameters: {sum(p.numel() for p in attention.parameters()):,}")
    
except Exception as e:
    print(f"✗ GroupedQueryAttention test failed: {e}")

# Test 5: Config and basic model structure
print("\n=== Test 5: Config and Model ===")
GEMMA3_CONFIG_270M = {
    "vocab_size": 50257,
    "context_length": 32_768,
    "emb_dim": 640,
    "n_heads": 4,
    "n_layers": 18,
    "hidden_dim": 2048,
    "head_dim": 256,
    "qk_norm": True,
    "n_kv_groups": 1,
    "rope_local_base": 10_000.0,
    "rope_base": 1_000_000.0,
    "sliding_window": 512,
    "layer_types": [
        "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
        "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
        "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"
    ],
    "dtype": torch.bfloat16,
    "query_pre_attn_scalar": 256,
}

print(f"✓ Config loaded: {len(GEMMA3_CONFIG_270M)} parameters")
print(f"✓ Model should have {GEMMA3_CONFIG_270M['n_layers']} layers")
print(f"✓ Vocab size: {GEMMA3_CONFIG_270M['vocab_size']:,}")
print(f"✓ Embedding dim: {GEMMA3_CONFIG_270M['emb_dim']}")

# Calculate approximate model size
embedding_params = GEMMA3_CONFIG_270M['vocab_size'] * GEMMA3_CONFIG_270M['emb_dim']
output_head_params = GEMMA3_CONFIG_270M['emb_dim'] * GEMMA3_CONFIG_270M['vocab_size']
# Rough estimate for transformer layers (very approximate)
layer_params_estimate = GEMMA3_CONFIG_270M['n_layers'] * (
    4 * GEMMA3_CONFIG_270M['emb_dim'] * GEMMA3_CONFIG_270M['hidden_dim'] +  # Feed forward
    4 * GEMMA3_CONFIG_270M['emb_dim'] * GEMMA3_CONFIG_270M['emb_dim']  # Attention projections
)
total_params_estimate = embedding_params + output_head_params + layer_params_estimate
print(f"✓ Estimated model size: ~{total_params_estimate/1e6:.1f}M parameters")

print("\n🎉 Model architecture tests completed!")
print("\nNext: Test data loading and training configuration")
