#!/usr/bin/env python3
"""
Gemma-270M Model Architecture

This module implements the complete Gemma-270M transformer model with:
- RoPE (Rotary Positional Encoding)
- Grouped Query Attention
- RMSNorm (Root Mean Square Layer Normalization)
- Sliding window and full attention layers
- Feed-forward networks with GELU activation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class GemmaConfig:
    """Configuration class for Gemma-270M model"""
    vocab_size: int = 50257
    context_length: int = 32768
    emb_dim: int = 640
    n_heads: int = 4
    n_layers: int = 18
    hidden_dim: int = 2048
    head_dim: int = 256
    qk_norm: bool = True
    n_kv_groups: int = 1
    rope_local_base: float = 10000.0
    rope_base: float = 1000000.0
    sliding_window: int = 512
    layer_types: List[str] = None
    dtype: torch.dtype = torch.bfloat16
    query_pre_attn_scalar: int = 256

    def __post_init__(self):
        if self.layer_types is None:
            # Default pattern: mostly sliding attention with periodic full attention
            self.layer_types = [
                "sliding_attention", "sliding_attention", "sliding_attention", 
                "sliding_attention", "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention", 
                "sliding_attention", "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention", 
                "sliding_attention", "sliding_attention", "full_attention"
            ]
        assert len(self.layer_types) == self.n_layers, f"layer_types length ({len(self.layer_types)}) must match n_layers ({self.n_layers})"


def compute_rope_params(head_dim: int, theta_base: float = 10_000, context_length: int = 4096, dtype: torch.dtype = torch.float32):
    """Compute RoPE (Rotary Positional Encoding) parameters"""
    assert head_dim % 2 == 0, "Embedding dimension must be even"
    
    # Compute the inverse frequencies
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))
    
    # Generate position indices
    positions = torch.arange(context_length, dtype=dtype)
    
    # Compute the angles
    angles = positions[:, None] * inv_freq[None, :]  # Shape: (context_length, head_dim // 2)
    
    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)
    
    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to query or key tensors"""
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"
    
    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2 :]  # Second half
    
    # Adjust sin and cos shapes
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)
    
    # Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)
    
    return x_rotated.to(dtype=x.dtype)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Gemma-style)"""
    
    def __init__(self, emb_dim: int, eps: float = 1e-6, bias: bool = False):
        super().__init__()
        self.eps = eps
        # Gemma3 stores zero-centered weights and uses (1 + weight) during forward
        self.scale = nn.Parameter(torch.zeros(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Match HF Gemma3: compute norm in float32, then scale by (1 + w)
        input_dtype = x.dtype
        x_f = x.float()
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_f * torch.rsqrt(var + self.eps)
        out = x_norm * (1.0 + self.scale.float())
        
        if self.shift is not None:
            out = out + self.shift.float()
        
        return out.to(input_dtype)


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention mechanism for Gemma-270M"""
    
    def __init__(
        self, 
        d_in: int, 
        num_heads: int, 
        num_kv_groups: int, 
        head_dim: Optional[int] = None, 
        qk_norm: bool = False,
        query_pre_attn_scalar: Optional[int] = None, 
        dtype: Optional[torch.dtype] = None,
    ):
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

    def forward(self, x: torch.Tensor, mask: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, num_tokens, _ = x.shape
        
        # Apply projections
        queries = self.W_query(x)  # (b, num_tokens, num_heads * head_dim)
        keys = self.W_key(x)       # (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)   # (b, num_tokens, num_kv_groups * head_dim)
        
        # Reshape
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        
        # Optional normalization
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys = self.k_norm(keys)
        
        # Apply RoPE
        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)
        
        # Expand K and V to match number of heads
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)
        
        # Scale queries
        queries = queries * self.scaling
        
        # Attention
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(context)


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation (Gemma-style)"""
    
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.emb_dim, config.hidden_dim, dtype=config.dtype, bias=False)
        self.fc2 = nn.Linear(config.emb_dim, config.hidden_dim, dtype=config.dtype, bias=False)
        self.fc3 = nn.Linear(config.hidden_dim, config.emb_dim, dtype=config.dtype, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = F.gelu(x_fc1, approximate="tanh") * x_fc2
        return self.fc3(x)


class TransformerBlock(nn.Module):
    """Transformer block with attention and feed-forward layers"""

    def __init__(self, config: GemmaConfig, attn_type: str):
        super().__init__()
        self.attn_type = attn_type

        self.att = GroupedQueryAttention(
            d_in=config.emb_dim,
            num_heads=config.n_heads,
            num_kv_groups=config.n_kv_groups,
            head_dim=config.head_dim,
            qk_norm=config.qk_norm,
            query_pre_attn_scalar=config.query_pre_attn_scalar,
            dtype=config.dtype,
        )
        self.ff = FeedForward(config)
        self.input_layernorm = RMSNorm(config.emb_dim, eps=1e-6)
        self.post_attention_layernorm = RMSNorm(config.emb_dim, eps=1e-6)
        self.pre_feedforward_layernorm = RMSNorm(config.emb_dim, eps=1e-6)
        self.post_feedforward_layernorm = RMSNorm(config.emb_dim, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        mask_global: torch.Tensor,
        mask_local: torch.Tensor,
        cos_global: torch.Tensor,
        sin_global: torch.Tensor,
        cos_local: torch.Tensor,
        sin_local: torch.Tensor,
    ) -> torch.Tensor:
        # Shortcut connection for attention block
        shortcut = x
        x = self.input_layernorm(x)

        if self.attn_type == "sliding_attention":
            attn_mask = mask_local
            cos = cos_local
            sin = sin_local
        else:
            attn_mask = mask_global
            cos = cos_global
            sin = sin_global

        x_attn = self.att(x, attn_mask, cos, sin)
        x_attn = self.post_attention_layernorm(x_attn)
        x = shortcut + x_attn

        # Shortcut connection for feed forward block
        shortcut = x
        x_ffn = self.pre_feedforward_layernorm(x)
        x_ffn = self.ff(x_ffn)
        x_ffn = self.post_feedforward_layernorm(x_ffn)
        x = shortcut + x_ffn
        return x


class GemmaModel(nn.Module):
    """Complete Gemma-270M model implementation"""
    
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        assert config.layer_types is not None and len(config.layer_types) == config.n_layers

        # Main model parameters
        self.tok_emb = nn.Embedding(config.vocab_size, config.emb_dim, dtype=config.dtype)

        self.blocks = nn.ModuleList([
            TransformerBlock(config, attn_type) for attn_type in config.layer_types
        ])

        self.final_norm = RMSNorm(config.emb_dim, eps=1e-6)
        self.out_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False, dtype=config.dtype)

        # Reusable utilities
        cos_local, sin_local = compute_rope_params(
            head_dim=config.head_dim,
            theta_base=config.rope_local_base,
            context_length=config.context_length,
            dtype=torch.float32,
        )
        cos_global, sin_global = compute_rope_params(
            head_dim=config.head_dim,
            theta_base=config.rope_base,
            context_length=config.context_length,
            dtype=torch.float32,
        )
        self.register_buffer("cos_local", cos_local, persistent=False)
        self.register_buffer("sin_local", sin_local, persistent=False)
        self.register_buffer("cos_global", cos_global, persistent=False)
        self.register_buffer("sin_global", sin_global, persistent=False)

    def _create_masks(self, seq_len: int, device: torch.device):
        """Create attention masks for sliding window and global attention"""
        ones = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)

        # mask_global (future is masked: j > i)
        mask_global = torch.triu(ones, diagonal=1)

        # far_past (too far back is masked: i - j >= sliding_window)
        far_past = torch.triu(ones, diagonal=self.config.sliding_window).T

        # Local (sliding_window) = future OR far-past
        mask_local = mask_global | far_past
        return mask_global, mask_local

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """Forward pass through the model"""
        b, seq_len = input_ids.shape
        x = self.tok_emb(input_ids) * (self.config.emb_dim ** 0.5)
        mask_global, mask_local = self._create_masks(seq_len, x.device)

        for block in self.blocks:
            x = block(
                x,
                mask_global=mask_global,
                mask_local=mask_local,
                cos_global=self.cos_global,
                sin_global=self.sin_global,
                cos_local=self.cos_local,
                sin_local=self.sin_local,
            )

        x = self.final_norm(x)
        logits = self.out_head(x.to(self.config.dtype))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """Generate text using the trained model"""
        for _ in range(max_new_tokens):
            ctx_len = self.config.context_length
            idx_cond = idx if idx.size(1) <= ctx_len else idx[:, -ctx_len:]
            logits, _ = self(idx_cond)  # targets=None by default
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return the number of parameters in the model"""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.tok_emb.weight.numel()
        return n_params


# Default configuration for Gemma-270M
def get_default_config() -> GemmaConfig:
    """Get the default configuration for Gemma-270M"""
    return GemmaConfig(
        vocab_size=50257,
        context_length=32768,
        emb_dim=640,
        n_heads=4,
        n_layers=18,
        hidden_dim=2048,
        head_dim=256,
        qk_norm=True,
        n_kv_groups=1,
        rope_local_base=10000.0,
        rope_base=1000000.0,
        sliding_window=512,
        layer_types=[
            "sliding_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "sliding_attention", "full_attention"
        ],
        dtype=torch.bfloat16,
        query_pre_attn_scalar=256,
    )


if __name__ == "__main__":
    # Test model creation
    config = get_default_config()
    model = GemmaModel(config)
    print(f"Gemma-270M Model created with {model.get_num_params():,} parameters")
    
    # Test forward pass
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        logits, _ = model(input_ids)
        print(f"Forward pass successful: input.shape={input_ids.shape}, logits.shape={logits.shape}")
