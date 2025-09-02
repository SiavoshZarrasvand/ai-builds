#!/usr/bin/env python3
"""
Test different model configurations to find true 270M setup
"""

import torch

def estimate_params(config):
    """Estimate model parameters"""
    emb_params = config['vocab_size'] * config['emb_dim']
    out_params = config['emb_dim'] * config['vocab_size'] 
    
    # Per layer: 4 attention projections + 3 MLP layers + norms
    attn_params = 4 * config['emb_dim']**2  # Q, K, V, O projections  
    mlp_params = 3 * config['emb_dim'] * config['hidden_dim']  # up, gate, down
    layer_params = config['n_layers'] * (attn_params + mlp_params + 6 * config['emb_dim'])  # +norms
    
    total = emb_params + out_params + layer_params
    return total / 1e6

# Test different configurations
configs = {
    'current': {'vocab_size': 50257, 'emb_dim': 640, 'n_layers': 18, 'hidden_dim': 2048},
    'true_270m_v1': {'vocab_size': 50257, 'emb_dim': 1024, 'n_layers': 20, 'hidden_dim': 4096},
    'true_270m_v2': {'vocab_size': 50257, 'emb_dim': 896, 'n_layers': 24, 'hidden_dim': 3584},
    'true_270m_v3': {'vocab_size': 50257, 'emb_dim': 768, 'n_layers': 28, 'hidden_dim': 3072},
}

print("Parameter estimates for different configs:")
print("=" * 50)

for name, config in configs.items():
    params = estimate_params(config)
    print(f"{name:15}: {params:6.1f}M params")
    print(f"{'':15}  emb_dim={config['emb_dim']}, layers={config['n_layers']}, hidden={config['hidden_dim']}")
    print()

# Test memory usage with current setup
print("\nTesting current model memory usage...")
try:
    from gemma_270m import GemmaModel, get_default_config
    
    config = get_default_config()
    model = GemmaModel(config).cuda()
    
    # Test with small batch first
    batch_size, seq_len = 8, 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).cuda()
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        logits, _ = model(input_ids)
    
    memory_used = torch.cuda.max_memory_allocated() / 1e9
    print(f"✓ Current model (165M): {memory_used:.2f} GB GPU memory")
    print(f"  Batch size: {batch_size}, Sequence length: {seq_len}")
    
except Exception as e:
    print(f"❌ Memory test failed: {e}")

print(f"\nGPU Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
