#!/usr/bin/env python3
"""
Debug script for the Gemma-270M notebook
Step-by-step execution to catch and fix issues
"""

import torch
import tiktoken
import os
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset

def step1_dataset():
    """Step 1: Load TinyStories dataset"""
    print("=== STEP 1: Loading TinyStories Dataset ===")
    try:
        ds = load_dataset("roneneldan/TinyStories")
        print(f"✓ Dataset loaded successfully!")
        print(f"✓ Train examples: {len(ds['train']):,}")
        print(f"✓ Validation examples: {len(ds['validation']):,}")
        return ds
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return None

def step2_tokenize(ds):
    """Step 2: Tokenize dataset and create binary files"""
    print("\n=== STEP 2: Tokenizing Dataset ===")
    
    def process_with_tokenizer(example):
        enc = tiktoken.get_encoding('gpt2')  # Define inside function for Windows
        ids = enc.encode_ordinary(example['text'])
        out = {'ids': ids, 'len': len(ids)}
        return out
    
    try:
        if not os.path.exists('train.bin'):
            print("Starting tokenization (single-process for Windows compatibility)...")
            tokenized = ds.map(
                process_with_tokenizer,
                remove_columns=['text'],
                desc='tokenizing the splits',
                num_proc=1,  # Windows compatibility
            )
            
            print("Creating binary files...")
            for split, dset in tokenized.items():
                arr_len = np.sum(dset['len'], dtype=np.uint64)
                filename = f'{split}.bin'
                dtype = np.uint16  # GPT-2 vocab size < 2^16
                arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
                total_batches = 1024
                
                idx = 0
                for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
                    batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
                    arr_batch = np.concatenate(batch['ids'])
                    arr[idx : idx + len(arr_batch)] = arr_batch
                    idx += len(arr_batch)
                arr.flush()
                print(f"✓ {filename}: {arr_len:,} tokens")
        else:
            print("✓ Binary files already exist, skipping tokenization")
        
        return True
    except Exception as e:
        print(f"✗ Error in tokenization: {e}")
        return False

def step3_test_batch_loading():
    """Step 3: Test batch loading function"""
    print("\n=== STEP 3: Testing Batch Loading ===")
    
    # First define the missing variables that will be needed
    batch_size = 32
    block_size = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    
    def get_batch(split):
        """Load batch from binary files"""
        if split == 'train':
            data = np.memmap('train.bin', dtype=np.uint16, mode='r')
        else:
            data = np.memmap('validation.bin', dtype=np.uint16, mode='r')
        
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        
        if device_type == 'cuda':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y
    
    try:
        # Test loading batches
        print(f"Device: {device}")
        X, y = get_batch("train")
        print(f"✓ Train batch loaded: X.shape={X.shape}, y.shape={y.shape}")
        
        X_val, y_val = get_batch("val")  
        print(f"✓ Validation batch loaded: X_val.shape={X_val.shape}, y_val.shape={y_val.shape}")
        
        return get_batch
        
    except Exception as e:
        print(f"✗ Error in batch loading: {e}")
        return None

def step4_test_model():
    """Step 4: Test model creation"""
    print("\n=== STEP 4: Testing Model Creation ===")
    
    try:
        # Import all model components
        import torch.nn as nn
        import torch.nn.functional as F
        import math
        from dataclasses import dataclass
        from contextlib import nullcontext
        
        # The model code from the notebook would go here...
        # For now, let's just test basic PyTorch functionality
        
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"✓ CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Test a simple model creation
        simple_model = nn.Linear(640, 50257)  # emb_dim to vocab_size
        print(f"✓ Simple model created: {sum(p.numel() for p in simple_model.parameters())} parameters")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in model creation: {e}")
        return False

def main():
    """Main debug function"""
    print("Gemma-270M Notebook Debug Script")
    print("=" * 50)
    
    # Step 1: Load dataset
    ds = step1_dataset()
    if ds is None:
        return
    
    # Step 2: Tokenization 
    if not step2_tokenize(ds):
        return
    
    # Step 3: Test batch loading
    get_batch = step3_test_batch_loading()
    if get_batch is None:
        return
    
    # Step 4: Test model
    if not step4_test_model():
        return
    
    print("\n🎉 All basic steps completed successfully!")
    print("\nNext steps:")
    print("1. Complete model architecture implementation")
    print("2. Test training configuration")
    print("3. Run a small training test")

if __name__ == "__main__":
    main()
