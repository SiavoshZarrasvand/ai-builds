#!/usr/bin/env python3
"""
Data loading and preprocessing for Gemma-270M

This module handles:
- Loading and tokenizing datasets
- Creating binary data files for efficient loading
- Batch generation for training and evaluation
- Dataset utilities
"""

import os
import torch
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm.auto import tqdm
from typing import Dict, Tuple, Optional, Union, List
from pathlib import Path


class TextDataset:
    """Dataset class for handling tokenized text data"""
    
    def __init__(
        self, 
        data_path: str, 
        block_size: int = 1024, 
        device: str = "cuda",
        dtype: torch.dtype = torch.long
    ):
        self.data_path = Path(data_path)
        self.block_size = block_size
        self.device = device
        self.dtype = dtype
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file {data_path} not found")
        
        # Load the memory-mapped data
        self.data = np.memmap(self.data_path, dtype=np.uint16, mode='r')
        self.length = len(self.data)
        
    def __len__(self) -> int:
        return max(0, self.length - self.block_size)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Extract a sequence of tokens starting at idx
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")
        
        # Get input sequence and target sequence (shifted by 1)
        x = torch.from_numpy(self.data[idx:idx + self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data[idx + 1:idx + 1 + self.block_size].astype(np.int64))
        
        return {"input_ids": x, "labels": y}


class GemmaDataLoader:
    """Data loader for Gemma-270M training and evaluation"""
    
    def __init__(
        self,
        train_path: str = "train.bin",
        val_path: str = "validation.bin", 
        batch_size: int = 32,
        block_size: int = 1024,
        device: str = "cuda",
        device_type: str = "cuda"
    ):
        self.train_path = train_path
        self.val_path = val_path
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.device_type = device_type
        
        # Validate files exist
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training data file {train_path} not found")
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"Validation data file {val_path} not found")
    
    def get_batch(self, split: str = "train") -> Tuple[torch.Tensor, torch.Tensor]:
        """Load a batch of data for training or validation"""
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        if split == 'train':
            data = np.memmap(self.train_path, dtype=np.uint16, mode='r')
        else:
            data = np.memmap(self.val_path, dtype=np.uint16, mode='r')
        
        # Generate random starting positions
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        
        # Extract sequences
        x = torch.stack([torch.from_numpy((data[i:i + self.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + self.block_size]).astype(np.int64)) for i in ix])
        
        # Move to device
        if self.device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        
        return x, y
    
    def get_train_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a training batch"""
        return self.get_batch("train")
    
    def get_val_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a validation batch"""
        return self.get_batch("val")


class DataProcessor:
    """Processes raw text data into tokenized binary files"""
    
    def __init__(self, tokenizer_name: str = "gpt2", num_proc: int = 8):
        self.enc = tiktoken.get_encoding(tokenizer_name)
        self.num_proc = num_proc
    
    def process_example(self, example: Dict[str, str]) -> Dict[str, List[int]]:
        """Process a single example by tokenizing the text"""
        ids = self.enc.encode_ordinary(example['text'])  # encode_ordinary ignores any special tokens
        return {'ids': ids, 'len': len(ids)}
    
    def create_binary_dataset(
        self, 
        dataset_name: str = "roneneldan/TinyStories", 
        output_dir: str = ".",
        force_reprocess: bool = False
    ) -> Tuple[str, str]:
        """
        Create binary dataset files from HuggingFace dataset
        
        Returns:
            Tuple of (train_path, validation_path)
        """
        train_path = os.path.join(output_dir, "train.bin")
        val_path = os.path.join(output_dir, "validation.bin")
        
        # Skip if files exist and not forcing reprocess
        if not force_reprocess and os.path.exists(train_path) and os.path.exists(val_path):
            print(f"Binary files already exist:")
            print(f"  {train_path} ({os.path.getsize(train_path) / 1e6:.0f} MB)")
            print(f"  {val_path} ({os.path.getsize(val_path) / 1e6:.0f} MB)")
            return train_path, val_path
        
        print(f"Loading dataset: {dataset_name}")
        ds = load_dataset(dataset_name)
        
        print("Tokenizing dataset...")
        tokenized = ds.map(
            self.process_example,
            remove_columns=['text'],
            desc="tokenizing the splits",
            num_proc=self.num_proc,
        )
        
        # concatenate all the ids in each dataset into one large file we can use for training
        for split, dset in tokenized.items():
            arr_len = np.sum(dset['len'], dtype=np.uint64)
            filename = os.path.join(output_dir, f'{split}.bin')
            dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
            arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
            total_batches = 1024

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
                # Batch together samples for faster write
                batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
                arr_batch = np.concatenate(batch['ids'])
                # Write into mmap
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()
        
        print(f"Created binary files:")
        print(f"  {train_path} ({os.path.getsize(train_path) / 1e6:.0f} MB)")
        print(f"  {val_path} ({os.path.getsize(val_path) / 1e6:.0f} MB)")
        
        return train_path, val_path
    
    def get_vocab_size(self) -> int:
        """Get the vocabulary size of the tokenizer"""
        return self.enc.n_vocab
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token ids"""
        return self.enc.encode_ordinary(text)
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token ids to text"""
        return self.enc.decode(tokens)


def prepare_data(
    dataset_name: str = "roneneldan/TinyStories",
    output_dir: str = ".",
    force_reprocess: bool = False,
    tokenizer_name: str = "gpt2",
    num_proc: int = 8
) -> Tuple[str, str, int]:
    """
    Prepare data for training
    
    Returns:
        Tuple of (train_path, val_path, vocab_size)
    """
    processor = DataProcessor(tokenizer_name=tokenizer_name, num_proc=num_proc)
    train_path, val_path = processor.create_binary_dataset(
        dataset_name=dataset_name, 
        output_dir=output_dir, 
        force_reprocess=force_reprocess
    )
    vocab_size = processor.get_vocab_size()
    
    return train_path, val_path, vocab_size


def get_data_info(data_path: str) -> Dict[str, Union[int, float]]:
    """Get information about a binary data file"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file {data_path} not found")
    
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    file_size = os.path.getsize(data_path)
    
    return {
        "num_tokens": len(data),
        "file_size_mb": file_size / 1e6,
        "min_token": int(data.min()),
        "max_token": int(data.max()),
        "unique_tokens": len(np.unique(data)),
    }


if __name__ == "__main__":
    # Test data processing
    print("Testing data processing...")
    
    # Check if data files exist
    if not os.path.exists("train.bin"):
        print("Creating binary dataset files...")
        train_path, val_path, vocab_size = prepare_data()
    else:
        train_path, val_path = "train.bin", "validation.bin"
        processor = DataProcessor()
        vocab_size = processor.get_vocab_size()
    
    print(f"Vocab size: {vocab_size}")
    
    # Test data info
    train_info = get_data_info(train_path)
    val_info = get_data_info(val_path)
    
    print(f"Train data: {train_info}")
    print(f"Val data: {val_info}")
    
    # Test data loader
    data_loader = GemmaDataLoader(
        train_path=train_path,
        val_path=val_path,
        batch_size=4,
        block_size=128,
        device="cpu"  # Use CPU for testing
    )
    
    # Test batch loading
    X_train, y_train = data_loader.get_train_batch()
    X_val, y_val = data_loader.get_val_batch()
    
    print(f"Train batch: X={X_train.shape}, y={y_train.shape}")
    print(f"Val batch: X={X_val.shape}, y={y_val.shape}")
    
    # Test tokenizer
    processor = DataProcessor()
    text = "Once upon a time there was a little girl."
    tokens = processor.encode(text)
    decoded = processor.decode(tokens)
    
    print(f"Original: {text}")
    print(f"Tokens: {tokens}")
    print(f"Decoded: {decoded}")
    
    print("✅ Data processing tests passed!")
