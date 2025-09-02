#!/usr/bin/env python3
"""
Inference and text generation for Gemma-270M

This module provides text generation capabilities with various sampling strategies:
- Greedy decoding
- Top-k sampling
- Top-p (nucleus) sampling  
- Temperature sampling
- Beam search
- Model loading utilities
- Batch generation
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, Union, Tuple
from pathlib import Path
import json
import time
from dataclasses import dataclass

from .model import GemmaModel
from .config import ExperimentConfig, ModelConfig
from .data import GemmaDataLoader


@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.9
    do_sample: bool = True
    num_return_sequences: int = 1
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    use_cache: bool = True
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    bad_words_ids: Optional[List[List[int]]] = None
    seed: Optional[int] = None


class GemmaGenerator:
    """Text generation pipeline for Gemma-270M"""
    
    def __init__(
        self,
        model: GemmaModel,
        tokenizer=None,
        config: Optional[GenerationConfig] = None,
        device: Optional[str] = None
    ):
        """Initialize generator
        
        Args:
            model: Trained Gemma model
            tokenizer: Tokenizer for encoding/decoding text
            config: Generation configuration
            device: Device to run inference on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or GenerationConfig()
        
        # Set device
        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device(device)
            self.model = self.model.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize special token IDs
        if self.tokenizer:
            self.config.pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)
            self.config.eos_token_id = getattr(self.tokenizer, 'eos_token_id', None)
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> torch.Tensor:
        """Generate text from input tokens
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            generation_config: Generation configuration (overrides default)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated token IDs [batch_size, seq_len + max_new_tokens]
        """
        # Use provided config or default
        config = generation_config or self.config
        
        # Update config with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Set random seed for reproducibility
        if config.seed is not None:
            torch.manual_seed(config.seed)
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize generation
        generated = input_ids.clone()
        past_key_values = None
        
        for step in range(config.max_new_tokens):
            # Get model output
            if config.use_cache and past_key_values is not None:
                # Only process the last token when using cache
                model_input = generated[:, -1:]
            else:
                model_input = generated
            
            # Forward pass
            logits, _ = self.model(model_input)
            
            # Get next token logits
            next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]
            
            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, generated, config.repetition_penalty
                )
            
            # Apply bad words filter
            if config.bad_words_ids:
                next_token_logits = self._apply_bad_words_filter(
                    next_token_logits, config.bad_words_ids
                )
            
            # Apply n-gram repetition filter
            if config.no_repeat_ngram_size > 0:
                next_token_logits = self._apply_no_repeat_ngram_filter(
                    next_token_logits, generated, config.no_repeat_ngram_size
                )
            
            # Sample next token
            if config.do_sample:
                next_tokens = self._sample(
                    next_token_logits,
                    temperature=config.temperature,
                    top_k=config.top_k,
                    top_p=config.top_p
                )
            else:
                # Greedy decoding
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append next token
            generated = torch.cat([generated, next_tokens], dim=1)
            
            # Check for early stopping
            if config.eos_token_id is not None:
                if torch.all(next_tokens.squeeze(-1) == config.eos_token_id):
                    break
        
        return generated
    
    def _apply_repetition_penalty(
        self, 
        logits: torch.Tensor, 
        generated: torch.Tensor, 
        penalty: float
    ) -> torch.Tensor:
        """Apply repetition penalty to discourage repeating tokens"""
        for batch_idx in range(logits.shape[0]):
            for token_id in set(generated[batch_idx].tolist()):
                if logits[batch_idx, token_id] < 0:
                    logits[batch_idx, token_id] *= penalty
                else:
                    logits[batch_idx, token_id] /= penalty
        return logits
    
    def _apply_bad_words_filter(
        self, 
        logits: torch.Tensor, 
        bad_words_ids: List[List[int]]
    ) -> torch.Tensor:
        """Filter out bad words by setting their logits to -inf"""
        for bad_word in bad_words_ids:
            if len(bad_word) == 1:
                logits[:, bad_word[0]] = float('-inf')
        return logits
    
    def _apply_no_repeat_ngram_filter(
        self,
        logits: torch.Tensor,
        generated: torch.Tensor,
        ngram_size: int
    ) -> torch.Tensor:
        """Prevent repeating n-grams"""
        batch_size = logits.shape[0]
        vocab_size = logits.shape[1]
        
        for batch_idx in range(batch_size):
            # Get the sequence for this batch
            seq = generated[batch_idx].tolist()
            seq_len = len(seq)
            
            if seq_len >= ngram_size:
                # Get the last (ngram_size - 1) tokens
                ngram_prefix = seq[-(ngram_size - 1):]
                
                # Find all occurrences of this prefix in the sequence
                for i in range(seq_len - ngram_size + 1):
                    if seq[i:i + ngram_size - 1] == ngram_prefix:
                        # The token that would complete this n-gram
                        forbidden_token = seq[i + ngram_size - 1]
                        logits[batch_idx, forbidden_token] = float('-inf')
        
        return logits
    
    def _sample(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """Sample next token using various strategies"""
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
        
        # Apply top-k filtering
        if top_k is not None and top_k > 0:
            logits = self._top_k_filtering(logits, top_k)
        
        # Apply top-p (nucleus) filtering
        if top_p is not None and top_p < 1.0:
            logits = self._top_p_filtering(logits, top_p)
        
        # Sample from the filtered distribution
        probs = F.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1)
        
        return next_tokens
    
    def _top_k_filtering(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Filter logits to only keep top k candidates"""
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k, dim=-1)
        min_values = values[:, -1].unsqueeze(-1).expand_as(logits)
        logits = torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)
        return logits
    
    def _top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Filter logits using nucleus (top-p) sampling"""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least one token
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False
        
        # Scatter to original indexing
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        return logits
    
    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        **kwargs
    ) -> Union[str, List[str]]:
        """Generate text from a string prompt
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling or greedy decoding
            num_return_sequences: Number of sequences to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text(s)
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer is required for text generation")
        
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        input_ids = input_ids.to(self.device)
        
        # Repeat for multiple sequences
        if num_return_sequences > 1:
            input_ids = input_ids.repeat(num_return_sequences, 1)
        
        # Generate
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            **kwargs
        )
        
        generated_ids = self.generate(input_ids, generation_config)
        
        # Decode results
        results = []
        for i in range(num_return_sequences):
            # Remove the input prompt from the generated sequence
            new_tokens = generated_ids[i][input_ids.shape[1]:]
            generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            results.append(generated_text)
        
        return results[0] if num_return_sequences == 1 else results
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate response for chat conversation
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response
        """
        # Format conversation into a single prompt
        prompt = self._format_chat_prompt(messages)
        
        response = self.generate_text(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs
        )
        
        return response
    
    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages into a prompt"""
        prompt = ""
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role == 'system':
                prompt += f"System: {content}\n\n"
            elif role == 'user':
                prompt += f"User: {content}\n\n"
            elif role == 'assistant':
                prompt += f"Assistant: {content}\n\n"
        
        prompt += "Assistant: "
        return prompt
    
    def benchmark(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        batch_size: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """Benchmark generation performance
        
        Args:
            prompts: List of prompts to generate from
            max_new_tokens: Maximum tokens per generation
            batch_size: Batch size for generation
            **kwargs: Additional generation parameters
            
        Returns:
            Benchmark results
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer is required for benchmarking")
        
        total_prompts = len(prompts)
        total_time = 0.0
        total_tokens_generated = 0
        results = []
        
        # Process in batches
        for i in range(0, total_prompts, batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            # Encode batch
            encoded = self.tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True)
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded.get('attention_mask')
            
            # Time the generation
            start_time = time.time()
            
            generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                **kwargs
            )
            
            generated_ids = self.generate(input_ids, generation_config)
            
            end_time = time.time()
            batch_time = end_time - start_time
            total_time += batch_time
            
            # Decode results
            batch_results = []
            for j, prompt_ids in enumerate(generated_ids):
                # Count new tokens
                new_tokens = generated_ids[j][input_ids.shape[1]:]
                tokens_generated = len(new_tokens)
                total_tokens_generated += tokens_generated
                
                # Decode text
                generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                
                batch_results.append({
                    'prompt': batch_prompts[j],
                    'generated_text': generated_text,
                    'tokens_generated': tokens_generated,
                    'time_taken': batch_time / len(batch_prompts)
                })
            
            results.extend(batch_results)
        
        # Calculate metrics
        avg_time_per_prompt = total_time / total_prompts
        avg_tokens_per_second = total_tokens_generated / total_time if total_time > 0 else 0
        avg_tokens_per_prompt = total_tokens_generated / total_prompts
        
        return {
            'results': results,
            'total_prompts': total_prompts,
            'total_time': total_time,
            'total_tokens_generated': total_tokens_generated,
            'avg_time_per_prompt': avg_time_per_prompt,
            'avg_tokens_per_second': avg_tokens_per_second,
            'avg_tokens_per_prompt': avg_tokens_per_prompt
        }


def load_model_for_inference(
    checkpoint_path: str,
    device: Optional[str] = None,
    config_override: Optional[Dict[str, Any]] = None
) -> Tuple[GemmaModel, ExperimentConfig]:
    """Load trained model for inference
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        config_override: Configuration overrides
        
    Returns:
        Tuple of (model, config)
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load configuration
    config_dict = checkpoint['config']
    if config_override:
        # Apply overrides
        for key, value in config_override.items():
            if '.' in key:
                # Nested key like 'model.emb_dim'
                parts = key.split('.')
                current = config_dict
                for part in parts[:-1]:
                    current = current[part]
                current[parts[-1]] = value
            else:
                config_dict[key] = value
    
    config = ExperimentConfig.from_dict(config_dict)
    
    # Create model
    model = GemmaModel(config.model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, config


def create_generator_from_checkpoint(
    checkpoint_path: str,
    tokenizer=None,
    device: Optional[str] = None,
    generation_config: Optional[GenerationConfig] = None
) -> GemmaGenerator:
    """Create generator from checkpoint
    
    Args:
        checkpoint_path: Path to model checkpoint
        tokenizer: Tokenizer for text generation
        device: Device for inference
        generation_config: Generation configuration
        
    Returns:
        Initialized GemmaGenerator
    """
    model, config = load_model_for_inference(checkpoint_path, device)
    return GemmaGenerator(model, tokenizer, generation_config, device)


if __name__ == "__main__":
    # Example usage
    print("Gemma-270M Inference Module")
    print("This module provides text generation capabilities.")
    print("Use create_generator_from_checkpoint() to load a trained model.")
    
    # Example with dummy data
    from .config import get_default_config
    
    config = get_default_config()
    model = GemmaModel(config.model)
    
    generator = GemmaGenerator(model)
    print("✅ Generator initialized successfully!")
