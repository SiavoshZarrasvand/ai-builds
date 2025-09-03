#!/usr/bin/env python3
"""
Simple Text Generation Example
=============================

Demonstrates basic text generation using a trained Gemma-270M model.
Run this after training a model with run_pipeline.py --quick or --full
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Generate text using the best trained model"""
    print("🎯 Simple Gemma-270M Text Generation")
    print("=" * 50)
    
    # Check if model exists
    checkpoint_path = "checkpoints/best_model.pt"
    if not os.path.exists(checkpoint_path):
        print(f"❌ No trained model found at {checkpoint_path}")
        print("Please run training first:")
        print("   python run_pipeline.py --quick")
        return
    
    # Import after checking checkpoint exists
    try:
        from gemma_270m.inference import create_generator_from_checkpoint  
        from transformers import GPT2Tokenizer
        import torch
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you have activated the virtual environment and installed dependencies")
        return
    
    print(f"✅ Loading model from: {checkpoint_path}")
    
    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    generator = create_generator_from_checkpoint(
        checkpoint_path=checkpoint_path,
        tokenizer=tokenizer
    )
    
    print(f"✅ Model loaded successfully!")
    print(f"📊 CUDA Available: {torch.cuda.is_available()}")
    
    # Example prompts
    prompts = [
        "Once upon a time in a magical forest,",
        "The future of artificial intelligence is",
        "In the year 2050, humans will",
        "The most important lesson I learned was",
        "Technology has changed our lives by"
    ]
    
    print("\n🚀 Generating text samples...")
    print("=" * 50)
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n📝 Example {i}: {prompt}")
        print("-" * 40)
        
        try:
            # Generate text
            response = generator.generate_text(
                prompt=prompt,
                max_new_tokens=100,
                temperature=0.8,
                top_p=0.9,
                do_sample=True
            )
            
            print(f"{prompt} {response}")
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
    
    print(f"\n🎉 Text generation complete!")
    print(f"💡 Try running with different prompts:")
    print(f"   python -c \"from examples.simple_generation import generate_custom; generate_custom('Your prompt here')\"")

def generate_custom(prompt: str):
    """Generate text from a custom prompt"""
    checkpoint_path = "checkpoints/best_model.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ No model found at {checkpoint_path}")
        return
    
    from gemma_270m.inference import create_generator_from_checkpoint
    from transformers import GPT2Tokenizer
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    generator = create_generator_from_checkpoint(
        checkpoint_path=checkpoint_path, 
        tokenizer=tokenizer
    )
    
    response = generator.generate_text(
        prompt=prompt,
        max_new_tokens=150,
        temperature=0.8,
        top_p=0.9
    )
    
    print(f"\n🎯 Generated text:")
    print(f"{prompt} {response}")

if __name__ == "__main__":
    main()
