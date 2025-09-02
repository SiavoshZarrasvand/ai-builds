#!/usr/bin/env python3
"""
Text generation script for Gemma-270M

This script provides a command-line interface for generating text using trained
Gemma-270M models with various sampling strategies and modes.

Usage:
    python generate.py --checkpoint checkpoints/best_model.pt --prompt "Once upon a time"
    python generate.py --checkpoint checkpoints/best_model.pt --interactive
    python generate.py --checkpoint checkpoints/best_model.pt --benchmark
"""

import argparse
import sys
import os
from pathlib import Path
import json
import torch
from transformers import GPT2Tokenizer

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from gemma_270m.inference import (
    GemmaGenerator, 
    GenerationConfig, 
    create_generator_from_checkpoint,
    load_model_for_inference
)
from gemma_270m.model import GemmaModel


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate text using Gemma-270M",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate from prompt
    python generate.py --checkpoint checkpoints/best_model.pt --prompt "Once upon a time"
    
    # Interactive generation
    python generate.py --checkpoint checkpoints/best_model.pt --interactive
    
    # Generate multiple sequences
    python generate.py --checkpoint checkpoints/best_model.pt --prompt "The future of AI" --num_sequences 3
    
    # Control sampling parameters
    python generate.py --checkpoint checkpoints/best_model.pt --prompt "Hello" --temperature 0.8 --top_k 40 --top_p 0.9
    
    # Benchmark generation speed
    python generate.py --checkpoint checkpoints/best_model.pt --benchmark
    
    # Chat mode
    python generate.py --checkpoint checkpoints/best_model.pt --chat
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    
    # Generation modes
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--prompt', '-p',
        type=str,
        help='Text prompt for generation'
    )
    mode_group.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Interactive generation mode'
    )
    mode_group.add_argument(
        '--chat',
        action='store_true',
        help='Chat conversation mode'
    )
    mode_group.add_argument(
        '--benchmark', '-b',
        action='store_true',
        help='Benchmark generation speed'
    )
    mode_group.add_argument(
        '--file', '-f',
        type=str,
        help='Generate from prompts in file (one per line)'
    )
    
    # Generation parameters
    gen_group = parser.add_argument_group('Generation Parameters')
    gen_group.add_argument('--max_tokens', type=int, default=100, help='Maximum tokens to generate')
    gen_group.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    gen_group.add_argument('--top_k', type=int, default=50, help='Top-k sampling')
    gen_group.add_argument('--top_p', type=float, default=0.9, help='Top-p (nucleus) sampling')
    gen_group.add_argument('--num_sequences', type=int, default=1, help='Number of sequences to generate')
    gen_group.add_argument('--no_sample', action='store_true', help='Use greedy decoding instead of sampling')
    gen_group.add_argument('--repetition_penalty', type=float, default=1.0, help='Repetition penalty')
    gen_group.add_argument('--seed', type=int, help='Random seed for reproducible generation')
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--output', '-o', type=str, help='Output file for generated text')
    output_group.add_argument('--json_output', action='store_true', help='Output results in JSON format')
    output_group.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Device options
    parser.add_argument('--device', type=str, help='Device to use (cuda/cpu)')
    
    return parser.parse_args()


def load_tokenizer():
    """Load tokenizer (GPT-2 compatible)"""
    try:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except Exception as e:
        print(f"⚠️  Warning: Could not load tokenizer: {e}")
        print("Text generation features will be limited without tokenizer.")
        return None


def generate_from_prompt(generator, prompt, args):
    """Generate text from a single prompt"""
    generation_config = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
        top_p=args.top_p if args.top_p < 1.0 else None,
        do_sample=not args.no_sample,
        num_return_sequences=args.num_sequences,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed
    )
    
    if args.verbose:
        print(f"🎯 Generation Config: {generation_config}")
        print(f"📝 Prompt: '{prompt}'")
        print("=" * 50)
    
    try:
        results = generator.generate_text(
            prompt=prompt,
            max_new_tokens=generation_config.max_new_tokens,
            temperature=generation_config.temperature,
            top_k=generation_config.top_k,
            top_p=generation_config.top_p,
            do_sample=generation_config.do_sample,
            num_return_sequences=generation_config.num_return_sequences,
            repetition_penalty=generation_config.repetition_penalty,
        )
        
        # Handle single vs multiple results
        if isinstance(results, str):
            results = [results]
        
        return results
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return None


def interactive_mode(generator, args):
    """Interactive text generation"""
    print("🔄 Interactive Generation Mode")
    print("Enter prompts (empty line to exit)")
    print("=" * 40)
    
    while True:
        try:
            prompt = input("\n📝 Prompt: ").strip()
            if not prompt:
                break
            
            results = generate_from_prompt(generator, prompt, args)
            if results:
                for i, result in enumerate(results):
                    if args.num_sequences > 1:
                        print(f"\n🤖 Generated {i+1}:")
                    else:
                        print(f"\n🤖 Generated:")
                    print(f"{result}")
                    
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def chat_mode(generator, args):
    """Chat conversation mode"""
    print("💬 Chat Mode")
    print("Start chatting! (type 'quit' to exit)")
    print("=" * 40)
    
    conversation = []
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            if user_input.lower() in ['quit', 'exit', 'bye']:
                break
            
            # Add user message to conversation
            conversation.append({"role": "user", "content": user_input})
            
            # Generate response
            try:
                response = generator.chat(
                    messages=conversation,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k if args.top_k > 0 else None,
                    top_p=args.top_p if args.top_p < 1.0 else None,
                    do_sample=not args.no_sample,
                )
                
                print(f"🤖 Assistant: {response}")
                
                # Add assistant response to conversation
                conversation.append({"role": "assistant", "content": response})
                
            except Exception as e:
                print(f"❌ Generation failed: {e}")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break


def benchmark_mode(generator, args):
    """Benchmark generation performance"""
    print("🏃 Benchmark Mode")
    print("Testing generation speed...")
    
    # Default benchmark prompts
    prompts = [
        "Once upon a time",
        "The future of artificial intelligence",
        "In a world where technology",
        "The greatest discovery in science",
        "Climate change is affecting"
    ]
    
    try:
        results = generator.benchmark(
            prompts=prompts,
            max_new_tokens=args.max_tokens,
            batch_size=1,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
            top_p=args.top_p if args.top_p < 1.0 else None,
            do_sample=not args.no_sample,
        )
        
        print("\n📊 Benchmark Results:")
        print("=" * 50)
        print(f"Total Prompts: {results['total_prompts']}")
        print(f"Total Time: {results['total_time']:.2f}s")
        print(f"Total Tokens Generated: {results['total_tokens_generated']:,}")
        print(f"Average Time/Prompt: {results['avg_time_per_prompt']:.2f}s")
        print(f"Average Tokens/Second: {results['avg_tokens_per_second']:.1f}")
        print(f"Average Tokens/Prompt: {results['avg_tokens_per_prompt']:.1f}")
        
        if args.verbose:
            print("\n📝 Individual Results:")
            for i, result in enumerate(results['results']):
                print(f"\nPrompt {i+1}: '{result['prompt']}'")
                print(f"Generated: {result['tokens_generated']} tokens in {result['time_taken']:.2f}s")
                print(f"Text: {result['generated_text'][:100]}...")
        
        return results
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return None


def generate_from_file(generator, file_path, args):
    """Generate from prompts in file"""
    print(f"📁 Generating from file: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        
        print(f"Found {len(prompts)} prompts")
        
        all_results = []
        for i, prompt in enumerate(prompts):
            print(f"\n🔄 Processing prompt {i+1}/{len(prompts)}")
            if args.verbose:
                print(f"Prompt: '{prompt}'")
            
            results = generate_from_prompt(generator, prompt, args)
            if results:
                all_results.extend([{
                    'prompt': prompt,
                    'generated': result
                } for result in results])
        
        return all_results
        
    except Exception as e:
        print(f"❌ Failed to process file: {e}")
        return None


def save_results(results, output_path, json_format=False):
    """Save results to file"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            if json_format:
                json.dump(results, f, indent=2, ensure_ascii=False)
            else:
                if isinstance(results, list) and all(isinstance(r, dict) for r in results):
                    # File-based results
                    for result in results:
                        f.write(f"Prompt: {result['prompt']}\n")
                        f.write(f"Generated: {result['generated']}\n")
                        f.write("-" * 50 + "\n")
                else:
                    # Simple text results
                    if isinstance(results, list):
                        for i, result in enumerate(results):
                            f.write(f"Generated {i+1}:\n{result}\n\n")
                    else:
                        f.write(str(results))
        
        print(f"💾 Results saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Failed to save results: {e}")


def main():
    """Main function"""
    args = parse_arguments()
    
    print("🤗 Gemma-270M Text Generation")
    print("=" * 40)
    
    try:
        # Load tokenizer
        print("📖 Loading tokenizer...")
        tokenizer = load_tokenizer()
        
        # Load model
        print(f"🏗️  Loading model from: {args.checkpoint}")
        generator = create_generator_from_checkpoint(
            checkpoint_path=args.checkpoint,
            tokenizer=tokenizer,
            device=args.device
        )
        
        print("✅ Model loaded successfully!")
        
        # Determine mode and execute
        results = None
        
        if args.prompt:
            # Single prompt generation
            results = generate_from_prompt(generator, args.prompt, args)
            if results:
                for i, result in enumerate(results):
                    if args.num_sequences > 1:
                        print(f"\n🤖 Generated {i+1}:")
                    else:
                        print(f"\n🤖 Generated:")
                    print(result)
        
        elif args.interactive:
            # Interactive mode
            interactive_mode(generator, args)
        
        elif args.chat:
            # Chat mode
            chat_mode(generator, args)
        
        elif args.benchmark:
            # Benchmark mode
            results = benchmark_mode(generator, args)
        
        elif args.file:
            # File-based generation
            results = generate_from_file(generator, args.file, args)
        
        else:
            # Default: prompt user for input
            prompt = input("📝 Enter prompt: ").strip()
            if prompt:
                results = generate_from_prompt(generator, prompt, args)
                if results:
                    for result in results:
                        print(f"\n🤖 Generated:\n{result}")
        
        # Save results if specified
        if results and args.output:
            save_results(results, args.output, args.json_output)
        
        print("\n✅ Generation completed!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Generation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
