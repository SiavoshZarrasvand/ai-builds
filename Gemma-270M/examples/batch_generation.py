#!/usr/bin/env python3
"""
Batch Text Generation Example
============================

Processes multiple prompts from a CSV file and generates completions.
Useful for generating training data, evaluations, or bulk content.

Expected CSV format:
    prompt,expected_length,temperature
    "Once upon a time,",100,0.8
    "The future is",150,0.9

Run this after training a model with run_pipeline.py --quick or --full
"""

import os
import sys
import csv
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Run batch text generation"""
    print("📋 Batch Gemma-270M Text Generation")
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
    
    # Create example input file if it doesn't exist
    input_file = "examples/batch_prompts.csv"
    if not os.path.exists(input_file):
        create_example_csv(input_file)
        print(f"📝 Created example input file: {input_file}")
    
    # Process the batch
    output_file = f"examples/batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    process_batch(generator, input_file, output_file)
    
    print(f"\n🎉 Batch generation complete!")
    print(f"📄 Results saved to: {output_file}")
    
def create_example_csv(filename):
    """Create an example CSV file with prompts"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    example_prompts = [
        ("Once upon a time in a distant galaxy,", 120, 0.8),
        ("The secret to happiness is", 100, 0.7),
        ("In the year 2030, technology will", 150, 0.9),
        ("The most important invention in history was", 80, 0.6),
        ("If I could travel anywhere, I would go to", 100, 0.8),
        ("The best advice I ever received was", 90, 0.7),
        ("Climate change will affect our future by", 140, 0.8),
        ("Artificial intelligence will revolutionize", 130, 0.9),
        ("The key to learning new skills is", 110, 0.75),
        ("My dream job would involve", 100, 0.8)
    ]
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['prompt', 'max_tokens', 'temperature'])
        writer.writerows(example_prompts)

def process_batch(generator, input_file, output_file):
    """Process batch of prompts from CSV file"""
    results = []
    
    print(f"\n🚀 Processing prompts from: {input_file}")
    
    # Read input prompts
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        prompts = list(reader)
    
    print(f"📊 Found {len(prompts)} prompts to process")
    
    # Process each prompt
    for i, row in enumerate(prompts, 1):
        prompt = row['prompt'].strip('"')
        max_tokens = int(row.get('max_tokens', 100))
        temperature = float(row.get('temperature', 0.8))
        
        print(f"\n🔄 Processing {i}/{len(prompts)}: {prompt[:50]}...")
        
        try:
            # Generate text
            start_time = datetime.now()
            response = generator.generate_text(
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True
            )
            end_time = datetime.now()
            generation_time = (end_time - start_time).total_seconds()
            
            # Store result
            result = {
                'prompt': prompt,
                'generated_text': response,
                'full_text': f"{prompt} {response}",
                'max_tokens': max_tokens,
                'temperature': temperature,
                'generation_time_seconds': round(generation_time, 2),
                'generated_tokens': len(response.split()),
                'timestamp': datetime.now().isoformat()
            }
            
            results.append(result)
            
            print(f"✅ Generated {len(response.split())} tokens in {generation_time:.2f}s")
            
        except Exception as e:
            print(f"❌ Error processing prompt: {e}")
            
            # Store error result
            result = {
                'prompt': prompt,
                'generated_text': '',
                'full_text': '',
                'max_tokens': max_tokens,
                'temperature': temperature,
                'generation_time_seconds': 0,
                'generated_tokens': 0,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
            results.append(result)
    
    # Save results
    save_results(results, output_file)
    print(f"\n📊 Batch Summary:")
    print(f"   Total prompts: {len(prompts)}")
    print(f"   Successful: {sum(1 for r in results if not r.get('error'))}")
    print(f"   Errors: {sum(1 for r in results if r.get('error'))}")
    
    if results:
        avg_time = sum(r['generation_time_seconds'] for r in results if r['generation_time_seconds'] > 0) / len(results)
        print(f"   Average generation time: {avg_time:.2f}s")

def save_results(results, output_file):
    """Save results to CSV and JSON files"""
    if not results:
        print("❌ No results to save")
        return
    
    # Save as CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = results[0].keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    # Also save as JSON for easier programmatic access
    json_file = output_file.replace('.csv', '.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Results saved to:")
    print(f"   CSV: {output_file}")
    print(f"   JSON: {json_file}")

if __name__ == "__main__":
    main()
