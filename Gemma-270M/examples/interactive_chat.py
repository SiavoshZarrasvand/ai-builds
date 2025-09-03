#!/usr/bin/env python3
"""
Interactive Chat Example
========================

Creates an interactive chat session with the trained Gemma-270M model.
Run this after training a model with run_pipeline.py --quick or --full
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Start interactive chat session"""
    print("💬 Interactive Gemma-270M Chat")
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
    print("\n💡 Chat Tips:")
    print("  - Type 'quit' or 'exit' to end the conversation")
    print("  - Type 'help' for more commands")
    print("  - Try conversation starters, questions, or story prompts")
    print("\n🚀 Chat session started! Say hello to your AI...")
    print("=" * 50)
    
    # Chat configuration
    chat_config = {
        'max_new_tokens': 120,
        'temperature': 0.9,
        'top_p': 0.85,
        'do_sample': True
    }
    
    conversation_history = []
    
    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
                
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\n👋 Goodbye! Thanks for chatting!")
                break
            elif user_input.lower() == 'help':
                show_help()
                continue
            elif user_input.lower() == 'clear':
                conversation_history.clear()
                print("\n🗑️ Conversation history cleared!")
                continue
            elif user_input.lower() == 'config':
                show_config(chat_config)
                continue
            elif user_input.lower().startswith('temp '):
                try:
                    new_temp = float(user_input.split()[1])
                    if 0.1 <= new_temp <= 2.0:
                        chat_config['temperature'] = new_temp
                        print(f"🌡️ Temperature set to {new_temp}")
                    else:
                        print("❌ Temperature must be between 0.1 and 2.0")
                except (IndexError, ValueError):
                    print("❌ Usage: temp 0.8")
                continue
            
            # Add to conversation history
            conversation_history.append(f"Human: {user_input}")
            
            # Create context from recent history (last 3 exchanges)
            recent_history = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
            context = "\n".join(recent_history)
            
            # Format prompt for conversation
            if context:
                prompt = f"{context}\nAI:"
            else:
                prompt = f"Human: {user_input}\nAI:"
            
            print("🤖 AI:", end=" ", flush=True)
            
            # Generate response
            response = generator.generate_text(
                prompt=prompt,
                **chat_config
            )
            
            # Clean up the response (remove the prompt part if it's included)
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            # Clean up common artifacts
            response = response.split('\n')[0].strip()  # Take first line only
            
            print(response)
            
            # Add AI response to history
            conversation_history.append(f"AI: {response}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! (Ctrl+C pressed)")
            break
        except Exception as e:
            print(f"\n❌ Error generating response: {e}")
            print("💡 Try a different prompt or restart the chat")

def show_help():
    """Show available commands"""
    print("\n📖 Available Commands:")
    print("  quit/exit/bye - End the conversation")
    print("  clear         - Clear conversation history")
    print("  config        - Show current generation settings")
    print("  temp X.X      - Set temperature (0.1-2.0, default 0.9)")
    print("  help          - Show this help message")
    print("\n💭 Chat Tips:")
    print("  - Ask questions, request stories, or have conversations")
    print("  - The model remembers recent context from your chat")
    print("  - Try different temperatures for varied creativity")

def show_config(config):
    """Show current chat configuration"""
    print("\n⚙️ Current Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
