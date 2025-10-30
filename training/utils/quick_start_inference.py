"""
Quick Start Script for TheraBot Inference
Run this to test your trained adapter with DBT system prompts.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from load_adapter_for_inference import TheraBotInference
import torch

def main():
    print("=" * 80)
    print("THERABOT QUICK START")
    print("=" * 80)
    
    # Configuration
    print("\n⚙️  Configuration:")
    
    # TODO: Update these paths
    ADAPTER_PATH = "./therapy-model-checkpoints"  # Your trained adapter path
    BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
    
    print(f"   Base Model: {BASE_MODEL}")
    print(f"   Adapter Path: {ADAPTER_PATH}")
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("   GPU: Not available (will use CPU - slower)")
    
    # Initialize
    print("\n🚀 Initializing TheraBot...")
    try:
        bot = TheraBotInference(
            base_model_name=BASE_MODEL,
            adapter_path=ADAPTER_PATH,
            use_4bit=True  # Use 4-bit for memory efficiency
        )
        print("✅ TheraBot initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing TheraBot: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Make sure your adapter path is correct")
        print("   2. Check that you have access to the base model")
        print("   3. Verify you have the required packages installed")
        return
    
    # Interactive chat loop
    print("\n" + "=" * 80)
    print("INTERACTIVE CHAT")
    print("=" * 80)
    print("\n💡 Type 'quit' to exit")
    print("💡 Type 'help' for available commands")
    print("-" * 80)
    
    conversation_history = []
    
    while True:
        try:
            # Get user input
            user_input = input("\n🤔 You: ").strip()
            
            if not user_input:
                continue
            
            # Check for commands
            if user_input.lower() == 'quit':
                print("\n👋 Goodbye!")
                break
            elif user_input.lower() == 'help':
                print("\n💬 Available commands:")
                print("   quit - Exit the chat")
                print("   history - Show conversation history")
                print("   clear - Clear conversation history")
                print("   crisis - Test crisis detection")
                continue
            elif user_input.lower() == 'history':
                print("\n📜 Conversation History:")
                for i, (user, bot) in enumerate(conversation_history, 1):
                    print(f"\n{i}. You: {user}")
                    print(f"   Bot: {bot}")
                continue
            elif user_input.lower() == 'clear':
                conversation_history = []
                print("✅ History cleared")
                continue
            elif user_input.lower() == 'crisis':
                user_input = "I want to kill myself"
                print(f"🤔 You: {user_input}")
            
            # Detect crisis
            is_crisis = bot.detect_crisis(user_input)
            
            # Generate response
            print("\n💭 TheraBot thinking...")
            
            response = bot.chat(
                user_message=user_input,
                conversation_history=conversation_history,
                use_dbt_prompt=True,
                dbt_context="crisis" if is_crisis else "general",
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9
            )
            
            # Display response
            print(f"\n💬 TheraBot: {response}")
            
            if is_crisis:
                print("⚠️  [CRISIS DETECTED - Crisis protocol activated]")
            
            # Update history
            conversation_history.append((user_input, response))
            
            # Keep history manageable
            if len(conversation_history) > 10:
                conversation_history = conversation_history[-10:]
            
            print("-" * 80)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("💡 Try again or type 'quit' to exit")


if __name__ == "__main__":
    main()

