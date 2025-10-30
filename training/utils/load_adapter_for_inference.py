"""
Load trained LoRA adapter for inference with DBT system prompts.
This demonstrates how to use your trained adapter with DBT knowledge injection.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from typing import List, Tuple, Optional
import logging

# Import DBT prompts
from dbt_system_prompts import (
    DBT_SYSTEM_PROMPT,
    DBT_SYSTEM_PROMPT_COMPACT,
    get_dbt_context_prompt,
    format_dbt_chat_prompt
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TheraBotInference:
    """
    Load and run inference with trained LoRA adapter + DBT system prompts.
    
    This combines:
    1. Your trained LoRA adapter (therapist conversation skills)
    2. DBT system prompts (DBT knowledge injection)
    """
    
    def __init__(
        self,
        base_model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        adapter_path: str = None,
        use_4bit: bool = True,
        load_in_8bit: bool = False
    ):
        """
        Initialize TheraBot with adapter and DBT prompts.
        
        Args:
            base_model_name: Base model name from HuggingFace
            adapter_path: Path to your trained LoRA adapter
            use_4bit: Use 4-bit quantization for memory efficiency
            load_in_8bit: Use 8-bit quantization (alternative to 4-bit)
        """
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        
        logger.info("🚀 Initializing TheraBot Inference Engine...")
        
        # Load tokenizer
        logger.info("📥 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure quantization if needed
        quantization_config = None
        if use_4bit:
            logger.info("🔧 Setting up 4-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        elif load_in_8bit:
            logger.info("🔧 Setting up 8-bit quantization...")
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Load base model
        logger.info(f"📥 Loading base model: {base_model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16 if not quantization_config else None,
            trust_remote_code=True
        )
        
        # Load LoRA adapter if provided
        if adapter_path:
            logger.info(f"📦 Loading trained LoRA adapter from: {adapter_path}...")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            logger.info("✅ Adapter loaded successfully")
        else:
            logger.warning("⚠️  No adapter path provided - using base model only")
        
        # Set to eval mode
        self.model.eval()
        logger.info("✅ TheraBot initialized successfully!")
    
    def chat(
        self,
        user_message: str,
        conversation_history: List[Tuple[str, str]] = None,
        use_dbt_prompt: bool = True,
        dbt_context: str = "general",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        use_compact_prompt: bool = False
    ) -> str:
        """
        Generate therapeutic response with DBT knowledge injection.
        
        Args:
            user_message: User's current message
            conversation_history: List of (user_msg, assistant_msg) tuples
            use_dbt_prompt: Whether to inject DBT system prompt
            dbt_context: DBT context type (general, crisis, skills, validation)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            use_compact_prompt: Use compact DBT prompt for faster inference
        
        Returns:
            Generated therapeutic response
        """
        # Select DBT prompt
        if use_dbt_prompt:
            if use_compact_prompt:
                system_prompt = DBT_SYSTEM_PROMPT_COMPACT
            else:
                system_prompt = get_dbt_context_prompt(dbt_context)
        else:
            system_prompt = "You are a helpful therapist."
        
        # Format prompt with chat template
        formatted_prompt = format_dbt_chat_prompt(
            system_prompt=system_prompt,
            user_message=user_message,
            conversation_history=conversation_history
        )
        
        # Tokenize
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode response
        # Get only the newly generated tokens (after the prompt)
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()
    
    def detect_crisis(self, user_message: str) -> bool:
        """
        Detect if user message contains crisis indicators.
        
        Args:
            user_message: User's message
        
        Returns:
            True if crisis detected
        """
        crisis_keywords = [
            "kill myself", "kill myself", "suicide", "end it all",
            "not worth living", "hurt myself", "self harm", "cut myself",
            "want to die", "better off dead", "give up"
        ]
        
        user_lower = user_message.lower()
        return any(keyword in user_lower for keyword in crisis_keywords)
    
    def chat_with_crisis_detection(
        self,
        user_message: str,
        conversation_history: List[Tuple[str, str]] = None,
        **kwargs
    ) -> Tuple[str, bool]:
        """
        Chat with automatic crisis detection.
        
        Returns:
            Tuple of (response, is_crisis)
        """
        is_crisis = self.detect_crisis(user_message)
        
        if is_crisis:
            # Override context to crisis mode
            response = self.chat(
                user_message,
                conversation_history,
                dbt_context="crisis",
                **kwargs
            )
        else:
            response = self.chat(
                user_message,
                conversation_history,
                **kwargs
            )
        
        return response, is_crisis


# Example usage and testing
def main():
    """Demo script showing how to use TheraBot."""
    
    print("=" * 80)
    print("THERABOT INFERENCE DEMO")
    print("=" * 80)
    
    # Initialize TheraBot
    # NOTE: Replace with your actual adapter path
    adapter_path = "./therapy-model-checkpoints"  # Your trained adapter path
    
    bot = TheraBotInference(
        base_model_name="meta-llama/Llama-3.1-8B-Instruct",
        adapter_path=adapter_path,
        use_4bit=True  # Use 4-bit for memory efficiency
    )
    
    # Example conversation
    print("\n" + "=" * 80)
    print("EXAMPLE CONVERSATION")
    print("=" * 80)
    
    # User messages to test
    test_messages = [
        "I'm feeling really overwhelmed with work and personal stuff.",
        "I don't know what to do. Everything feels hopeless.",
        "Can you teach me the TIPP skills?",
    ]
    
    conversation_history = []
    
    for i, user_msg in enumerate(test_messages):
        print(f"\n🤔 User {i+1}: {user_msg}")
        
        # Get response with crisis detection
        response, is_crisis = bot.chat_with_crisis_detection(
            user_message=user_msg,
            conversation_history=conversation_history,
            use_dbt_prompt=True,
            use_compact_prompt=False
        )
        
        print(f"\n💬 TheraBot: {response}")
        if is_crisis:
            print("⚠️  [CRISIS DETECTED - Crisis protocol activated]")
        
        # Update history
        conversation_history.append((user_msg, response))
        
        print("-" * 80)
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

