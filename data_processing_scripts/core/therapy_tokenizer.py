#!/usr/bin/env python3
"""
Stage 2: Resume-Capable Tokenization
Tokenizes instruction examples with checkpoint system for resuming.
"""

import sys
import json
import os
import signal
from pathlib import Path
from therapy_data_pipeline import TherapyDataProcessor, ProcessingConfig
import logging

class CheckpointManager:
    """Manages checkpoint saving and loading for tokenization."""
    
    def __init__(self, checkpoint_file):
        self.checkpoint_file = checkpoint_file
        self.current_chunk = 0
        self.total_chunks = 0
        
    def save_checkpoint(self, chunk_index, processed_count):
        """Save current progress."""
        checkpoint_data = {
            'current_chunk': chunk_index,
            'total_chunks': self.total_chunks,
            'processed_count': processed_count,
            'last_processed_chunk': chunk_index
        }
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        print(f"Checkpoint saved: chunk {chunk_index}/{self.total_chunks}")
    
    def load_checkpoint(self):
        """Load previous progress if exists."""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
            
            self.current_chunk = data.get('current_chunk', 0)
            self.total_chunks = data.get('total_chunks', 0)
            print(f"Resuming from chunk {self.current_chunk}/{self.total_chunks}")
            return True
        return False
    
    def clear_checkpoint(self):
        """Clear checkpoint file."""
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)

def tokenize_with_resume(input_file, output_dir, checkpoint_file="tokenization_checkpoint.json"):
    """Tokenize instruction examples with resume capability."""
    
    print("STAGE 2: RESUME-CAPABLE TOKENIZATION")
    print("=" * 60)
    print("Tokenizing instruction examples with checkpoint system")
    print("Can be safely terminated and resumed")
    print("=" * 60)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configuration
    config = ProcessingConfig(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        max_length=2048,
        input_data_dir="",  # Not needed for this stage
        output_dir=output_dir,
        cache_dir=".cache"
    )
    
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    print()
    
    try:
        # Load instruction examples
        print("Loading instruction examples...")
        with open(input_file, 'r', encoding='utf-8') as f:
            examples = json.load(f)
        
        print(f"Loaded {len(examples)} instruction examples")
        
        # Create processor
        processor = TherapyDataProcessor(config)
        processor.load_tokenizer()
        
        # Setup checkpoint manager
        checkpoint_manager = CheckpointManager(checkpoint_file)
        checkpoint_manager.total_chunks = len(examples)
        
        # Try to resume from checkpoint
        start_chunk = 0
        if checkpoint_manager.load_checkpoint():
            start_chunk = checkpoint_manager.current_chunk
            print(f"Resuming from chunk {start_chunk}")
        else:
            print("Starting fresh tokenization")
        
        # Setup signal handler for graceful shutdown
        def signal_handler(signum, frame):
            print(f"\nReceived signal {signum}. Saving checkpoint...")
            checkpoint_manager.save_checkpoint(start_chunk, 0)
            print("Checkpoint saved. You can resume later.")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Process examples in batches
        batch_size = 100
        tokenized_examples = []
        
        print(f"Starting tokenization from chunk {start_chunk}...")
        print("Progress will be shown below:")
        print("-" * 50)
        
        for i in range(start_chunk, len(examples), batch_size):
            batch_end = min(i + batch_size, len(examples))
            batch = examples[i:batch_end]
            
            # Show progress
            progress = (i + 1) / len(examples) * 100
            bar_length = 40
            filled_length = int(bar_length * (i + 1) // len(examples))
            bar = '#' * filled_length + '-' * (bar_length - filled_length)
            print(f'\rTokenizing: |{bar}| {progress:.1f}% ({i + 1}/{len(examples)})', end='', flush=True)
            
            # Tokenize batch
            batch_tokenized = tokenize_batch(processor, batch)
            tokenized_examples.extend(batch_tokenized)
            
            # Save checkpoint every batch
            checkpoint_manager.save_checkpoint(batch_end, len(tokenized_examples))
        
        print()  # New line after progress bar
        
        # Create dataset
        print("Creating HuggingFace dataset...")
        from datasets import Dataset
        dataset = Dataset.from_list(tokenized_examples)
        
        # Create train/val/test splits
        print("Creating train/validation/test splits...")
        dataset_dict = processor.create_train_val_split(dataset)
        
        # Save processed data
        print("Saving processed data...")
        processor.save_processed_data(dataset_dict)
        
        # Clear checkpoint on successful completion
        checkpoint_manager.clear_checkpoint()
        
        print(f"\nSTAGE 2 COMPLETED SUCCESSFULLY!")
        print(f"Tokenized {len(tokenized_examples)} examples")
        print(f"Data saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"Error during tokenization: {e}")
        logging.exception("Tokenization failed")
        return False

def tokenize_batch(processor, batch):
    """Tokenize a batch of examples."""
    
    def tokenize_function(examples):
        """Tokenization function for batched processing."""
        # Combine input and output
        texts = []
        for input_text, output in zip(examples['input'], examples['output']):
            full_text = f"{input_text}\n\nResponse: {output}"
            texts.append(full_text)
        
        # Tokenize
        tokenized = processor.tokenizer(
            texts,
            truncation=processor.config.truncation,
            max_length=processor.config.max_length,
            padding=False,
            return_tensors=None
        )
        
        return tokenized
    
    # Create temporary dataset for batch processing
    from datasets import Dataset
    temp_dataset = Dataset.from_list(batch)
    
    # Apply tokenization
    tokenized_dataset = temp_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=temp_dataset.column_names,
        desc="Tokenizing batch"
    )
    
    # Convert to list
    return [dict(tokenized_dataset[i]) for i in range(len(tokenized_dataset))]

def main():
    """Main function."""
    if len(sys.argv) != 3:
        print("Usage: python therapy_tokenizer.py <input_file> <output_dir>")
        print("Example: python therapy_tokenizer.py ../data/instruction_examples/instruction_examples.json ../data/processed")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    success = tokenize_with_resume(input_file, output_dir)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()


