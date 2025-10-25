#!/usr/bin/env python3
"""
Comprehensive Therapy Data Processing Pipeline
Following HuggingFace LLM Course Chapter 3.2: https://huggingface.co/learn/llm-course/chapter3/2

This pipeline implements:
- Batched processing with Dataset.map() for efficiency
- Dynamic padding with DataCollatorWithPadding
- Proper tokenization for Llama 3.1
- Session-level chunking with overlap for long conversations
- Instruction-following format conversion
"""

import json
import os
import re
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging

import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    DataCollatorWithPadding,
    PreTrainedTokenizer
)
from transformers.tokenization_utils_base import BatchEncoding
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configuration for data processing pipeline."""
    # Model and tokenization
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    max_length: int = 2048  # Maximum sequence length
    padding: str = "max_length"
    truncation: bool = True
    
    # Session chunking
    max_session_exchanges: int = 300  # Split sessions longer than this
    overlap_percentage: float = 0.2   # 20% overlap for context preservation
    
    # Data filtering
    min_exchange_length: int = 1      # Minimum words per exchange
    max_exchange_length: int = 200    # Maximum words per exchange
    
    # Output format
    instruction_template: str = ""  # No instruction - just client-therapist conversation pattern
    
    # File paths
    input_data_dir: str = "data/training_data"
    output_dir: str = "data/processed"
    cache_dir: str = ".cache"

class TherapyDataProcessor:
    """
    Main data processing class following HuggingFace best practices.
    
    Reference: https://huggingface.co/learn/llm-course/chapter3/2
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.tokenizer = None
        self.data_collator = None
        
        # Create output directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.cache_dir, exist_ok=True)
        
    def load_tokenizer(self) -> PreTrainedTokenizer:
        """
        Load and configure tokenizer for Llama 3.1.
        
        Reference: https://huggingface.co/docs/transformers/main/en/tokenizer_summary
        """
        logger.info(f"Loading tokenizer for {self.config.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir,
            trust_remote_code=True
        )
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Configure data collator for dynamic padding
        self.data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True,
            return_tensors="pt"
        )
        
        logger.info(f"Tokenizer loaded. Vocab size: {len(self.tokenizer)}")
        return self.tokenizer
    
    def load_therapy_sessions(self) -> List[Dict]:
        """
        Load all therapy sessions from the data directory.
        
        Returns:
            List of session dictionaries with dialogue data
        """
        logger.info(f"Loading therapy sessions from {self.config.input_data_dir}")
        
        sessions = []
        data_path = Path(self.config.input_data_dir)
        
        # Process all JSON files in subdirectories
        for json_file in data_path.rglob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract dialogue from various possible structures
                dialogue = self._extract_dialogue(data)
                if dialogue:
                    session = {
                        'session_id': self._extract_session_id(data, json_file),
                        'dialogue': dialogue,
                        'source_file': str(json_file)
                    }
                    sessions.append(session)
                    
            except Exception as e:
                logger.warning(f"Error loading {json_file}: {e}")
                continue
        
        logger.info(f"Loaded {len(sessions)} therapy sessions")
        return sessions
    
    def _extract_dialogue(self, data: Dict) -> Optional[List[Dict]]:
        """Extract dialogue from various JSON structures."""
        if isinstance(data, list):
            return data
        elif 'conversations' in data:
            return data['conversations']
        elif 'dialogue' in data:
            return data['dialogue']
        elif 'messages' in data:
            return data['messages']
        return None
    
    def _extract_session_id(self, data: Dict, file_path: Path) -> str:
        """Extract or generate session ID."""
        if isinstance(data, dict) and 'metadata' in data:
            return data['metadata'].get('session_id', file_path.stem)
        elif isinstance(data, dict) and 'session_id' in data:
            return data['session_id']
        else:
            return file_path.stem
    
    def filter_exchanges(self, exchanges: List[Dict]) -> List[Dict]:
        """
        Filter exchanges based on length and quality criteria.
        
        Args:
            exchanges: List of dialogue exchanges
            
        Returns:
            Filtered list of exchanges
        """
        filtered = []
        
        for exchange in exchanges:
            if not isinstance(exchange, dict) or 'message' not in exchange:
                continue
                
            message = exchange['message'].strip()
            word_count = len(message.split())
            
            # Apply length filters
            if (word_count >= self.config.min_exchange_length and 
                word_count <= self.config.max_exchange_length):
                filtered.append(exchange)
        
        return filtered
    
    def chunk_long_sessions(self, sessions: List[Dict]) -> List[Dict]:
        """
        Split long sessions into chunks with overlap for context preservation.
        
        Reference: Session-level chunking strategy from analysis
        
        Args:
            sessions: List of session dictionaries
            
        Returns:
            List of session chunks (original + split sessions)
        """
        logger.info("Processing session-level chunking...")
        
        chunked_sessions = []
        
        for session in sessions:
            dialogue = session['dialogue']
            num_exchanges = len(dialogue)
            
            if num_exchanges <= self.config.max_session_exchanges:
                # Keep short sessions intact
                chunked_sessions.append(session)
            else:
                # Split long sessions with overlap
                logger.info(f"Splitting session {session['session_id']} ({num_exchanges} exchanges)")
                
                chunks = self._split_session_with_overlap(
                    session, 
                    self.config.max_session_exchanges,
                    self.config.overlap_percentage
                )
                chunked_sessions.extend(chunks)
        
        logger.info(f"Created {len(chunked_sessions)} session chunks from {len(sessions)} original sessions")
        return chunked_sessions
    
    def _split_session_with_overlap(self, session: Dict, max_exchanges: int, overlap_pct: float) -> List[Dict]:
        """
        Split a long session into chunks with overlap.
        
        Args:
            session: Session dictionary
            max_exchanges: Maximum exchanges per chunk
            overlap_pct: Overlap percentage (0.0 to 1.0)
            
        Returns:
            List of session chunks
        """
        dialogue = session['dialogue']
        overlap_size = int(max_exchanges * overlap_pct)
        step_size = max_exchanges - overlap_size
        
        chunks = []
        start_idx = 0
        
        while start_idx < len(dialogue):
            end_idx = min(start_idx + max_exchanges, len(dialogue))
            
            chunk_dialogue = dialogue[start_idx:end_idx]
            
            chunk_session = {
                'session_id': f"{session['session_id']}_chunk_{len(chunks)}",
                'dialogue': chunk_dialogue,
                'source_file': session['source_file'],
                'chunk_info': {
                    'original_session': session['session_id'],
                    'chunk_index': len(chunks),
                    'start_exchange': start_idx,
                    'end_exchange': end_idx,
                    'total_exchanges': len(chunk_dialogue)
                }
            }
            
            chunks.append(chunk_session)
            
            # Move to next chunk
            if end_idx >= len(dialogue):
                break
            start_idx += step_size
        
        return chunks
    
    def convert_to_instruction_format(self, sessions: List[Dict]) -> List[Dict]:
        """
        Convert therapy sessions to instruction-following format.
        
        Format: instruction + input + output
        """
        logger.info("Converting to instruction-following format...")
        print(f"Converting {len(sessions)} sessions to instruction format...")
        print("This may take a few minutes...")
        print("Progress will be shown below:")
        print("-" * 50)
        
        formatted_sessions = []
        
        for i, session in enumerate(sessions):
            # Show progress every 10 sessions
            if i % 10 == 0 or i == len(sessions) - 1:
                progress = (i + 1) / len(sessions) * 100
                bar_length = 40
                filled_length = int(bar_length * (i + 1) // len(sessions))
                bar = '#' * filled_length + '-' * (bar_length - filled_length)
                print(f'\rConverting: |{bar}| {progress:.1f}% ({i + 1}/{len(sessions)})', end='', flush=True)
            dialogue = session['dialogue']
            
            # Filter exchanges
            filtered_dialogue = self.filter_exchanges(dialogue)
            
            if len(filtered_dialogue) < 2:  # Need at least 2 exchanges
                continue
            
            # Convert to instruction format
            instruction_examples = self._create_instruction_examples(
                filtered_dialogue, 
                session['session_id']
            )
            
            formatted_sessions.extend(instruction_examples)
        
        print()  # New line after progress bar
        logger.info(f"Created {len(formatted_sessions)} instruction examples")
        return formatted_sessions
    
    def _create_instruction_examples(self, dialogue: List[Dict], session_id: str) -> List[Dict]:
        """
        Create training examples from dialogue using hybrid approach.
        
        Creates examples with weighted exchange counts:
        - 50% - 7 exchanges
        - 25% - 10 exchanges  
        - 15% - 12 exchanges
        - 10% - 15 exchanges
        
        Ensures output is always therapist response. If the last exchange
        in the chunk is not a therapist response, adds more exchanges until
        a therapist response is found.
        """
        examples = []
        
        if not self.tokenizer:
            self.load_tokenizer()
        
        # Weighted distribution for exchange counts
        exchange_weights = {
            7: 0.50,   # 50% - 7 exchanges
            10: 0.25,  # 25% - 10 exchanges  
            12: 0.15,  # 15% - 12 exchanges
            15: 0.10   # 10% - 15 exchanges
        }
        
        # Calculate how many examples of each type to create
        total_examples = min(50, len(dialogue) // 2)  # Limit examples per session
        
        for exchange_count, weight in exchange_weights.items():
            num_examples = int(total_examples * weight)
            
            for _ in range(num_examples):
                if len(dialogue) < exchange_count + 1:
                    continue
                    
                # Random starting position
                max_start = len(dialogue) - exchange_count - 1
                if max_start < 0:
                    continue
                    
                start_idx = random.randint(0, max_start)
                end_idx = start_idx + exchange_count
                
                # Get the chunk starting from minimum size
                chunk = dialogue[start_idx:end_idx + 1]
                
                # Create example ensuring output is therapist response, expanding if needed
                example = self._create_therapist_output_example_adaptive(
                    dialogue, start_idx, end_idx, session_id, exchange_count
                )
                if example:
                    examples.append(example)
        
        return examples
    
    def _create_therapist_output_example_adaptive(self, dialogue: List[Dict], start_idx: int, end_idx: int, session_id: str, min_exchanges: int) -> Optional[Dict]:
        """Create example ensuring output is always therapist response, expanding if needed."""
        
        # Start with minimum chunk size and expand until we find therapist response
        current_end = end_idx
        max_expand = len(dialogue) - 1
        
        # Try to find a therapist response
        while current_end <= max_expand:
            chunk = dialogue[start_idx:current_end + 1]
            
            # Format the context
            chunk_text = self._format_context(chunk)
            lines = chunk_text.strip().split('\n')
            
            if len(lines) < 2:
                break
            
            # Check if last exchange is from therapist
            last_line = lines[-1]
            if last_line.startswith('Therapist:'):
                # Found therapist response! Create example
                input_lines = lines[:-1]
                output_line = last_line
                
                # Extract just the message (without "Therapist:")
                output_text = output_line.split(':', 1)[1].strip() if ':' in output_line else output_line
                input_text = '\n'.join(input_lines)
                
                # Calculate token count
                input_tokens = self.tokenizer.encode(input_text)
                output_tokens = self.tokenizer.encode(output_text)
                total_tokens = len(input_tokens) + len(output_tokens)
                
                return {
                    'input': input_text,
                    'output': output_text,
                    'session_id': session_id,
                    'chunk_index': start_idx,
                    'token_count': total_tokens,
                    'exchanges': len(chunk),
                    'exchange_count': min_exchanges,
                    'actual_exchanges': len(chunk)
                }
            
            # Last exchange is not from therapist, expand by one
            current_end += 1
        
        # Couldn't find therapist response
        return None
    
    def _create_therapist_output_example(self, chunk: List[Dict], session_id: str, start_idx: int, exchange_count: int) -> Optional[Dict]:
        """Create example ensuring output is always therapist response."""
        
        # Format the context
        chunk_text = self._format_context(chunk)
        lines = chunk_text.strip().split('\n')
        
        if len(lines) < 2:
            return None
        
        # Find the last therapist response
        last_therapist_idx = -1
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith('Therapist:'):
                last_therapist_idx = i
                break
        
        if last_therapist_idx == -1:
            return None
        
        # Input is everything before the last therapist response
        input_lines = lines[:last_therapist_idx]
        output_line = lines[last_therapist_idx]
        
        # Extract just the message (without "Therapist:")
        output_text = output_line.split(':', 1)[1].strip() if ':' in output_line else output_line
        
        input_text = '\n'.join(input_lines)
        
        # Calculate token count
        input_tokens = self.tokenizer.encode(input_text)
        output_tokens = self.tokenizer.encode(output_text)
        total_tokens = len(input_tokens) + len(output_tokens)
        
        return {
            'input': input_text,
            'output': output_text,
            'session_id': session_id,
            'chunk_index': start_idx,
            'token_count': total_tokens,
            'exchanges': len(chunk),
            'exchange_count': exchange_count
        }
    
    def _format_context(self, context: List[Dict]) -> str:
        """Format conversation context as input text."""
        formatted_lines = []
        
        for exchange in context:
            speaker = exchange.get('speaker', 'Unknown')
            message = exchange.get('message', '')
            
            # Clean speaker name
            cleaned_speaker = self._clean_speaker_name(speaker)
            
            # Skip if speaker was filtered out
            if cleaned_speaker is None:
                continue
            
            formatted_lines.append(f"{cleaned_speaker}: {message}")
        
        return "\n".join(formatted_lines)
    
    def _clean_speaker_name(self, speaker: str) -> str:
        """Clean and standardize speaker names."""
        speaker = speaker.lower().strip()
        
        # Map various speaker names to standard format
        if speaker in ['therapist', 'dr.', 'dr', 'doctor', 'counselor', 'counsellor', 'instructor']:
            return "Therapist"
        elif speaker in ['client', 'loretta', 'patient', 'client1', 'client2', 
                        'male participant', 'female participant', 'participant',
                        'andreas', 'ann larkin', 'betty', 'christina', 'connirae',
                        'earla', 'heather', 'linda', 'lori', 'lucy', 'marge',
                        'michelle', 'sarah', 'steve', 'sylvia', 'virginia',
                        'male speaker', 'male voice', 'unknown woman']:
            return "Client"
        elif speaker in ['moderator', 'interviewer', 'narrator', 'unknown speaker', 'unidentified participant']:
            # These should have been filtered out by the moderator filter
            # If they still exist, skip them
            return None
        else:
            # Unknown speakers - skip them to be safe
            return None
    
    def tokenize_examples(self, examples: List[Dict]) -> Dataset:
        """
        Tokenize examples using HuggingFace Datasets with batched processing.
        
        Reference: https://huggingface.co/learn/llm-course/chapter3/2
        """
        logger.info("Tokenizing examples with batched processing...")
        
        if not self.tokenizer:
            self.load_tokenizer()
        
        # Create dataset
        dataset = Dataset.from_list(examples)
        
        def tokenize_function(examples):
            """
            Tokenization function for batched processing.
            
            Reference: https://huggingface.co/learn/llm-course/chapter3/2
            """
            # Combine input and output (no instruction)
            texts = []
            for input_text, output in zip(examples['input'], examples['output']):
                # Format as conversation
                full_text = f"{input_text}\n\nResponse: {output}"
                texts.append(full_text)
            
            # Tokenize with proper settings
            tokenized = self.tokenizer(
                texts,
                truncation=self.config.truncation,
                max_length=self.config.max_length,
                padding=False,  # We'll use dynamic padding
                return_tensors=None
            )
            
            return tokenized
        
        # Apply tokenization with batched processing and progress bar
        print(f"Starting tokenization of {len(dataset)} examples...")
        print("This may take several minutes for large datasets...")
        print("Progress tracking enabled - you should see a progress bar below:")
        print("-" * 50)
        
        # Use tqdm for better progress tracking
        from tqdm import tqdm
        import os
        
        # Enable progress bars
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        print("Setting up tokenization...")
        print("Processing batches...")
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing examples"
        )
        
        print("\nTokenization completed!")
        
        logger.info(f"Tokenized {len(tokenized_dataset)} examples")
        return tokenized_dataset
    
    def create_train_val_split(self, dataset: Dataset, val_split: float = 0.1, test_split: float = 0.1) -> DatasetDict:
        """
        Create train/validation/test splits.
        
        Args:
            dataset: Tokenized dataset
            val_split: Validation split ratio
            test_split: Test split ratio
            
        Returns:
            DatasetDict with train/val/test splits
        """
        logger.info(f"Creating train/val/test splits (val: {val_split}, test: {test_split})")
        
        # Calculate split sizes
        total_size = len(dataset)
        test_size = int(total_size * test_split)
        val_size = int(total_size * val_split)
        train_size = total_size - test_size - val_size
        
        # Create splits
        train_dataset = dataset.select(range(train_size))
        val_dataset = dataset.select(range(train_size, train_size + val_size))
        test_dataset = dataset.select(range(train_size + val_size, total_size))
        
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })
        
        logger.info(f"Created splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        return dataset_dict
    
    def save_processed_data(self, dataset_dict: DatasetDict, output_path: str = None):
        """
        Save processed dataset to disk.
        
        Args:
            dataset_dict: Processed dataset dictionary
            output_path: Output path for saving
        """
        if output_path is None:
            output_path = os.path.join(self.config.output_dir, "therapy_dataset")
        
        logger.info(f"Saving processed dataset to {output_path}")
        
        # Save dataset
        dataset_dict.save_to_disk(output_path)
        
        # Save tokenizer
        tokenizer_path = os.path.join(output_path, "tokenizer")
        self.tokenizer.save_pretrained(tokenizer_path)
        
        # Save data collator info
        collator_info = {
            'type': 'DataCollatorWithPadding',
            'padding': True,
            'return_tensors': 'pt'
        }
        
        with open(os.path.join(output_path, "data_collator_info.json"), 'w') as f:
            json.dump(collator_info, f, indent=2)
        
        logger.info("Dataset saved successfully")
    
    def _show_random_examples(self, data: List[Dict], stage: str, num_examples: int = 2):
        """Show random examples of data at each stage."""
        if not data:
            return
        
        print(f"\nRANDOM EXAMPLES FROM {stage.upper()}:")
        print("-" * 50)
        
        # Get random examples
        sample_size = min(num_examples, len(data))
        random_indices = random.sample(range(len(data)), sample_size)
        
        for i, idx in enumerate(random_indices):
            example = data[idx]
            print(f"\nExample {i+1}:")
            
            if stage == "raw_sessions":
                print(f"  Session ID: {example.get('session_id', 'unknown')}")
                print(f"  Exchanges: {len(example.get('dialogue', []))}")
                if example.get('dialogue'):
                    sample_dialogue = example['dialogue'][:3]  # First 3 exchanges
                    for exchange in sample_dialogue:
                        speaker = exchange.get('speaker', 'Unknown')
                        message = exchange.get('message', '')[:100]
                        print(f"    {speaker}: {message}...")
            
            elif stage == "chunked_sessions":
                print(f"  Session ID: {example.get('session_id', 'unknown')}")
                print(f"  Exchanges: {len(example.get('dialogue', []))}")
                if example.get('chunk_info'):
                    chunk_info = example['chunk_info']
                    print(f"  Chunk: {chunk_info.get('chunk_index', 0)} of original session")
                    print(f"  Range: exchanges {chunk_info.get('start_exchange', 0)}-{chunk_info.get('end_exchange', 0)}")
            
            elif stage == "instruction_examples":
                print(f"  Input: {example.get('input', '')[:150]}...")
                print(f"  Output: {example.get('output', '')[:100]}...")
                if 'token_count' in example:
                    print(f"  Tokens: {example['token_count']}")
            
            elif stage == "tokenized_examples":
                print(f"  Token count: {len(example.get('input_ids', []))}")
                print(f"  Attention mask: {example.get('attention_mask', [])[:10]}...")
                # Try to decode a sample
                try:
                    if self.tokenizer and 'input_ids' in example:
                        decoded = self.tokenizer.decode(example['input_ids'][:50], skip_special_tokens=True)
                        print(f"  Sample text: {decoded[:100]}...")
                except Exception as e:
                    print(f"  Could not decode: {e}")
    
    def process_all_data(self) -> DatasetDict:
        """
        Complete data processing pipeline.
        
        Returns:
            Processed dataset dictionary
        """
        logger.info("Starting complete data processing pipeline...")
        print("\nTHERAPY DATA PROCESSING PIPELINE")
        print("=" * 50)
        
        # Step 1: Load therapy sessions
        print("\nSTEP 1: Loading therapy sessions...")
        start_time = time.time()
        sessions = self.load_therapy_sessions()
        step_time = time.time() - start_time
        print(f"Loaded {len(sessions)} therapy sessions (took {step_time:.1f}s)")
        
        # Show session statistics
        if sessions:
            session_lengths = [len(session['dialogue']) for session in sessions]
            print(f"   Session lengths: {min(session_lengths)}-{max(session_lengths)} exchanges")
            print(f"   Average session: {sum(session_lengths)/len(session_lengths):.1f} exchanges")
        
        # Skip showing examples to speed up processing
        # self._show_random_examples(sessions, "raw_sessions", 2)
        # self._show_full_examples(sessions, "raw_sessions", 1)
        
        # Step 2: Chunk long sessions
        print(f"\nSTEP 2: Chunking long sessions...")
        start_time = time.time()
        chunked_sessions = self.chunk_long_sessions(sessions)
        step_time = time.time() - start_time
        print(f"Created {len(chunked_sessions)} session chunks from {len(sessions)} original sessions (took {step_time:.1f}s)")
        
        # Show chunking statistics
        if len(chunked_sessions) > len(sessions):
            print(f"   Split {len(chunked_sessions) - len(sessions)} long sessions")
        
        # Show random examples
        # Skip showing examples to speed up processing
        # self._show_random_examples(chunked_sessions, "chunked_sessions", 2)
        # self._show_full_examples(chunked_sessions, "chunked_sessions", 1)
        
        # Step 3: Convert to instruction format
        print(f"\nSTEP 3: Converting to instruction format...")
        start_time = time.time()
        examples = self.convert_to_instruction_format(chunked_sessions)
        step_time = time.time() - start_time
        print(f"Created {len(examples)} training examples (took {step_time:.1f}s)")
        
        # Show example statistics
        if examples:
            token_counts = [ex.get('token_count', 0) for ex in examples if 'token_count' in ex]
            if token_counts:
                print(f"   Token counts: {min(token_counts)}-{max(token_counts)} tokens")
                print(f"   Average tokens: {sum(token_counts)/len(token_counts):.1f}")
        
        # Show random examples
        # Skip showing examples to speed up processing
        # self._show_random_examples(examples, "instruction_examples", 2)
        # self._show_full_examples(examples, "instruction_examples", 2)
        
        # Step 4: Tokenize examples
        print(f"\nSTEP 4: Tokenizing examples...")
        start_time = time.time()
        tokenized_dataset = self.tokenize_examples(examples)
        step_time = time.time() - start_time
        print(f"Tokenized {len(tokenized_dataset)} examples (took {step_time:.1f}s)")
        
        # Show tokenization sample
        if len(tokenized_dataset) > 0:
            sample = tokenized_dataset[0]
            print(f"   Sample token count: {len(sample['input_ids'])}")
            print(f"   Sample attention mask length: {len(sample['attention_mask'])}")
        
        # Show random examples
        tokenized_list = [dict(tokenized_dataset[i]) for i in range(min(5, len(tokenized_dataset)))]
        self._show_random_examples(tokenized_list, "tokenized_examples", 2)
        
        # Step 5: Create train/val/test splits
        print(f"\nSTEP 5: Creating train/validation/test splits...")
        start_time = time.time()
        dataset_dict = self.create_train_val_split(tokenized_dataset)
        step_time = time.time() - start_time
        print(f"Created splits (took {step_time:.1f}s):")
        print(f"   Train: {len(dataset_dict['train']):,} examples")
        print(f"   Validation: {len(dataset_dict['validation']):,} examples")
        print(f"   Test: {len(dataset_dict['test']):,} examples")
        
        # Step 6: Save processed data
        print(f"\nSTEP 6: Saving processed data...")
        start_time = time.time()
        self.save_processed_data(dataset_dict)
        step_time = time.time() - start_time
        print(f"Data saved to: {self.config.output_dir} (took {step_time:.1f}s)")
        
        print(f"\nPIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        logger.info("Data processing pipeline completed successfully!")
        return dataset_dict

    def _show_random_examples(self, data: List[Dict], stage_name: str, max_examples: int = 2) -> None:
        """Show random examples from data at each stage."""
        if not data:
            print(f"  No data to show for {stage_name}")
            return
        
        print(f"\nRANDOM EXAMPLES FROM {stage_name.upper()}:")
        print("-" * 50)
        
        # Show random examples
        import random
        sample_indices = random.sample(range(len(data)), min(max_examples, len(data)))
        
        for i, idx in enumerate(sample_indices):
            example = data[idx]
            print(f"\nExample {i+1}:")
            
            if 'session_id' in example:
                print(f"  Session ID: {example['session_id']}")
            
            if 'exchanges' in example:
                print(f"  Exchanges: {example['exchanges']}")
                if 'chunk' in example:
                    print(f"  Chunk: {example['chunk']} of original session")
                if 'range' in example:
                    print(f"  Range: {example['range']}")
            
            if 'input' in example and 'output' in example:
                print(f"  Input: {example['input'][:200]}...")
                print(f"  Output: {example['output']}")
                if 'tokens' in example:
                    print(f"  Tokens: {example['tokens']}")
            
            elif 'dialogue' in example:
                dialogue = example['dialogue']
                print(f"  Exchanges: {len(dialogue)}")
                for j, exchange in enumerate(dialogue[:5]):  # Show first 5 exchanges
                    speaker = exchange.get('speaker', 'Unknown')
                    message = exchange.get('message', '')[:80]
                    print(f"    {speaker}: {message}...")
                if len(dialogue) > 5:
                    print(f"    ... and {len(dialogue) - 5} more exchanges")
            
            elif isinstance(example, dict) and 'message' in example:
                speaker = example.get('speaker', 'Unknown')
                message = example.get('message', '')[:100]
                print(f"  {speaker}: {message}...")
    
    def _show_full_examples(self, data: List[Dict], stage_name: str, max_examples: int = 2) -> None:
        """Show FULL examples from data at each stage for manual verification."""
        if not data:
            print(f"  No data to show for {stage_name}")
            return
        
        print(f"\nFULL EXAMPLES FROM {stage_name.upper()} (FOR MANUAL VERIFICATION):")
        print("=" * 80)
        
        # Show random examples
        import random
        sample_indices = random.sample(range(len(data)), min(max_examples, len(data)))
        
        for i, idx in enumerate(sample_indices):
            example = data[idx]
            print(f"\nEXAMPLE {i+1} (Index {idx}):")
            print("-" * 60)
            
            if 'session_id' in example:
                print(f"Session ID: {example['session_id']}")
            
            if 'exchanges' in example:
                print(f"Exchanges: {example['exchanges']}")
                if 'chunk' in example:
                    print(f"Chunk: {example['chunk']} of original session")
                if 'range' in example:
                    print(f"Range: {example['range']}")
            
            if 'input' in example and 'output' in example:
                print(f"\nINPUT (Context):")
                print(example['input'])
                print(f"\nOUTPUT (Response):")
                print(example['output'])
                if 'tokens' in example:
                    print(f"\nToken count: {example['tokens']}")
            
            elif 'dialogue' in example:
                dialogue = example['dialogue']
                print(f"\nDIALOGUE ({len(dialogue)} exchanges):")
                for j, exchange in enumerate(dialogue):
                    speaker = exchange.get('speaker', 'Unknown')
                    message = exchange.get('message', '')
                    print(f"{j+1:2d}. {speaker}: {message}")
            
            elif isinstance(example, dict) and 'message' in example:
                speaker = example.get('speaker', 'Unknown')
                message = example.get('message', '')
                print(f"{speaker}: {message}")
            
            print("\n" + "=" * 80)
        
        # Skip manual verification in automated runs
        print(f"\n{stage_name} examples shown above.")
        print("Proceeding with processing...")

def main():
    """Main function to run the data processing pipeline."""
    
    # Configuration
    config = ProcessingConfig(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        max_length=2048,
        max_session_exchanges=300,
        overlap_percentage=0.2,
        input_data_dir="data/training_data",
        output_dir="data/processed"
    )
    
    # Create processor and run pipeline
    processor = TherapyDataProcessor(config)
    dataset_dict = processor.process_all_data()
    
    # Print summary
    print("\n" + "="*50)
    print("DATA PROCESSING SUMMARY")
    print("="*50)
    print(f"Train examples: {len(dataset_dict['train']):,}")
    print(f"Validation examples: {len(dataset_dict['validation']):,}")
    print(f"Test examples: {len(dataset_dict['test']):,}")
    print(f"Total examples: {sum(len(split) for split in dataset_dict.values()):,}")
    print("="*50)

if __name__ == "__main__":
    main()
