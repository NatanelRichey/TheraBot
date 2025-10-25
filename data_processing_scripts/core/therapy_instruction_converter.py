#!/usr/bin/env python3
"""
Convert therapy sessions to instruction format with custom exchange distributions.
This script allows you to specify custom exchange count distributions for different runs.
"""

import sys
import json
import random
from pathlib import Path
from therapy_data_pipeline import TherapyDataProcessor, ProcessingConfig
import logging

def convert_instruction_custom_distribution(input_dir, output_dir, exchange_distribution, run_name="custom"):
    """
    Convert therapy sessions to instruction format with custom exchange distribution.
    
    Args:
        input_dir: Directory containing JSON transcript files
        output_dir: Directory to save instruction examples
        exchange_distribution: Dict of {exchange_count: weight} e.g. {2: 0.5, 3: 0.25, 6: 0.15, 10: 0.1}
        run_name: Name for this run (used in output files)
    """
    
    print(f"STAGE 1: CUSTOM INSTRUCTION FORMAT CONVERSION - {run_name.upper()}")
    print("=" * 70)
    print("Creating instruction examples with custom exchange counts:")
    for exchange_count, weight in sorted(exchange_distribution.items()):
        print(f"  {weight*100:.0f}% - {exchange_count} exchanges")
    print("Output will ALWAYS be therapist response")
    print("If minimum exchanges end with client, will expand until therapist")
    print("=" * 70)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configuration for instruction conversion only
    config = ProcessingConfig(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        max_length=2048,
        max_session_exchanges=300,
        overlap_percentage=0.2,
        min_exchange_length=1,
        max_exchange_length=200,
        input_data_dir=input_dir,
        output_dir=output_dir,
        cache_dir=".cache"
    )
    
    print(f"Input directory: {config.input_data_dir}")
    print(f"Output directory: {config.output_dir}")
    print()
    
    try:
        # Create processor
        processor = TherapyDataProcessor(config)
        
        # Step 1: Load therapy sessions from JSON files
        print("STEP 1: Loading therapy sessions from JSON files...")
        sessions = load_sessions_from_json(input_dir)
        print(f"Loaded {len(sessions)} therapy sessions")
        
        # Step 2: Chunk long sessions
        print(f"\nSTEP 2: Chunking long sessions...")
        
        # Normalize session structure for the processor
        normalized_sessions = []
        for i, session in enumerate(sessions):
            # Convert 'dialogues' to 'dialogue' if needed
            if 'dialogues' in session and 'dialogue' not in session:
                session_copy = session.copy()
                session_copy['dialogue'] = session_copy['dialogues']
                # Add required fields if missing
                if 'session_id' not in session_copy:
                    session_copy['session_id'] = session.get('file', f'session_{i}')
                if 'source_file' not in session_copy:
                    session_copy['source_file'] = session.get('file', f'session_{i}.json')
                normalized_sessions.append(session_copy)
            else:
                # Add required fields if missing
                if 'session_id' not in session:
                    session['session_id'] = session.get('file', f'session_{i}')
                if 'source_file' not in session:
                    session['source_file'] = session.get('file', f'session_{i}.json')
                normalized_sessions.append(session)
        
        chunked_sessions = processor.chunk_long_sessions(normalized_sessions)
        print(f"Created {len(chunked_sessions)} session chunks from {len(sessions)} original sessions")
        
        # Step 3: Convert to instruction format with custom distribution
        print(f"\nSTEP 3: Converting to instruction format with custom distribution...")
        print("This may take a few minutes...")
        print("Progress will be shown below:")
        print("-" * 50)
        
        examples = convert_to_instruction_format_custom(processor, chunked_sessions, exchange_distribution)
        print(f"\nCreated {len(examples)} instruction examples")
        
        # Save examples for inspection
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save all examples
        examples_file = output_path / f"instruction_examples_{run_name}.json"
        with open(examples_file, 'w', encoding='utf-8') as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)
        print(f"Saved all examples to: {examples_file}")
        
        # Save sample examples for inspection
        sample_file = output_path / f"sample_instruction_examples_{run_name}.json"
        sample_examples = examples[:10]  # First 10 examples
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(sample_examples, f, indent=2, ensure_ascii=False)
        print(f"Saved sample examples to: {sample_file}")
        
        # Show detailed examples
        print(f"\nDETAILED EXAMPLES:")
        print("=" * 50)
        
        for i, example in enumerate(examples[:3]):  # Show first 3 examples
            print(f"\nExample {i+1}:")
            print(f"  Input: {example['input'][:200]}...")
            print(f"  Output: {example['output'][:200]}...")
            print(f"  Full text length: {len(example['input']) + len(example['output'])} characters")
            print(f"  Exchange count: {example.get('exchange_count', 'N/A')}")
        
        # Analyze distribution
        print(f"\nDISTRIBUTION ANALYSIS:")
        print("=" * 50)
        
        # Count by exchange count
        exchange_counts = {}
        for example in examples:
            count = example.get('exchange_count', 0)
            exchange_counts[count] = exchange_counts.get(count, 0) + 1
        
        print("Examples by exchange count:")
        for count in sorted(exchange_counts.keys()):
            percentage = (exchange_counts[count] / len(examples)) * 100
            print(f"  {count} exchanges: {exchange_counts[count]} ({percentage:.1f}%)")
        
        # Check that all outputs are therapist responses
        therapist_outputs = 0
        client_outputs = 0
        
        for example in examples[:100]:  # Check first 100
            output = example.get('output', '')
            if not any(output.startswith(prefix) for prefix in ['Client:', 'Patient:']):
                therapist_outputs += 1
            else:
                client_outputs += 1
        
        print(f"\nOutput validation (first 100 examples):")
        print(f"  Therapist outputs: {therapist_outputs}")
        print(f"  Client outputs: {client_outputs}")
        
        if client_outputs == 0:
            print(f"  [SUCCESS] All outputs are therapist responses!")
        else:
            print(f"  [WARNING] Some outputs are client responses!")
        
        # Check for common issues
        print(f"\nQUALITY CHECKS:")
        print("=" * 50)
        
        empty_inputs = sum(1 for ex in examples if not ex['input'].strip())
        empty_outputs = sum(1 for ex in examples if not ex['output'].strip())
        very_short = sum(1 for ex in examples if len(ex['input']) + len(ex['output']) < 50)
        very_long = sum(1 for ex in examples if len(ex['input']) + len(ex['output']) > 2000)
        
        print(f"Empty inputs: {empty_inputs}")
        print(f"Empty outputs: {empty_outputs}")
        print(f"Very short examples (<50 chars): {very_short}")
        print(f"Very long examples (>2000 chars): {very_long}")
        
        if empty_inputs == 0 and empty_outputs == 0:
            print(f"[SUCCESS] All examples have valid input and output!")
        else:
            print(f"[WARNING] Some examples have empty input or output!")
        
        print(f"\nCONVERSION COMPLETED SUCCESSFULLY!")
        print(f"Total examples created: {len(examples)}")
        print(f"Examples saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        logging.exception("Conversion failed")
        return False

def load_sessions_from_json(input_dir):
    """Load therapy sessions from JSON files in the input directory."""
    sessions = []
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Get all JSON files
    json_files = list(input_path.glob("*.json"))
    json_files.sort()  # Sort for consistent ordering
    
    print(f"Found {len(json_files)} JSON files")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                # If it's a list of sessions
                for session in data:
                    if 'exchanges' in session or 'dialogue' in session or 'dialogues' in session:
                        sessions.append(session)
            elif isinstance(data, dict):
                # If it's a single session
                if 'exchanges' in data or 'dialogue' in data or 'dialogues' in data:
                    sessions.append(data)
            
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
            continue
    
    return sessions

def convert_to_instruction_format_custom(processor, sessions, exchange_distribution):
    """Convert sessions to instruction format with custom exchange distribution."""
    examples = []
    
    if not processor.tokenizer:
        processor.load_tokenizer()
    
    total_sessions = len(sessions)
    
    for session_idx, session in enumerate(sessions):
        # Show progress
        progress = (session_idx + 1) / total_sessions * 100
        bar_length = 40
        filled_length = int(bar_length * (session_idx + 1) // total_sessions)
        bar = '#' * filled_length + '-' * (bar_length - filled_length)
        print(f'\rProcessing: |{bar}| {progress:.1f}% ({session_idx + 1}/{total_sessions})', end='', flush=True)
        
        # Get dialogue from session
        dialogue = session.get('exchanges', session.get('dialogue', session.get('dialogues', [])))
        if not dialogue:
            continue
        
        session_id = session.get('session_id', f"session_{session_idx}")
        
        # Create examples with custom distribution
        session_examples = create_instruction_examples_custom(processor, dialogue, session_id, exchange_distribution)
        examples.extend(session_examples)
    
    print()  # New line after progress bar
    return examples

def create_instruction_examples_custom(processor, dialogue, session_id, exchange_distribution):
    """Create training examples from dialogue using custom exchange distribution."""
    examples = []
    
    # Calculate how many examples of each type to create
    total_examples = min(50, len(dialogue) // 2)  # Limit examples per session
    
    for exchange_count, weight in exchange_distribution.items():
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
            example = create_therapist_output_example_adaptive(
                processor, dialogue, start_idx, end_idx, session_id, exchange_count
            )
            if example:
                examples.append(example)
    
    return examples

def create_therapist_output_example_adaptive(processor, dialogue, start_idx, end_idx, session_id, min_exchanges):
    """Create an example ensuring the output is a therapist response."""
    
    # Start with the minimum exchanges
    current_end = end_idx
    
    # Keep expanding until we get a therapist response
    while current_end < len(dialogue):
        chunk = dialogue[start_idx:current_end + 1]
        
        # Check if the last exchange is from therapist
        if chunk and is_therapist_speaker(chunk[-1]):
            # Create the example
            input_text = format_dialogue_as_input(chunk[:-1])
            output_text = format_speaker_output(chunk[-1])
            
            return {
                'input': input_text,
                'output': output_text,
                'session_id': session_id,
                'start_idx': start_idx,
                'end_idx': current_end,
                'exchange_count': current_end - start_idx + 1
            }
        
        # Expand by one more exchange
        current_end += 1
    
    return None

def format_dialogue_as_input(context):
    """Format conversation context as input text."""
    formatted_lines = []
    
    for exchange in context:
        speaker = exchange.get('speaker', 'Unknown')
        message = exchange.get('text', exchange.get('message', ''))  # Handle both 'text' and 'message'
        
        # Clean speaker name
        cleaned_speaker = clean_speaker_name(speaker)
        
        # Skip if speaker was filtered out
        if cleaned_speaker is None:
            continue
        
        formatted_lines.append(f"{cleaned_speaker}: {message}")
    
    return "\n".join(formatted_lines)

def format_speaker_output(exchange):
    """Format speaker output."""
    speaker = exchange.get('speaker', 'Unknown')
    message = exchange.get('text', exchange.get('message', ''))  # Handle both 'text' and 'message'
    
    # Clean speaker name
    cleaned_speaker = clean_speaker_name(speaker)
    
    if cleaned_speaker is None:
        return ""
    
    return f"{cleaned_speaker}: {message}"

def clean_speaker_name(speaker):
    """Clean and standardize speaker names."""
    speaker = speaker.lower().strip()
    
    # Map various speaker names to standard format
    if 'therapist' in speaker or 'counselor' in speaker:
        return 'Therapist'
    elif 'client' in speaker or 'patient' in speaker:
        return 'Client'
    else:
        # Keep original if not recognized
        return speaker.title()

def is_therapist_speaker(exchange):
    """Check if an exchange is from a therapist."""
    speaker = exchange.get('speaker', '').lower()
    return any(role in speaker for role in ['therapist', 'counselor', 'doctor', 'dr.', 'therapist:'])

def main():
    """Main function."""
    if len(sys.argv) < 4:
        print("Usage: python therapy_instruction_converter.py <input_dir> <output_dir> <run_name> [exchange_counts...]")
        print("Example: python therapy_instruction_converter.py ../data/transcripts_json ../data/instruction_examples run1 2:0.5 3:0.25 6:0.15 10:0.1")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    run_name = sys.argv[3]
    
    # Parse exchange distribution from command line
    exchange_distribution = {}
    for arg in sys.argv[4:]:
        if ':' in arg:
            count_str, weight_str = arg.split(':')
            try:
                count = int(count_str)
                weight = float(weight_str)
                exchange_distribution[count] = weight
            except ValueError:
                print(f"Invalid argument: {arg}")
                sys.exit(1)
        else:
            print(f"Invalid argument format: {arg} (expected count:weight)")
            sys.exit(1)
    
    if not exchange_distribution:
        print("No exchange distribution provided!")
        sys.exit(1)
    
    # Validate weights sum to 1.0
    total_weight = sum(exchange_distribution.values())
    if abs(total_weight - 1.0) > 0.01:
        print(f"Warning: Weights sum to {total_weight:.3f}, not 1.0")
    
    success = convert_instruction_custom_distribution(input_dir, output_dir, exchange_distribution, run_name)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
