#!/usr/bin/env python3
"""
Add labels to therapy datasets.
The labels mask all tokens except the response (the part after "Response:").
This ensures the model only learns to predict the therapist's response, 
conditioned on the full context.
"""

import sys
import os
from pathlib import Path
from datasets import load_from_disk, DatasetDict, Dataset
from transformers import AutoTokenizer

def find_response_start(input_ids, tokenizer):
    """
    Find the starting index of the response tokens.
    
    The format is: {context}\n\nResponse: {response}
    We need to find where "Response:" starts and return the token after ":"
    
    Args:
        input_ids: List of token IDs
        tokenizer: Tokenizer instance
        
    Returns:
        Index where response starts, or -1 if not found
    """
    # Tokenize "Response:"
    response_tokens = tokenizer.encode("Response:", add_special_tokens=False)
    
    # Search for "Response:" in the sequence
    for i in range(len(input_ids) - len(response_tokens) + 1):
        if input_ids[i:i+len(response_tokens)] == response_tokens:
            # Return the token after "Response:"
            return i + len(response_tokens)
    
    return -1

def add_labels_to_dataset(dataset_path, output_path, tokenizer):
    """
    Add labels to a dataset where only response tokens are labeled.
    
    Args:
        dataset_path: Path to the input dataset
        output_path: Path to save the labeled dataset
        tokenizer: Tokenizer instance
    """
    print(f"\nProcessing {dataset_path}...")
    
    # Load dataset
    dataset = load_from_disk(dataset_path)
    
    def add_labels(example):
        """Add labels to a single example."""
        input_ids = example['input_ids']
        
        # Find where response starts
        response_start_idx = find_response_start(input_ids, tokenizer)
        
        # Create labels
        # -100 means ignore in loss computation
        labels = [-100] * len(input_ids)
        
        if response_start_idx >= 0:
            # Everything after response_start_idx should be learned
            for i in range(response_start_idx, len(input_ids)):
                labels[i] = input_ids[i]
        
        return {'labels': labels}
    
    # Apply to all splits
    dataset_dict = dataset if isinstance(dataset, DatasetDict) else {'train': dataset}
    
    print(f"Processing dataset with {len(dataset_dict.get('train', []))} examples...")
    
    labeled_datasets = {}
    for split_name, split_data in dataset_dict.items():
        print(f"  Processing {split_name} split ({len(split_data)} examples)...")
        labeled_split = split_data.map(add_labels, desc=f"Adding labels to {split_name}")
        labeled_datasets[split_name] = labeled_split
    
    # Create DatasetDict
    labeled_dict = DatasetDict(labeled_datasets) if isinstance(dataset, DatasetDict) else labeled_datasets
    
    # Save labeled dataset
    print(f"Saving to {output_path}...")
    labeled_dict.save_to_disk(output_path)
    
    # Print statistics
    print(f"\nLabeling complete for {dataset_path}")
    for split_name, split_data in labeled_dict.items():
        if len(split_data) > 0:
            example = split_data[0]
            labels = example['labels']
            total_tokens = len(labels)
            labeled_tokens = sum(1 for l in labels if l != -100)
            print(f"  {split_name}: {total_tokens} total tokens, {labeled_tokens} labeled tokens ({labeled_tokens/total_tokens*100:.1f}%)")
    
    return labeled_dict

def main():
    """Main function to add labels to all three datasets."""
    if len(sys.argv) < 2:
        print("Usage: python add_labels.py <model_name>")
        print("Example: python add_labels.py meta-llama/Llama-3.1-8B-Instruct")
        sys.exit(1)
    
    model_name = sys.argv[1]
    
    print("=" * 70)
    print("ADDING LABELS TO THERAPY DATASETS")
    print("=" * 70)
    print(f"Model: {model_name}")
    print("\nThis will add labels to the datasets so that only the")
    print("response (after 'Response:') is used in loss computation.")
    print("All context tokens will be ignored (label = -100).")
    print("=" * 70)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")
    
    # Define datasets
    base_dir = Path("data")
    datasets = [
        ("processed_run1_short_processed/therapy_dataset", "processed_run1_short_labeled/therapy_dataset"),
        ("processed_run2_medium_processed/therapy_dataset", "processed_run2_medium_labeled/therapy_dataset"),
        ("processed_run3_long_processed/therapy_dataset", "processed_run3_long_labeled/therapy_dataset"),
    ]
    
    # Process each dataset
    for input_path, output_path in datasets:
        input_full = base_dir / input_path
        output_full = base_dir / output_path
        
        if not input_full.exists():
            print(f"\nWARNING: {input_full} not found, skipping...")
            continue
        
        try:
            labeled_dataset = add_labels_to_dataset(
                str(input_full),
                str(output_full),
                tokenizer
            )
            
            # Validate a few examples
            print(f"\nValidating examples from {output_path}...")
            for split_name in ['train', 'validation', 'test']:
                if split_name in labeled_dataset and len(labeled_dataset[split_name]) > 0:
                    example = labeled_dataset[split_name][0]
                    input_ids = example['input_ids']
                    labels = example['labels']
                    
                    # Decode to show what's being learned
                    response_start_idx = find_response_start(input_ids, tokenizer)
                    if response_start_idx >= 0:
                        # Decode the labeled part (the response)
                        response_tokens = labels[response_start_idx:response_start_idx+20]  # First 20 labeled tokens
                        response_text = tokenizer.decode([t for t in response_tokens if t != -100][:10])
                        
                        print(f"\n  Sample from {split_name} split:")
                        print(f"    Response starts at token {response_start_idx}")
                        print(f"    First labeled tokens: {response_text[:100]}...")
                    
                    break  # Only show one example
            
        except Exception as e:
            print(f"\nERROR processing {input_path}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("ALL DATASETS LABELED SUCCESSFULLY!")
    print("=" * 70)
    print("\nLabeled datasets saved to:")
    for _, output_path in datasets:
        print(f"  - {base_dir / output_path}")
    
    print("\nNext steps:")
    print("  1. Use these labeled datasets for training")
    print("  2. The model will only learn to predict the response tokens")
    print("  3. Context tokens are ignored in loss computation")

if __name__ == "__main__":
    main()

