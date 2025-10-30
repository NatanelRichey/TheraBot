#!/usr/bin/env python3
"""
Quick validation of cleaned datasets to confirm all examples are valid.
"""

import sys
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer

def find_response_start(input_ids, tokenizer):
    """Find where the response starts."""
    response_tokens = tokenizer.encode("Response:", add_special_tokens=False)
    for i in range(len(input_ids) - len(response_tokens) + 1):
        if input_ids[i:i+len(response_tokens)] == response_tokens:
            return i + len(response_tokens)
    return -1

def validate_dataset(dataset_path, tokenizer):
    """Validate that ALL examples in a dataset are valid."""
    print(f"\nValidating {dataset_path}...")
    
    dataset = load_from_disk(dataset_path)
    splits = dataset if hasattr(dataset, 'keys') else {'dataset': dataset}
    
    total_invalid = 0
    total_examples = 0
    
    for split_name, split_data in splits.items():
        if len(split_data) == 0:
            continue
        
        print(f"  {split_name}: {len(split_data)} examples")
        split_invalid = 0
        
        for i in range(len(split_data)):
            example = split_data[i]
            input_ids = example['input_ids']
            
            response_start_idx = find_response_start(input_ids, tokenizer)
            
            if response_start_idx == -1:
                split_invalid += 1
                print(f"    ERROR: Example {i+1} has no 'Response:'")
        
        total_invalid += split_invalid
        total_examples += len(split_data)
        
        if split_invalid == 0:
            print(f"    Status: OK (all examples valid)")
        else:
            print(f"    Status: {split_invalid} invalid examples")
    
    return total_invalid == 0, total_invalid, total_examples

def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_clean_datasets.py <model_name>")
        sys.exit(1)
    
    model_name = sys.argv[1]
    
    print("=" * 70)
    print("VALIDATING CLEANED DATASETS")
    print("=" * 70)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Tokenizer loaded.")
    
    # Define datasets to validate
    base_dir = Path("data")
    datasets = [
        "processed_run1_short_labeled_clean/therapy_dataset",
        "processed_run2_medium_labeled_clean/therapy_dataset",
        "processed_run3_long_labeled_clean/therapy_dataset",
    ]
    
    all_valid = True
    total_invalid = 0
    total_examples = 0
    
    for dataset_path in datasets:
        full_path = base_dir / dataset_path
        
        if not full_path.exists():
            print(f"\nWARNING: {full_path} not found")
            all_valid = False
            continue
        
        try:
            is_valid, invalid_count, example_count = validate_dataset(str(full_path), tokenizer)
            all_valid = all_valid and is_valid
            total_invalid += invalid_count
            total_examples += example_count
        except Exception as e:
            print(f"\nERROR validating {full_path}: {e}")
            all_valid = False
    
    print("\n" + "=" * 70)
    print(f"VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Total examples: {total_examples}")
    print(f"Invalid examples: {total_invalid}")
    
    if all_valid and total_invalid == 0:
        print("\n✓ ALL CLEANED DATASETS ARE VALID!")
        print("✓ All examples have valid 'Response:' markers.")
        print("✓ Ready for training!")
    else:
        print(f"\n✗ {total_invalid} INVALID EXAMPLES FOUND!")
    
    return 0 if all_valid else 1

if __name__ == "__main__":
    sys.exit(main())

