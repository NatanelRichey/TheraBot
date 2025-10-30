#!/usr/bin/env python3
"""
Remove examples from labeled datasets where 'Response:' is not found.
"""

import sys
from pathlib import Path
from datasets import load_from_disk, DatasetDict
from transformers import AutoTokenizer

def find_response_start(input_ids, tokenizer):
    """Find where the response starts."""
    response_tokens = tokenizer.encode("Response:", add_special_tokens=False)
    for i in range(len(input_ids) - len(response_tokens) + 1):
        if input_ids[i:i+len(response_tokens)] == response_tokens:
            return i + len(response_tokens)
    return -1

def remove_invalid_examples(dataset_path, output_path, tokenizer):
    """
    Remove examples that don't have 'Response:' from a dataset.
    
    Args:
        dataset_path: Path to the input dataset
        output_path: Path to save the cleaned dataset
        tokenizer: Tokenizer instance
    """
    print(f"\nProcessing {dataset_path}...")
    
    # Load dataset
    dataset = load_from_disk(dataset_path)
    
    # Get all splits
    splits = dataset if hasattr(dataset, 'keys') else {'dataset': dataset}
    
    cleaned_splits = {}
    total_removed = 0
    
    for split_name, split_data in splits.items():
        if len(split_data) == 0:
            cleaned_splits[split_name] = split_data
            continue
        
        print(f"  Processing {split_name} split ({len(split_data)} examples)...")
        
        # Find indices of valid examples
        valid_indices = []
        removed_count = 0
        
        for i in range(len(split_data)):
            example = split_data[i]
            input_ids = example['input_ids']
            
            # Find response start
            response_start_idx = find_response_start(input_ids, tokenizer)
            
            if response_start_idx >= 0:
                valid_indices.append(i)
            else:
                removed_count += 1
        
        print(f"    Keeping {len(valid_indices)} examples, removing {removed_count}")
        total_removed += removed_count
        
        # Create new split with only valid examples
        if valid_indices:
            cleaned_split = split_data.select(valid_indices)
        else:
            # Empty split
            cleaned_split = split_data.select([])
        
        cleaned_splits[split_name] = cleaned_split
    
    # Create DatasetDict
    if isinstance(dataset, DatasetDict):
        cleaned_dataset = DatasetDict(cleaned_splits)
    else:
        cleaned_dataset = cleaned_splits['dataset']
    
    # Save cleaned dataset
    print(f"  Saving cleaned dataset to {output_path}...")
    cleaned_dataset.save_to_disk(output_path)
    
    print(f"\n  Removed {total_removed} invalid examples from {dataset_path}")
    return total_removed

def main():
    """Main function to clean all labeled datasets."""
    if len(sys.argv) < 2:
        print("Usage: python remove_invalid_examples.py <model_name>")
        print("Example: python remove_invalid_examples.py meta-llama/Llama-3.1-8B-Instruct")
        sys.exit(1)
    
    model_name = sys.argv[1]
    
    print("=" * 70)
    print("REMOVING INVALID EXAMPLES FROM LABELED DATASETS")
    print("=" * 70)
    print(f"Model: {model_name}")
    print("\nThis will remove all examples that don't contain 'Response:'")
    print("=" * 70)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Tokenizer loaded.")
    
    # Define datasets to clean
    base_dir = Path("data")
    datasets = [
        ("processed_run1_short_labeled/therapy_dataset", "processed_run1_short_labeled_clean/therapy_dataset"),
        ("processed_run2_medium_labeled/therapy_dataset", "processed_run2_medium_labeled_clean/therapy_dataset"),
        ("processed_run3_long_labeled/therapy_dataset", "processed_run3_long_labeled_clean/therapy_dataset"),
    ]
    
    total_removed = 0
    
    for input_path, output_path in datasets:
        input_full = base_dir / input_path
        output_full = base_dir / output_path
        
        if not input_full.exists():
            print(f"\nWARNING: {input_full} not found, skipping...")
            continue
        
        try:
            removed = remove_invalid_examples(
                str(input_full),
                str(output_full),
                tokenizer
            )
            total_removed += removed
        except Exception as e:
            print(f"\nERROR processing {input_path}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print(f"COMPLETE! Removed {total_removed} invalid examples total")
    print("=" * 70)
    print("\nCleaned datasets saved to:")
    for _, output_path in datasets:
        print(f"  - {base_dir / output_path}")
    
    print("\nYou can now use the '_clean' datasets for training.")
    print("All examples in the cleaned datasets have valid 'Response:' markers.")

if __name__ == "__main__":
    main()

