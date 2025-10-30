#!/usr/bin/env python3
"""
Analyze the size of responses that are being labeled in the cleaned datasets.
This shows how many tokens the model is learning to predict.
"""

import sys
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer
import numpy as np

def find_response_start(input_ids, tokenizer):
    """Find where the response starts."""
    response_tokens = tokenizer.encode("Response:", add_special_tokens=False)
    for i in range(len(input_ids) - len(response_tokens) + 1):
        if input_ids[i:i+len(response_tokens)] == response_tokens:
            return i + len(response_tokens)
    return -1

def analyze_response_sizes(dataset_path, tokenizer, dataset_name):
    """
    Analyze response sizes in a dataset.
    
    Args:
        dataset_path: Path to the dataset
        tokenizer: Tokenizer instance
        dataset_name: Name of the dataset for display
        
    Returns:
        Dictionary with statistics
    """
    print(f"\n{'='*80}")
    print(f"ANALYZING: {dataset_name}")
    print(f"{'='*80}")
    
    dataset = load_from_disk(dataset_path)
    splits = dataset if hasattr(dataset, 'keys') else {'dataset': dataset}
    
    all_response_sizes = []
    
    for split_name, split_data in splits.items():
        if len(split_data) == 0:
            continue
        
        print(f"\n  Split: {split_name} ({len(split_data)} examples)")
        
        split_response_sizes = []
        
        for i, example in enumerate(split_data):
            input_ids = example['input_ids']
            labels = example['labels']
            
            # Find response start
            response_start_idx = find_response_start(input_ids, tokenizer)
            
            if response_start_idx >= 0:
                # Count labeled tokens (not -100)
                response_size = sum(1 for label in labels if label != -100)
                split_response_sizes.append(response_size)
                all_response_sizes.append(response_size)
        
        # Statistics for this split
        if split_response_sizes:
            sizes_array = np.array(split_response_sizes)
            print(f"    Mean response size: {np.mean(sizes_array):.1f} tokens")
            print(f"    Median response size: {np.median(sizes_array):.1f} tokens")
            print(f"    Min response size: {np.min(sizes_array)} tokens")
            print(f"    Max response size: {np.max(sizes_array)} tokens")
            print(f"    Std deviation: {np.std(sizes_array):.1f} tokens")
            print(f"    Percentiles:")
            print(f"      25th: {np.percentile(sizes_array, 25):.1f} tokens")
            print(f"      50th: {np.percentile(sizes_array, 50):.1f} tokens")
            print(f"      75th: {np.percentile(sizes_array, 75):.1f} tokens")
            print(f"      95th: {np.percentile(sizes_array, 95):.1f} tokens")
            print(f"      99th: {np.percentile(sizes_array, 99):.1f} tokens")
            
            # Distribution bins
            bins = [0, 5, 10, 20, 30, 50, 100, 200, 500, 1000, float('inf')]
            bin_labels = ['0-5', '6-10', '11-20', '21-30', '31-50', '51-100', '101-200', '201-500', '501-1000', '1000+']
            counts, _ = np.histogram(sizes_array, bins=bins)
            print(f"    Distribution:")
            for i, (label, count) in enumerate(zip(bin_labels, counts)):
                if count > 0:
                    pct = (count / len(sizes_array)) * 100
                    print(f"      {label:10s}: {count:6d} examples ({pct:5.1f}%)")
    
    # Overall statistics for this dataset
    if all_response_sizes:
        all_sizes_array = np.array(all_response_sizes)
        print(f"\n  OVERALL FOR {dataset_name}:")
        print(f"    Total examples: {len(all_sizes_array)}")
        print(f"    Mean response size: {np.mean(all_sizes_array):.1f} tokens")
        print(f"    Median response size: {np.median(all_sizes_array):.1f} tokens")
        print(f"    Min response size: {np.min(all_sizes_array)} tokens")
        print(f"    Max response size: {np.max(all_sizes_array)} tokens")
    
    return all_response_sizes

def main():
    """Main function to analyze all cleaned datasets."""
    if len(sys.argv) < 2:
        print("Usage: python analyze_response_sizes.py <model_name>")
        print("Example: python analyze_response_sizes.py meta-llama/Llama-3.1-8B-Instruct")
        sys.exit(1)
    
    model_name = sys.argv[1]
    
    print("=" * 80)
    print("RESPONSE SIZE ANALYTICS FOR LABELED DATASETS")
    print("=" * 80)
    print(f"Model: {model_name}")
    print("\nThis analyzes how many tokens are in the responses that the model")
    print("is learning to predict (the labeled portion of each example).")
    print("=" * 80)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Tokenizer loaded.")
    
    # Define datasets
    base_dir = Path("data")
    datasets = [
        ("processed_run1_short_labeled_clean/therapy_dataset", "Short"),
        ("processed_run2_medium_labeled_clean/therapy_dataset", "Medium"),
        ("processed_run3_long_labeled_clean/therapy_dataset", "Long"),
    ]
    
    all_sizes = {}
    
    # Analyze each dataset
    for dataset_path, dataset_name in datasets:
        full_path = base_dir / dataset_path
        
        if not full_path.exists():
            print(f"\nWARNING: {full_path} not found, skipping...")
            continue
        
        try:
            sizes = analyze_response_sizes(str(full_path), tokenizer, dataset_name)
            all_sizes[dataset_name] = sizes
        except Exception as e:
            print(f"\nERROR analyzing {full_path}: {e}")
            import traceback
            traceback.print_exc()
    
    # Overall comparison
    print("\n" + "="*80)
    print("OVERALL COMPARISON ACROSS ALL DATASETS")
    print("="*80)
    
    combined_sizes = []
    for dataset_name, sizes in all_sizes.items():
        combined_sizes.extend(sizes)
    
    if combined_sizes:
        combined_array = np.array(combined_sizes)
        print(f"\nCombined Statistics (all datasets):")
        print(f"  Total examples: {len(combined_array):,}")
        print(f"  Mean response size: {np.mean(combined_array):.1f} tokens")
        print(f"  Median response size: {np.median(combined_array):.1f} tokens")
        print(f"  Min response size: {np.min(combined_array)} tokens")
        print(f"  Max response size: {np.max(combined_array)} tokens")
        print(f"  Std deviation: {np.std(combined_array):.1f} tokens")
        print(f"\n  Percentiles:")
        print(f"    25th: {np.percentile(combined_array, 25):.1f} tokens")
        print(f"    50th: {np.percentile(combined_array, 50):.1f} tokens")
        print(f"    75th: {np.percentile(combined_array, 75):.1f} tokens")
        print(f"    95th: {np.percentile(combined_array, 95):.1f} tokens")
        print(f"    99th: {np.percentile(combined_array, 99):.1f} tokens")
        
        # Distribution bins
        bins = [0, 5, 10, 20, 30, 50, 100, 200, 500, 1000, float('inf')]
        bin_labels = ['0-5', '6-10', '11-20', '21-30', '31-50', '51-100', '101-200', '201-500', '501-1000', '1000+']
        counts, _ = np.histogram(combined_array, bins=bins)
        print(f"\n  Distribution (all datasets):")
        for i, (label, count) in enumerate(zip(bin_labels, counts)):
            if count > 0:
                pct = (count / len(combined_array)) * 100
                print(f"    {label:10s}: {count:8,d} examples ({pct:5.1f}%)")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nKey insights:")
    print("  - The model is learning to predict therapist responses")
    print("  - Context is fully masked (ignored in loss computation)")
    print("  - Only the response tokens contribute to the loss")
    print("  - This ensures the model learns to generate appropriate therapeutic responses")

if __name__ == "__main__":
    main()

