#!/usr/bin/env python3
"""
Validation script to verify labels are correctly applied.
This script checks that:
1. Only response tokens are labeled (have non -100 values)
2. All context tokens have label = -100
3. Labels match the input_ids for response tokens
4. Shows clear examples for verification
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

def validate_all_labels_comprehensive(dataset_paths, tokenizer):
    """
    Comprehensive validation that checks EVERY SINGLE LABEL across all datasets.
    
    Args:
        dataset_paths: List of paths to labeled datasets
        tokenizer: Tokenizer instance
        
    Returns:
        True if all labels are valid, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE LABEL VALIDATION - CHECKING EVERY SINGLE LABEL")
    print(f"{'='*80}")
    
    total_examples_checked = 0
    total_labels_checked = 0
    total_invalid_labels = 0
    total_invalid_examples = 0
    
    for dataset_path in dataset_paths:
        print(f"\n{'='*80}")
        print(f"DATASET: {dataset_path}")
        print(f"{'='*80}")
        
        try:
            dataset = load_from_disk(dataset_path)
            splits = dataset if hasattr(dataset, 'keys') else {'dataset': dataset}
            
            for split_name, split_data in splits.items():
                if len(split_data) == 0:
                    continue
                    
                print(f"\n{'-'*60}")
                print(f"Split: {split_name} ({len(split_data)} examples)")
                print(f"{'-'*60}")
                
                split_invalid_labels = 0
                split_invalid_examples = 0
                
                # Check EVERY example in this split
                for i in range(len(split_data)):
                    example = split_data[i]
                    input_ids = example['input_ids']
                    labels = example['labels']
                    
                    # Find response start
                    response_start_idx = find_response_start(input_ids, tokenizer)
                    
                    if response_start_idx == -1:
                        print(f"  Example {i+1}: ERROR - Could not find 'Response:'")
                        # Decode the full text to show what's actually there
                        full_text = tokenizer.decode(input_ids, skip_special_tokens=False)
                        print(f"    Full decoded text: {full_text}")
                        print(f"    First 50 tokens: {input_ids[:50]}")
                        split_invalid_examples += 1
                        continue
                    
                    # Check EVERY label
                    example_invalid_labels = 0
                    
                    # Check context labels (should all be -100)
                    for j in range(response_start_idx):
                        if labels[j] != -100:
                            example_invalid_labels += 1
                            if example_invalid_labels == 1:  # Only print first error per example
                                print(f"  Example {i+1}: ERROR - Context label at position {j} = {labels[j]} (should be -100)")
                    
                    # Check response labels (should match input_ids)
                    for j in range(response_start_idx, len(labels)):
                        if labels[j] != input_ids[j]:
                            example_invalid_labels += 1
                            if example_invalid_labels == 1:  # Only print first error per example
                                print(f"  Example {i+1}: ERROR - Response label at position {j} = {labels[j]} (should be {input_ids[j]})")
                    
                    if example_invalid_labels > 0:
                        split_invalid_examples += 1
                        split_invalid_labels += example_invalid_labels
                    
                    total_examples_checked += 1
                    total_labels_checked += len(labels)
                    total_invalid_labels += example_invalid_labels
                
                total_invalid_examples += split_invalid_examples
                
                print(f"\nSplit Summary:")
                print(f"  Examples checked: {len(split_data)}")
                print(f"  Invalid examples: {split_invalid_examples}")
                print(f"  Invalid labels: {split_invalid_labels}")
                
                if split_invalid_examples == 0:
                    print(f"  STATUS: ✓ ALL LABELS VALID")
                else:
                    print(f"  STATUS: ✗ {split_invalid_examples} EXAMPLES HAVE INVALID LABELS")
        
        except Exception as e:
            print(f"\nERROR processing {dataset_path}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total examples checked: {total_examples_checked}")
    print(f"Total labels checked: {total_labels_checked}")
    print(f"Invalid examples: {total_invalid_examples}")
    print(f"Invalid labels: {total_invalid_labels}")
    
    if total_invalid_examples == 0:
        print(f"\n✓ ALL {total_examples_checked} EXAMPLES HAVE VALID LABELS!")
        print(f"✓ ALL {total_labels_checked} LABELS ARE CORRECTLY SET!")
    else:
        print(f"\n✗ {total_invalid_examples} EXAMPLES HAVE INVALID LABELS!")
        print(f"✗ {total_invalid_labels} LABELS ARE INCORRECT!")
    
    return total_invalid_examples == 0

def validate_labels(dataset_path, tokenizer, num_examples=5):
    """
    Validate labels in a dataset.
    
    Args:
        dataset_path: Path to the dataset
        tokenizer: Tokenizer instance
        num_examples: Number of examples to validate and show
    """
    print(f"\n{'='*70}")
    print(f"VALIDATING: {dataset_path}")
    print(f"{'='*70}")
    
    # Load dataset
    dataset = load_from_disk(dataset_path)
    
    # Get all splits
    splits = dataset if hasattr(dataset, 'keys') else {'dataset': dataset}
    
    total_valid = 0
    total_invalid = 0
    
    for split_name, split_data in splits.items():
        if len(split_data) == 0:
            continue
            
        print(f"\n{'-'*70}")
        print(f"Split: {split_name} ({len(split_data)} examples)")
        print(f"{'-'*70}")
        
        # Validate a few examples
        num_to_check = min(num_examples, len(split_data))
        split_valid = 0
        split_invalid = 0
        
        for i in range(num_to_check):
            example = split_data[i]
            input_ids = example['input_ids']
            labels = example['labels']
            
            print(f"\nExample {i+1}:")
            print(f"  Total length: {len(input_ids)} tokens")
            
            # Find response start
            response_start_idx = find_response_start(input_ids, tokenizer)
            
            if response_start_idx == -1:
                print(f"  WARNING: Could not find 'Response:' in this example!")
                print(f"  First 20 tokens: {input_ids[:20]}")
                decoded = tokenizer.decode(input_ids[:50])
                print(f"  Decoded start: {decoded}")
                split_invalid += 1
                continue
            
            # Check that context has label = -100
            context_labels = labels[:response_start_idx]
            context_valid = all(l == -100 for l in context_labels)
            
            # Check that response has non -100 labels
            response_labels = labels[response_start_idx:]
            response_valid = any(l != -100 for l in response_labels)
            
            # Count labeled tokens
            num_labeled = sum(1 for l in labels if l != -100)
            num_ignored = sum(1 for l in labels if l == -100)
            
            print(f"  Response starts at token: {response_start_idx}")
            print(f"  Context tokens (should be -100): {num_ignored}")
            print(f"  Labeled tokens: {num_labeled}")
            
            if context_valid and response_valid:
                print(f"  STATUS: ✓ VALID")
                split_valid += 1
            else:
                print(f"  STATUS: ✗ INVALID")
                if not context_valid:
                    print(f"    ERROR: Context has non -100 labels!")
                    # Find the first non -100 in context
                    for j, l in enumerate(context_labels):
                        if l != -100:
                            print(f"    First error at position {j}: label = {l}")
                            break
                if not response_valid:
                    print(f"    ERROR: Response has all -100 labels!")
                split_invalid += 1
            
            # Show decoded example
            print(f"\n  DECODED EXAMPLE:")
            print(f"  {'-'*70}")
            
            # Decode full sequence
            full_text = tokenizer.decode(input_ids, skip_special_tokens=False)
            
            # Find where response starts in text
            response_text_start = full_text.find("Response:")
            if response_text_start >= 0:
                context_text = full_text[:response_text_start]
                response_text = full_text[response_text_start + len("Response:"):]
                
                print(f"  CONTEXT (ignored in loss):")
                # Show last 200 chars of context
                print(f"  {context_text[-200:]}")
                print(f"\n  RESPONSE (learned by model):")
                print(f"  {response_text}")
            else:
                print(f"  Could not find 'Response:' in decoded text")
                print(f"  Full text: {full_text}")
            
            # Show label distribution
            print(f"\n  LABEL CHECK:")
            print(f"    Positions {0} to {response_start_idx-1}: context (should all be -100)")
            print(f"    Positions {response_start_idx} to {len(labels)-1}: response (should be actual token IDs)")
            
            # Check a few positions
            if response_start_idx > 0:
                check_positions = [
                    0,  # First token
                    response_start_idx // 2,  # Middle of context
                    response_start_idx - 1,  # Last context token
                    response_start_idx,  # First response token
                    response_start_idx + 5,  # 5th response token
                    min(response_start_idx + 20, len(labels) - 1),  # Far into response
                ]
                
                print(f"    Sample labels at key positions:")
                for pos in check_positions:
                    if pos < len(labels):
                        label = labels[pos]
                        token = input_ids[pos]
                        decoded_token = tokenizer.decode([token])
                        status = "CONTEXT" if pos < response_start_idx else "RESPONSE"
                        label_status = "✓" if (pos < response_start_idx and label == -100) or (pos >= response_start_idx and label != -100) else "✗"
                        print(f"      [{pos:4d}]: {label:6d} (token: {token:6d}, decoded: '{decoded_token:20s}') [{status}] {label_status}")
            
            print(f"\n  {'-'*70}")
        
        print(f"\nSplit Summary:")
        print(f"  Valid examples: {split_valid}/{num_to_check}")
        print(f"  Invalid examples: {split_invalid}/{num_to_check}")
        
        total_valid += split_valid
        total_invalid += split_invalid
    
    print(f"\n{'='*70}")
    print(f"TOTAL VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"Valid examples: {total_valid}")
    print(f"Invalid examples: {total_invalid}")
    print(f"Total checked: {total_valid + total_invalid}")
    
    if total_invalid == 0:
        print(f"\n✓ ALL EXAMPLES VALID!")
    else:
        print(f"\n✗ SOME EXAMPLES INVALID!")
    
    return total_invalid == 0

def main():
    """Main function to validate all labeled datasets."""
    if len(sys.argv) < 2:
        print("Usage: python validate_labels.py <model_name>")
        print("Example: python validate_labels.py meta-llama/Llama-3.1-8B-Instruct")
        sys.exit(1)
    
    model_name = sys.argv[1]
    
    print("=" * 70)
    print("VALIDATING LABELS IN THERAPY DATASETS")
    print("=" * 70)
    print(f"Model: {model_name}")
    print("\nThis will verify that labels are correctly applied:")
    print("- Context tokens should have label = -100 (ignored)")
    print("- Response tokens should have label = actual token ID")
    print("=" * 70)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Tokenizer loaded.")
    
    # Define datasets to validate
    base_dir = Path("data")
    datasets = [
        "processed_run1_short_labeled/therapy_dataset",
        "processed_run2_medium_labeled/therapy_dataset",
        "processed_run3_long_labeled/therapy_dataset",
    ]
    
    # First, show sample examples with full responses
    print("\n" + "="*80)
    print("SAMPLE VALIDATION WITH FULL RESPONSES")
    print("="*80)
    
    all_valid = True
    
    # Validate each dataset with samples
    for dataset_path in datasets:
        full_path = base_dir / dataset_path
        
        if not full_path.exists():
            print(f"\nWARNING: {full_path} not found, skipping...")
            continue
        
        try:
            is_valid = validate_labels(str(full_path), tokenizer)
            all_valid = all_valid and is_valid
        except Exception as e:
            print(f"\nERROR validating {full_path}: {e}")
            import traceback
            traceback.print_exc()
            all_valid = False
    
    # Now run comprehensive validation on ALL labels
    print("\n" + "="*80)
    print("COMPREHENSIVE VALIDATION - CHECKING EVERY SINGLE LABEL")
    print("="*80)
    
    # Get all dataset paths that exist
    existing_datasets = []
    for dataset_path in datasets:
        full_path = base_dir / dataset_path
        if full_path.exists():
            existing_datasets.append(str(full_path))
    
    if existing_datasets:
        comprehensive_valid = validate_all_labels_comprehensive(existing_datasets, tokenizer)
        all_valid = all_valid and comprehensive_valid
    else:
        print("No datasets found for comprehensive validation!")
        all_valid = False
    
    print(f"\n{'='*70}")
    if all_valid:
        print("✓ ALL DATASETS VALIDATED SUCCESSFULLY!")
    else:
        print("✗ SOME DATASETS HAVE VALIDATION ERRORS!")
    print("=" * 70)
    
    sys.exit(0 if all_valid else 1)

if __name__ == "__main__":
    main()

