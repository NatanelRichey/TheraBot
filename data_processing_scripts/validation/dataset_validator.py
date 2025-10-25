#!/usr/bin/env python3
"""
Detailed validation script that shows examples at each stage of data processing.
This helps you understand exactly what the data looks like at each point.
"""

import sys
import json
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer, DataCollatorWithPadding

def detailed_validation(data_dir):
    """Detailed validation with examples at each stage."""
    
    print("DETAILED THERAPY DATA VALIDATION WITH EXAMPLES")
    print("=" * 70)
    print("This script will show you exactly what your data looks like at each stage")
    print("=" * 70)
    
    # Load dataset
    print("\n1. LOADING DATASET")
    print("-" * 30)
    dataset = load_from_disk(data_dir)
    print(f"[OK] Dataset loaded: {len(dataset)} splits")
    
    # Show dataset structure
    print(f"\nDataset structure:")
    for split_name, split_data in dataset.items():
        print(f"  {split_name}: {len(split_data)} examples")
    
    # Show features
    if len(dataset['train']) > 0:
        features = dataset['train'].features
        print(f"\nDataset features:")
        for feature_name, feature_type in features.items():
            print(f"  {feature_name}: {feature_type}")
    
    # Load tokenizer
    print(f"\n2. LOADING TOKENIZER")
    print("-" * 30)
    tokenizer_path = Path(data_dir) / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    print(f"[OK] Tokenizer loaded: {tokenizer.name_or_path}")
    print(f"[OK] Vocab size: {len(tokenizer)}")
    
    # Show special tokens
    print(f"\nSpecial tokens:")
    print(f"  BOS (Beginning): {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    print(f"  EOS (End): {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"  PAD (Padding): {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"  UNK (Unknown): {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")
    
    if len(dataset['train']) == 0:
        print("\n[ERROR] No training examples found!")
        return False
    
    # Get sample data
    sample = dataset['train'][0]
    input_ids = sample['input_ids']
    attention_mask = sample['attention_mask']
    
    print(f"\n3. SAMPLE DATA ANALYSIS")
    print("-" * 30)
    print(f"Sample from training set:")
    print(f"  Input IDs length: {len(input_ids)}")
    print(f"  Attention mask length: {len(attention_mask)}")
    print(f"  Lengths match: {len(input_ids) == len(attention_mask)}")
    
    # Show raw token IDs
    print(f"\nRaw token IDs (first 20):")
    print(f"  {input_ids[:20]}")
    
    # Show attention mask
    print(f"\nAttention mask (first 20):")
    print(f"  {attention_mask[:20]}")
    
    # Decode the full sequence
    print(f"\n4. DECODED TEXT ANALYSIS")
    print("-" * 30)
    full_decoded = tokenizer.decode(input_ids, skip_special_tokens=True)
    print(f"Full decoded text:")
    print(f"  Length: {len(full_decoded)} characters")
    print(f"  Content preview:")
    print(f"  {full_decoded[:300]}...")
    
    # Show special tokens in the sequence
    print(f"\nSpecial tokens in sequence:")
    bos_count = input_ids.count(tokenizer.bos_token_id) if tokenizer.bos_token_id else 0
    eos_count = input_ids.count(tokenizer.eos_token_id) if tokenizer.eos_token_id else 0
    pad_count = input_ids.count(tokenizer.pad_token_id) if tokenizer.pad_token_id else 0
    
    print(f"  BOS tokens: {bos_count}")
    print(f"  EOS tokens: {eos_count}")
    print(f"  PAD tokens: {pad_count}")
    
    # Show token-by-token breakdown
    print(f"\n5. TOKEN-BY-TOKEN BREAKDOWN")
    print("-" * 30)
    print(f"First 10 tokens with their text:")
    for i in range(min(10, len(input_ids))):
        token_id = input_ids[i]
        token_text = tokenizer.decode([token_id])
        attention = attention_mask[i]
        print(f"  Token {i:2d}: ID={token_id:6d}, Text='{token_text}', Attention={attention}")
    
    # Show instruction format
    print(f"\n6. INSTRUCTION FORMAT ANALYSIS")
    print("-" * 30)
    
    # Try to identify instruction structure
    decoded_parts = full_decoded.split('\n\n')
    if len(decoded_parts) >= 2:
        print(f"Text appears to be split into {len(decoded_parts)} parts:")
        for i, part in enumerate(decoded_parts[:3]):  # Show first 3 parts
            print(f"  Part {i+1}: {part[:100]}...")
    
    # Show conversation structure
    lines = full_decoded.split('\n')
    print(f"\nConversation structure (first 10 lines):")
    for i, line in enumerate(lines[:10]):
        if line.strip():
            print(f"  Line {i+1:2d}: {line}")
    
    # Show data statistics
    print(f"\n7. DATA STATISTICS")
    print("-" * 30)
    
    # Token length statistics
    all_lengths = [len(ex['input_ids']) for ex in dataset['train']]
    print(f"Token length statistics (training set):")
    print(f"  Min length: {min(all_lengths)}")
    print(f"  Max length: {max(all_lengths)}")
    print(f"  Average length: {sum(all_lengths)/len(all_lengths):.1f}")
    
    # Length distribution
    length_ranges = [
        (0, 100, "Very short"),
        (100, 500, "Short"),
        (500, 1000, "Medium"),
        (1000, 1500, "Long"),
        (1500, float('inf'), "Very long")
    ]
    
    print(f"\nLength distribution:")
    for min_len, max_len, label in length_ranges:
        count = sum(1 for length in all_lengths if min_len <= length < max_len)
        percentage = (count / len(all_lengths)) * 100
        print(f"  {label:10s}: {count:4d} examples ({percentage:5.1f}%)")
    
    # Show multiple examples
    print(f"\n8. MULTIPLE EXAMPLES")
    print("-" * 30)
    print(f"Showing 3 different examples from training set:")
    
    for i in range(min(3, len(dataset['train']))):
        example = dataset['train'][i]
        decoded = tokenizer.decode(example['input_ids'][:100], skip_special_tokens=True)
        print(f"\nExample {i+1}:")
        print(f"  Length: {len(example['input_ids'])} tokens")
        print(f"  Preview: {decoded[:150]}...")
    
    # Test data collator
    print(f"\n9. DATA COLLATOR TEST")
    print("-" * 30)
    try:
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
        sample_batch = [dataset['train'][i] for i in range(min(4, len(dataset['train'])))]
        collated = data_collator(sample_batch)
        
        print(f"[OK] Data collator works correctly")
        print(f"  Batch input_ids shape: {collated['input_ids'].shape}")
        print(f"  Batch attention_mask shape: {collated['attention_mask'].shape}")
        
        # Explain the tensor shape
        batch_size, max_length = collated['input_ids'].shape
        print(f"\nTensor shape explanation:")
        print(f"  Batch size: {batch_size} (number of examples in batch)")
        print(f"  Max length: {max_length} (longest sequence in this batch)")
        print(f"  This means: {batch_size} examples, each with {max_length} tokens")
        
        # Show batch example
        print(f"\nBatch example (first example in batch):")
        batch_decoded = tokenizer.decode(collated['input_ids'][0][:50], skip_special_tokens=True)
        print(f"  {batch_decoded[:200]}...")
        
    except Exception as e:
        print(f"[ERROR] Data collator failed: {e}")
        return False
    
    # Test encode-decode consistency
    print(f"\n10. ENCODE-DECODE CONSISTENCY TEST")
    print("-" * 30)
    try:
        # Get original text
        original_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        
        # Re-encode the decoded text
        reencoded = tokenizer.encode(original_text, add_special_tokens=True)
        
        # Check if they match
        if input_ids == reencoded:
            print(f"[OK] Perfect encode-decode match!")
        else:
            print(f"[WARNING] Encode-decode mismatch!")
            print(f"  Original length: {len(input_ids)}")
            print(f"  Re-encoded length: {len(reencoded)}")
            print(f"  Difference: {len(input_ids) - len(reencoded)} tokens")
            
            # Show differences
            min_len = min(len(input_ids), len(reencoded))
            differences = 0
            for i in range(min_len):
                if input_ids[i] != reencoded[i]:
                    differences += 1
            print(f"  Token differences: {differences}")
        
        # Test with special tokens
        print(f"\nTesting with special tokens:")
        original_with_special = tokenizer.decode(input_ids, skip_special_tokens=False)
        reencoded_with_special = tokenizer.encode(original_with_special, add_special_tokens=False)
        
        if input_ids == reencoded_with_special:
            print(f"[OK] Perfect match with special tokens!")
        else:
            print(f"[WARNING] Mismatch with special tokens!")
        
    except Exception as e:
        print(f"[ERROR] Encode-decode test failed: {e}")
        return False
    
    # Quality checks
    print(f"\n11. QUALITY CHECKS")
    print("-" * 30)
    
    # Check for empty sequences
    empty_sequences = 0
    for split_name, split_data in dataset.items():
        for example in split_data:
            if len(example['input_ids']) == 0:
                empty_sequences += 1
    
    print(f"Empty sequences: {empty_sequences}")
    
    # Check attention mask consistency
    inconsistent_masks = 0
    for split_name, split_data in dataset.items():
        for example in split_data:
            if len(example['input_ids']) != len(example['attention_mask']):
                inconsistent_masks += 1
    
    print(f"Inconsistent attention masks: {inconsistent_masks}")
    
    # Check for reasonable token lengths
    very_short = sum(1 for length in all_lengths if length < 10)
    very_long = sum(1 for length in all_lengths if length > 2000)
    
    print(f"Very short sequences (<10 tokens): {very_short}")
    print(f"Very long sequences (>2000 tokens): {very_long}")
    
    # Final assessment
    print(f"\n12. FINAL ASSESSMENT")
    print("-" * 30)
    
    total_examples = sum(len(split) for split in dataset.values())
    print(f"Total examples: {total_examples:,}")
    
    issues = []
    if total_examples == 0:
        issues.append("No examples found")
    if empty_sequences > 0:
        issues.append(f"{empty_sequences} empty sequences")
    if inconsistent_masks > 0:
        issues.append(f"{inconsistent_masks} inconsistent attention masks")
    if very_short > total_examples * 0.1:  # More than 10% very short
        issues.append(f"Too many very short sequences ({very_short})")
    
    if not issues:
        print(f"[SUCCESS] Dataset is ready for training!")
        print(f"[SUCCESS] All quality checks passed!")
        print(f"[SUCCESS] Data looks good for fine-tuning!")
        return True
    else:
        print(f"[WARNING] Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python dataset_validator.py <data_directory>")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    success = detailed_validation(data_dir)
    sys.exit(0 if success else 1)
