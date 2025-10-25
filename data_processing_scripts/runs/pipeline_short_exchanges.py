#!/usr/bin/env python3
"""
Run 1: Complete pipeline with short exchanges
50% 2 exchanges, 25% 3 exchanges, 15% 6 exchanges, 10% 10 exchanges
"""

import sys
import subprocess
from pathlib import Path

def run_complete_pipeline(run_name, exchange_distribution, input_dir, base_output_dir):
    """Run the complete pipeline: instruction conversion + tokenization + validation."""
    
    print(f"RUNNING COMPLETE PIPELINE - {run_name}")
    print("=" * 60)
    
    # Define paths
    instruction_output_dir = f"{base_output_dir}_{run_name}_instructions"
    processed_output_dir = f"{base_output_dir}_{run_name}_processed"
    
    print(f"Input directory: {input_dir}")
    print(f"Instruction output: {instruction_output_dir}")
    print(f"Processed output: {processed_output_dir}")
    
    # Step 1: Instruction Conversion
    print(f"\n{'='*60}")
    print("STEP 1: INSTRUCTION CONVERSION")
    print(f"{'='*60}")
    
    print("Exchange distribution:")
    for count, weight in sorted(exchange_distribution.items()):
        print(f"  {weight*100:.0f}% - {count} exchanges")
    
    # Build command for instruction conversion
    cmd = [
        "python", "therapy_instruction_converter.py",
        input_dir,
        instruction_output_dir,
        run_name
    ]
    
    # Add exchange distribution arguments
    for count, weight in exchange_distribution.items():
        cmd.append(f"{count}:{weight}")
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Instruction conversion completed successfully!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Instruction conversion failed!")
        print(f"Error: {e.stderr}")
        return False
    
    # Step 2: Tokenization
    print(f"\n{'='*60}")
    print("STEP 2: TOKENIZATION")
    print(f"{'='*60}")
    
    instruction_file = f"{instruction_output_dir}/instruction_examples_{run_name}.json"
    
    if not Path(instruction_file).exists():
        print(f"❌ Instruction file not found: {instruction_file}")
        return False
    
    cmd = [
        "python", "therapy_tokenizer.py",
        instruction_file,
        processed_output_dir
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Tokenization completed successfully!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Tokenization failed!")
        print(f"Error: {e.stderr}")
        return False
    
    # Step 3: Validation
    print(f"\n{'='*60}")
    print("STEP 3: VALIDATION")
    print(f"{'='*60}")
    
    cmd = [
        "python", "dataset_validator.py",
        f"{processed_output_dir}/therapy_dataset"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Validation completed successfully!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Validation failed!")
        print(f"Error: {e.stderr}")
        return False
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"RUN {run_name.upper()} COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"Instruction examples: {instruction_output_dir}")
    print(f"Processed dataset: {processed_output_dir}")
    print(f"Ready for training!")
    
    return True

def main():
    """Main function for Run 1."""
    
    # Run 1 configuration
    run_name = "run1_short"
    exchange_distribution = {
        2: 0.50,   # 50% - 2 exchanges
        3: 0.25,   # 25% - 3 exchanges
        6: 0.15,   # 15% - 6 exchanges
        10: 0.10   # 10% - 10 exchanges
    }
    
    input_dir = "../data/transcripts_json"
    base_output_dir = "../data/processed"
    
    print("RUN 1: SHORT EXCHANGES")
    print("Distribution: 50% 2 exchanges, 25% 3 exchanges, 15% 6 exchanges, 10% 10 exchanges")
    
    success = run_complete_pipeline(run_name, exchange_distribution, input_dir, base_output_dir)
    
    if success:
        print("\n🎉 RUN 1 COMPLETED SUCCESSFULLY!")
    else:
        print("\n❌ RUN 1 FAILED!")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
