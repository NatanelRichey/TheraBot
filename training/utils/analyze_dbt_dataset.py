"""
Analyze the DBT manual dataset to understand its structure and content.
"""

from datasets import load_dataset
import pandas as pd

def analyze_dbt_dataset():
    """Load and analyze the DBT manual dataset."""
    
    print("=" * 80)
    print("DBT MANUAL DATASET ANALYSIS")
    print("=" * 80)
    
    try:
        print("\n📥 Loading dataset: vjain/dbt...")
        ds = load_dataset("vjain/dbt")
        
        print("\n✅ Dataset loaded successfully!")
        
        # Show dataset structure
        print("\n" + "-" * 80)
        print("DATASET STRUCTURE")
        print("-" * 80)
        print(ds)
        
        # Get splits
        splits = list(ds.keys())
        print(f"\nSplits: {splits}")
        
        # Analyze each split
        for split_name in splits:
            split = ds[split_name]
            
            print("\n" + "-" * 80)
            print(f"ANALYZING SPLIT: {split_name}")
            print("-" * 80)
            print(f"Number of examples: {len(split)}")
            
            # Show column names
            if len(split) > 0:
                print(f"Columns: {split.column_names}")
                
                # Show sample rows
                print("\nFirst 3 examples:")
                for i in range(min(3, len(split))):
                    example = dict(split[i])
                    print(f"\n--- Example {i+1} ---")
                    for key, value in example.items():
                        if isinstance(value, str):
                            # Truncate long strings
                            display_value = value[:200] + "..." if len(value) > 200 else value
                            print(f"{key}: {display_value}")
                        else:
                            print(f"{key}: {value}")
                
                # Statistical analysis if text fields exist
                print("\n" + "-" * 80)
                print("STATISTICAL ANALYSIS")
                print("-" * 80)
                
                for col in split.column_names:
                    if col in split.features and split.features[col] and hasattr(split.features[col], 'dtype'):
                        # Try to convert to pandas for analysis
                        df = split.to_pandas()
                        if col in df.columns:
                            if df[col].dtype == 'object':  # String columns
                                avg_length = df[col].str.len().mean()
                                print(f"{col} average length: {avg_length:.1f} characters")
                                print(f"{col} min length: {df[col].str.len().min()}")
                                print(f"{col} max length: {df[col].str.len().max()}")
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        
        return ds
        
    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Check your HuggingFace credentials (HF_TOKEN environment variable)")
        print("   2. Verify dataset exists: https://huggingface.co/datasets/vjain/dbt")
        print("   3. May need to request access if dataset is private")
        
        return None

if __name__ == "__main__":
    ds = analyze_dbt_dataset()

