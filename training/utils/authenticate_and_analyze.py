"""
Authenticate with HuggingFace and analyze the DBT dataset.
Run this to get started with the RAG system.
"""

import os
from datasets import load_dataset
from huggingface_hub import login, whoami
import getpass

def authenticate_hf():
    """Authenticate with HuggingFace."""
    print("=" * 80)
    print("HUGGINGFACE AUTHENTICATION")
    print("=" * 80)
    
    # Check if already logged in
    try:
        user = whoami()
        print(f"\n✅ Already logged in as: {user['name']}")
        return True
    except Exception as e:
        print("\n⚠️  Not logged in yet")
    
    # Try environment variable first
    hf_token = os.getenv('HF_TOKEN')
    
    if not hf_token or hf_token in ['your_huggingface_token_here', '']:
        print("\n📝 HuggingFace token not found in environment")
        print("\n💡 To get your token:")
        print("   1. Go to: https://huggingface.co/settings/tokens")
        print("   2. Create a new token with 'read' permission")
        print("   3. Copy the token")
        
        print("\n🔑 Enter your HuggingFace token:")
        hf_token = getpass.getpass("Token: ")
    
    if hf_token:
        try:
            login(token=hf_token)
            user = whoami()
            print(f"✅ Successfully logged in as: {user['name']}")
            
            # Save to environment for this session
            os.environ['HF_TOKEN'] = hf_token
            return True
        except Exception as e:
            print(f"❌ Authentication failed: {e}")
            return False
    
    return False

def analyze_dbt_dataset():
    """Load and analyze the DBT dataset."""
    print("\n" + "=" * 80)
    print("ANALYZING DBT DATASET: vjain/dbt")
    print("=" * 80)
    
    try:
        print("\n📥 Loading dataset...")
        ds = load_dataset("vjain/dbt")
        
        print("\n✅ Dataset loaded successfully!")
        print("\n" + "-" * 80)
        print("DATASET STRUCTURE")
        print("-" * 80)
        print(ds)
        
        # Get splits
        splits = list(ds.keys())
        print(f"\n📊 Splits: {splits}")
        
        # Analyze each split
        for split_name in splits:
            split = ds[split_name]
            
            print("\n" + "=" * 80)
            print(f"SPLIT: {split_name}")
            print("=" * 80)
            print(f"Number of examples: {len(split):,}")
            
            if len(split) > 0:
                print(f"\nColumns: {split.column_names}")
                
                # Show first example
                print("\n" + "-" * 80)
                print("FIRST EXAMPLE:")
                print("-" * 80)
                first = dict(split[0])
                for key, value in first.items():
                    if isinstance(value, str):
                        display = value[:300] + "..." if len(value) > 300 else value
                        print(f"\n{key}:")
                        print(f"  {display}")
                    else:
                        print(f"\n{key}: {value}")
                
                # Statistical summary
                print("\n" + "-" * 80)
                print("STATISTICAL SUMMARY:")
                print("-" * 80)
                
                # Convert to pandas if possible
                if 'text' in split.column_names:
                    df = split.to_pandas()
                    if 'text' in df.columns:
                        lengths = df['text'].str.len()
                        print(f"Text length - Mean: {lengths.mean():.1f}, Min: {lengths.min()}, Max: {lengths.max()}")
                        print(f"Total characters: {lengths.sum():,}")
                
                # Show field names
                print(f"\nAll columns: {', '.join(split.column_names)}")
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        
        # Recommendations
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS FOR RAG")
        print("=" * 80)
        print("""
Based on your dataset analysis:

1. Identify the 'text' column for chunking
2. Set chunk_size based on average text length
3. Consider metadata fields for filtering
4. Test retrieval with sample queries

Next steps:
   python training/utils/dbt_rag_system.py
        """)
        
        return ds
        
    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Make sure you're authenticated (run this script first)")
        print("   2. Check dataset exists: https://huggingface.co/datasets/vjain/dbt")
        print("   3. May need to request access if dataset is private")
        
        return None

def main():
    """Main function."""
    print("\n🚀 Starting authentication and analysis...")
    
    # Authenticate
    if authenticate_hf():
        # Analyze dataset
        ds = analyze_dbt_dataset()
        
        if ds:
            print("\n✅ Ready to set up RAG system!")
            print("\n📝 Next steps:")
            print("   1. Review dataset structure above")
            print("   2. Update dbt_rag_system.py with correct column names")
            print("   3. Run: python training/utils/dbt_rag_system.py")
        else:
            print("\n⚠️  Dataset analysis failed")
    else:
        print("\n❌ Authentication failed - cannot proceed")

if __name__ == "__main__":
    main()

