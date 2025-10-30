"""
Detailed analysis of the DBT manual dataset.
"""

from datasets import load_dataset
import pandas as pd

def detailed_analysis():
    """Perform detailed analysis of DBT dataset."""
    
    print("=" * 80)
    print("DETAILED DBT MANUAL ANALYSIS")
    print("=" * 80)
    
    # Load dataset
    print("\n📥 Loading dataset...")
    ds = load_dataset("vjain/dbt")
    
    # Convert to DataFrame
    train_split = ds['train']
    df = train_split.to_pandas()
    
    print(f"\n✅ Loaded {len(df)} examples")
    
    # Basic stats
    print("\n" + "=" * 80)
    print("BASIC STATISTICS")
    print("=" * 80)
    
    df['content_length'] = df['page_content'].str.len()
    
    print(f"\nContent length statistics:")
    print(f"  Mean: {df['content_length'].mean():.1f} characters")
    print(f"  Median: {df['content_length'].median():.1f} characters")
    print(f"  Min: {df['content_length'].min()} characters")
    print(f"  Max: {df['content_length'].max()} characters")
    print(f"  Total characters: {df['content_length'].sum():,}")
    
    print(f"\nSource breakdown:")
    print(df['source'].value_counts().to_string())
    
    # Sample content analysis
    print("\n" + "=" * 80)
    print("SAMPLE CONTENT (First 5 Examples)")
    print("=" * 80)
    
    for i in range(min(5, len(df))):
        row = df.iloc[i]
        print(f"\n--- Example {i+1} (ID: {row['id']}) ---")
        print(f"Source: {row['source']}")
        print(f"Length: {len(row['page_content'])} chars")
        print(f"\nContent preview:")
        # Show first 300 chars
        preview = row['page_content'][:300].replace('\n', ' ')
        print(f"  {preview}...")
    
    # Look for DBT-specific terms
    print("\n" + "=" * 80)
    print("DBT TERM SEARCH")
    print("=" * 80)
    
    dbt_terms = [
        'TIPP', 'PLEASE', 'DEAR MAN', 'GIVE', 'FAST',
        'mindfulness', 'distress tolerance', 'emotion regulation',
        'interpersonal effectiveness', 'validation', 'dialectical'
    ]
    
    print("\nSearching for DBT-specific terms:")
    for term in dbt_terms:
        count = df['page_content'].str.contains(term, case=False, regex=True).sum()
        if count > 0:
            print(f"  {term}: found in {count} examples ({count/len(df)*100:.1f}%)")
    
    # Long examples
    print("\n" + "=" * 80)
    print("LONGEST EXAMPLES")
    print("=" * 80)
    
    longest = df.nlargest(5, 'content_length')
    for idx, row in longest.iterrows():
        print(f"\nLength: {row['content_length']} chars")
        preview = row['page_content'][:200].replace('\n', ' ')
        print(f"  {preview}...")
    
    # Short examples (likely headers)
    print("\n" + "=" * 80)
    print("SHORTEST EXAMPLES (Likely Headers)")
    print("=" * 80)
    
    shortest = df.nsmallest(5, 'content_length')
    for idx, row in shortest.iterrows():
        print(f"\nLength: {row['content_length']} chars")
        print(f"  {row['page_content']}")
    
    # Find examples with DBT skills
    print("\n" + "=" * 80)
    print("DBT SKILL EXAMPLES")
    print("=" * 80)
    
    skill_examples = df[df['page_content'].str.contains('TIPP|PLEASE|DEAR MAN', case=False, regex=True)]
    print(f"\nFound {len(skill_examples)} examples mentioning DBT skills")
    
    if len(skill_examples) > 0:
        print("\nSample skill example:")
        sample = skill_examples.iloc[0]
        preview = sample['page_content'][:500].replace('\n', ' ')
        print(f"  {preview}...")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RAG RECOMMENDATIONS")
    print("=" * 80)
    
    avg_length = df['content_length'].mean()
    print(f"""
Based on analysis:

📊 Dataset: 691 examples from DBT manual

📏 Chunking Strategy:
   - Average length: {avg_length:.0f} characters
   - Recommend chunk_size: {int(avg_length * 0.5)} - {int(avg_length * 2)}
   - chunk_overlap: 100-200 characters

🔍 Retrieval:
   - 691 raw examples will be chunked into ~1000-2000 chunks
   - top_k=3-5 should work well for most queries
   - DBT skills are well-represented

✨ Setup:
   1. Run: python training/utils/dbt_rag_system.py
   2. Will auto-chunk and create embeddings
   3. Estimated time: 10-20 minutes for first run

✅ Ready to implement RAG!
    """)
    
    return df

if __name__ == "__main__":
    df = detailed_analysis()

