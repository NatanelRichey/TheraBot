"""
Show DBT dataset content and how RAG will use it.
"""

from datasets import load_dataset

def show_dataset_samples():
    """Show actual samples from your DBT dataset."""
    print("=" * 80)
    print("YOUR DBT DATASET: WHAT IT CONTAINS")
    print("=" * 80)
    
    # Load dataset
    ds = load_dataset("vjain/dbt")
    train_split = ds['train']
    
    print("\n📊 Dataset Overview:")
    print(f"   Total examples: {len(train_split)}")
    print(f"   Columns: {train_split.column_names}")
    
    # Sample 1: Show headers
    print("\n" + "=" * 80)
    print("SAMPLE 1: Section Headers")
    print("=" * 80)
    headers = train_split.filter(lambda x: len(x['page_content']) < 100)
    for i, idx in enumerate(range(min(3, len(headers))), 1):
        ex = headers[idx]
        print(f"\nHeader {i}:")
        print(f"  {ex['page_content']}")
    
    # Sample 2: Typical content
    print("\n" + "=" * 80)
    print("SAMPLE 2: Typical Content Chunk (This is what gets embedded)")
    print("=" * 80)
    
    typical = train_split.filter(lambda x: 500 < len(x['page_content']) < 1200)
    example = typical[10]
    
    print(f"\nID: {example['id']}")
    print(f"Length: {len(example['page_content'])} characters")
    print(f"\nContent:")
    print("-" * 80)
    print(example['page_content'])
    print("-" * 80)
    
    print("\n✅ This entire chunk → becomes 384 numbers (embedding vector)")
    print("✅ All 691 chunks stored in vector database for retrieval")
    
    # Sample 3: DBT skill content
    print("\n" + "=" * 80)
    print("SAMPLE 3: DBT Skill Content (High-value for RAG)")
    print("=" * 80)
    
    # Find different skills
    skills_to_find = {
        'TIPP': 'tipp',
        'PLEASE': 'please',
        'DEAR MAN': 'dear man',
        'Mindfulness': 'mindfulness'
    }
    
    for skill_name, keyword in skills_to_find.items():
        matches = train_split.filter(
            lambda x: keyword in x['page_content'].lower()
        )
        if len(matches) > 0:
            ex = matches[0]
            print(f"\n{skill_name} skill example:")
            print("-" * 80)
            print(f"ID: {ex['id']}")
            print(f"Length: {len(ex['page_content'])} characters")
            
            # Show excerpt around the keyword
            content = ex['page_content']
            keyword_idx = content.lower().find(keyword)
            if keyword_idx != -1:
                start = max(0, keyword_idx - 50)
                end = min(len(content), keyword_idx + 500)
                excerpt = content[start:end]
                print(f"\nExcerpt:")
                print(excerpt)
                if end < len(content):
                    print("...")
            break
    
    # Long example
    print("\n" + "=" * 80)
    print("SAMPLE 4: Long Detailed Content")
    print("=" * 80)
    long_ex = train_split.filter(lambda x: len(x['page_content']) > 1800)[0]
    print(f"\nLength: {len(long_ex['page_content'])} characters")
    print(f"\nFirst 600 characters:")
    print(long_ex['page_content'][:600])
    print("...")

def explain_rag_flow():
    """Explain how the model will see data through RAG."""
    print("\n\n" + "=" * 80)
    print("HOW YOUR MODEL WILL SEE DATA THROUGH RAG")
    print("=" * 80)
    
    print("""
┌─────────────────────────────────────────────────────────────────┐
│ BEFORE RAG (System Prompts)                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ User: "Teach me TIPP skills"                                    │
│                                                                  │
│ Model sees:                                                      │
│ ┌──────────────────────────────────────────────────────────┐   │
│ │ System Prompt (static, always the same):                 │   │
│ │ "You are a DBT therapist. Use DBT skills..."            │   │
│ │ [~500 words of DBT guidance]                            │   │
│ └──────────────────────────────────────────────────────────┘   │
│                                 └─────────↓──────────┘          │
│                            Model generates response             │
│                            (using only its training)            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────┐
│ WITH RAG (Dynamic Retrieval) ⭐                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ User: "Teach me TIPP skills"                                    │
│                                                                  │
│ STEP 1: Search DBT manual                                       │
│          "TIPP skills" → Find most relevant chunks               │
│                                                                  │
│ STEP 2: Retrieve top matches                                    │
│          Chunk 245: "TIPP stands for Temperature..."            │
│          Chunk 312: "TIPP skills are used in crisis..."         │
│          Chunk 189: "Temperature technique activates..."        │
│                                                                  │
│ STEP 3: Model sees:                                             │
│ ┌──────────────────────────────────────────────────────────┐   │
│ │ System: "You are a DBT therapist..."                     │   │
│ │                                                          │   │
│ │ Retrieved Knowledge Base (DYNAMIC - changes per query): │   │
│ │ ┌────────────────────────────────────────────────────┐  │   │
│ │ │ [DBT Manual - Chunk 245]                           │  │   │
│ │ │ TIPP stands for Temperature, Intense exercise,     │  │   │
│ │ │ Paced breathing, and Paired muscle relaxation...   │  │   │
│ │ └────────────────────────────────────────────────────┘  │   │
│ │                                                          │   │
│ │ ┌────────────────────────────────────────────────────┐  │   │
│ │ │ [DBT Manual - Chunk 312]                           │  │   │
│ │ │ TIPP skills are used in crisis situations when...  │  │   │
│ │ └────────────────────────────────────────────────────┘  │   │
│ │                                                          │   │
│ │ ┌────────────────────────────────────────────────────┐  │   │
│ │ │ [DBT Manual - Chunk 189]                           │  │   │
│ │ │ Temperature: Splash cold water on face activates   │  │   │
│ │ │ dive reflex and calms nervous system quickly...    │  │   │
│ │ └────────────────────────────────────────────────────┘  │   │
│ └──────────────────────────────────────────────────────────┘   │
│                                 └─────────↓──────────┘          │
│                          Model generates response                │
│                          (using training + retrieved knowledge)  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

KEY DIFFERENCES:
─────────────────────────────────────────────────────────────────
Before RAG:   Model answers from memory (training data)
With RAG:     Model answers from exact DBT manual (retrieved)

Before RAG:   One-size-fits-all prompt
With RAG:     Custom context per question

Before RAG:   Can't cite sources
With RAG:     Can cite specific manual sections

Before RAG:   Limited to ~4K tokens of knowledge
With RAG:     Full 691-chunk DBT manual available
""")

def show_retrieval_example():
    """Show what happens when retrieving."""
    print("\n\n" + "=" * 80)
    print("EXAMPLE: WHAT HAPPENS WHEN YOU ASK 'TEACH ME TIPP'")
    print("=" * 80)
    
    ds = load_dataset("vjain/dbt")
    
    # Find TIPP-related chunks
    tipp_chunks = ds['train'].filter(
        lambda x: 'tipp' in x['page_content'].lower() or 
                  'temperature' in x['page_content'].lower()[:200]
    )
    
    print(f"\n🔍 Searching for 'TIPP skills'...")
    print(f"✅ Found {len(tipp_chunks)} relevant chunks in DBT manual")
    
    if len(tipp_chunks) > 0:
        print("\n📚 Top retrieved chunks that will be sent to model:")
        
        for i, chunk in enumerate(tipp_chunks[:3], 1):
            print(f"\n{'='*80}")
            print(f"RETRIEVED CHUNK {i}")
            print(f"{'='*80}")
            print(f"ID: {chunk['id']}")
            print(f"Source: {chunk['source']}")
            print(f"\nContent:")
            print(chunk['page_content'][:600])
            if len(chunk['page_content']) > 600:
                print("...")
        
        print("\n" + "=" * 80)
        print("💬 MODEL'S ENRICHED PROMPT")
        print("=" * 80)
        print("""
System: You are a DBT therapist. Provide empathetic, evidence-based
responses using DBT techniques.

──────────────────────────────────────────────────────────────────
Retrieved DBT Manual Knowledge:

[Chunk 1]
""" + tipp_chunks[0]['page_content'][:400] + """...

[Chunk 2]
""" + (tipp_chunks[1]['page_content'][:400] if len(tipp_chunks) > 1 else "") + """...

[Chunk 3]
""" + (tipp_chunks[2]['page_content'][:400] if len(tipp_chunks) > 2 else "") + """...

──────────────────────────────────────────────────────────────────
User: "Can you teach me TIPP skills?"

──────────────────────────────────────────────────────────────────
Model generates response using:
  1. Your trained adapter (therapist conversation skills)
  2. Retrieved DBT manual content (exact TIPP information)
  3. Context-aware combination

""")

def main():
    """Show complete flow."""
    print("\n🎓 UNDERSTANDING YOUR DBT DATASET & RAG FLOW")
    print("=" * 80)
    
    # 1. Show what's in the dataset
    show_dataset_samples()
    
    # 2. Explain RAG flow
    explain_rag_flow()
    
    # 3. Show actual retrieval example
    show_retrieval_example()
    
    print("\n\n" + "=" * 80)
    print("✅ SUMMARY")
    print("=" * 80)
    print("""
Your DBT dataset contains:
  • 691 chunks of DBT manual content
  • Covers all major DBT skills & concepts
  • Average ~1000 characters per chunk

RAG will:
  1. Take each chunk → Convert to embedding (384 numbers)
  2. Store all embeddings in vector database
  3. When user asks question:
     a. Convert question to embedding
     b. Find most similar chunks (cosine similarity)
     c. Retrieve top 3-5 chunks
     d. Send to model along with question
  4. Model generates response using retrieved knowledge

Result: Model has access to entire DBT manual, not just prompts!
    """)

if __name__ == "__main__":
    main()

