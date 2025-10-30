"""
Visualize RAG Flow: Show exactly how data flows through the RAG system.
"""

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np

def show_dataset_samples():
    """Show actual samples from your DBT dataset."""
    print("=" * 80)
    print("YOUR DBT DATASET SAMPLES")
    print("=" * 80)
    
    # Load dataset
    ds = load_dataset("vjain/dbt")
    train_split = ds['train']
    
    # Show different types of content
    samples_shown = []
    
    print("\n📊 Dataset Overview:")
    print(f"   Total examples: {len(train_split)}")
    print(f"   Columns: {train_split.column_names}")
    
    # Sample 1: Short (likely header)
    print("\n" + "-" * 80)
    print("SAMPLE 1: Short Example (Header/Title)")
    print("-" * 80)
    short_example = train_split.filter(lambda x: len(x['page_content']) < 100)[0]
    print(f"ID: {short_example['id']}")
    print(f"Length: {len(short_example['page_content'])} characters")
    print(f"Source: {short_example['source']}")
    print(f"\nContent:\n{short_example['page_content']}")
    
    # Sample 2: Medium (typical chunk)
    print("\n" + "-" * 80)
    print("SAMPLE 2: Medium Example (Typical Content)")
    print("-" * 80)
    medium_examples = train_split.filter(lambda x: 500 < len(x['page_content']) < 1500)
    if len(medium_examples) > 0:
        medium_example = medium_examples[10]
        print(f"ID: {medium_example['id']}")
        print(f"Length: {len(medium_example['page_content'])} characters")
        print(f"Source: {medium_example['source']}")
        print(f"\nContent:\n{medium_example['page_content'][:800]}...")
    
    # Sample 3: Find DBT skill example
    print("\n" + "-" * 80)
    print("SAMPLE 3: DBT Skill Example (TIPP, PLEASE, etc.)")
    print("-" * 80)
    skill_keywords = ['TIPP', 'PLEASE', 'DEAR MAN', 'mindfulness']
    
    for keyword in skill_keywords:
        skill_examples = train_split.filter(
            lambda x: keyword.lower() in x['page_content'].lower()
        )
        if len(skill_examples) > 0:
            skill_example = skill_examples[0]
            print(f"Found: '{keyword}' example")
            print(f"ID: {skill_example['id']}")
            print(f"Length: {len(skill_example['page_content'])} characters")
            print(f"Source: {skill_example['source']}")
            
            # Show relevant excerpt
            content = skill_example['page_content']
            keyword_idx = content.lower().find(keyword.lower())
            if keyword_idx != -1:
                start = max(0, keyword_idx - 100)
                end = min(len(content), keyword_idx + 400)
                print(f"\nContent excerpt (around '{keyword}'):")
                print(content[start:end])
            else:
                print(f"\nContent:\n{content[:800]}...")
            break
    
    # Sample 4: Long example
    print("\n" + "-" * 80)
    print("SAMPLE 4: Long Example (Detailed Content)")
    print("-" * 80)
    long_example = train_split.filter(lambda x: len(x['page_content']) > 1800)[0]
    print(f"ID: {long_example['id']}")
    print(f"Length: {len(long_example['page_content'])} characters")
    print(f"Source: {long_example['source']}")
    print(f"\nContent:\n{long_example['page_content'][:600]}...")

def demonstrate_embedding():
    """Show how text becomes embeddings."""
    print("\n\n" + "=" * 80)
    print("HOW EMBEDDINGS WORK")
    print("=" * 80)
    
    # Load embedding model
    print("\n📥 Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"✅ Model loaded: {model.get_sentence_embedding_dimension()} dimensions")
    
    # Create sample texts
    sample_texts = [
        "TIPP skills help with crisis situations",
        "PLEASE skills are for emotion regulation",
        "I'm feeling overwhelmed and anxious"
    ]
    
    print("\n" + "-" * 80)
    print("Sample Texts:")
    print("-" * 80)
    for i, text in enumerate(sample_texts, 1):
        print(f"{i}. {text}")
    
    # Create embeddings
    print("\n🔢 Creating embeddings...")
    embeddings = model.encode(sample_texts, show_progress_bar=False)
    
    print(f"\nEmbedding dimensions: {embeddings.shape}")
    print(f"Each text → vector of {embeddings.shape[1]} numbers")
    
    # Show similarity
    print("\n📊 Similarity Matrix:")
    print("-" * 80)
    for i, text1 in enumerate(sample_texts):
        for j, text2 in enumerate(sample_texts):
            if i >= j:
                similarity = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                print(f"  Text {i+1} ↔ Text {j+1}: {similarity:.3f}")

def demonstrate_chunking():
    """Show how text gets chunked."""
    print("\n\n" + "=" * 80)
    print("HOW CHUNKING WORKS")
    print("=" * 80)
    
    # Load a sample
    ds = load_dataset("vjain/dbt")
    sample = ds['train'][50]  # Get a sample
    
    print(f"\nOriginal text length: {len(sample['page_content'])} characters")
    
    # Simulate chunking
    def chunk_text(text, chunk_size, overlap):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append({
                'text': chunk,
                'start': start,
                'end': end,
                'length': len(chunk)
            })
            start += chunk_size - overlap
            if start + overlap >= len(text):
                break
        return chunks
    
    chunks = chunk_text(sample['page_content'], chunk_size=1000, overlap=200)
    
    print(f"\nAfter chunking (size=1000, overlap=200):")
    print(f"   Created {len(chunks)} chunks")
    
    print("\n" + "-" * 80)
    print("Chunk Details:")
    print("-" * 80)
    for i, chunk in enumerate(chunks[:3], 1):  # Show first 3
        print(f"\nChunk {i}:")
        print(f"  Position: chars {chunk['start']}-{chunk['end']}")
        print(f"  Length: {chunk['length']} chars")
        print(f"  Content: {chunk['text'][:150]}...")

def demonstrate_retrieval_flow():
    """Show complete RAG flow."""
    print("\n\n" + "=" * 80)
    print("COMPLETE RAG FLOW")
    print("=" * 80)
    
    print("""
How RAG works when you ask a question:

STEP 1: USER ASKS QUESTION
─────────────────────────────────────────────────────────
User: "Can you teach me TIPP skills?"


STEP 2: CREATE EMBEDDING
─────────────────────────────────────────────────────────
Question → Embedding Model → [384 numbers]
  
  "Can you teach me TIPP skills?"
          ↓
  [0.23, -0.45, 0.12, ..., 0.89]  # 384 dimensions


STEP 3: SEARCH VECTOR DATABASE
─────────────────────────────────────────────────────────
Compare question embedding with all chunk embeddings

  Question: [0.23, -0.45, 0.12, ...]
  Chunk 1:  [0.21, -0.42, 0.15, ...] ← Similarity: 0.89 ✓
  Chunk 2:  [0.98, 0.03, -0.45, ...] ← Similarity: 0.32 ✗
  Chunk 3:  [0.18, -0.48, 0.11, ...] ← Similarity: 0.92 ✓
  Chunk 4:  [0.45, 0.67, -0.23, ...] ← Similarity: 0.25 ✗
  ...

  Returns top 3 most similar chunks


STEP 4: RETRIEVE RELEVANT CONTEXT
─────────────────────────────────────────────────────────
Chunk 1: "TIPP stands for Temperature, Intense exercise,
         Paced breathing, and Paired muscle relaxation..."
         
Chunk 2: "When to use TIPP skills: In crisis situations
         when emotions feel overwhelming..."
         
Chunk 3: "Temperature technique: Splash cold water on your
         face to activate the dive reflex..."


STEP 5: CREATE ENRICHED PROMPT
─────────────────────────────────────────────────────────
System: "You are a DBT therapist..."

Knowledge Base (Retrieved):
  [Chunk 1]
  [Chunk 2]  
  [Chunk 3]

User: "Can you teach me TIPP skills?"


STEP 6: GENERATE RESPONSE
─────────────────────────────────────────────────────────
Your trained adapter + retrieved DBT knowledge → Response

  "I'd be happy to teach you TIPP skills! TIPP stands for
   Temperature, Intense exercise, Paced breathing, and Paired
   muscle relaxation. When your emotions feel overwhelming,
   these skills can help..."
""")

def show_full_example():
    """Show a complete end-to-end example."""
    print("\n\n" + "=" * 80)
    print("FULL END-TO-END EXAMPLE")
    print("=" * 80)
    
    # Load dataset
    ds = load_dataset("vjain/dbt")
    
    # Find a TIPP example
    tipp_examples = ds['train'].filter(
        lambda x: 'tipp' in x['page_content'].lower()
    )
    
    if len(tipp_examples) > 0:
        example = tipp_examples[0]
        
        print("\n📚 Retrieved from DBT Manual:")
        print("-" * 80)
        print(f"ID: {example['id']}")
        print(f"Source: {example['source']}")
        print(f"\n{example['page_content']}")
        
        print("\n" + "-" * 80)
        print("🤔 User asks: 'Can you teach me TIPP skills?'")
        print("-" * 80)
        
        print("\n💬 Model Response (generated by combining):")
        print("-" * 80)
        print("""
I'd be happy to teach you TIPP skills! TIPP is a distress 
tolerance skill that stands for:

🔹 Temperature: Splash cold water on your face (this activates 
   the dive reflex and can quickly calm your nervous system)

🔹 Intense Exercise: Brief, vigorous movement to shift your 
   emotional state

🔹 Paced Breathing: Slow, deep breathing (try inhaling for 4, 
   holding for 4, exhaling for 6)

🔹 Paired Muscle Relaxation: Tense and release muscle groups

These skills work best when you're in a crisis and need 
immediate relief. Would you like to try one of them right now?
        """)
        
        print("\n📝 This response combines:")
        print("  ✓ Your trained adapter (empathy, validation, therapy skills)")
        print("  ✓ Retrieved DBT manual content (exact TIPP information)")
        print("  ✓ Dynamic context (relevant to the question)")

def main():
    """Run all demonstrations."""
    print("\n🎓 RAG FLOW VISUALIZATION")
    print("=" * 80)
    
    # 1. Show dataset samples
    show_dataset_samples()
    
    # 2. Show embeddings
    demonstrate_embedding()
    
    # 3. Show chunking
    demonstrate_chunking()
    
    # 4. Show retrieval flow
    demonstrate_retrieval_flow()
    
    # 5. Show full example
    show_full_example()
    
    print("\n\n" + "=" * 80)
    print("✅ VISUALIZATION COMPLETE")
    print("=" * 80)
    print("\nNow you understand how RAG works!")
    print("The model sees: Your question + Retrieved DBT chunks → Combined response")

if __name__ == "__main__":
    main()

