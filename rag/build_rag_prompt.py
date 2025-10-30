"""
Build prompts from RAG-retrieved DBT chunks.
Integrates retrieval results into a formatted prompt for the LLM.
"""

from typing import List, Dict, Tuple
from hybrid_rag_retrieval import HybridRAGRetrieval


def format_dbt_context(chunks: List[Dict]) -> str:
    """
    Format retrieved DBT chunks into context for the prompt.
    
    Args:
        chunks: List of chunk dicts from format_results()
    
    Returns:
        Formatted context string or None if no chunks
    """
    if not chunks:
        return None
    
    context_parts = []
    
    for i, chunk in enumerate(chunks, 1):
        section = chunk.get('section', 'Unknown Section')
        text = chunk.get('text', '')[:600]  # Limit to 600 chars per chunk
        techniques = chunk.get('techniques', '')
        
        chunk_text = f"[{i}] Section: {section}\n{text}"
        
        if techniques:
            chunk_text += f"\n   Related Techniques: {techniques[:100]}"
        
        context_parts.append(chunk_text)
    
    return "\n\n".join(context_parts)


def build_rag_prompt(
    user_query: str,
    retrieved_chunks: List[Dict],
    system_prompt: str = None,
    max_context_length: int = 1000
) -> Tuple[str, str]:
    """
    Build full prompt with RAG context for LLM.
    
    Args:
        user_query: User's question
        retrieved_chunks: Formatted chunks from retriever
        system_prompt: Optional custom system prompt
        max_context_length: Max chars for context
    
    Returns:
        Tuple of (system_prompt, user_prompt_with_context)
    """
    
    # Default system prompt
    if system_prompt is None:
        system_prompt = """You are a compassionate DBT (Dialectical Behavior Therapy) therapist.

Core DBT Principles:
1. VALIDATION: Always validate emotions first ("It makes sense that...")
2. DIALECTICS: Balance acceptance and change
3. SKILLS: Teach practical coping strategies
4. MINDFULNESS: Incorporate present-moment awareness

You have access to DBT manual excerpts below. Use them to inform your responses when relevant.

CRITICAL GUIDELINES:
- The DBT context below is SUPPLEMENTARY - your training is your primary knowledge base
- Use context ONLY when it directly and clearly relates to the user's question
- DO NOT force-fit unrelated context into your response
- If context seems irrelevant or unclear, IGNORE IT and use your training
- If context would lead to a nonsensical or awkward response, IGNORE IT
- Always prioritize: validation, empathy, and natural conversation flow
- Be concrete and practical in your suggestions
- It's better to give a good response from training than a forced response from context
- If you're unsure about using context, DON'T use it"""

    # Format context from chunks
    dbt_context = format_dbt_context(retrieved_chunks)
    
    # Build user prompt with or without RAG context
    if dbt_context:
        # Truncate if too long
        if len(dbt_context) > max_context_length:
            dbt_context = dbt_context[:max_context_length] + "..."
        
        # With RAG context
        user_prompt = f"""DBT Knowledge Base (SUPPLEMENTARY - Use Only If Relevant):
{dbt_context}

---

User's Question: {user_query}

Respond naturally and compassionately:
1. FIRST: Validate the user's experience
2. THEN: Check if any DBT context above is truly relevant to the question
   - If YES and context is clear: Incorporate it naturally
   - If NO or context is unclear: Ignore context, use your training
3. FINALLY: Offer concrete, practical suggestions
4. REMEMBER: Natural conversation > forced context usage"""
    else:
        # Without RAG context (casual/positive queries)
        user_prompt = f"""User's Question: {user_query}

Provide a warm, compassionate response using your DBT training."""
    
    return system_prompt, user_prompt


def demonstrate_rag_prompt_building():
    """Demo: Show how prompts are built from RAG results."""
    
    print("=" * 80)
    print("RAG PROMPT BUILDING DEMO")
    print("=" * 80)
    
    # Initialize retriever
    print("\nInitializing RAG system...")
    retriever = HybridRAGRetrieval(enable_reranking=False)  # Use fast for demo
    
    # Test queries
    test_queries = [
        "I'm feeling overwhelmed, what should I do?",
        "Can you teach me TIPP skills?",
        "What is mindfulness?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}: {query}")
        print(f"{'-'*80}")
        
        # Retrieve
        print("\n1️⃣ Retrieving DBT context...")
        results, metadata = retriever.retrieve(query, k=3)
        formatted = retriever.format_results(results, metadata)
        
        print(f"   Found {len(formatted)} chunks")
        print(f"   Time: {metadata['time_ms']:.1f}ms")
        
        # Build prompt
        print("\n2️⃣ Building prompt...")
        system_prompt, user_prompt = build_rag_prompt(query, formatted)
        
        # Display
        print(f"\n📋 SYSTEM PROMPT:")
        print(system_prompt[:300] + "..." if len(system_prompt) > 300 else system_prompt)
        
        print(f"\n💬 USER PROMPT (with context):")
        print(user_prompt[:800] + "..." if len(user_prompt) > 800 else user_prompt)
        
        print(f"\n   Context length: {len(user_prompt)} chars")
    
    print("\n" + "=" * 80)
    print("INTEGRATION NOTES")
    print("=" * 80)
    print("""
How to use in your chatbot:

Step 1: Retrieve (RAG)
  results, metadata = retriever.retrieve(user_query, k=5)
  chunks = retriever.format_results(results, metadata)

Step 2: Build Prompt  
  system_msg, user_msg = build_rag_prompt(user_query, chunks)

Step 3: Generate Response
  response = llm.chat(system=system_msg, user=user_msg)

This happens at INFERENCE TIME (not during training).
RAG retrieval → prompt building → LLM generation → user sees response.
    """)


if __name__ == "__main__":
    demonstrate_rag_prompt_building()

