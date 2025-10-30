"""
Simple test script for RAG retrieval.
Just run this and enter your queries!
"""

import sys
from hybrid_rag_retrieval import HybridRAGRetrieval
from build_rag_prompt import build_rag_prompt

def main():
    print("=" * 80)
    print("RAG RETRIEVAL TESTER")
    print("=" * 80)
    print("Enter queries to test. Type 'quit' to exit.")
    print("-" * 80)
    
    # Initialize retriever
    print("\nInitializing...")
    retriever = HybridRAGRetrieval(enable_reranking=True)
    print("Ready!\n")
    
    while True:
        try:
            # Get query
            query = input("\n💬 Your query: ").strip()
            
            if not query:
                continue
            
            if query.lower() == 'quit':
                print("\n👋 Goodbye!")
                break
            
            # Retrieve
            print("\n🔍 Retrieving...")
            results, metadata = retriever.retrieve(query, k=5)
            
            # Show path used
            path_emoji = "⚡" if metadata["path"] == "fast" else "🔍" if metadata["path"] == "detailed" else "🚫"
            print(f"\n{path_emoji} Used: {metadata['path']} path")
            print(f"   Reason: {metadata['reason']}")
            if metadata.get("detected_pillar"):
                pillar_icons = {
                    "mindfulness": "🧠",
                    "distress_tolerance": "🛡️",
                    "emotion_regulation": "💭",
                    "interpersonal_effectiveness": "🤝",
                    "general": "📚"
                }
                pillar = metadata["detected_pillar"]
                conf = metadata.get("pillar_confidence", 0)
                icon = pillar_icons.get(pillar, "📚")
                print(f"   🎯 Detected Pillar: {icon} {pillar.replace('_', ' ').title()} (confidence: {conf:.1f})")
            print(f"   Time: {metadata['time_ms']:.1f}ms")
            
            # Handle no-RAG case
            if metadata["path"] == "none":
                print(f"\n✅ No RAG context needed - use model's training only")
                formatted = []
            else:
                print(f"\n📋 Results:")
                # Show results (compact)
                formatted = retriever.format_results(results, metadata)
                for r in formatted:
                    technique_mark = "⭐ TECHNIQUE" if r["is_technique"] else ""
                    pillar = r.get("pillar", "general")
                    pillar_icons = {
                        "mindfulness": "🧠",
                        "distress_tolerance": "🛡️",
                        "emotion_regulation": "💭",
                        "interpersonal_effectiveness": "🤝",
                        "general": "📚"
                    }
                    pillar_icon = pillar_icons.get(pillar, "📚")
                    print(f"\n   [{r['rank']}] {technique_mark} {pillar_icon} {pillar.replace('_', ' ').title()}")
                    print(f"   Section: {r['section'][:70]}")
                    if r['techniques']:
                        print(f"   Topics: {r['techniques'][:80]}")
            
            # Build and show prompts (FULL, no truncation)
            print(f"\n{'='*80}")
            print("📝 FULL GENERATED PROMPTS (what gets sent to LLM):")
            print(f"{'-'*80}")
            
            system_msg, user_msg = build_rag_prompt(query, formatted)
            
            print(f"\n🤖 SYSTEM PROMPT ({len(system_msg)} chars):")
            print(system_msg)
            
            print(f"\n💬 USER PROMPT ({len(user_msg)} chars):")
            print(user_msg)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Try again or type 'quit' to exit")

if __name__ == "__main__":
    main()

