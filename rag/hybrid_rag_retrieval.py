"""
Hybrid RAG Retrieval with Intent Detection
Routes queries to fast or detailed retrieval based on clarity/intent.
"""

import json
import numpy as np
import time
import re
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss


class IntentClassifier:
    """Classify query intent to route to appropriate retrieval path."""
    
    # Clear intent patterns - use fast path
    CLEAR_INTENT_PATTERNS = {
        'specific_skill': [
            r'\btipp\b', r'\bplease\b', r'\bdear man\b', r'\bgive\b', r'\bfast\b',
            r'\baccepts\b', r'\bself[- ]soothing\b', r'\bwise mind\b',
            r'\bopposite action\b', r'\bcheck the facts\b'
        ],
        'short_query': lambda text: len(text.split()) < 15,
        'definition_request': [
            r'what is', r'what are', r'tell me about', r'explain',
            r'how do i practice', r'can you teach me'
        ],
        'specific_technique': [
            r'temperature', r'paced breathing', r'intense exercise',
            r'paired muscle relaxation'
        ]
    }
    
    # Ambiguous intent patterns - use detailed path
    AMBIGUOUS_INTENT_PATTERNS = {
        'multiple_emotions': [
            r'\b(and|but also|plus)\b.*\b(feel|anxious|overwhelmed|angry|sad|depressed)\b'
        ],
        'why_questions': [
            r'why do i', r'why am i', r'why does'
        ],
        'complex_choice': [
            r'should i.*or', r'whether.*or', r'choose between'
        ],
        'long_query': lambda text: len(text.split()) > 30,
        'multiple_issues': [
            r'\band\b.*\band\b',  # Multiple "and"s
        ]
    }
    
    # Patterns indicating no RAG needed (positive, social, casual)
    NO_RAG_PATTERNS = {
        'positive_update': [
            r'doing well', r'doing great', r'feeling good', r'feeling better',
            r'much better', r'good day', r'great day'
        ],
        'social_greeting': [
            r'^hi\b', r'^hey\b', r'^hello\b', r'how are you'
        ],
        'thanks': [
            r'thank you', r'thanks', r'appreciate'
        ]
    }
    
    # Query patterns for DBT pillars
    PILLAR_QUERY_PATTERNS = {
        "mindfulness": [
            r'\b(mindful|mindfulness|wise mind|meditation|meditate|present moment|awareness)',
            r'\b(observe|describe|participate|nonjudgmental)',
            r'focus.*breath', r'present.*moment'
        ],
        "distress_tolerance": [
            r'\b(distress|crisis|overwhelmed|intense|urgent|emergency)',
            r'\b(TIPP|ACCEPTS|self[- ]soothing|radical acceptance)',
            r'cant.*stand', r'unbearable'
        ],
        "emotion_regulation": [
            r'\b(emotion|feeling|feel|angry|sad|anxious|afraid|disgusted|envious)',
            r'\b(regulate|control|manage.*emotion|PLEASE|opposite action)',
            r'feeling.*way', r'emotional'
        ],
        "interpersonal_effectiveness": [
            r'\b(relationship|friend|partner|someone|person|people|they|them)',
            r'\b(DEAR MAN|GIVE|FAST|conflict|argument|boundary)',
            r'ask.*someone', r'say.*no', r'communicat'
        ]
    }
    
    @classmethod
    def detect_pillar_intent(cls, query: str) -> Tuple[str, float]:
        """
        Detect which DBT pillar the query is most related to.
        
        Returns:
            Tuple of (pillar_name, confidence_score)
            pillar_name: 'mindfulness', 'distress_tolerance', 'emotion_regulation', 
                        'interpersonal_effectiveness', or 'general'
            confidence_score: 0.0 to 1.0
        """
        query_lower = query.lower()
        pillar_scores = {}
        
        for pillar, patterns in cls.PILLAR_QUERY_PATTERNS.items():
            matches = sum(1 for pat in patterns if re.search(pat, query_lower))
            if matches > 0:
                pillar_scores[pillar] = matches
        
        if not pillar_scores:
            return "general", 0.0
        
        # Find pillar with most matches
        best_pillar = max(pillar_scores.items(), key=lambda x: x[1])[0]
        max_matches = pillar_scores[best_pillar]
        
        # Confidence based on match count (0.5 = 1 match, 1.0 = 2+ matches)
        confidence = min(0.5 + (max_matches - 1) * 0.5, 1.0)
        
        return best_pillar, confidence
    
    @classmethod
    def needs_rag(cls, query: str) -> Tuple[bool, str]:
        """
        Determine if query needs RAG at all.
        
        Returns:
            Tuple of (needs_rag: bool, reason: str)
        """
        query_lower = query.lower()
        
        # Check for no-RAG patterns
        for pattern_name, patterns in cls.NO_RAG_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, query_lower):
                    return False, f"Casual/Positive query - no RAG needed"
        
        return True, "RAG may be helpful"
    
    @classmethod
    def should_use_reranking(cls, query: str) -> Tuple[bool, str]:
        """
        Determine if query needs detailed reranking.
        
        Returns:
            Tuple of (use_reranking: bool, reason: str)
        """
        query_lower = query.lower()
        word_count = len(query.split())
        
        # FIRST: Check for ambiguous/complex patterns (these override everything)
        for pattern_name, pattern in cls.AMBIGUOUS_INTENT_PATTERNS.items():
            if pattern_name == 'long_query':
                if pattern(query):
                    return True, f"Long query ({word_count} words) - detailed path"
            elif isinstance(pattern, list):
                for pat in pattern:
                    if re.search(pat, query_lower):
                        return True, f"Complex pattern ({pattern_name}) - detailed path"
        
        # SECOND: Check for clear/specific patterns (fast path)
        for pattern_name, pattern in cls.CLEAR_INTENT_PATTERNS.items():
            if pattern_name == 'short_query':
                if pattern(query):
                    return False, f"Short query ({word_count} words) - fast path"
            elif isinstance(pattern, list):
                for pat in pattern:
                    if re.search(pat, query_lower):
                        return False, f"Clear pattern ({pattern_name}) - fast path"
        
        # LAST: Check word count as tiebreaker
        if word_count > 25:
            return True, f"Long query ({word_count} words) - detailed path"
        elif word_count < 10:
            return False, f"Short query ({word_count} words) - fast path"
        
        # Default: use fast path (most queries are clear enough)
        return False, "Default - fast path"


class HybridRAGRetrieval:
    """
    Hybrid RAG retrieval system with intent-based routing.
    """
    
    def __init__(
        self,
        index_path: str = "rag/index.faiss",
        docstore_path: str = "rag/docstore.json",
        embedding_model: str = "intfloat/e5-small-v2",
        enable_reranking: bool = True
    ):
        """Initialize hybrid retrieval system."""
        print("🚀 Initializing Hybrid RAG Retrieval...")
        
        # Load FAISS index and docstore
        self.index = faiss.read_index(index_path)
        with open(docstore_path, "r", encoding="utf-8") as f:
            store = json.load(f)
        self.chunks = store["chunks"]
        self.meta = store["meta"]
        
        # Load embedding model
        self.embed_model = SentenceTransformer(embedding_model)
        
        # Load cross-encoder if reranking enabled
        self.enable_reranking = enable_reranking
        if enable_reranking:
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        else:
            self.reranker = None
        
        # Intent classifier
        self.intent_classifier = IntentClassifier
        
        print(f"✅ Loaded {len(self.chunks)} chunks")
        print(f"   Reranking: {'Enabled' if enable_reranking else 'Disabled'}")
    
    def apply_technique_boost(self, candidates: List[Tuple[int, float]], boost_factor: float = 1.5, pillar_boost_factor: float = 1.3, target_pillar: str = None) -> List[Tuple[int, float]]:
        """
        Apply boosting to technique chunks and pillar-matched chunks.
        
        Args:
            candidates: List of (chunk_idx, score) tuples
            boost_factor: Multiplier for technique chunks
            pillar_boost_factor: Multiplier for chunks matching target pillar
            target_pillar: Pillar to boost (None to disable pillar boosting)
        
        Returns:
            List of (chunk_idx, boosted_score) tuples
        """
        boosted = []
        for idx, score in candidates:
            chunk_pillar = self.meta[idx].get("pillar", "general")
            is_technique = self.meta[idx].get("is_technique", False)
            
            boosted_score = score
            
            # Apply pillar boost if target pillar specified
            if target_pillar and chunk_pillar == target_pillar:
                boosted_score *= pillar_boost_factor
            
            # Apply technique boost
            if is_technique:
                boosted_score *= boost_factor
            
            boosted.append((idx, boosted_score))
        return boosted
    
    def fast_search(self, query: str, k: int = 5, retrieve_more: int = 3, target_pillar: str = None) -> List[Tuple[int, float]]:
        """
        Fast retrieval: FAISS + technique + pillar boosting.
        
        Returns:
            List of (chunk_idx, final_score) tuples
        """
        # Embed query
        qv = self.embed_model.encode([f"query: {query}"], normalize_embeddings=True)
        
        # FAISS search
        D, I = self.index.search(np.asarray(qv, dtype="float32"), k * retrieve_more)
        
        # Get candidates
        candidates = [(int(idx), float(score)) for score, idx in zip(D[0], I[0])]
        
        # Apply technique boost + pillar boost
        boosted = self.apply_technique_boost(candidates, target_pillar=target_pillar)
        
        # Sort by boosted score
        sorted_results = sorted(boosted, key=lambda x: x[1], reverse=True)
        
        # Deduplicate by section (take best from each unique section)
        deduplicated = self._deduplicate_by_section(sorted_results, k)
        
        return deduplicated
    
    def _deduplicate_by_section(self, results: List[Tuple[int, float]], k: int) -> List[Tuple[int, float]]:
        """
        Remove duplicates by section, keeping highest-scoring chunk from each.
        
        Args:
            results: List of (chunk_idx, score) tuples sorted by score
            k: Max number of results to return
        
        Returns:
            Deduplicated list
        """
        seen_sections = set()
        deduplicated = []
        
        for idx, score in results:
            section = self.meta[idx].get('section', '')
            
            # Take first occurrence of each section (highest score since sorted)
            if section not in seen_sections:
                seen_sections.add(section)
                deduplicated.append((idx, score))
                
                # Stop when we have enough unique sections
                if len(deduplicated) >= k:
                    break
        
        return deduplicated
    
    def detailed_search(self, query: str, k: int = 5, ann_k: int = 30, target_pillar: str = None) -> List[Tuple[int, float]]:
        """
        Detailed retrieval: FAISS → Cross-encoder → technique + pillar boost.
        
        Returns:
            List of (chunk_idx, final_score) tuples
        """
        if not self.reranker:
            # Fallback to fast if reranker not available
            return self.fast_search(query, k, target_pillar=target_pillar)
        
        # Embed query
        qv = self.embed_model.encode([f"query: {query}"], normalize_embeddings=True)
        
        # FAISS search (get more candidates)
        D, I = self.index.search(np.asarray(qv, dtype="float32"), ann_k)
        candidates = [(int(idx), float(score)) for score, idx in zip(D[0], I[0])]
        
        # Cross-encoder reranking
        pair_inputs = [(query, self.chunks[idx]) for idx, _ in candidates]
        rerank_scores = self.reranker.predict(pair_inputs)
        
        # Combine original scores with rerank scores
        # Weight: 30% original FAISS, 70% cross-encoder
        combined = [
            (idx, 0.3 * orig_score + 0.7 * rerank_score)
            for (idx, orig_score), rerank_score in zip(candidates, rerank_scores)
        ]
        
        # Apply technique boost + pillar boost
        boosted = self.apply_technique_boost(combined, target_pillar=target_pillar)
        
        # Sort by combined score
        sorted_results = sorted(boosted, key=lambda x: x[1], reverse=True)
        
        # Deduplicate by section
        deduplicated = self._deduplicate_by_section(sorted_results, k)
        
        return deduplicated
    
    def retrieve(self, query: str, k: int = 5) -> Tuple[List[Tuple[int, float]], Dict]:
        """
        Hybrid retrieval: route based on intent.
        
        Args:
            query: User query
            k: Number of results to return
        
        Returns:
            Tuple of (results, metadata)
            results: List of (chunk_idx, score) tuples
            metadata: Dict with path used, timing, etc.
        """
        start_time = time.time()
        
        # FIRST: Check if RAG is needed at all
        needs_rag_flag, rag_reason = self.intent_classifier.needs_rag(query)
        
        if not needs_rag_flag:
            # Return empty results - no RAG needed
            elapsed = (time.time() - start_time) * 1000
            metadata = {
                "path": "none",
                "reason": rag_reason,
                "time_ms": elapsed,
                "results_count": 0
            }
            return [], metadata
        
        # SECOND: Detect pillar intent
        target_pillar, pillar_confidence = self.intent_classifier.detect_pillar_intent(query)
        
        # THIRD: Classify intent for retrieval path
        use_reranking, reason = self.intent_classifier.should_use_reranking(query)
        
        # Route to appropriate retrieval method with pillar boost
        if use_reranking:
            results = self.detailed_search(query, k, target_pillar=target_pillar)
            path = "detailed"
        else:
            results = self.fast_search(query, k, target_pillar=target_pillar)
            path = "fast"
        
        # Timing
        elapsed = (time.time() - start_time) * 1000  # ms
        
        metadata = {
            "path": path,
            "reason": reason,
            "detected_pillar": target_pillar,
            "pillar_confidence": pillar_confidence,
            "time_ms": elapsed,
            "results_count": len(results)
        }
        
        return results, metadata
    
    def format_results(self, results: List[Tuple[int, float]], metadata: Dict) -> List[Dict]:
        """
        Format retrieval results for display.
        
        Returns:
            List of formatted result dicts
        """
        formatted = []
        for rank, (idx, score) in enumerate(results, 1):
            formatted.append({
                "rank": rank,
                "chunk_idx": idx,
                "score": score,
                "text": self.chunks[idx],
                "section": self.meta[idx].get("section"),
                "pillar": self.meta[idx].get("pillar"),
                "quality": self.meta[idx].get("quality"),
                "techniques": self.meta[idx].get("techniques"),
                "is_technique": self.meta[idx].get("is_technique", False)
            })
        return formatted


def main():
    """Demo hybrid retrieval with intent detection."""
    print("=" * 80)
    print("HYBRID RAG RETRIEVAL DEMO")
    print("=" * 80)
    
    # Initialize
    retriever = HybridRAGRetrieval(
        enable_reranking=True  # Enable both paths
    )
    
    # Test queries
    test_queries = [
        ("What are TIPP skills?", "Should use fast - specific skill"),
        ("Can you teach me PLEASE?", "Should use fast - specific skill"),
        ("I had a bad day today, I feel like I'm not doing well.", "Should use fast - emotional support"),
        ("I feel overwhelmed and anxious but also frustrated, what should I do?", "Should use detailed - multiple emotions"),
        ("Why do I keep feeling this way even though I know it doesn't make sense?", "Should use detailed - why question"),
        ("Should I try mindfulness or should I talk to someone about this?", "Should use detailed - complex choice"),
        ("What should I do to feel better?", "Should use fast - clear request"),
    ]
    
    print("\n" + "=" * 80)
    print("TESTING INTENT DETECTION & ROUTING")
    print("=" * 80)
    
    for query, expected in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"Expected: {expected}")
        print(f"{'-'*80}")
        
        # Retrieve
        results, metadata = retriever.retrieve(query, k=3)
        
        # Show results
        path_emoji = "⚡" if metadata["path"] == "fast" else "🔍"
        print(f"\n{path_emoji} Path: {metadata['path']} ({metadata['reason']})")
        print(f"⏱️  Time: {metadata['time_ms']:.1f}ms")
        
        # Show top results
        formatted = retriever.format_results(results, metadata)
        for r in formatted[:3]:
            boost_mark = "⭐" if r["is_technique"] else " "
            print(f"\n{boost_mark} Rank {r['rank']}: {r['section'][:60]}")
            if r["techniques"]:
                print(f"   Techniques: {r['techniques'][:80]}")
            print(f"   Text: {r['text'][:200]}...")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Intent-based hybrid routing:
  ⚡ Fast path: Clear, specific queries (20-40ms)
  🔍 Detailed path: Complex, ambiguous queries (2-3s)
  
Technique boosting applies to both paths.
Most queries (90%+) use fast path for speed.
    """)

if __name__ == "__main__":
    main()

