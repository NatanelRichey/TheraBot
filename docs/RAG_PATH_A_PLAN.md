<!-- 94332114-0d18-48c7-9ac0-94bbce2186b7 55a6f57e-cf6d-4602-9258-83564fe3c011 -->
# Path A: RAG on Free CPU Space (llama.cpp + GGUF)

## Phase 1 — Light notebook updates (train/adapt your model)

- Update your existing Llama 3.1 8B Instruct notebook to run cleanly:
- Ensure correct tokenizer and `pad_token_id`; set EOS/BOS properly.
- Verify context length (e.g., 4096) and enable gradient checkpointing if needed.
- If using LoRA/QLoRA, keep adapters separate during training.
- Save checkpoints in standard HF format (`.safetensors`).
- Optional for deployment via llama.cpp:
- If you used LoRA: merge adapters into the base model to produce merged `.safetensors`.
- Convert HF model → GGUF using `llama.cpp` conversion scripts.
- Quantize to Q4_K_M for CPU-friendly footprint.
- Practical note: Free CPU Spaces have tight RAM; prefer deploying a 3B GGUF for the Space and keep your 8B for local/GPU usage.

## Phase 2 — Prepare RAG data (DBT manual)

- Collect the DBT manual in clean text/markdown.
- Chunking: split into 300–800 token chunks with overlap (e.g., 50–100 tokens).
- Embeddings: pick a small model (e.g., `sentence-transformers/all-MiniLM-L6-v2`).
- Build FAISS index and save alongside a `docstore.json` with chunk texts and optional metadata (title, section, url).
- **Metadata enrichment**: Classify chunks as "technique", "handout", "worksheet", "other" for retrieval boosting.
- **Technique detection**: Tag chunks with detected DBT techniques (TIPP, PLEASE, DEAR MAN, etc.)

## Phase 3 — Space repository layout

- Files to include:
- `requirements.txt` (gradio, faiss-cpu, sentence-transformers, requests, llama-cpp-python)
- `build_index.py` (one-time: build FAISS + docstore from DBT manual)
- `app.py` (Gradio chat UI + retrieval + calls llama.cpp server)
- `download_model.py` (optional: download `.gguf` at startup)
- `models/llama3-3b-instruct.Q4_K_M.gguf` (or download on launch)
- `data/dbt_manual/…` (raw manual) and `index.faiss`, `docstore.json`

## Phase 4 — Start llama.cpp server in the Space

- Launch command (in Space Settings → App → Start command) or from `app.py` before Gradio starts:
- `python -m llama_cpp.server --model models/llama3-3b-instruct.Q4_K_M.gguf --host 0.0.0.0 --port 8000 --n_ctx 4096 --chat_format llama-3`
- Confirm the server is reachable at `/v1/chat/completions`.

## Phase 5 — Wire RAG to llama.cpp (OpenAI-compatible)

### 5.1 — Hybrid RAG Strategy: Fast vs Detailed Retrieval

**Concept**: Route queries based on clarity/intent to optimize speed vs quality trade-off.

#### Fast RAG Path (20–40ms) — Most queries
- Use FAISS search + technique boosting only (no cross-encoder).
- Apply 1.5× score boost to chunks tagged "technique" or "handout".
- Retrieval time: ~20–40ms
- Best for: Clear queries, most chat interactions

#### Detailed RAG Path (2–3s) — Complex queries
- FAISS search → Cross-encoder reranking → technique boosting
- Retrieval time: ~2–3 seconds
- Best for: Ambiguous queries needing maximum accuracy

#### Intent Detection Logic
```python
def should_use_reranking(query: str) -> bool:
    """Decide if cross-encoder reranking is needed."""
    
    # Patterns suggesting simple/clear intent
    clear_patterns = [
        len(query.split()) < 15,  # Short query
        any(skill in query.lower() for skill in ['tipp', 'please', 'dear man']),  # Specific skill
        query.endswith('?') and 'what' in query.lower(),  # Definition request
    ]
    
    # Patterns suggesting complex/ambiguous intent
    ambiguous_patterns = [
        'feel' in query.lower() and 'why' in query.lower(),  # Why do I feel X?
        'should' in query.lower() and 'whether' in query.lower(),  # Should I whether...
        multiple_emotions_in_query,  # Multiple issues
    ]
    
    # Use reranking for ambiguous, not for clear
    return not any(clear_patterns) and any(ambiguous_patterns)
```

#### Implementation
```python
def retrieve_with_hybrid(query: str, k: int = 5):
    """Hybrid retrieval: fast path or detailed path."""
    
    # Decide path
    use_reranking = should_use_reranking(query)
    
    if use_reranking:
        # Detailed path with cross-encoder
        candidates = fast_faiss_search(query, k * 3)
        ranked = cross_encoder_rerank(query, candidates)
        boosted = apply_technique_boost(ranked)
        return boosted[:k]
    else:
        # Fast path with boosting only
        candidates = fast_faiss_search(query, k * 3)
        boosted = apply_technique_boost(candidates)
        return sorted(boosted, reverse=True)[:k]
```

### 5.2 — RAG Integration Flow

On each user message:
1) **Intent detection**: Route to fast or detailed RAG path
2) **Retrieve**: Get top-k chunks with hybrid strategy above
3) **Build prompt**: System message with grounding rules + user message + top-k DBT context
4) **POST**: `/v1/chat/completions` with `stream=false`
5) **Display**: Assistant response in Gradio chat
- Keep `temperature` low (≈0.2–0.4) for factuality

## Phase 6 — Deploy, test, and iterate

### 6.1 — Testing Hybrid RAG Performance

**Test Queries by Type:**
- **Clear queries** (should use fast path):
  - "What are TIPP skills?"
  - "Can you teach me PLEASE?"
  - "How do I practice DEAR MAN?"
  
- **Ambiguous queries** (should use detailed path):
  - "I feel overwhelmed and anxious but also frustrated, what should I do?"
  - "Should I try mindfulness or should I talk to someone about this?"
  - "Why do I keep feeling this way even though I know it doesn't make sense?"

**Success Criteria:**
- Fast path: 90%+ of queries, < 50ms latency
- Detailed path: 10% of queries, < 3s latency
- Technique chunks prioritized in results
- Retrieval quality meets accuracy threshold

### 6.2 — Deployment Testing

- Push repo to the Space; first build runs `pip install` and (optionally) `build_index.py`.
- Test:
- **Latency**: Adjust quantization and `n_ctx` if slow or OOM.
- **Retrieval quality**: Tune chunk size, overlap, and top-k.
- **Intent detection**: Fine-tune thresholds for fast vs detailed routing.
- **Prompt**: Enforce use of context; instruct to say "I don't know" when needed.
- **Citations**: Return snippet titles/sections with the answer.
- **Hybrid routing**: Monitor which path is used and adjust intent patterns.

## Notes on using your 8B model

- If you need to deploy your fine-tuned 8B:
- Merge LoRA → convert to GGUF → heavy quantization (Q4_K_M/ Q3_K) → test memory.
- Likely requires GPU/paid Space; for free CPU, prefer 3B GGUF.

### To-dos

- [ ] Fix tokenizer/pad/EOS, context length, LoRA config in your training notebook
- [ ] Save trained or baseline weights in HF format (.safetensors)
- [ ] If using LoRA, merge adapters; convert to GGUF; quantize Q4_K_M
- [x] Clean and chunk DBT manual into 300–800 token segments
- [x] Compute embeddings and build FAISS; write index.faiss and docstore.json
- [x] Add metadata tags (technique, handout, worksheet, pillar) to chunks
- [x] Implement intent detection logic for hybrid routing
- [x] Build hybrid retrieval with fast/detailed paths
- [x] Implement pillar detection and boosting
- [x] Add deduplication and no-RAG detection
- [x] Create requirements_deploy.txt
- [x] Implement retrieval + prompt + POST logic in app.py
- [ ] Train and convert model to GGUF
- [ ] Run llama.cpp server with GGUF model in Space (port 8000)
- [ ] Push to Space, test latency, retrieval quality, and prompt grounding
- [ ] Show source snippets and enable output streaming in UI
- [ ] Tune intent detection thresholds based on performance


