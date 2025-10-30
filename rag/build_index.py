import json
import os
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class ChunkingConfig:
    target_chunk_tokens: int = 512  # Target average
    overlap_tokens: int = 100
    min_chunk_tokens: int = 200  # More reasonable minimum (allows 1-2 sentences)
    max_chunk_tokens: int = 800  # Strict maximum


def normalize_text(text: str) -> str:
    # Normalize newlines and whitespace but PRESERVE line breaks for heading detection
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    cleaned_lines: List[str] = []
    for line in lines:
        line = line.replace("\t", " ")
        line = line.replace("\u00A0", " ")  # non-breaking space
        # Collapse internal spaces but keep the line
        line = re.sub(r"\s+", " ", line).strip()
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


HEADING_PATTERNS = [
    r"^\s*[A-Z][A-Za-z ]+(Skills|Overview|Introduction|General|Mindfulness|Emotion|Interpersonal|Distress|Tolerance|Worksheets?)\b.*$",
    r"^\s*[A-Z][A-Za-z ]+\s*$",
]

FILTER_PATTERNS = [
    r"^\s*\d+\s*$",  # bare page numbers
    r"permission to photocopy",
    r"copyright",
    r"www\.guilford\.com",
    r"^\s*[/•\-]{3,}\s*$",
    r"worksheet\s*\d+",
    r"^about the author$",
    r"^table of contents$",
]

# DBT Technique patterns
TECHNIQUE_PATTERNS = [
    r"\bTIPP?\s*(skills?|technique|method)",
    r"Temperature.*Intense.*Paced.*Paired",
    r"PLEASE\s*(skills?|technique|method)",
    r"PhysicaL.*Eating.*Avoid.*Sleep.*Exercise",
    r"DEAR\s*MAN",
    r"Describe.*Express.*Assert.*Reinforce",
    r"GIVE\s*(skills?|technique)",
    r"Gentle.*Interested.*Validate.*Easy",
    r"FAST\s*(skills?|technique)",
    r"Fair.*Apologies.*Stick.*Truthful",
    r"ACCEPTS?\s*(distraction|technique)",
    r"Activities.*Contributing.*Comparisons",
    r"Self.Soothing",
    r"Wise\s*Mind",
    r"Opposite\s*Action",
    r"Check\s*the\s*Facts",
    r"Mindfulness.*skills?",
    r"Distress\s*Tolerance",
    r"Emotion\s*Regulation",
    r"Interpersonal\s*Effectiveness",
    r".*Handout\s*\d+.*:",  # Handout sections
    r"Guidelines for.*Effectiveness",
    r"How to.*practice.*skill",
    r"step by step",
]

# Content quality indicators
WORKSHEET_PATTERNS = [
    r"worksheet\s*\d+",
    r"fill out",
    r"track your practice",
    r"practice log",
    r"___+\s*\(1\)",  # fill-in-the-blank pattern
    r"___+\s*\(2\)",
]

SECTION_QUALITY_KEYWORDS = {
    "high": ["skills", "technique", "method", "practice", "how to", "step by step"],
    "medium": ["handout", "guideline", "principle", "introduction", "overview"],
    "low": ["worksheet", "track", "record", "fill out", "log"]
}

# DBT 4 Pillars
DBT_PILLARS = {
    "mindfulness": [
        r"mindfulness\s*(skills?|practice|module|handout)",
        r"wise mind",
        r"core mindfulness",
        r"observe.*describe.*participate",
        r"judgment.*nonjudgment"
    ],
    "distress_tolerance": [
        r"distress tolerance\s*(skills?|module|handout)",
        r"\bTIPP?\b",
        r"ACCEPTS?",
        r"self[- ]soothing",
        r"radical acceptance",
        r"crisis survival"
    ],
    "emotion_regulation": [
        r"emotion regulation\s*(skills?|module|handout)",
        r"\bPLEASE\b",
        r"opposite action",
        r"check the facts",
        r"understand.*name.*emotions"
    ],
    "interpersonal_effectiveness": [
        r"interpersonal\s*(skills?|effectiveness|module|handout)",
        r"\bDEAR\s*MAN\b",
        r"\bGIVE\b",
        r"\bFAST\b",
        r"relationship",
        r"getting what.*want"
    ]
}

def is_heading(line: str) -> bool:
    for pat in HEADING_PATTERNS:
        if re.search(pat, line.strip(), flags=re.IGNORECASE):
            return True
    return False

def should_filter(line: str) -> bool:
    l = line.strip().lower()
    for pat in FILTER_PATTERNS:
        if re.search(pat, l):
            return True
    return False

def detect_techniques_in_text(text: str) -> List[str]:
    """Detect which DBT techniques are mentioned in the text."""
    detected = []
    text_lower = text.lower()
    for pat in TECHNIQUE_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE):
            # Extract the technique name
            match = re.search(pat, text, flags=re.IGNORECASE)
            if match:
                detected.append(match.group(0))
    return list(set(detected))  # Remove duplicates

def classify_content_quality(text: str) -> str:
    """Classify if content is a technique section, handout, or worksheet."""
    text_lower = text.lower()
    
    # Check for worksheet patterns
    for pat in WORKSHEET_PATTERNS:
        if re.search(pat, text_lower):
            return "worksheet"
    
    # Check for technique patterns
    if detect_techniques_in_text(text):
        return "technique"
    
    # Check quality keywords
    high_count = sum(1 for kw in SECTION_QUALITY_KEYWORDS["high"] if kw in text_lower)
    low_count = sum(1 for kw in SECTION_QUALITY_KEYWORDS["low"] if kw in text_lower)
    
    if high_count > 0 and low_count == 0:
        return "technique"
    elif high_count > low_count:
        return "handout"
    else:
        return "other"

def is_technique_section(text: str) -> bool:
    """Determine if this section contains actual DBT techniques."""
    quality = classify_content_quality(text)
    return quality in ["technique", "handout"]

def detect_dbt_pillar(text: str) -> str:
    """
    Detect which DBT pillar this text belongs to.
    Returns one of: 'mindfulness', 'distress_tolerance', 'emotion_regulation', 
    'interpersonal_effectiveness', or 'general'
    """
    text_lower = text.lower()
    
    # Count matches for each pillar
    pillar_scores = {}
    for pillar, patterns in DBT_PILLARS.items():
        matches = sum(1 for pat in patterns if re.search(pat, text_lower))
        if matches > 0:
            pillar_scores[pillar] = matches
    
    # Return pillar with most matches
    if pillar_scores:
        return max(pillar_scores.items(), key=lambda x: x[1])[0]
    
    return "general"

def split_by_headings(clean_text: str) -> List[Tuple[str, str]]:
    lines = clean_text.split("\n")
    sections: List[Tuple[str, List[str]]] = []
    current_title = "Preamble"
    current_buf: List[str] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if should_filter(line):
            continue
        if is_heading(line):
            # start new section
            if current_buf:
                sections.append((current_title, current_buf))
                current_buf = []
            current_title = line
        else:
            current_buf.append(line)
    if current_buf:
        sections.append((current_title, current_buf))
    # join buffers
    return [(title, "\n".join(buf)) for title, buf in sections]


def chunk_text_by_tokens(
    text: str,
    tokenizer,
    cfg: ChunkingConfig,
) -> List[str]:
    """
    Chunk text by tokens with strict bounds enforcement.
    Ensures all chunks are between min_chunk_tokens and max_chunk_tokens.
    """
    # Tokenize whole text
    ids: List[int] = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]
    if not ids:
        return []

    # Use target as base
    chunk_tokens = cfg.target_chunk_tokens
    overlap = cfg.overlap_tokens

    chunks: List[str] = []
    start = 0
    
    while start < len(ids):
        # Target end position
        target_end = min(len(ids), start + chunk_tokens)
        
        # Fetch chunk
        chunk_ids = ids[start:target_end]
        # Decode back to text
        tokens = tokenizer.convert_ids_to_tokens(chunk_ids)
        chunk_text = tokenizer.convert_tokens_to_string(tokens)
        chunk_text = chunk_text.strip()
        
        # Only add if it meets minimum requirements
        token_count = len(chunk_ids)
        if chunk_text and token_count >= cfg.min_chunk_tokens:
            chunks.append(chunk_text)
        
        # Move to next position
        if target_end >= len(ids):
            break
        
        # Calculate next start with overlap
        start = max(start + 1, target_end - overlap)  # Ensure we make progress

    return chunks


def build_faiss_index(
    chunks: List[str],
    embedding_model_name: str = "intfloat/e5-small-v2",
) -> Tuple["faiss.Index", List[Dict]]:  # type: ignore[name-defined]
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import faiss

    model = SentenceTransformer(embedding_model_name)
    # E5 expects "passage: " prefix for corpus entries
    corpus_inputs = [f"passage: {c}" for c in chunks]
    embs = model.encode(corpus_inputs, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    embs = np.asarray(embs, dtype="float32")

    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine via normalized vectors + inner product
    index.add(embs)

    # Rich metadata per chunk with technique detection
    metadata: List[Dict] = []
    for i, chunk in enumerate(chunks):
        # Detect techniques
        techniques = detect_techniques_in_text(chunk)
        quality = classify_content_quality(chunk)
        is_technique = is_technique_section(chunk)
        pillar = detect_dbt_pillar(chunk)
        
        metadata.append({
            "source": "dbt_manual",
            "title": "DBT Manual",
            "section": f"chunk_{i}",
            "pillar": pillar,  # mindfulness, distress_tolerance, emotion_regulation, interpersonal_effectiveness, general
            "quality": quality,  # technique, handout, worksheet, other
            "is_technique": is_technique,  # True/False
            "techniques": ",".join(techniques) if techniques else "",  # comma-separated list
            "technique_count": len(techniques),
            "boost": "high" if is_technique else "normal"  # For retrieval boosting
        })

    return index, metadata


def save_outputs(index, chunks: List[str], metadata: List[Dict], out_dir: str) -> None:
    import faiss

    os.makedirs(out_dir, exist_ok=True)
    faiss_path = os.path.join(out_dir, "index.faiss")
    store_path = os.path.join(out_dir, "docstore.json")

    faiss.write_index(index, faiss_path)
    with open(store_path, "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks, "meta": metadata}, f, ensure_ascii=False)


def main():
    from transformers import AutoTokenizer

    # Inputs/outputs
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    manual_path = os.path.join(project_root, "data", "dbt_manual.txt")
    out_dir = os.path.join(project_root, "rag")

    if not os.path.exists(manual_path):
        raise FileNotFoundError(f"DBT manual not found at: {manual_path}")

    print(f"📖 Reading manual: {manual_path}")
    with open(manual_path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    text = normalize_text(raw)
    # Show a preview of cleaned text
    print("🧽 Cleaned text preview:\n" + text[:500] + ("..." if len(text) > 500 else ""))

    print("🔪 Chunking by tokens with overlap...")
    cfg = ChunkingConfig()
    # Use a tokenizer for token counting only (set huge max len to avoid warnings)
    tok = AutoTokenizer.from_pretrained("intfloat/e5-small-v2")
    # Prevent HF warning about 512-token limit; we are not forwarding to this model
    tok.model_max_length = int(1e9)
    # Section-aware split, then chunk per section, store section titles for metadata later
    sections = split_by_headings(text)
    all_chunks: List[str] = []
    section_titles: List[str] = []
    for title, body in sections:
        cs = chunk_text_by_tokens(body, tok, cfg)
        all_chunks.extend(cs)
        section_titles.extend([title] * len(cs))

    print(f"✅ Created {len(all_chunks)} chunks (target={cfg.target_chunk_tokens}, overlap={cfg.overlap_tokens})")
    if len(all_chunks) == 0:
        raise RuntimeError("No chunks produced from the input text.")

    # Show a few chunk samples with lengths
    preview_n = min(3, len(all_chunks))
    for i in range(preview_n):
        print(f"\n🔎 Chunk {i} (len={len(tok(all_chunks[i], add_special_tokens=False)['input_ids'])} tokens):\n" + all_chunks[i][:400] + ("..." if len(all_chunks[i]) > 400 else ""))

    print("🧠 Building embeddings and FAISS index...")
    index, metadata = build_faiss_index(all_chunks)
    # Overwrite section names in metadata with detected titles when available
    for i in range(min(len(metadata), len(section_titles))):
        metadata[i]["section"] = section_titles[i]

    # Print technique detection stats
    technique_chunks = sum(1 for m in metadata if m.get("is_technique", False))
    worksheet_chunks = sum(1 for m in metadata if m.get("quality") == "worksheet")
    high_quality_chunks = sum(1 for m in metadata if m.get("quality") in ["technique", "handout"])
    
    print(f"\n📊 Content Quality Analysis:")
    print(f"   Technique/Handout chunks: {high_quality_chunks} ({high_quality_chunks/len(metadata)*100:.1f}%)")
    print(f"   Worksheet chunks: {worksheet_chunks} ({worksheet_chunks/len(metadata)*100:.1f}%)")
    print(f"   Other chunks: {len(metadata) - high_quality_chunks - worksheet_chunks}")
    
    # Print DBT pillar distribution
    pillar_counts = {}
    for m in metadata:
        pillar = m.get("pillar", "general")
        pillar_counts[pillar] = pillar_counts.get(pillar, 0) + 1
    
    print(f"\n🎯 DBT Pillar Distribution:")
    pillar_labels = {
        "mindfulness": "🧠 Mindfulness",
        "distress_tolerance": "🛡️ Distress Tolerance",
        "emotion_regulation": "💭 Emotion Regulation",
        "interpersonal_effectiveness": "🤝 Interpersonal Effectiveness",
        "general": "📚 General"
    }
    for pillar in ["mindfulness", "distress_tolerance", "emotion_regulation", "interpersonal_effectiveness", "general"]:
        count = pillar_counts.get(pillar, 0)
        pct = count / len(metadata) * 100 if metadata else 0
        label = pillar_labels.get(pillar, pillar)
        print(f"   {label}: {count} ({pct:.1f}%)")
    
    # Show sample technique chunks
    technique_examples = [m for m in metadata if m.get("is_technique", False)][:5]
    if technique_examples:
        print(f"\n🔍 Sample technique chunks detected:")
        for ex in technique_examples:
            idx = metadata.index(ex)
            techniques = ex.get("techniques", "")
            section = ex.get("section", "")
            print(f"   Chunk {idx}: {section[:80]}...")
            if techniques:
                print(f"      Techniques: {techniques[:100]}")

    print(f"\n💾 Saving FAISS and docstore to: {out_dir}")
    save_outputs(index, all_chunks, metadata, out_dir)
    print("🎉 Done. Files written:")
    print(f" - {os.path.join(out_dir, 'index.faiss')}")
    print(f" - {os.path.join(out_dir, 'docstore.json')}")


if __name__ == "__main__":
    main()


