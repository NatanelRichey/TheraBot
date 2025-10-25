"""
Universal extraction script for therapy transcripts
Handles: single sessions, multiple volumes, or series collections
"""
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple

def extract_dialogue_from_html(html_text: str) -> List[Dict[str, str]]:
    """Extract speaker-message pairs from HTML paragraphs"""
    dialogues = []
    
    # Find all paragraphs with dialogue
    pattern = r'<p data-id="[^"]+"><span data-id="[^"]+">([^<]+)</span>'
    matches = re.findall(pattern, html_text)
    
    for text in matches:
        text = text.strip()
        
        # Skip empty lines
        if not text or ':' not in text:
            continue
        
        # Skip metadata lines
        if any(skip in text for skip in [
            'TRANSCRIPT OF AUDIO',
            'BEGIN TRANSCRIPT',
            'END TRANSCRIPT',
            'INTRODUCTION:',
            'Print page',
            '---',
            'Volume no.'
        ]):
            continue
        
        # Extract speaker and message
        match = re.match(r'^([A-Z][A-Z\s\.\-\']+?):\s*(.+)$', text)
        if match:
            speaker = match.group(1).strip()
            message = match.group(2).strip()
            
            # Clean HTML entities
            message = clean_html_entities(message)
            
            # Sanity checks
            if len(speaker) < 50 and len(message) > 0:
                dialogues.append({
                    'speaker': speaker,
                    'message': message
                })
    
    return dialogues

def clean_html_entities(text: str) -> str:
    """Clean HTML entities from text"""
    entities = {
        '&nbsp;': ' ', '&#39;': "'", '&quot;': '"', 
        '&amp;': '&', '&lt;': '<', '&gt;': '>',
        '&ndash;': '–', '&mdash;': '—'
    }
    for entity, char in entities.items():
        text = text.replace(entity, char)
    
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_sessions_boundaries(html_text: str) -> List[Tuple[int, int, str]]:
    """Find session boundaries and titles"""
    lines = html_text.split('\n')
    sessions = []
    
    for i, line in enumerate(lines):
        # Look for END TRANSCRIPT markers
        if 'END TRANSCRIPT' in line:
            # Try to find the previous BEGIN TRANSCRIPT or session start
            start_line = max(0, i - 500)  # Look back up to 500 lines
            
            # Extract title from ucv-section-title
            title = "Unknown Session"
            for j in range(i, max(0, i-50), -1):
                title_match = re.search(r'ucv-section-title[^>]*>(?:<span>)?([^<]+)', lines[j])
                if title_match:
                    title = clean_html_entities(title_match.group(1))
                    if 'Series' in title or 'Session' in title or 'Volume' in title:
                        break
            
            sessions.append((start_line, i, title))
    
    return sessions

def main(input_file: str, output_subdir: str = ""):
    """Extract transcripts from any HTML file"""
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"[ERROR] File not found: {input_file}")
        return
    
    # Determine output directory
    if output_subdir:
        output_dir = Path(f"data/processed/{output_subdir}")
    else:
        # Use filename as subdir
        base_name = input_path.stem  # e.g., "2" from "2.txt"
        output_dir = Path(f"data/processed/file_{base_name}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Processing: {input_path.name}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    # Read file
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        html_text = f.read()
    
    # Check if it's a single session or multiple
    end_transcript_count = html_text.count('END TRANSCRIPT')
    
    if end_transcript_count == 0:
        print("[WARNING] No END TRANSCRIPT markers found. Treating as single session.")
        end_transcript_count = 1
    
    print(f"Detected: {end_transcript_count} session(s)")
    
    if end_transcript_count == 1:
        # Single session
        extract_single_session(html_text, output_dir, input_path.stem)
    else:
        # Multiple sessions - extract each
        extract_multiple_sessions(html_text, output_dir)

def extract_single_session(html_text: str, output_dir: Path, file_id: str):
    """Extract a single session"""
    # Extract title
    title_match = re.search(r'ucv-section-title[^>]*>(?:<span>)?([^<]+)', html_text)
    title = clean_html_entities(title_match.group(1)) if title_match else "Unknown Session"
    
    # Extract dialogues
    dialogues = extract_dialogue_from_html(html_text)
    
    if len(dialogues) == 0:
        print(f"  [WARNING] No dialogues extracted!")
        return
    
    # Create metadata
    metadata = {
        'session_id': file_id,
        'title': title,
        'num_exchanges': len(dialogues),
        'format': 'single_session'
    }
    
    # Save files
    safe_title = re.sub(r'[^\w\s-]', '', title)[:50].strip().replace(' ', '_')
    save_session(output_dir, safe_title, metadata, dialogues)
    
    print(f"  [OK] Extracted session: {title}")
    print(f"       Exchanges: {len(dialogues)}")

def extract_multiple_sessions(html_text: str, output_dir: Path):
    """Extract multiple sessions"""
    sessions = extract_sessions_boundaries(html_text)
    lines = html_text.split('\n')
    
    summary = []
    
    for idx, (start, end, title) in enumerate(sessions, 1):
        print(f"\nSession {idx}: {title[:60]}...")
        
        # Extract session text
        session_text = '\n'.join(lines[start:end+1])
        
        # Extract dialogues
        dialogues = extract_dialogue_from_html(session_text)
        
        if len(dialogues) == 0:
            print(f"  [WARNING] No dialogues extracted")
            continue
        
        # Create metadata
        metadata = {
            'session_id': f"session_{idx:02d}",
            'title': title,
            'num_exchanges': len(dialogues),
            'format': 'series_session'
        }
        
        # Save files
        safe_title = re.sub(r'[^\w\s-]', '', title)[:50].strip().replace(' ', '_')
        filename = f"session_{idx:02d}_{safe_title}"
        save_session(output_dir, filename, metadata, dialogues)
        
        print(f"  [OK] {len(dialogues)} exchanges")
        
        summary.append({
            'session_id': idx,
            'title': title,
            'exchanges': len(dialogues),
            'file': f"{filename}.json"
        })
    
    # Save summary
    summary_file = output_dir / "_sessions_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"[COMPLETE] Extracted {len(summary)} sessions")
    print(f"Total exchanges: {sum(s['exchanges'] for s in summary)}")
    print(f"Summary: {summary_file}")

def save_session(output_dir: Path, filename: str, metadata: Dict, dialogues: List[Dict]):
    """Save session in both JSON and JSONL formats"""
    # JSON format
    output = {
        'metadata': metadata,
        'conversations': dialogues
    }
    
    json_file = output_dir / f"{filename}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    # JSONL format
    jsonl_file = output_dir / f"{filename}.jsonl"
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for dialogue in dialogues:
            f.write(json.dumps(dialogue, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python extract_any_transcript.py <input_file> [output_subdir]")
        print("\nExamples:")
        print("  python extract_any_transcript.py data/transcripts/2.txt sessions")
        print("  python extract_any_transcript.py data/transcripts/3.txt ellis_collection")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_subdir = sys.argv[2] if len(sys.argv) > 2 else ""
    
    main(input_file, output_subdir)

