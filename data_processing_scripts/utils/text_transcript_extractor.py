"""
Extract therapy transcripts from plain text files (non-HTML)
For files like 4.txt and 5.txt that have simple text format
"""
import re
import json
from pathlib import Path
from typing import List, Dict

def extract_dialogues_from_plaintext(text: str) -> List[Dict[str, str]]:
    """Extract speaker-message pairs from plain text"""
    dialogues = []
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # Skip metadata/navigation lines
        if any(skip in line.lower() for skip in [
            'transcript of',
            'begin transcript',
            'end transcript',
            'skip to main',
            'you are here',
            'cite',
            'email',
            'embed',
            'page ',
            'session ',
            'client 0',
            'presented by'
        ]):
            continue
        
        # Match pattern: SPEAKER: message
        # Can be COUNSELOR:, PATIENT:, THERAPIST:, CLIENT:, etc.
        match = re.match(r'^([A-Z][A-Z\s\.\-\']+?):\s*(.+)$', line)
        if match:
            speaker = match.group(1).strip()
            message = match.group(2).strip()
            
            # Skip if message is too short or speaker name too long
            if len(message) > 0 and len(speaker) < 30:
                dialogues.append({
                    'speaker': speaker,
                    'message': message
                })
    
    return dialogues

def extract_title_from_plaintext(text: str) -> str:
    """Extract title from plain text file"""
    lines = text.split('\n')
    
    # Check first 20 lines for title
    for line in lines[:20]:
        line = line.strip()
        if len(line) > 10 and len(line) < 150:
            # Skip common headers
            if any(skip in line.lower() for skip in [
                'transcript of',
                'begin transcript',
                'skip to',
                'search',
                'menu',
                'you are here'
            ]):
                continue
            
            # Likely a title
            if line and not line.startswith('Page'):
                return line
    
    return "Unknown Session"

def detect_session_boundaries(text: str) -> List[tuple]:
    """Detect multiple sessions in plain text"""
    lines = text.split('\n')
    sessions = []
    
    current_start = 0
    current_title = "Unknown Session"
    
    for i, line in enumerate(lines):
        # Look for session markers
        session_match = re.match(r'Session (\d+):\s*(.+)', line, re.IGNORECASE)
        
        if session_match:
            # Save previous session if exists
            if i > current_start + 10:  # At least 10 lines
                sessions.append((current_start, i, current_title))
            
            # Start new session
            current_start = i
            current_title = line.strip()
    
    # Add final session
    if len(lines) > current_start + 10:
        sessions.append((current_start, len(lines), current_title))
    
    return sessions

def main(input_file: str, output_subdir: str = ""):
    """Extract transcripts from plain text file"""
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"[ERROR] File not found: {input_file}")
        return
    
    # Determine output directory
    if not output_subdir:
        base_name = input_path.stem
        output_subdir = f"file_{base_name}_plaintext"
    
    output_dir = Path(f"data/processed/{output_subdir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Processing plain text: {input_path.name}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    # Read file
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    
    # Check for multiple sessions
    sessions = detect_session_boundaries(text)
    
    if len(sessions) > 1:
        print(f"Detected: {len(sessions)} sessions\n")
        extract_multiple_sessions(text, sessions, output_dir)
    else:
        print(f"Detected: Single session\n")
        extract_single_session(text, output_dir, input_path.stem)

def extract_single_session(text: str, output_dir: Path, file_id: str):
    """Extract single session from plain text"""
    title = extract_title_from_plaintext(text)
    dialogues = extract_dialogues_from_plaintext(text)
    
    if len(dialogues) == 0:
        print(f"  [WARNING] No dialogues extracted!")
        return
    
    metadata = {
        'session_id': file_id,
        'title': title,
        'num_exchanges': len(dialogues),
        'format': 'plaintext_single'
    }
    
    save_session(output_dir, file_id, metadata, dialogues)
    
    print(f"  [OK] {title}")
    print(f"       Exchanges: {len(dialogues)}")

def extract_multiple_sessions(text: str, sessions: List[tuple], output_dir: Path):
    """Extract multiple sessions from plain text"""
    lines = text.split('\n')
    summary = []
    
    for idx, (start, end, title) in enumerate(sessions, 1):
        session_text = '\n'.join(lines[start:end])
        dialogues = extract_dialogues_from_plaintext(session_text)
        
        if len(dialogues) == 0:
            print(f"Session {idx}: {title[:50]} - [WARNING] No dialogues")
            continue
        
        metadata = {
            'session_id': f"session_{idx:02d}",
            'title': title,
            'num_exchanges': len(dialogues),
            'format': 'plaintext_series'
        }
        
        safe_title = re.sub(r'[^\w\s-]', '', title)[:50].strip().replace(' ', '_')
        filename = f"session_{idx:02d}_{safe_title}"
        
        save_session(output_dir, filename, metadata, dialogues)
        
        print(f"Session {idx}: {title[:60]}")
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

def save_session(output_dir: Path, filename: str, metadata: Dict, dialogues: List[Dict]):
    """Save session in JSON and JSONL formats"""
    output = {
        'metadata': metadata,
        'conversations': dialogues
    }
    
    # JSON
    json_file = output_dir / f"{filename}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    # JSONL
    jsonl_file = output_dir / f"{filename}.jsonl"
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for dialogue in dialogues:
            f.write(json.dumps(dialogue, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python extract_plaintext_transcripts.py <input_file> [output_subdir]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_subdir = sys.argv[2] if len(sys.argv) > 2 else ""
    
    main(input_file, output_subdir)

