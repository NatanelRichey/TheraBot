import re
import json
from pathlib import Path
from datetime import datetime


THERAPISTS = {
    "susan johnson": "therapist",
    "johnson": "therapist",
    "patricia love": "therapist",
    "love": "therapist",
}

# Ordered mapping for participant -> patient_N labels
PARTICIPANTS_ORDERED = [
    "carlson",
    "kjos",
    "dave",
    "kathy",
    "scott",
    "leslie",
]


def normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def build_speaker_patterns():
    # Build regex alternation for all known names and surnames
    names = [
        "Susan Johnson",
        "Johnson",
        "Patricia Love",
        "Love",
        "Carlson",
        "Kjos",
        "Dave",
        "Kathy",
        "Scott",
        "Leslie",
    ]
    # Allow optional titles and whitespace, then colon
    # Example matches: "Susan Johnson:", "Love:", "Dave:", "Scott :"
    alternation = "|".join([re.escape(n) for n in names])
    # Also allow bracketed or uppercase forms and possible leading bullets/dashes
    pattern = rf"^\s*(?:-|\*|\d+\.|\[?)*\s*(?P<speaker>{alternation})\s*:\s*(?P<utterance>.*)$"
    return re.compile(pattern, re.IGNORECASE)


def role_for_speaker(speaker_norm: str, participant_to_label: dict) -> str:
    if speaker_norm in THERAPISTS:
        return "therapist"
    # Assign stable patient_N
    if speaker_norm not in participant_to_label:
        # map by provided ordering if matches, else next index
        if speaker_norm in [p for p in PARTICIPANTS_ORDERED]:
            idx = PARTICIPANTS_ORDERED.index(speaker_norm) + 1
        else:
            idx = len([k for k, v in participant_to_label.items() if v.startswith("patient_")]) + 1
        participant_to_label[speaker_norm] = f"patient_{idx:02d}"
    return participant_to_label[speaker_norm]


def extract_dialogues(text: str):
    pattern = build_speaker_patterns()
    participant_to_label: dict[str, str] = {}
    dialogues = []
    current = None

    def flush_current():
        nonlocal current
        if current and current["content"].strip():
            dialogues.append(current)
        current = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\n\r")
        m = pattern.match(line)
        if m:
            # New turn
            flush_current()
            speaker = normalize_name(m.group("speaker"))
            role = role_for_speaker(speaker, participant_to_label)
            utterance = m.group("utterance").strip()
            current = {"role": role, "speaker": speaker, "content": utterance}
        else:
            # Continuation of current speaker block
            if current is None:
                # Skip navigation/headers until first speaker appears
                continue
            # Append paragraph text; keep spacing
            if line.strip() == "":
                current["content"] += "\n"
            else:
                if current["content"]:
                    current["content"] += " "
                current["content"] += line.strip()

    flush_current()
    return dialogues


def save_outputs(dialogues, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # JSON (single object with dialogues array)
    json_path = out_dir / "session_01_group_expert_couples.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({
            "session_id": "76_group_couples_experts",
            "title": "Couples Therapy with the Experts (Custom Tagged)",
            "date_processed": datetime.utcnow().isoformat(),
            "dialogues": dialogues,
        }, f, ensure_ascii=False, indent=2)

    # JSONL (one turn per line)
    jsonl_path = out_dir / "session_01_group_expert_couples.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for turn in dialogues:
            f.write(json.dumps(turn, ensure_ascii=False) + "\n")

    # Summary
    summary_path = out_dir / "_sessions_summary.json"
    summary = {
        "source_file": "data/transcripts/76.txt",
        "sessions": 1,
        "turns": len(dialogues),
        "notes": "Therapists: Susan Johnson, Patricia Love (Love). Others mapped to patient_01..06 based on provided names.",
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main():
    import sys
    src = Path("data/transcripts/76.txt") if len(sys.argv) < 2 else Path(sys.argv[1])
    if not src.exists():
        print(f"[ERROR] Not found: {src}")
        sys.exit(1)
    text = src.read_text(encoding="utf-8", errors="ignore")
    dialogues = extract_dialogues(text)
    out_dir = Path("data/processed/collection_couples_therapy_with_the_experts")
    save_outputs(dialogues, out_dir)
    print(f"[OK] Extracted {len(dialogues)} turns -> {out_dir}")


if __name__ == "__main__":
    main()


