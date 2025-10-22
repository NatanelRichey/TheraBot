# TheraBot Training Data - Complete Collection

## 🎉 Overview

You now have **29,369 real therapy dialogue exchanges** from **95 therapy sessions** across **12 transcript files**, ready for training TheraBot!

---

## 📊 Quick Stats

| Metric | Count |
|--------|-------|
| **Total Files Processed** | 12 |
| **Total Sessions Extracted** | 95 |
| **Total Dialogue Exchanges** | 29,369 |
| **Output Files Created** | 190+ |
| **Therapeutic Approaches** | 5+ |
| **Clinical Topics** | 10+ |

---

## 📁 What You Have

### **12 Collections in `data/processed/`:**

1. **AAP Therapy Transcripts/** - 16 volumes, 4,847 exchanges
2. **bj_amy_sessions/** - 1 session, 395 exchanges
3. **ellis_institute/** - 9 sessions, 4,004 exchanges
4. **brief_therapy_anxiety/** - 1 session, 327 exchanges
5. **client_002_sessions/** - 12 sessions, 3,357 exchanges
6. **client_003_sessions/** - 12 sessions, 1,976 exchanges
7. **file_7_sessions/** - 8 sessions, 2,521 exchanges
8. **file_15_sessions/** - 10 sessions, 3,226 exchanges
9. **file_17_sessions/** - 14 sessions, 3,853 exchanges
10. **file_18_sessions/** - 4 sessions, 1,722 exchanges
11. **file_19_sessions/** - 6 sessions, 2,626 exchanges
12. **file_20_sessions/** - 2 sessions, 515 exchanges

---

## 🎯 File Formats Explained

Each session = **2 files**:

### **JSON Format** (`.json`)
- Complete session with metadata
- Easy to load and read
- Best for: Analysis, review, full session loading

```json
{
  "metadata": {
    "session_id": "01",
    "title": "Client discusses...",
    "num_exchanges": 217
  },
  "conversations": [
    {"speaker": "THERAPIST", "message": "How are you feeling?"},
    {"speaker": "CLIENT", "message": "I've been anxious..."}
  ]
}
```

### **JSONL Format** (`.jsonl`)
- One dialogue per line
- Memory efficient streaming
- Best for: Training pipelines, embeddings, RAG

```jsonl
{"speaker": "THERAPIST", "message": "How are you feeling?"}
{"speaker": "CLIENT", "message": "I've been anxious..."}
{"speaker": "THERAPIST", "message": "Tell me more..."}
```

**Use whichever format fits your needs!**

---

## 🚀 Quick Start

### **1. Browse Collections**
```python
import json

# View all collections
with open('data/processed/COLLECTIONS_INDEX.md') as f:
    print(f.read())
```

### **2. Load a Specific Collection**
```python
import json

# Load Client 002 summary
with open('data/processed/client_002_sessions/_sessions_summary.json') as f:
    sessions = json.load(f)

for session in sessions:
    print(f"Session {session['session_id']}: {session['exchanges']} exchanges")
```

### **3. Stream Training Data (JSONL)**
```python
import json

# Stream efficiently for training
with open('data/processed/client_002_sessions/session_02_....jsonl') as f:
    for line in f:
        dialogue = json.loads(line)
        # Process for training...
        print(f"{dialogue['speaker']}: {dialogue['message']}")
```

### **4. Load Complete Session (JSON)**
```python
import json

# Load full session with metadata
with open('data/processed/client_002_sessions/session_02_....json') as f:
    session = json.load(f)

print(f"Title: {session['metadata']['title']}")
print(f"Exchanges: {len(session['conversations'])}")

# Access conversations
for conv in session['conversations'][:5]:
    print(f"{conv['speaker']}: {conv['message']}")
```

---

## 📚 Training Recommendations

### **By Therapeutic Approach:**

**CBT/REBT Training (9,000+ exchanges):**
- `ellis_institute/` - 9 pure REBT sessions
- `AAP Therapy Transcripts/volume_11...` - John Jones REBT

**Client-Centered Training (20,000+ exchanges):**
- `client_002_sessions/` - 12 sessions
- `client_003_sessions/` - 12 sessions
- `file_7_sessions/` - 8 sessions
- `file_15_sessions/` - 10 sessions
- `file_17_sessions/` - 14 sessions
- And more...

**Brief/Solution-Focused (722 exchanges):**
- `brief_therapy_anxiety/` - 1 session
- `bj_amy_sessions/` - 1 session

### **By Clinical Topic:**

**Sexual Dysfunction (5,333 exchanges):**
- Client 002 & 003 combined

**Relationship Issues (5,662 exchanges):**
- Files 7, 19, 20

**Self-Esteem & Body Image (8,801 exchanges):**
- Files 15, 17, 18

---

## 🎓 Training Splits

### **Recommended Split:**
```python
# Split by complete sessions (not individual exchanges!)

train_sessions = 76  # 80%
val_sessions = 9     # 10%
test_sessions = 10   # 10%

# Approximate exchange counts
train_exchanges = ~23,500
val_exchanges = ~2,900
test_exchanges = ~2,900
```

**Important:** Always split by **complete sessions**, not individual dialogue exchanges, to prevent data leakage!

---

## 📖 Documentation Files

1. **`COMPLETE_EXTRACTION_SUMMARY.md`** - Complete overview of all 12 files
2. **`COLLECTIONS_INDEX.md`** - Quick reference table
3. **`FILE_STRUCTURE_GUIDE.md`** - How files are organized
4. **`data/processed/MASTER_CATALOG.json`** - Searchable catalog
5. **Collection-specific `_sessions_summary.json`** files

---

## 🔍 Find What You Need

### **All Summary Files:**
```
data/processed/AAP Therapy Transcripts/_volumes_summary.json
data/processed/ellis_institute/_sessions_summary.json
data/processed/client_002_sessions/_sessions_summary.json
data/processed/client_003_sessions/_sessions_summary.json
data/processed/file_7_sessions/_sessions_summary.json
data/processed/file_15_sessions/_sessions_summary.json
data/processed/file_17_sessions/_sessions_summary.json
data/processed/file_18_sessions/_sessions_summary.json
data/processed/file_19_sessions/_sessions_summary.json
data/processed/file_20_sessions/_sessions_summary.json
```

### **Main Catalog:**
```
data/processed/MASTER_CATALOG.json
data/processed/COLLECTIONS_INDEX.md
```

---

## ⚙️ Extraction Scripts (Reusable)

All extraction scripts are in the project root:

- `extract_all_volumes.py` - For multi-volume HTML files
- `extract_any_transcript.py` - Universal HTML extractor
- `extract_plaintext_transcripts.py` - For plain text files

**Reusable for future transcript files!**

---

## 💡 Usage Examples

### **Convert to Training Format (OpenAI-style):**
```python
import json

def convert_to_training_format(jsonl_file):
    """Convert JSONL to training pairs"""
    training_data = []
    
    with open(jsonl_file) as f:
        dialogues = [json.loads(line) for line in f]
    
    # Create user-assistant pairs
    for i in range(0, len(dialogues)-1, 2):
        if i+1 < len(dialogues):
            training_data.append({
                "messages": [
                    {"role": "user", "content": dialogues[i]['message']},
                    {"role": "assistant", "content": dialogues[i+1]['message']}
                ]
            })
    
    return training_data

# Use it
training_data = convert_to_training_format(
    'data/processed/ellis_institute/session_01_Unknown_Session.jsonl'
)
```

### **Batch Load All Collections:**
```python
import json
from pathlib import Path

def load_all_collections():
    """Load summary of all collections"""
    processed_dir = Path('data/processed')
    collections = {}
    
    # Find all summary files
    for summary_file in processed_dir.glob('**/_sessions_summary.json'):
        collection_name = summary_file.parent.name
        with open(summary_file) as f:
            collections[collection_name] = json.load(f)
    
    return collections

collections = load_all_collections()
for name, sessions in collections.items():
    total = sum(s['exchanges'] for s in sessions)
    print(f"{name}: {len(sessions)} sessions, {total} exchanges")
```

---

## ✅ Quality Assurance

- ✅ Clean speaker-message format
- ✅ HTML entities decoded
- ✅ Whitespace normalized
- ✅ Preserved therapeutic context
- ✅ (inaudible), (laughter) markers kept
- ✅ Complete dialogue flow maintained
- ✅ Metadata attached to each session

---

## 🎯 Ready to Use!

**Everything is extracted, formatted, and documented.**

Your training data is in:
```
data/processed/
```

Start training TheraBot with:
- 95 real therapy sessions
- 29,369 dialogue exchanges
- Multiple therapeutic approaches
- Diverse clinical presentations

**All ready to go!** 🚀

---

## 📞 Need Help?

Check these files:
- `COMPLETE_EXTRACTION_SUMMARY.md` - Full details
- `COLLECTIONS_INDEX.md` - Quick reference
- `FILE_STRUCTURE_GUIDE.md` - How it's organized

---

**Happy Training!** 🎓

