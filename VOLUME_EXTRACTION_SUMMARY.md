# AAP Therapy Transcripts - Volume Extraction Summary

## Overview
Successfully extracted and formatted **16 therapy session volumes** from the American Academy of Psychotherapists (AAP) HTML transcript file into structured training data.

## Extraction Results

### Statistics
- **Total Volumes Processed**: 16
- **Total Dialogue Exchanges**: 4,847
- **Format**: JSON + JSONL (dual format for flexibility)
- **Source File**: `data/transcripts/1.txt` (9,146 lines of HTML)

### Volume Breakdown

| Vol | Title | Exchanges | Therapist(s) | Clinical Focus |
|-----|-------|-----------|--------------|----------------|
| 01 | Loretta | 374 | Ellis, Felder, Rogers | Schizophrenia |
| 02 | 19-Year-Old Male | 482 | Unknown | Therapeutic relationship development |
| 03 | Mr. Vac | 134 | Unknown | Depression/silence |
| 05 | Miss Mun | 123 | Unknown | Interpersonal difficulties |
| 06 | Mr. Lin | 153 | Unknown | Homosexuality |
| 07 | Life Goals | 331 | Unknown | Ambivalence |
| 08 | Mrs. P.S. | 144 | Carl Rogers | Childhood/family |
| 09 | Composite ABC | 210 | Multiple | Anxiety/phobias |
| 11 | John Jones | 552 | Albert Ellis | Homosexuality (REBT) |
| 12 | Gregg | 389 | Unknown | Dream analysis |
| 13 | Harry | 414 | Unknown | Recording anxiety |
| 14 | Jim | 140 | Unknown | Paranoid episode |
| 15 | Don | 134 | Unknown | Frustration |
| 17 | Miss E.S.T. | 132 | Unknown | Post-ECT therapy |
| 19 | Betty & Dick | 276 | Group therapist | Group therapy |
| 21 | Couple | 859 | Whitaker & Felder | Couple therapy |

## File Structure

### Output Directory
```
data/processed/volumes/
├── _volumes_summary.json          # Master index of all volumes
├── README.md                       # Documentation
├── volume_01_Loretta_....json     # Complete session (JSON)
├── volume_01_Loretta_....jsonl    # Streaming format (JSONL)
├── volume_02_19-Year-Old_....json
├── volume_02_19-Year-Old_....jsonl
└── ... (16 volumes × 2 formats = 32 files)
```

### Metadata
```
data/metadata/
└── aap_volumes_metadata.csv       # Comprehensive metadata for all volumes
```

## Data Format

### JSON Format (Complete Session)
```json
{
  "metadata": {
    "volume_id": 1,
    "title": "Loretta - Interviews with Ellis, Felder, and Rogers",
    "description": "Client with schizophrenia...",
    "therapists": [],
    "client_info": "",
    "num_exchanges": 374
  },
  "conversations": [
    {
      "speaker": "DR. FELDER",
      "message": "Hi."
    },
    {
      "speaker": "LORETTA",
      "message": "How are you?"
    }
  ]
}
```

### JSONL Format (Line-Delimited for Streaming)
```jsonl
{"speaker": "DR. FELDER", "message": "Hi."}
{"speaker": "LORETTA", "message": "How are you?"}
{"speaker": "DR. FELDER", "message": "You want to sit over here..."}
```

## Answer to Your Question

### **Should you split each volume into a different file?**

**YES - ABSOLUTELY!** ✓

Here's why this is the right approach:

#### ✓ Advantages of Splitting:

1. **Different Cases** - Each volume is a unique client/situation
2. **Different Therapeutic Approaches**
   - Volume 1: REBT (Ellis), Client-Centered (Rogers), Holistic (Felder)
   - Volume 11: Pure REBT (Ellis)
   - Volume 21: Family Systems (Whitaker & Felder)
   
3. **Training Flexibility**
   - Can train on specific therapeutic styles
   - Easy to exclude/include specific cases
   - Better for cross-validation
   
4. **Easier Management**
   - Simpler to reference specific cases
   - Can update individual volumes
   - Better for version control
   
5. **Performance**
   - Faster to load specific sessions
   - Better for streaming (JSONL format)
   - Easier to process in parallel

#### File Naming Convention
Each volume has a clear, descriptive filename:
- `volume_[ID]_[Title].json` - For complete loading
- `volume_[ID]_[Title].jsonl` - For streaming/processing

## Usage Examples

### Load a Specific Volume (Python)
```python
import json

# Load complete session
with open('data/processed/volumes/volume_01_Loretta_....json') as f:
    session = json.load(f)
    
print(f"Volume: {session['metadata']['title']}")
print(f"Exchanges: {len(session['conversations'])}")

for exchange in session['conversations'][:5]:
    print(f"{exchange['speaker']}: {exchange['message']}")
```

### Stream JSONL Format
```python
import json

with open('data/processed/volumes/volume_01_Loretta_....jsonl') as f:
    for line in f:
        dialogue = json.loads(line)
        print(f"{dialogue['speaker']}: {dialogue['message']}")
```

### Load All Volumes
```python
import json
from pathlib import Path

volumes_dir = Path('data/processed/volumes')
summary_file = volumes_dir / '_volumes_summary.json'

with open(summary_file) as f:
    volumes = json.load(f)

for vol in volumes:
    print(f"Volume {vol['volume_id']}: {vol['title']} ({vol['exchanges']} exchanges)")
```

## Training Data Conversion

### For LLM Fine-Tuning (OpenAI Format)
```python
def convert_to_training_format(conversations):
    training_data = []
    for i in range(0, len(conversations)-1, 2):
        if i+1 < len(conversations):
            training_data.append({
                "messages": [
                    {"role": "user", "content": conversations[i]['message']},
                    {"role": "assistant", "content": conversations[i+1]['message']}
                ]
            })
    return training_data
```

### For RAG / Vector Database
```python
# Each JSONL line can be embedded separately
# Preserve speaker role as metadata
{
    "text": "DR. FELDER: Hi.",
    "metadata": {
        "volume_id": 1,
        "speaker": "DR. FELDER",
        "therapist": "Richard Felder",
        "approach": "Holistic"
    }
}
```

## Notable Volumes for Specific Training

### **Best for DBT/CBT Training:**
- Volume 1 (Loretta) - Albert Ellis REBT approach
- Volume 11 (John Jones) - Pure REBT by Ellis

### **Best for Client-Centered Training:**
- Volume 2 (19-Year-Old Male) - Relationship development
- Volume 8 (Mrs. P.S.) - Carl Rogers session

### **Best for Family/Systems Training:**
- Volume 21 (Couple Therapy) - Whitaker & Felder co-therapy

### **Best for Group Therapy:**
- Volume 19 (Betty & Dick) - Group dynamics

## Quality Notes

### What's Preserved:
- ✓ Original speaker labels
- ✓ Conversational markers: (inaudible), (laughter), (background noise)
- ✓ Complete dialogue flow
- ✓ Therapeutic context

### What's Cleaned:
- ✓ HTML tags removed
- ✓ HTML entities decoded
- ✓ Whitespace normalized
- ✓ Non-dialogue metadata excluded

## Recommended Next Steps

1. **For TheraBot Training:**
   - Focus on volumes with specific therapeutic approaches
   - Volume 1, 11 for CBT/REBT
   - Volume 8 for client-centered
   
2. **Data Augmentation:**
   - Can create therapist-role specific datasets
   - Extract patterns by therapeutic approach
   
3. **Evaluation Sets:**
   - Hold out 1-2 volumes for testing
   - Use different volumes for validation

4. **Fine-Tuning Strategy:**
   - Start with specific approach (e.g., all REBT volumes)
   - Then expand to multi-approach training

## Files Created

### Main Outputs:
- `extract_all_volumes.py` - Extraction script (reusable)
- `data/processed/volumes/` - 16 volumes × 2 formats = 32 files
- `data/metadata/aap_volumes_metadata.csv` - Complete metadata
- `data/processed/volumes/_volumes_summary.json` - Quick reference
- `data/processed/volumes/README.md` - Format documentation

### Total Data:
- **4,847 dialogue exchanges** ready for training
- **16 unique therapy cases** across multiple approaches
- **Multiple therapeutic styles** represented

## Conclusion

✓ **Splitting was the right choice!**

You now have:
- Clean, structured training data
- Easy-to-use file organization  
- Flexible formats (JSON + JSONL)
- Comprehensive metadata
- Documentation for usage

Each volume can be used independently or combined as needed for your TheraBot project.

---

**Date Extracted**: October 22, 2025  
**Source**: American Academy of Psychotherapists Archive  
**Extraction Script**: `extract_all_volumes.py`

