# TheraBot: Therapy Transcript Processing Framework

A comprehensive framework for processing and structuring therapy transcripts into training data for AI models, with a focus on privacy and ethical data handling.

## 🚨 Privacy & Ethics Notice

**This repository contains NO sensitive therapy data.** All personal information, therapy transcripts, and processed training data remain local and are excluded from this public repository for:

- **Privacy Protection**: Therapy sessions contain highly sensitive personal information
- **Legal Compliance**: Source materials are likely copyrighted
- **Professional Ethics**: Maintains client confidentiality principles

## 📊 Dataset Overview

The framework processes **173,145 dialogue exchanges** across **529 therapy sessions** from **92 collections**, including:

- **REBT/CBT**: Rational Emotive Behavior Therapy and Cognitive Behavioral Therapy
- **Psychoanalytic**: Psychodynamic therapy approaches  
- **Group Therapy**: Multi-participant therapeutic sessions
- **Individual Therapy**: One-on-one counseling sessions
- **Couples Therapy**: Relationship counseling sessions

## 🛠️ Processing Pipeline

### Core Scripts

- **`extract_any_transcript.py`**: Universal extractor for HTML and plain text formats
- **`extract_plaintext_transcripts.py`**: Specialized plain text processor
- **`custom_group_extractor.py`**: Group therapy session handler
- **`clean_and_label_data.py`**: Data cleaning and labeling utilities

### Supported Formats

- **Input**: HTML files, plain text transcripts, various therapy session formats
- **Output**: JSON (complete sessions) and JSONL (streaming format)

## 📁 Repository Structure

```
TheraBot/
├── data/
│   ├── metadata/           # Collection metadata and summaries
│   ├── processed/          # [LOCAL ONLY] Processed training data
│   └── transcripts/        # [LOCAL ONLY] Source transcript files
├── scripts/                # Processing and extraction scripts
├── docs/                   # Documentation and guides
└── requirements.txt        # Python dependencies
```

## 🚀 Quick Start

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd TheraBot
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Process transcripts** (with your own data):
   ```python
   from extract_any_transcript import process_transcript
   
   # Process a therapy transcript
   result = process_transcript('path/to/transcript.txt')
   ```

## 📋 Usage Examples

### Basic Transcript Processing

```python
from extract_any_transcript import process_transcript

# Process HTML therapy session
sessions = process_transcript('session.html')

# Process plain text transcript  
sessions = process_transcript('transcript.txt')
```

### Group Therapy Processing

```python
from custom_group_extractor import process_group_session

# Process group therapy with specific speaker mapping
sessions = process_group_session('group_session.txt', 
                                therapists=['Dr. Smith', 'Dr. Jones'],
                                participants=['Client A', 'Client B'])
```

## 🔧 Configuration

### Speaker Mapping

The framework supports flexible speaker identification:

- **Automatic detection**: THERAPIST/CLIENT labels
- **Custom mapping**: Specify therapist and client names
- **Group therapy**: Multiple therapists and participants

### Output Formats

- **JSON**: Complete session with metadata
- **JSONL**: Line-delimited for streaming processing
- **CSV**: Metadata summaries for analysis

## 📚 Documentation

- **`README_TRAINING_DATA.md`**: Comprehensive dataset documentation
- **`ALL_EXTRACTIONS_SUMMARY.md`**: Processing summary and statistics
- **`data/README.md`**: Data structure and organization guide

## ⚖️ Legal & Ethical Considerations

- **No sensitive data** in public repository
- **Local processing only** for privacy protection
- **Copyright compliance** with source materials
- **Professional ethics** maintained throughout

## 🤝 Contributing

This framework is designed for research and educational purposes. When contributing:

1. **Never commit sensitive data**
2. **Follow privacy guidelines**
3. **Respect copyright restrictions**
4. **Maintain ethical standards**

## 📄 License

This project is licensed for research and educational use. Users are responsible for ensuring compliance with all applicable laws and regulations regarding therapy data and patient privacy.

## 🔍 Research Applications

The processed data supports various AI research applications:

- **Therapeutic conversation modeling**
- **Mental health AI training**
- **Dialogue system development**
- **Clinical decision support systems**

## 📞 Contact

For questions about the framework or research collaboration, please contact the project maintainers.

---

**Remember**: Always handle therapy data with the utmost care for privacy and ethical considerations. This framework provides the tools, but responsible use is essential.
