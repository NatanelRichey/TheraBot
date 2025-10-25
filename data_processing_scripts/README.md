# Therapy Data Processing Pipeline

A comprehensive data processing pipeline for therapy conversation data, following HuggingFace LLM Course Chapter 3.2 best practices.

## 📚 References

This pipeline implements techniques from:
- [HuggingFace LLM Course Chapter 3.1](https://huggingface.co/learn/llm-course/chapter3/1) - Fine-tuning fundamentals
- [HuggingFace LLM Course Chapter 3.2](https://huggingface.co/learn/llm-course/chapter3/2) - Data preprocessing and tokenization
- [HuggingFace Datasets Documentation](https://huggingface.co/docs/datasets/) - Data processing and management
- [Tokenizers Summary](https://huggingface.co/docs/transformers/main/en/tokenizer_summary) - Tokenization strategies

## 🎯 Features

### Core Processing
- **Batched Processing**: Uses `Dataset.map()` with `batched=True` for efficiency
- **Dynamic Padding**: Implements `DataCollatorWithPadding` for optimal memory usage
- **Session-Level Chunking**: Keeps conversations intact, splits only very long sessions
- **Instruction Format**: Converts therapy conversations to instruction-following format
- **Quality Filtering**: Filters exchanges by length and quality criteria

### HuggingFace Best Practices
- ✅ Batched preprocessing for speed
- ✅ Dynamic padding instead of fixed-length padding
- ✅ Proper tokenization with Llama 3.1 tokenizer
- ✅ Efficient data collation
- ✅ Memory-optimized processing

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Data Processing

```bash
python run_data_processing.py
```

### 3. Validate Results

```bash
python validate_processing.py
```

## 📁 File Structure

```
data_processing_scripts/
├── therapy_data_processor.py    # Main processing pipeline
├── run_data_processing.py       # Execution script
├── validate_processing.py       # Validation script
├── simple_length_analysis.py    # Data analysis tool
├── requirements.txt             # Dependencies
└── README.md                   # This file
```

## ⚙️ Configuration

The pipeline uses a `ProcessingConfig` class for configuration:

```python
config = ProcessingConfig(
    # Model and tokenization
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    max_length=2048,
    
    # Session chunking (your specifications)
    max_session_exchanges=300,  # Split sessions longer than 300 exchanges
    overlap_percentage=0.2,     # 20% overlap for context preservation
    
    # Data filtering
    min_exchange_length=1,      # Minimum words per exchange
    max_exchange_length=200,    # Maximum words per exchange
    
    # File paths
    input_data_dir="data/training_data",
    output_dir="data/processed",
    cache_dir=".cache"
)
```

## 🔄 Processing Pipeline

### Step 1: Load Therapy Sessions
- Scans `data/training_data` for JSON files
- Extracts dialogue from various JSON structures
- Ignores AI-generated labels (as requested)

### Step 2: Session-Level Chunking
- **Short sessions** (<300 exchanges): Keep intact
- **Long sessions** (>300 exchanges): Split with 20% overlap
- Preserves conversation context and flow

### Step 3: Instruction Format Conversion
- Converts to instruction/input/output format
- Creates sliding window examples (2-3 exchanges per example)
- Filters by exchange length and quality

### Step 4: Tokenization
- Uses Llama 3.1 SentencePiece tokenizer
- Batched processing with `Dataset.map()`
- Dynamic padding with `DataCollatorWithPadding`

### Step 5: Train/Val/Test Split
- 80% train, 10% validation, 10% test
- Random split for balanced distribution

### Step 6: Save Processed Data
- Saves dataset in HuggingFace format
- Includes tokenizer and data collator info
- Ready for fine-tuning

## 📊 Data Analysis

The pipeline includes analysis tools:

```bash
python simple_length_analysis.py
```

**Your Data Statistics:**
- **4,847 total exchanges** across 16 sessions
- **Average exchange**: 18.7 words (median: 12 words)
- **90th percentile**: 53 words per exchange
- **Sessions range**: 243-859 exchanges

## 🎯 Output Format

### Processed Dataset Structure
```python
{
    'train': Dataset,
    'validation': Dataset,
    'test': Dataset
}
```

### Example Features
```python
{
    'input_ids': [1, 2, 3, ...],           # Token IDs
    'attention_mask': [1, 1, 1, ...],      # Attention mask
    'labels': [1, 2, 3, ...]               # Labels (same as input_ids for causal LM)
}
```

### Instruction Format
```
Instruction: "You are a DBT therapist. Respond to the client's concerns using DBT techniques."

Input: "Therapist: How are you feeling today?
Client: I'm really anxious about work."

Output: "I understand you're feeling anxious about work. Let's use some DBT skills to help you manage this anxiety..."
```

## 🔧 Usage in Training

### Load Processed Data
```python
from datasets import load_from_disk
from transformers import DataCollatorWithPadding

# Load dataset
dataset_dict = load_from_disk("data/processed")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("data/processed/tokenizer")

# Create data collator
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True,
    return_tensors="pt"
)
```

### Training Configuration
```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    dataloader_pin_memory=False,  # Important for dynamic padding
    # ... other args
)
```

## 📈 Performance Optimizations

### Batched Processing
- Uses `Dataset.map(batched=True)` for 10x faster preprocessing
- Processes multiple examples simultaneously
- Reduces memory overhead

### Dynamic Padding
- `DataCollatorWithPadding` pads to longest sequence in batch
- More memory efficient than fixed-length padding
- Better for variable-length conversations

### Memory Management
- Processes data in chunks
- Clears cache between operations
- Optimized for large datasets

## 🐛 Troubleshooting

### Common Issues

1. **Memory Error**: Reduce batch size or max_length
2. **Tokenizer Error**: Check model name and cache directory
3. **Data Loading Error**: Verify JSON file structure

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📋 Validation Checklist

- [ ] Dataset loads correctly
- [ ] Tokenization is valid
- [ ] Train/val/test splits are balanced
- [ ] Dynamic padding works
- [ ] Data collator functions properly
- [ ] Ready for fine-tuning

## 🎯 Next Steps

After running the data processing pipeline:

1. **Review processed data** with validation script
2. **Proceed with model fine-tuning** using the processed dataset
3. **Use DataCollatorWithPadding** for dynamic padding during training
4. **Monitor training progress** with validation metrics

## 📚 Additional Resources

- [HuggingFace Fine-tuning Guide](https://huggingface.co/learn/llm-course/chapter3/1)
- [Data Processing Best Practices](https://huggingface.co/learn/llm-course/chapter3/2)
- [Tokenization Strategies](https://huggingface.co/docs/transformers/main/en/tokenizer_summary)
- [Datasets Documentation](https://huggingface.co/docs/datasets/)
