# Colab Cell: Convert Model to GGUF Q4_K_M

## Cell 1: Install llama.cpp tools

```python
# Install llama.cpp tools for conversion
!pip install -q llama-cpp-python
!git clone --depth 1 https://github.com/ggerganov/llama.cpp.git
!cd llama.cpp && make
```

---

## Cell 2: Convert HuggingFace Model to GGUF

```python
import os

# Configuration
HF_MODEL_PATH = "your-checkpoint-path"  # UPDATE THIS - your trained adapter or merged model
OUTPUT_DIR = "gguf_models"
MODEL_NAME = "llama3-3b"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Run conversion
!python llama.cpp/convert-hf-to-gguf.py {HF_MODEL_PATH} \
    --outdir {OUTPUT_DIR} \
    --outfile {MODEL_NAME}.gguf
```

---

## Cell 3: Quantize to Q4_K_M

```python
# Quantize to Q4_K_M
!./llama.cpp/quantize {OUTPUT_DIR}/{MODEL_NAME}.gguf {OUTPUT_DIR}/{MODEL_NAME}.Q4_K_M.gguf Q4_K_M

print(f"✅ Model saved to: {OUTPUT_DIR}/{MODEL_NAME}.Q4_K_M.gguf")
```

---

## Cell 4: Download or Save

```python
# Option A: Download to local machine
from google.colab import files
files.download(f'{OUTPUT_DIR}/{MODEL_NAME}.Q4_K_M.gguf')

# Option B: Upload to HuggingFace Hub (recommended)
from huggingface_hub import login, upload_file

login()  # You'll need to enter your HF token

upload_file(
    path_or_fileobj=f'{OUTPUT_DIR}/{MODEL_NAME}.Q4_K_M.gguf',
    path_in_repo=f'{MODEL_NAME}.Q4_K_M.gguf',
    repo_id='YOUR_USERNAME/llama3-3b-dbt-therabot'  # UPDATE THIS
)
```

---

## Complete Combined Cell (All-in-One)

```python
# Install llama.cpp
!pip install -q llama-cpp-python
!git clone --depth 1 https://github.com/ggerganov/llama.cpp.git
!cd llama.cpp && make

# Configuration
HF_MODEL_PATH = "your-checkpoint-path"  # UPDATE THIS
OUTPUT_DIR = "gguf_models"
MODEL_NAME = "llama3-3b"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Convert to GGUF
!python llama.cpp/convert-hf-to-gguf.py {HF_MODEL_PATH} \
    --outdir {OUTPUT_DIR} \
    --outfile {MODEL_NAME}.gguf

# Quantize to Q4_K_M
!./llama.cpp/quantize {OUTPUT_DIR}/{MODEL_NAME}.gguf {OUTPUT_DIR}/{MODEL_NAME}.Q4_K_M.gguf Q4_K_M

print(f"✅ Model ready: {OUTPUT_DIR}/{MODEL_NAME}.Q4_K_M.gguf")

# Download
from google.colab import files
files.download(f'{OUTPUT_DIR}/{MODEL_NAME}.Q4_K_M.gguf')
```

---

## Important Notes

1. **Update `HF_MODEL_PATH`**: Point to your trained checkpoint
2. **LoRA Users**: Merge adapters first! (see next section)
3. **File Size**: Q4_K_M is ~1.8GB for 3B model
4. **Download Time**: ~5-10 minutes

---

## If Using LoRA: Merge First!

```python
# Merge LoRA adapters into base model before conversion
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-3B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-3B-Instruct")

# Load LoRA adapters
lora_model = PeftModel.from_pretrained(base_model, "path/to/lora/adapter")

# Merge and save
merged_model = lora_model.merge_and_unload()
merged_model.save_pretrained("merged_model")
tokenizer.save_pretrained("merged_model")

print("✅ Merged model saved to 'merged_model/'")
print("Now use this as HF_MODEL_PATH in conversion above")
```

