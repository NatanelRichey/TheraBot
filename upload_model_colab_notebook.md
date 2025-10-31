# Upload Model from Colab to HuggingFace Space

## Quick Setup

Run this in a Colab cell to upload your GGUF model file directly to your Space.

## Option 1: Using HuggingFace Hub (Recommended - Simpler)

```python
# Install dependencies
!pip install huggingface_hub

from huggingface_hub import HfApi, login
import os

# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_PATH = "/content/llama3-3b.Q4_K_M.gguf"  # ⬅️ Your model path in Colab
HF_USERNAME = "NatanelRichey"
SPACE_NAME = "TheraBot"

# Get your HuggingFace token
# Option A: Use Colab secrets (recommended)
from google.colab import userdata
HF_TOKEN = userdata.get('HF_TOKEN')

# Option B: Paste directly (less secure)
# HF_TOKEN = "your_token_here"

# =============================================================================
# UPLOAD
# =============================================================================
# Login
login(token=HF_TOKEN)

# Upload
api = HfApi()
api.upload_file(
    path_or_fileobj=MODEL_PATH,
    path_in_repo=f"models/{os.path.basename(MODEL_PATH)}",
    repo_id=f"{HF_USERNAME}/{SPACE_NAME}",
    repo_type="space",
)

print("✅ Model uploaded successfully!")
print(f"   https://huggingface.co/spaces/{HF_USERNAME}/{SPACE_NAME}")
```

## Option 2: Using Git (For Large Files)

```python
# Install git
!apt-get install git -y

import subprocess
import os

# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_PATH = "/content/llama3-3b.Q4_K_M.gguf"  # ⬅️ Your model path
HF_USERNAME = "NatanelRichey"
SPACE_NAME = "TheraBot"

# Get token
from google.colab import userdata
HF_TOKEN = userdata.get('HF_TOKEN')

# =============================================================================
# UPLOAD
# =============================================================================
SPACE_URL = f"https://{HF_USERNAME}:{HF_TOKEN}@huggingface.co/spaces/{HF_USERNAME}/{SPACE_NAME}"
CLONE_DIR = f"/tmp/{SPACE_NAME}"

# Clean up if exists
!rm -rf {CLONE_DIR}

# Clone
!git clone {SPACE_URL} {CLONE_DIR}

# Create models directory
import os
os.makedirs(f"{CLONE_DIR}/models", exist_ok=True)

# Copy model
MODEL_NAME = os.path.basename(MODEL_PATH)
!cp {MODEL_PATH} {CLONE_DIR}/models/{MODEL_NAME}

# Commit and push
%cd {CLONE_DIR}
!git config user.name {HF_USERNAME}
!git config user.email "{HF_USERNAME}@users.noreply.huggingface.co"
!git add models/
!git commit -m "Add model file: {MODEL_NAME}"
!git push

print("✅ Model uploaded successfully!")
```

## Setting Up Colab Secrets (Recommended)

1. Go to your Colab notebook
2. Click the 🔑 icon (left sidebar) → "Secrets"
3. Click "+ Add Secret"
4. Name: `HF_TOKEN`
5. Value: Your HuggingFace token (get from https://huggingface.co/settings/tokens)
6. Click "Add Secret"

Then use:
```python
from google.colab import userdata
HF_TOKEN = userdata.get('HF_TOKEN')
```

## Verify Upload

After uploading, check your Space:
```python
from huggingface_hub import list_repo_files

files = list_repo_files(
    repo_id=f"{HF_USERNAME}/{SPACE_NAME}",
    repo_type="space"
)

print("Files in Space:")
for f in files:
    print(f"  - {f}")
```

