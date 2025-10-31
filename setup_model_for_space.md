# Solution: Host Model Separately & Download on Space Startup

## The Problem
HuggingFace Spaces have a **1 GB storage limit** on the free tier. Your GGUF model (2-3 GB) exceeds this.

## Solution: Store Model in Model Repo, Download on Startup

### Step 1: Upload Model to a Model Repository

Create a separate Model repository (not Space) to host your model. Model repos have higher limits.

**In Colab, run:**

```python
!pip install huggingface_hub

from huggingface_hub import HfApi, login
import os

# Configuration
MODEL_PATH = "/content/llama3-3b.Q4_K_M.gguf"  # Your model file
HF_USERNAME = "NatanelRichey"
MODEL_REPO = "therabot-llama3-3b-gguf"  # New model repo name

# Get token
from google.colab import userdata
HF_TOKEN = userdata.get('HF_TOKEN')

# Login
login(token=HF_TOKEN)

# Create model repo (if doesn't exist)
api = HfApi()
try:
    api.create_repo(
        repo_id=f"{HF_USERNAME}/{MODEL_REPO}",
        repo_type="model",
        private=False,  # Set True if you want it private
    )
    print("✅ Created model repository")
except:
    print("ℹ️  Model repository already exists")

# Upload model
print(f"📦 Uploading model...")
api.upload_file(
    path_or_fileobj=MODEL_PATH,
    path_in_repo=os.path.basename(MODEL_PATH),
    repo_id=f"{HF_USERNAME}/{MODEL_REPO}",
    repo_type="model",
)

print("✅ Model uploaded to Model repository!")
print(f"   View: https://huggingface.co/{HF_USERNAME}/{MODEL_REPO}")
```

### Step 2: Update app.py to Download Model on Startup

The model will be downloaded automatically when the Space starts.

