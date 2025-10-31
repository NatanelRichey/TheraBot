# Upload Model to Persistent Storage in HuggingFace Space

## Important: Persistent Storage is NOT the Git Repo

When you add persistent storage to a Space, it's mounted at `/persistent/` in the running container, but **you cannot upload files to it via Git**. 

## Solution: Upload via Space Runtime (Terminal Access)

Since persistent storage can't be accessed via git push, you need to upload the file **after the Space is running** using one of these methods:

### Method 1: Upload via Space Terminal (Easiest)

1. Go to your Space: https://huggingface.co/spaces/NatanelRichey/TheraBot
2. Click the **"..."** menu (top right) → **"Duplicate Space"** or open Terminal
3. Go to **Settings** → Enable **"Web Terminal"** (if available)
4. Or use **"Files and versions"** → Upload via web interface if possible

**However, if terminal access isn't available, use Method 2:**

### Method 2: Download in app.py on First Run (Recommended)

The app already has download logic. Just upload your model to a **Model repository** (not Space), and the app will download it to persistent storage on first run.

**In Colab, upload to Model repo:**

```python
!pip install huggingface_hub

from huggingface_hub import HfApi, login
import os

# Configuration
MODEL_PATH = "/content/llama3-3b.Q4_K_M.gguf"
HF_USERNAME = "NatanelRichey"
MODEL_REPO = "therabot-llama3-3b-gguf"  # Different from Space name

# Get token
from google.colab import userdata
HF_TOKEN = userdata.get('HF_TOKEN')

# Login and upload
login(token=HF_TOKEN)
api = HfApi()

# Create repo if needed
try:
    api.create_repo(
        repo_id=f"{HF_USERNAME}/{MODEL_REPO}",
        repo_type="model",
        private=False,
    )
except:
    pass

# Upload
api.upload_file(
    path_or_fileobj=MODEL_PATH,
    path_in_repo=os.path.basename(MODEL_PATH),
    repo_id=f"{HF_USERNAME}/{MODEL_REPO}",
    repo_type="model",
)

print(f"✅ Uploaded to: https://huggingface.co/{HF_USERNAME}/{MODEL_REPO}")
```

### Method 3: Use HuggingFace CLI from Local Machine

If you have the model file locally:

```bash
# Install HuggingFace CLI
pip install huggingface-cli

# Login
huggingface-cli login

# Upload to Model repo
huggingface-cli upload NatanelRichey/therabot-llama3-3b-gguf llama3-3b.Q4_K_M.gguf --repo-type model
```

## Then Update Your Space

1. Set environment variable in Space Settings → Variables:
   - `HF_MODEL_REPO=NatanelRichey/therabot-llama3-3b-gguf`

2. The app will automatically download to `/persistent/models/` on first run

## Verify Upload

After the Space starts, check the logs - you should see:
```
📥 Downloading model from NatanelRichey/therabot-llama3-3b-gguf...
✅ Model downloaded to /persistent/models/llama3-3b.Q4_K_M.gguf
```

The model will persist in `/persistent/models/` across restarts!

