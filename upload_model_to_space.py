"""
Upload GGUF model file from Google Colab to HuggingFace Space
Run this in a Colab notebook cell
"""

import os
import subprocess
import sys

def upload_model_via_git(model_path, space_repo, username, token):
    """
    Upload model file using git.
    
    Args:
        model_path: Path to the GGUF file in Colab
        space_repo: Space name (e.g., "TheraBot")
        username: Your HuggingFace username
        token: Your HuggingFace token (with write permissions)
    """
    print("=" * 80)
    print("Uploading Model to HuggingFace Space via Git")
    print("=" * 80)
    
    space_url = f"https://{username}:{token}@huggingface.co/spaces/{username}/{space_repo}"
    
    # Clone the space
    clone_dir = f"/tmp/{space_repo}"
    if os.path.exists(clone_dir):
        print(f"⚠️  Removing existing clone directory...")
        subprocess.run(["rm", "-rf", clone_dir], check=True)
    
    print(f"📥 Cloning space repository...")
    subprocess.run(["git", "clone", space_url, clone_dir], check=True)
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join(clone_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Copy model file
    model_filename = os.path.basename(model_path)
    dest_path = os.path.join(models_dir, model_filename)
    
    print(f"📦 Copying model file...")
    print(f"   From: {model_path}")
    print(f"   To:   {dest_path}")
    
    subprocess.run(["cp", model_path, dest_path], check=True)
    
    # Commit and push
    print(f"💾 Committing and pushing...")
    os.chdir(clone_dir)
    
    subprocess.run(["git", "config", "user.name", username], check=True)
    subprocess.run(["git", "config", "user.email", f"{username}@users.noreply.huggingface.co"], check=True)
    
    subprocess.run(["git", "add", "models/"], check=True)
    subprocess.run(["git", "commit", "-m", f"Add model file: {model_filename}"], check=True)
    subprocess.run(["git", "push"], check=True)
    
    print("=" * 80)
    print("✅ Model uploaded successfully!")
    print(f"   Space URL: https://huggingface.co/spaces/{username}/{space_repo}")
    print("=" * 80)


def upload_model_via_hub(model_path, space_repo, username, token):
    """
    Upload model file using HuggingFace Hub library.
    
    Args:
        model_path: Path to the GGUF file in Colab
        space_repo: Space name (e.g., "TheraBot")
        username: Your HuggingFace username
        token: Your HuggingFace token (with write permissions)
    """
    try:
        from huggingface_hub import HfApi, login
    except ImportError:
        print("❌ huggingface_hub not installed. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)
        from huggingface_hub import HfApi, login
    
    print("=" * 80)
    print("Uploading Model to HuggingFace Space via Hub")
    print("=" * 80)
    
    # Login
    print("🔐 Logging in to HuggingFace...")
    login(token=token)
    
    # Upload file
    api = HfApi()
    model_filename = os.path.basename(model_path)
    repo_id = f"{username}/{space_repo}"
    repo_type = "space"
    
    print(f"📦 Uploading model file...")
    print(f"   File: {model_path}")
    print(f"   To:   {repo_id}/models/{model_filename}")
    
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=f"models/{model_filename}",
        repo_id=repo_id,
        repo_type=repo_type,
    )
    
    print("=" * 80)
    print("✅ Model uploaded successfully!")
    print(f"   Space URL: https://huggingface.co/spaces/{username}/{space_repo}")
    print("=" * 80)


if __name__ == "__main__":
    # =============================================================================
    # CONFIGURATION - EDIT THESE VALUES
    # =============================================================================
    
    # Path to your GGUF model file in Colab
    MODEL_PATH = "/content/llama3-3b.Q4_K_M.gguf"  # ⬅️ EDIT THIS
    
    # HuggingFace Space details
    HF_USERNAME = "NatanelRichey"  # ⬅️ EDIT IF NEEDED
    SPACE_NAME = "TheraBot"  # ⬅️ EDIT IF NEEDED
    
    # HuggingFace token (get from https://huggingface.co/settings/tokens)
    # For security, paste it when prompted or use Colab secrets
    HF_TOKEN = ""  # ⬅️ PASTE YOUR TOKEN HERE or use: os.environ["HF_TOKEN"]
    
    # =============================================================================
    # UPLOAD METHOD - Choose one
    # =============================================================================
    
    # Method 1: Using Git (recommended for large files)
    USE_GIT = True
    
    # Method 2: Using HuggingFace Hub (simpler, but may be slower for large files)
    # USE_GIT = False
    
    # =============================================================================
    # EXECUTION
    # =============================================================================
    
    # Get token from environment or prompt
    if not HF_TOKEN:
        if "HF_TOKEN" in os.environ:
            HF_TOKEN = os.environ["HF_TOKEN"]
        else:
            print("⚠️  HF_TOKEN not set. Please set it above or use:")
            print("   from google.colab import userdata")
            print("   HF_TOKEN = userdata.get('HF_TOKEN')")
            sys.exit(1)
    
    # Verify model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: Model file not found: {MODEL_PATH}")
        print("\nPlease check:")
        print("  1. The model file path is correct")
        print("  2. The file has been generated/converted")
        sys.exit(1)
    
    file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)  # MB
    print(f"📊 Model file size: {file_size:.1f} MB")
    
    # Choose upload method
    if USE_GIT:
        # Make sure git is available
        try:
            subprocess.run(["git", "--version"], check=True, capture_output=True)
        except:
            print("❌ Git not available. Install with: !apt-get install git -y")
            sys.exit(1)
        
        upload_model_via_git(MODEL_PATH, SPACE_NAME, HF_USERNAME, HF_TOKEN)
    else:
        upload_model_via_hub(MODEL_PATH, SPACE_NAME, HF_USERNAME, HF_TOKEN)

