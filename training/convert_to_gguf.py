"""
Convert trained model to GGUF Q4_K_M format for deployment.
Run this in Google Colab or local machine with CUDA.
"""

# =============================================================================
# CELL 1: Install llama.cpp tools
# =============================================================================

def install_llama_cpp():
    """Install llama.cpp and build tools."""
    import subprocess
    import os
    
    print("Installing llama.cpp...")
    subprocess.run(["pip", "install", "-q", "llama-cpp-python"], check=True)
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp.git"], check=True)
    
    os.chdir("llama.cpp")
    subprocess.run(["make"], check=True)
    os.chdir("..")
    print("✅ llama.cpp installed")

# =============================================================================
# CELL 2: Convert HuggingFace Model to GGUF
# =============================================================================

def convert_to_gguf(hf_model_path: str, output_dir: str = "gguf_models", model_name: str = "llama3-3b"):
    """Convert HF model to GGUF format."""
    import subprocess
    import os
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run conversion
    cmd = [
        "python", "llama.cpp/convert-hf-to-gguf.py",
        hf_model_path,
        "--outdir", output_dir,
        "--outfile", f"{model_name}.gguf"
    ]
    
    print(f"Converting {hf_model_path} to GGUF...")
    subprocess.run(cmd, check=True)
    print(f"✅ GGUF model created: {output_dir}/{model_name}.gguf")

# =============================================================================
# CELL 3: Quantize to Q4_K_M
# =============================================================================

def quantize_to_q4km(gguf_path: str, output_path: str):
    """Quantize GGUF model to Q4_K_M."""
    import subprocess
    
    print(f"Quantizing to Q4_K_M...")
    cmd = ["./llama.cpp/quantize", gguf_path, output_path, "Q4_K_M"]
    subprocess.run(cmd, check=True)
    print(f"✅ Quantized model: {output_path}")

# =============================================================================
# CELL 4: Download or Upload
# =============================================================================

def download_model(model_path: str):
    """Download model from Colab."""
    try:
        from google.colab import files
        files.download(model_path)
        print("✅ Model downloaded")
    except ImportError:
        print("Not in Colab - model saved locally")

def upload_to_hub(model_path: str, repo_id: str):
    """Upload to HuggingFace Hub."""
    from huggingface_hub import login, upload_file
    
    login()  # Enter your HF token
    
    filename = os.path.basename(model_path)
    upload_file(
        path_or_fileobj=model_path,
        path_in_repo=filename,
        repo_id=repo_id
    )
    print(f"✅ Model uploaded to {repo_id}")

# =============================================================================
# MAIN: Complete Conversion Pipeline
# =============================================================================

def convert_pipeline(hf_model_path: str, output_dir: str = "gguf_models", model_name: str = "llama3-3b", upload_repo: str = None):
    """
    Complete conversion pipeline.
    
    Args:
        hf_model_path: Path to HuggingFace model (merged if using LoRA)
        output_dir: Directory to save GGUF models
        model_name: Name for output files
        upload_repo: Optional HF repo_id to upload to
    """
    
    # Step 1: Install
    install_llama_cpp()
    
    # Step 2: Convert to GGUF
    convert_to_gguf(hf_model_path, output_dir, model_name)
    
    # Step 3: Quantize
    gguf_path = f"{output_dir}/{model_name}.gguf"
    quantized_path = f"{output_dir}/{model_name}.Q4_K_M.gguf"
    quantize_to_q4km(gguf_path, quantized_path)
    
    # Step 4: Download or Upload
    try:
        download_model(quantized_path)
    except ImportError:
        pass
    
    if upload_repo:
        upload_to_hub(quantized_path, upload_repo)
    
    print(f"\n🎉 Conversion complete!")
    print(f"Model ready: {quantized_path}")
    
    return quantized_path


# =============================================================================
# USAGE
# =============================================================================

if __name__ == "__main__":
    # UPDATE THESE PATHS
    HF_MODEL = "your-checkpoint-path"  # Your trained model or merged model
    OUTPUT_DIR = "gguf_models"
    MODEL_NAME = "llama3-3b"
    
    # Optional: Upload to Hub
    # UPLOAD_REPO = "your-username/llama3-3b-dbt-therabot"
    UPLOAD_REPO = None
    
    # Run conversion
    convert_pipeline(
        hf_model_path=HF_MODEL,
        output_dir=OUTPUT_DIR,
        model_name=MODEL_NAME,
        upload_repo=UPLOAD_REPO
    )

