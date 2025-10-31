"""
TheraBot - DBT Therapy Chatbot with RAG
Main Gradio app integrating llama.cpp, RAG retrieval, and DBT knowledge base.
"""

import os
import sys
import time
import subprocess
import gradio as gr
import requests
from typing import List, Tuple, Optional

# Import RAG system
sys.path.append(os.path.join(os.path.dirname(__file__), "rag"))
from hybrid_rag_retrieval import HybridRAGRetrieval
from build_rag_prompt import build_rag_prompt

def _persistent_root() -> Optional[str]:
    """Return persistent storage root if mounted (Spaces mount is /data or /persistent)."""
    for path in ("/data", "/persistent"):
        if os.path.exists(path) and os.access(path, os.W_OK):
            return path
    return None


def _ensure_rag_assets() -> None:
    """
    Ensure FAISS index and docstore exist locally. If RAG_ASSETS_REPO is set,
    download `index.faiss` and `docstore.json` from that repository.
    Prefer persistent storage when available to persist across restarts.
    """
    persistent = _persistent_root()
    target_dir = os.path.join(persistent, "rag") if persistent else "rag"
    os.makedirs(target_dir, exist_ok=True)
    index_path = os.path.join(target_dir, "index.faiss")
    store_path = os.path.join(target_dir, "docstore.json")

    # If both exist, nothing to do (mirror to ./rag if needed)
    if os.path.exists(index_path) and os.path.exists(store_path):
        print(f"✅ RAG assets detected at: {target_dir}")
        # If assets live in /persistent but code expects `rag/`, ensure symlinks/copies
        if target_dir != "rag":
            os.makedirs("rag", exist_ok=True)
            try:
                if not os.path.exists("rag/index.faiss"):
                    os.symlink(index_path, "rag/index.faiss")
                if not os.path.exists("rag/docstore.json"):
                    os.symlink(store_path, "rag/docstore.json")
            except Exception:
                # Fallback to copy on systems without symlink permissions
                if not os.path.exists("rag/index.faiss"):
                    subprocess.run(["cp", index_path, "rag/index.faiss"])  # best-effort
                if not os.path.exists("rag/docstore.json"):
                    subprocess.run(["cp", store_path, "rag/docstore.json"])  # best-effort
        return

    # If persistent storage is available but missing assets there,
    # and local ./rag has them, copy them into persistent to persist across restarts.
    if persistent:
        local_index = os.path.join("rag", "index.faiss")
        local_store = os.path.join("rag", "docstore.json")
        if os.path.exists(local_index) and os.path.exists(local_store):
            try:
                os.makedirs(target_dir, exist_ok=True)
                subprocess.run(["cp", local_index, index_path], check=False)
                subprocess.run(["cp", local_store, store_path], check=False)
                print(f"✅ RAG assets copied to persistent: {target_dir}")
            except Exception:
                pass
            # Ensure local references exist
            if target_dir != "rag":
                try:
                    if not os.path.exists("rag/index.faiss"):
                        os.symlink(index_path, "rag/index.faiss")
                    if not os.path.exists("rag/docstore.json"):
                        os.symlink(store_path, "rag/docstore.json")
                except Exception:
                    subprocess.run(["cp", index_path, "rag/index.faiss"], check=False)
                    subprocess.run(["cp", store_path, "rag/docstore.json"], check=False)
            return

    repo = os.getenv("RAG_ASSETS_REPO")
    if not repo:
        # No repo configured; let the caller decide to build
        return

    try:
        from huggingface_hub import hf_hub_download
        print(f"📥 Downloading RAG assets from {repo} → {target_dir}")
        hf_hub_download(repo_id=repo, filename="index.faiss", repo_type="model", local_dir=target_dir, local_dir_use_symlinks=False)
        hf_hub_download(repo_id=repo, filename="docstore.json", repo_type="model", local_dir=target_dir, local_dir_use_symlinks=False)

        # Mirror to rag/ for code that expects relative paths
        if target_dir != "rag":
            os.makedirs("rag", exist_ok=True)
            try:
                if not os.path.exists("rag/index.faiss"):
                    os.symlink(index_path, "rag/index.faiss")
                if not os.path.exists("rag/docstore.json"):
                    os.symlink(store_path, "rag/docstore.json")
            except Exception:
                subprocess.run(["cp", index_path, "rag/index.faiss"])  # best-effort
                subprocess.run(["cp", store_path, "rag/docstore.json"])  # best-effort

        print("✅ RAG assets ready")
    except Exception as e:
        print(f"⚠️  Failed to download RAG assets from {repo}: {e}")
        print("   Will attempt to build locally if needed.")


def _download_model_if_needed() -> Optional[str]:
    """
    Download model from HuggingFace Model repository if not found locally.
    Uses persistent storage if available.
    
    Returns:
        Path to model file, or None if download failed
    """
    model_filename = "llama3-3b.Q4_K_M.gguf"
    hf_model_repo = os.getenv("HF_MODEL_REPO", "NatanelRichey/therabot-llama3-3b-gguf")
    
    # Determine destination path (prefer persistent storage)
    persistent = _persistent_root()
    if persistent:
        dest_dir = os.path.join(persistent, "models")
        model_path = os.path.join(dest_dir, model_filename)
    else:
        dest_dir = "models"
        model_path = os.path.join(dest_dir, model_filename)
    
    os.makedirs(dest_dir, exist_ok=True)
    
    # Check if already downloaded
    if os.path.exists(model_path):
        print(f"✅ Model already exists at {model_path}")
        return model_path
    
    try:
        print(f"📥 Downloading model from {hf_model_repo}...")
        from huggingface_hub import hf_hub_download
        
        downloaded_path = hf_hub_download(
            repo_id=hf_model_repo,
            filename=model_filename,
            local_dir=dest_dir,
            local_dir_use_symlinks=False
        )
        
        print(f"✅ Model downloaded to {downloaded_path}")
        # Ensure a path under ./models exists for llama.cpp server if it expects it
        try:
            os.makedirs("models", exist_ok=True)
            link_path = os.path.join("models", model_filename)
            if not os.path.exists(link_path):
                try:
                    os.symlink(downloaded_path, link_path)
                except Exception:
                    subprocess.run(["cp", downloaded_path, link_path], check=False)
        except Exception:
            pass
        return downloaded_path
    except ImportError:
        print("⚠️  huggingface_hub not available. Install with: pip install huggingface_hub")
        return None
    except Exception as e:
        print(f"⚠️  Failed to download model: {e}")
        return None


class TheraBot:
    """Main chatbot class integrating RAG and llama.cpp."""
    
    def __init__(
        self,
        model_path: str = "models/llama3-3b.Q4_K_M.gguf",
        llama_server_url: str = "http://localhost:8000",
        rag_dir: str = "rag",
        enable_reranking: bool = True
    ):
        """Initialize TheraBot with model and RAG system."""
        self.model_path = model_path
        self.llama_server_url = llama_server_url
        
        # Initialize RAG retrieval system
        index_path = os.path.join(rag_dir, "index.faiss")
        docstore_path = os.path.join(rag_dir, "docstore.json")
        
        self.retriever = HybridRAGRetrieval(
            index_path=index_path,
            docstore_path=docstore_path,
            enable_reranking=enable_reranking
        )
        
        print("✅ TheraBot initialized with RAG system")
    
    def chat(
        self,
        user_message: str,
        history: List[Tuple[str, str]]
    ) -> Tuple[List[Tuple[str, str]], str]:
        """
        Process user message with RAG and generate response.
        
        Args:
            user_message: User's input
            history: Conversation history [(user, bot), ...]
        
        Returns:
            Updated history and empty string (for Gradio)
        """
        if not user_message.strip():
            return history, ""
        
        # Step 1: Retrieve DBT context with RAG
        print(f"\n🔍 Query: {user_message}")
        results, metadata = self.retriever.retrieve(user_message, k=5)
        chunks = self.retriever.format_results(results, metadata)
        
        # Log retrieval metadata
        path_emoji = "⚡" if metadata["path"] == "fast" else "🔍" if metadata["path"] == "detailed" else "🚫"
        print(f"   {path_emoji} RAG: {metadata['path']} path ({metadata['time_ms']:.1f}ms)")
        
        if metadata.get("detected_pillar"):
            pillar = metadata["detected_pillar"]
            conf = metadata.get("pillar_confidence", 0)
            print(f"   🎯 Pillar: {pillar} ({conf:.1f})")
        
        # Step 2: Build prompts with context
        system_msg, user_msg = build_rag_prompt(user_message, chunks)
        print(f"   📝 Prompts built ({len(system_msg)+len(user_msg)} chars)")
        
        # Step 3: Call llama.cpp server
        try:
            response = self._call_llama(system_msg, user_msg)
            print(f"   ✅ Response generated")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            response = "I apologize, but I'm having trouble generating a response right now. Please try again."
        
        # Step 4: Update history
        history.append((user_message, response))
        
        return history, ""
    
    def _call_llama(
        self,
        system_message: str,
        user_message: str,
        temperature: float = 0.3,
        max_tokens: int = 256
    ) -> str:
        """
        Call llama.cpp server for text generation.
        
        Args:
            system_message: System prompt
            user_message: User prompt with context
            temperature: Sampling temperature
            max_tokens: Max response length
        
        Returns:
            Generated response text
        """
        url = f"{self.llama_server_url}/v1/chat/completions"
        
        payload = {
            "model": "therabot",
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        # Try once, then retry on timeout with a longer timeout
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
        except requests.exceptions.ReadTimeout:
            print("   ⏳ Timeout at 60s; retrying with 180s timeout...")
            response = requests.post(url, json=payload, timeout=180)
            response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"]


def create_gradio_interface():
    """
    Create Gradio chat interface.
    Note: llama.cpp server should be started separately before calling this.
    """
    
    # Check if llama.cpp server is running
    server_check_url = "http://localhost:8000/v1/models"
    
    try:
        requests.get(server_check_url, timeout=2)
        print("✅ llama.cpp server detected")
    except:
        print("⚠️  WARNING: llama.cpp server not detected at http://localhost:8000")
        print("   The app may fail when processing messages")
    
    # Initialize bot
    bot = TheraBot()
    
    # Create Gradio interface
    with gr.Blocks(title="TheraBot - DBT Therapy Chatbot") as demo:
        gr.Markdown("# TheraBot 🤖")
        gr.Markdown("A DBT-informed therapeutic chatbot with RAG-enhanced responses.")
        
        chatbot = gr.Chatbot(
            label="Conversation",
            height=500,
            show_label=True,
            type="messages"
        )
        
        msg = gr.Textbox(
            label="Your Message",
            placeholder="Type your message here...",
            lines=2
        )
        
        with gr.Row():
            submit_btn = gr.Button("Send", variant="primary")
            clear_btn = gr.Button("Clear History")
        
        # Event handlers
        def respond(message, history):
            """Process message and return updated history."""
            # Convert messages format to tuples for bot.chat (legacy support)
            tuple_history = []
            if history:
                for h in history:
                    if isinstance(h, dict):
                        tuple_history.append((h["role"], h["content"]))
                    else:
                        tuple_history.append(h)
            
            new_tuple_history, _ = bot.chat(message, tuple_history)
            # Convert back to messages format - get the last exchange (user message + bot response)
            # bot.chat returns tuples as (user_message, bot_response) pairs
            messages_to_add = []
            if len(new_tuple_history) > len(tuple_history):
                # Get the newly added exchange (last entry)
                last_exchange = new_tuple_history[-1]
                if isinstance(last_exchange, tuple) and len(last_exchange) == 2:
                    user_msg, bot_msg = last_exchange
                    messages_to_add = [
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": bot_msg}
                    ]
            return history + messages_to_add, ""
        
        msg.submit(respond, [msg, chatbot], [chatbot, msg])
        submit_btn.click(respond, [msg, chatbot], [chatbot, msg])
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])
        
        # Examples
        gr.Examples(
            examples=[
                "I'm feeling overwhelmed today",
                "What are TIPP skills?",
                "Can you teach me mindfulness?",
                "I'm having trouble with my emotions"
            ],
            inputs=msg
        )
    
    return demo


if __name__ == "__main__":
    print("=" * 80)
    print("THERABOT - Starting Gradio Interface")
    print("=" * 80)
    
    # Ensure RAG assets; if not present, try to download via RAG_ASSETS_REPO; if still missing, build locally
    _ensure_rag_assets()

    # Optional: force rebuild via env var (e.g., set RAG_REBUILD=1 in Space)
    if os.getenv("RAG_REBUILD") == "1":
        print("♻️  RAG_REBUILD=1 set: rebuilding FAISS index and docstore...")
        try:
            subprocess.check_call([sys.executable, "rag/build_index.py"]) 
            print("✅ RAG index rebuilt successfully")
        except Exception as e:
            print(f"❌ Failed to rebuild RAG index: {e}")
        # Re-run ensure to persist/mirror
        _ensure_rag_assets()
    index_path = "rag/index.faiss"
    store_path = "rag/docstore.json"
    if not (os.path.exists(index_path) and os.path.exists(store_path)):
        print("⚠️  RAG assets still missing. Attempting local build now...")
        try:
            subprocess.check_call([sys.executable, "rag/build_index.py"]) 
            print("✅ RAG index built successfully")
        except Exception as e:
            print(f"❌ Failed to build RAG index automatically: {e}")
            print("   Please run: python rag/build_index.py")
            sys.exit(1)
    else:
        print("✅ RAG assets available; proceeding")
    
    # Auto-start llama.cpp server for HuggingFace Spaces
    # Check if server is already running
    server_check_url = "http://localhost:8000/v1/models"
    server_running = False
    
    try:
        requests.get(server_check_url, timeout=2)
        print("✅ llama.cpp server already running")
        server_running = True
    except:
        print("⚠️  llama.cpp server not detected, starting automatically...")
    
    # Start server if not running (always for HuggingFace Spaces)
    if not server_running:
        model_filename = "llama3-3b.Q4_K_M.gguf"
        
        # Check persistent storage first, then local, then download
        persistent_root = _persistent_root()
        persistent_path = os.path.join(persistent_root, "models", model_filename) if persistent_root else None
        local_path = f"models/{model_filename}"
        
        if persistent_path and os.path.exists(persistent_path):
            model_path = persistent_path
            print(f"✅ Found model in persistent storage: {model_path}")
        elif os.path.exists(local_path):
            model_path = local_path
            print(f"✅ Found model in local directory: {model_path}")
        else:
            # Download model to persistent storage (if available) or local directory
            print(f"📥 Model not found, downloading from HuggingFace Model repository...")
            
            # Configuration - Update with your model repo name
            model_repo = os.getenv("HF_MODEL_REPO", "NatanelRichey/therabot-llama3-3b-gguf")
            
            # Determine download destination (prefer persistent storage)
            if persistent_root:
                download_dir = os.path.join(persistent_root, "models")
                model_path = os.path.join(download_dir, model_filename)
                print(f"   Using persistent storage: {download_dir}")
            else:
                download_dir = "models"
                model_path = f"models/{model_filename}"
                print(f"   Using local directory: {download_dir}")
            
            os.makedirs(download_dir, exist_ok=True)
            
            try:
                from huggingface_hub import hf_hub_download
                
                print(f"   Repository: {model_repo}")
                print(f"   File: {model_filename}")
                print("   This may take 5-10 minutes for a 2-3 GB model...")
                
                downloaded_path = hf_hub_download(
                    repo_id=model_repo,
                    filename=model_filename,
                    local_dir=download_dir,
                    local_dir_use_symlinks=False,
                )
                
                model_path = downloaded_path
                print(f"✅ Model downloaded successfully to {model_path}!")
            except ImportError:
                print("❌ ERROR: huggingface_hub not installed")
                print("   Add to requirements.txt: huggingface_hub")
                sys.exit(1)
            except Exception as e:
                print(f"❌ ERROR: Failed to download model: {e}")
                print(f"   Model repo: {model_repo}")
                print("\n   Please ensure:")
                print("   1. Model is uploaded to a HuggingFace Model repository (not Space)")
                print("   2. Repository is public or you have access")
                print("   3. Set HF_MODEL_REPO environment variable in Space settings")
                print("      Example: HF_MODEL_REPO=NatanelRichey/therabot-llama3-3b-gguf")
                sys.exit(1)
        
        if not os.path.exists(model_path):
            print(f"❌ ERROR: Model not found at {model_path}")
            print("   Please ensure the model file is available")
            sys.exit(1)
        
        print(f"\n🚀 Starting llama.cpp server...")
        # Choose a sensible default for threads; allow override via env
        threads = os.getenv("LLAMA_THREADS") or str(max(2, min(8, (os.cpu_count() or 2))))
        server_cmd = [
            sys.executable, "-m", "llama_cpp.server",
            "--model", model_path,
            "--host", "0.0.0.0",
            "--port", "8000",
            "--n_ctx", "2048",
            "--chat_format", "llama-3",
            "--n_threads", threads
        ]
        print(f"   Command: {' '.join(server_cmd)}")
        
        try:
            # Start server in background
            # Stream server logs directly to Space logs for debugging
            process = subprocess.Popen(server_cmd)
            print("   Server starting in background...")
            
            # Wait for server to be ready (up to 5 minutes on cold start)
            max_wait = 300
            for i in range(max_wait):
                time.sleep(1)
                try:
                    # Try multiple readiness endpoints
                    ok = False
                    for url in [
                        server_check_url,
                        "http://localhost:8000/health",
                        "http://localhost:8000/v1/models"
                    ]:
                        try:
                            r = requests.get(url, timeout=1)
                            if r.status_code < 500:
                                ok = True
                                break
                        except Exception:
                            pass
                    if not ok:
                        raise RuntimeError("not ready")
                    print(f"✅ llama.cpp server ready! (took {i+1}s)")
                    server_running = True
                    break
                except:
                    if i % 10 == 0:
                        print(f"   Waiting for server... ({i+1}s)")
            
            if not server_running:
                print("❌ WARNING: Server may not be ready yet, continuing anyway...")
                print("   The app will continue but may fail when processing messages")
        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            print("   The app will continue but may fail when processing messages")

    # Optional warm-up request to reduce first-token latency
    try:
        print("🔥 Sending warm-up request to llama server...")
        warmup_payload = {
            "model": "therabot",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "hi"}
            ],
            "temperature": 0.0,
            "max_tokens": 1,
            "stream": False
        }
        requests.post("http://localhost:8000/v1/chat/completions", json=warmup_payload, timeout=30)
        print("✅ Warm-up completed")
    except Exception as e:
        print(f"⚠️  Warm-up skipped: {e}")
    
    # Create and launch interface
    demo = create_gradio_interface()
    
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False
    )

