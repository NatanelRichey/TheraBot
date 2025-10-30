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
from typing import List, Tuple

# Import RAG system
sys.path.append(os.path.join(os.path.dirname(__file__), "rag"))
from hybrid_rag_retrieval import HybridRAGRetrieval
from build_rag_prompt import build_rag_prompt


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
        max_tokens: int = 1024
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
        
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]


def create_gradio_interface(auto_start_server: bool = False):
    """
    Create Gradio chat interface.
    
    Args:
        auto_start_server: If True, automatically start llama.cpp server
    """
    
    # Check if llama.cpp server is running
    server_check_url = "http://localhost:8000/v1/models"
    server_running = False
    
    try:
        requests.get(server_check_url, timeout=2)
        print("✅ llama.cpp server detected")
        server_running = True
    except:
        print("⚠️  llama.cpp server not detected at http://localhost:8000")
    
    # Auto-start server if requested (for local testing)
    if not server_running and auto_start_server:
        print("\n🚀 Starting llama.cpp server automatically...")
        model_path = "models/llama3-3b.Q4_K_M.gguf"
        
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            print("   Skipping auto-start")
        else:
            server_cmd = [
                sys.executable, "-m", "llama_cpp.server",
                "--model", model_path,
                "--host", "0.0.0.0",
                "--port", "8000",
                "--n_ctx", "4096",
                "--chat_format", "llama-3"
            ]
            print(f"   Command: {' '.join(server_cmd)}")
            
            try:
                subprocess.Popen(server_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print("   Server starting in background...")
                time.sleep(3)  # Wait for server to start
                print("✅ Server should be ready")
            except Exception as e:
                print(f"❌ Failed to start server: {e}")
    
    # Initialize bot
    bot = TheraBot()
    
    # Create Gradio interface
    with gr.Blocks(title="TheraBot - DBT Therapy Chatbot") as demo:
        gr.Markdown("# TheraBot 🤖")
        gr.Markdown("A DBT-informed therapeutic chatbot with RAG-enhanced responses.")
        
        chatbot = gr.Chatbot(
            label="Conversation",
            height=500,
            show_label=True
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
            new_history, _ = bot.chat(message, history)
            return new_history, ""
        
        msg.submit(respond, [msg, chatbot], [chatbot, msg])
        submit_btn.click(respond, [msg, chatbot], [chatbot, msg])
        clear_btn.click(lambda: ([], ""), None, [chatbot, msg])
        
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
    
    # Check if index files exist
    if not os.path.exists("rag/index.faiss"):
        print("❌ ERROR: rag/index.faiss not found!")
        print("   Run: python rag/build_index.py")
        sys.exit(1)
    
    if not os.path.exists("rag/docstore.json"):
        print("❌ ERROR: rag/docstore.json not found!")
        print("   Run: python rag/build_index.py")
        sys.exit(1)
    
    # Create and launch interface
    # Set AUTO_START_SERVER=1 to auto-start llama.cpp server (local only)
    auto_start = os.getenv("AUTO_START_SERVER", "0") == "1"
    demo = create_gradio_interface(auto_start_server=auto_start)
    
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False
    )

