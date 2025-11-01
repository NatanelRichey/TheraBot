---
title: TheraBot - DBT Therapy Chatbot
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# TheraBot - DBT Therapy Chatbot

A compassionate DBT-informed therapeutic chatbot with RAG-enhanced responses, built on Llama 3.1 3B with the official DBT manual knowledge base.

## Features

- 🧠 **DBT-Informed Responses** - Trained on therapy data with DBT principles
- 🔍 **RAG-Enhanced** - Retrieves relevant techniques from DBT manual
- ⚡ **Hybrid Retrieval** - Fast path (20-40ms) or detailed path (2-3s)
- 🎯 **Pillar Detection** - Automatically routes to relevant DBT modules:
  - 🧠 Mindfulness
  - 🛡️ Distress Tolerance
  - 💭 Emotion Regulation
  - 🤝 Interpersonal Effectiveness
- 🚀 **Smart & Fast** - Skips RAG for casual queries, uses it for therapeutic needs

## How It Works

1. **User sends message** → Intent detection
2. **Smart RAG retrieval** → Fast or detailed based on complexity
3. **Pillar detection** → Routes to relevant DBT modules
4. **Prompt building** → Injects context when helpful
5. **Model generation** → Compassionate DBT-informed response

## Usage

Simply type your message in the chat interface.

**Example queries:**
- "I'm feeling overwhelmed"
- "What are TIPP skills?"
- "Can you teach me mindfulness?"
- "How do I handle relationship conflicts?"

## Technical Details

**Model:** Llama 3.1 3B Instruct (Q4_K_M quantized)  
**RAG:** FAISS + E5-small-v2 embeddings  
**Knowledge Base:** 108 DBT manual chunks  
**Infrastructure:** llama.cpp + Gradio on HuggingFace Spaces

## Deployment Options

### HuggingFace Spaces
The app is deployed on HuggingFace Spaces with automatic GPU support:
- Visit the Space URL to use the chatbot
- No setup required

### Google Colab (GPU Runtime)
You can also run TheraBot on Google Colab with GPU acceleration:

1. **Prepare files in Google Drive:**
   - Upload `TheraBot_Training/models/llama3-3b.Q4_K_M.gguf` to your Drive
   - Upload `TheraBot_Training/rag/` folder (index.faiss, docstore.json)
   - Upload/clone this repository code to Colab

2. **Open the notebook:**
   - Open `TheraBot_Gradio_Deployment.ipynb` in Colab

3. **Set up GPU:**
   - Go to: Runtime → Change runtime type
   - Select: GPU (T4 or better recommended)
   - Click: Save

4. **Run all cells:**
   - The notebook will automatically:
     - Mount Google Drive
     - Copy model and RAG assets (much faster than downloading!)
     - Install dependencies
     - Start llama.cpp server with GPU offloading
     - Launch Gradio interface

5. **Access the app:**
   - Gradio provides a public URL
   - Share the link to use the chatbot

**Benefits of Colab deployment:**
- ✅ Much faster file access from Google Drive (30-60s vs 5-10min download)
- ✅ Faster inference with GPU acceleration (3-5x speedup)
- ✅ Full control over configuration
- ✅ Easy experimentation with settings
- ✅ Free GPU access (with usage limits)

## Important

This is a therapeutic tool, not a replacement for professional therapy.  
For crisis situations, contact a mental health professional or crisis hotline immediately.
