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

## Important

This is a therapeutic tool, not a replacement for professional therapy.  
For crisis situations, contact a mental health professional or crisis hotline immediately.
