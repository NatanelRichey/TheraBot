# How to Deploy TheraBot to HuggingFace Space

## Prerequisites

✅ You have:
- Trained 3B model
- Model converted to GGUF (Q4_K_M)
- All RAG files ready

---

## Step 1: Prepare Repository

### Files to Push to GitHub

```
your-repo/
├── app.py                      ← Main app
├── README.md                   ← Space metadata
├── requirements.txt            ← Copy from requirements_deploy.txt
│
├── rag/
│   ├── index.faiss
│   ├── docstore.json
│   ├── hybrid_rag_retrieval.py
│   └── build_rag_prompt.py
│
└── models/
    └── llama3-3b.Q4_K_M.gguf  ← Your trained model
```

### Prepare requirements.txt

For Space deployment, rename:
```bash
cp requirements_deploy.txt requirements.txt
```

---

## Step 2: Create HuggingFace Space

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Fill in:
   - **Name:** `therabot-dbt-chatbot`
   - **SDK:** `Gradio`
   - **Hardware:** `CPU Basic` (for 3B model) or `T4 Small` (if you upgrade)
   - **Visibility:** Public or Private

---

## Step 3: Configure Space

### Option A: Auto llama.cpp Server (Recommended)

**Space Settings → App → Start command:**
```bash
python -m llama_cpp.server --model models/llama3-3b.Q4_K_M.gguf --host 0.0.0.0 --port 8000 --n_ctx 4096 --chat_format llama-3
```

**Important:** Set this BEFORE uploading files!

### Option B: Manual llama.cpp Start

Start llama.cpp separately, set `LLAMA_SERVER_URL` in Space env vars.

---

## Step 4: Upload Files

### Method 1: Git Push
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/therabot-dbt-chatbot
git push
```

### Method 2: Web Upload
- Upload files via Spaces UI
- Drag and drop `app.py`, `requirements.txt`, README.md
- Upload entire `rag/` folder
- Upload entire `models/` folder

---

## Step 5: Wait for Build

Space will:
1. Install dependencies from `requirements.txt`
2. Start llama.cpp server (if configured)
3. Load Gradio app
4. Start serving

**Build time:** ~5-10 minutes

---

## Step 6: Test Your Space

Visit: `https://huggingface.co/spaces/YOUR_USERNAME/therabot-dbt-chatbot`

### Test Queries
1. "hi how are you" - Should skip RAG
2. "I'm feeling overwhelmed" - Should use RAG
3. "what is mindfulness" - Should detect 🧠 pillar
4. "I had a bad day" - Should get distress tolerance chunks

---

## Troubleshooting

### Issue: "llama.cpp server not detected"
**Solution:** Check Space logs, ensure llama.cpp started properly

### Issue: "RAG files not found"
**Solution:** Verify `rag/` folder uploaded with `index.faiss` and `docstore.json`

### Issue: "Module not found"
**Solution:** Check `requirements.txt` has all dependencies

### Issue: Slow responses
**Solutions:**
- Check CPU usage in Space logs
- Consider upgrading to T4 GPU
- Reduce `max_tokens` in app.py

### Issue: Model OOM
**Solutions:**
- Try Q3_K_M quantization instead of Q4_K_M
- Use smaller context window (`--n_ctx 2048`)
- Upgrade to paid tier with more RAM

---

## Cost Estimation

**Free Tier (CPU Basic):**
- ✅ 3B model should work
- ⚠️ Slow response times (10-30s)
- ✅ Unlimited usage

**Starter Tier ($5/month):**
- ✅ 2 vCPU, 16GB RAM
- ✅ Much faster responses

**Standard Tier ($11/month):**
- ✅ 4 vCPU, 32GB RAM  
- ✅ Can handle larger models if you upgrade

---

## Environment Variables

Optional env vars you can set in Space:

```bash
AUTO_START_SERVER=0          # Don't auto-start (server started separately)
LLAMA_SERVER_URL=http://localhost:8000  # Change if remote server
```

---

## Success Criteria

✅ Chat interface loads  
✅ Can send messages  
✅ Receives responses  
✅ RAG retrieval works (check logs)  
✅ Pillar detection working (check logs)  
✅ Response times < 30s  

---

**You're live!** 🎉

