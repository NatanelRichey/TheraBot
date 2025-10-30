# TheraBot Training & DBT Integration

## 🎯 Overview

You have a **trained LoRA adapter** with general therapy conversation skills. This directory contains everything you need to inject DBT (Dialectical Behavior Therapy) knowledge into your model.

---

## 📚 Complete Documentation

### **Start Here:**
- **[GET_STARTED.md](GET_STARTED.md)** - Quick start guide, choose your approach
- **[NEXT_STEPS.md](NEXT_STEPS.md)** - Detailed implementation guide
- **[OPTIONS_SUMMARY.md](OPTIONS_SUMMARY.md)** - Complete comparison of all approaches

### **Deep Dives:**
- **[DBT_INJECTION_GUIDE.md](DBT_INJECTION_GUIDE.md)** - Technical DBT integration guide
- **[RAG_IMPLEMENTATION_GUIDE.md](RAG_IMPLEMENTATION_GUIDE.md)** - RAG system deep dive

---

## 🚀 Your Options

### Option 1: System Prompts (30 min) ⚡

**Files:**
- `utils/dbt_system_prompts.py` - DBT prompt templates
- `utils/load_adapter_for_inference.py` - Inference engine
- `utils/quick_start_inference.py` - **Run this!**

**Quick Start:**
```bash
python utils/quick_start_inference.py
```

**Best for:** Quick testing, learning, demos

---

### Option 2: RAG System (2-4 hours) ⭐ **RECOMMENDED**

**Files:**
- `utils/dbt_rag_system.py` - Complete RAG implementation
- `utils/authenticate_and_analyze.py` - Dataset analyzer
- `utils/analyze_dbt_dataset.py` - Dataset stats

**Quick Start:**
```bash
# Step 1: Authenticate
python utils/authenticate_and_analyze.py

# Step 2: Run RAG system
python utils/dbt_rag_system.py
```

**Best for:** Production, best quality, comprehensive DBT knowledge

---

### Option 3: Additional Fine-Tuning (4-8 hours)

**Status:** Not yet implemented (can create if needed)

**Best for:** Maximum consistency, deepest integration

---

## 📁 Directory Structure

```
training/
├── GET_STARTED.md                  # Start here!
├── NEXT_STEPS.md                   # Implementation guide
├── OPTIONS_SUMMARY.md              # Complete comparison
├── DBT_INJECTION_GUIDE.md          # Technical guide
├── RAG_IMPLEMENTATION_GUIDE.md     # RAG deep dive
│
└── utils/
    ├── dbt_system_prompts.py       # DBT prompts
    ├── load_adapter_for_inference.py  # Inference engine
    ├── quick_start_inference.py    # Interactive chat
    ├── dbt_rag_system.py           # RAG implementation
    ├── authenticate_and_analyze.py # HF auth & analysis
    └── analyze_dbt_dataset.py      # Dataset stats
```

---

## 🎯 Quick Decision Guide

**Choose based on your goal:**

| Goal | Recommended Approach | Time | File to Run |
|------|---------------------|------|-------------|
| Quick demo | System Prompts | 30 min | `quick_start_inference.py` |
| Best quality | RAG ⭐ | 2-4 hours | `dbt_rag_system.py` |
| Maximum integration | Fine-Tuning | 4-8 hours | Contact me |

---

## 📋 Checklist

### Prerequisites
- [x] Trained LoRA adapter
- [x] Access to `vjain/dbt` dataset
- [ ] Chosen approach (System Prompts, RAG, or Fine-Tuning)

### System Prompts Setup
- [ ] Install dependencies: `pip install torch transformers peft bitsandbytes`
- [ ] Update adapter path in `quick_start_inference.py`
- [ ] Run: `python utils/quick_start_inference.py`
- [ ] Test conversations
- [ ] Adjust prompts if needed

### RAG Setup
- [ ] Install: `pip install sentence-transformers chromadb datasets`
- [ ] Authenticate with HuggingFace
- [ ] Run: `python utils/authenticate_and_analyze.py`
- [ ] Update `dbt_rag_system.py` column names
- [ ] Run: `python utils/dbt_rag_system.py`
- [ ] Test retrieval quality
- [ ] Compare with system prompts

---

## 🔗 Key Resources

### External
- [vjain/dbt Dataset](https://huggingface.co/datasets/vjain/dbt) - DBT manual
- [HuggingFace PEFT Docs](https://huggingface.co/docs/peft/) - LoRA documentation
- [Sentence Transformers](https://www.sbert.net/) - Embeddings
- [ChromaDB](https://www.trychroma.com/) - Vector database

### Internal
- `data/dbt_knowledge.md` - DBT knowledge base
- `THERABOT_PLAN` - Complete project plan
- `backend/README.md` - Backend architecture

---

## 🆘 Troubleshooting

**Common Issues:**

1. **"Dataset not found"** → Run `authenticate_and_analyze.py` first
2. **"Out of memory"** → Already optimized with 4-bit, contact for more help
3. **"Column not found"** → Check dataset columns with analysis script
4. **"Poor responses"** → Adjust temperature, prompts, or chunk size

**Need help?** Check specific guide files or ask!

---

## 🎉 What You've Built

✅ Trained LoRA adapter on 173K therapy conversations  
✅ DBT knowledge base (manual dataset)  
✅ Complete inference pipeline  
✅ RAG system ready to use  
✅ Multiple integration approaches  

**Next:** Choose an approach and deploy! 🚀

---

## 📞 Next Steps

1. **Read:** [GET_STARTED.md](GET_STARTED.md)
2. **Choose:** Your approach (I recommend RAG ⭐)
3. **Setup:** Follow the steps
4. **Test:** Try it out
5. **Deploy:** Integrate into backend
6. **Share:** Your TheraBot!

**Questions?** All guides are comprehensive, but I'm here to help!
