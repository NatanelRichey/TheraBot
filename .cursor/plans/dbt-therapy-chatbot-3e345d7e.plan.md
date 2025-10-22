<!-- 3e345d7e-feda-46fc-b49e-34758532a4dc e8622f0b-6438-4fa4-99fc-c1c01c4a8f07 -->
# DBT Therapy Chatbot Implementation Plan

## Technology Stack

- **Model**: Llama 3.1 8B Instruct (open-source, good quality, free)
- **Fine-tuning**: LoRA (Parameter Efficient) via HuggingFace PEFT
- **Training**: Google Colab (free tier initially, Pro if needed $10/month)
- **Backend**: FastAPI (Python) + HuggingFace Inference API
- **Frontend**: Next.js + Tailwind CSS
- **Deployment**: Vercel (frontend, free) + HuggingFace Spaces (backend, free)

## Phase 1: Environment Setup & Learning (Day 1)

### Setup Development Environment

```bash
# Local project structure
TheraBot/
├── data/              # Training data preparation
├── training/          # Fine-tuning notebooks
├── backend/           # FastAPI server
└── frontend/          # Next.js app
```

**Key files to create:**

- `requirements.txt` - Python dependencies (transformers, datasets, peft, torch, fastapi)
- `training/finetune_notebook.ipynb` - Colab notebook for training
- `data/dbt_conversations.json` - Training dataset

### Learning Resources

1. **Fine-tuning basics**: HuggingFace Course (Chapter 3) - https://huggingface.co/learn/nlp-course
2. **LoRA/PEFT**: "Parameter-Efficient Fine-Tuning" guide
3. **DBT Protocol**: Review your workbook for core techniques (distress tolerance, emotion regulation, interpersonal effectiveness, mindfulness)

## Phase 2: Data Preparation (Days 1-2)

### Current Dataset Status

**Available Data**: 171,623 dialogue exchanges from 359 therapy sessions

- **Well-labeled files**: 359 sessions with therapy approaches, clinical topics, demographics
- **Unknown files**: 100 sessions (can be excluded for initial training)
- **Therapy approaches**: CBT, psychoanalytic, client-centered, family systems, group, EMDR, etc.
- **Clinical topics**: anxiety, depression, relationships, work stress, self-esteem, etc.

### Data Quality & Labeling Strategy

**AI-Generated Labels**: The therapy approach and clinical topic labels were automatically generated using keyword matching and pattern recognition.

**Validation Approach**:

- Manually verify 20-30 random files to check label accuracy
- If accuracy >80%, use labels confidently
- If accuracy <80%, treat as "soft labels" for filtering only

**Conservative Usage**:

- Use labels for **data filtering and organization** (not as training targets)
- Don't train model to predict therapy approaches
- Focus on therapeutic response quality rather than label accuracy
- Treat clinical topics as context, not hard categories

### Training Data Selection

**Recommended approach**:

1. Start with 359 well-labeled files (171k+ exchanges)
2. Filter by therapy approach for specialized training:

   - CBT sessions for cognitive-behavioral responses
   - Client-centered sessions for empathetic responses
   - Family systems for relationship dynamics

3. Use clinical topics for context-aware training
4. Exclude 100 unknown files initially (can add later if needed)

### Create Training Dataset Format

Structure conversations in `data/dbt_conversations.json`:

```json
[
  {
    "instruction": "You are a DBT-trained therapeutic chatbot...",
    "input": "User message showing distress",
    "output": "Empathetic, DBT-guided response"
  }
]
```

**Target**: 100-300 high-quality examples minimum

- 50-100 from workbook scenarios
- Additional from DBT resources/websites if available
- If needed, use GPT-4 to help structure workbook content into conversations (cheaper than full generation)

### Data Categories to Cover

- Emotional distress scenarios
- Skill teaching (TIPP, PLEASE, DEAR MAN)
- Validation and empathy
- Crisis situations (with appropriate disclaimers)
- Daily check-ins and mood tracking

## Phase 3: Model Fine-Tuning (Days 3-4)

### Training Notebook (`training/finetune_notebook.ipynb`)

**Run in Google Colab (GPU runtime)**

```python
# Key components:
1. Load base model (Llama-3.1-8B-Instruct)
2. Configure LoRA (rank=8, alpha=16, dropout=0.1)
3. Setup training arguments (2-3 epochs, ~2 hours)
4. Train with monitored loss
5. Save adapter weights to HuggingFace Hub
```

**Important settings:**

- Use 4-bit quantization (reduces memory)
- LoRA only fine-tunes ~1% of parameters (fast + cheap)
- Monitor validation loss to prevent overfitting
- Save checkpoints every epoch

**Cost**: Free tier works, or Colab Pro ($10) if needed

### Model Evaluation

Create `training/evaluate.py`:

- Test on held-out examples
- Manual review of responses
- Check for empathy, DBT accuracy, safety

## Phase 4: Backend API Development (Day 4-5)

### FastAPI Server (`backend/main.py`)

```python
# Core endpoints:
POST /chat - Main conversation endpoint
GET /health - Service health check

# Features:
- Load fine-tuned model from HuggingFace
- Conversation history management
- Safety filters (crisis detection)
- Response generation with DBT context
```

### Key Files

- `backend/main.py` - FastAPI routes
- `backend/model.py` - Model loading and inference
- `backend/prompts.py` - System prompts and DBT context
- `backend/safety.py` - Crisis detection and disclaimers

### Crisis Detection

Implement simple keyword detection for severe cases:

- Detect self-harm/suicide mentions
- Provide crisis resources (988, Crisis Text Line)
- Clear disclaimer: "I'm an AI, not a replacement for professional help"

## Phase 5: Frontend Development (Days 5-6)

### Next.js App (`frontend/`)

**Structure:**

```
frontend/
├── pages/
│   ├── index.tsx          # Landing page
│   └── chat.tsx           # Chat interface
├── components/
│   ├── ChatMessage.tsx    # Message bubbles
│   ├── ChatInput.tsx      # User input
│   └── Disclaimer.tsx     # Initial disclaimer
└── styles/
    └── globals.css        # Tailwind styling
```

### Features

- Clean, calming UI design (soft colors, comfortable spacing)
- Conversation history (client-side)
- Typing indicators
- Mobile-responsive
- Initial disclaimer modal
- About page explaining DBT and limitations

### Design Principles

- Warm, non-clinical aesthetic
- Easy to read typography
- Accessibility (ARIA labels, keyboard navigation)
- Clear "This is not therapy" messaging

## Phase 6: Deployment (Day 6-7)

### Backend Deployment Options

**Option A: HuggingFace Inference API (Easiest)**

- Upload fine-tuned model to HuggingFace Hub
- Use free Inference API (rate limited but sufficient for demo)
- No server management needed

**Option B: HuggingFace Spaces (More control)**

- Create Spaces app with FastAPI
- Free tier available
- Slightly more setup but better for portfolio

### Frontend Deployment

- Deploy to Vercel (free tier, perfect for Next.js)
- Environment variable for backend API URL
- Custom domain optional (free with Vercel)

### Configuration

```env
# frontend/.env.local
NEXT_PUBLIC_API_URL=https://your-hf-space.com/api
```

## Phase 7: Testing & Refinement (Day 7)

### Testing Checklist

- [ ] Test various DBT scenarios
- [ ] Verify empathetic tone
- [ ] Check crisis detection triggers
- [ ] Mobile responsiveness
- [ ] Load testing (minimal traffic expected)

### Feedback Collection

- Share with 5 testers
- Create simple feedback form
- Iterate on responses based on feedback

## Cost Breakdown (Minimal Budget)

| Item | Cost |

|------|------|

| Google Colab Pro (optional) | $10 (one month) |

| HuggingFace (hosting) | Free |

| Vercel (frontend) | Free |

| Domain (optional) | $12/year |

| **Total** | **$10-22** |

## Portfolio Presentation Tips

1. **Documentation**: Write clear README with:

   - Problem statement (BPD support need)
   - Technical approach (LoRA fine-tuning)
   - DBT methodology explanation
   - Results and learnings

2. **Demo**: Include screenshots and example conversations

3. **Code Quality**: 

   - Type hints in Python
   - Clean component structure in React
   - Comments explaining DBT-specific logic

4. **Ethical Considerations**: 

   - Clear disclaimers
   - Privacy policy (data handling)
   - Limitations section

## Safety & Ethical Notes

- **Always** include crisis resources (988 Suicide & Crisis Lifeline)
- Clear "not a replacement for therapy" messaging
- Consider data privacy (don't log sensitive conversations)
- Add rate limiting to prevent abuse
- Include resources for finding real therapists

## Learning Outcomes

By completing this project, you'll gain:

- ✓ Fine-tuning transformer models with LoRA
- ✓ Training loop implementation and monitoring
- ✓ Model deployment and inference optimization
- ✓ API design for ML applications
- ✓ Full-stack integration (ML + web)
- ✓ Ethical AI considerations

## Next Steps After Demo

If you want to continue development:

1. Implement the "patient bot" training loop idea
2. Add conversation memory/personalization
3. Integrate mood tracking features
4. Expand to other therapeutic modalities
5. Conduct user studies for research

### To-dos

- [ ] Create project structure and install dependencies (transformers, peft, datasets, torch, fastapi, next.js)
- [ ] Extract DBT knowledge from workbook and create training dataset (100-300 examples in JSON format)
- [ ] Create Colab notebook for fine-tuning Llama 3.1 8B with LoRA on DBT conversations
- [ ] Run training on Google Colab GPU and save fine-tuned model to HuggingFace Hub
- [ ] Create FastAPI backend with chat endpoint, model loading, and crisis detection
- [ ] Build Next.js chat interface with disclaimer, message history, and responsive design
- [ ] Deploy backend to HuggingFace Spaces and frontend to Vercel
- [ ] Test with various scenarios, collect feedback from 5 testers, and refine responses