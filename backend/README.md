# Backend API

## Overview
FastAPI backend for TheraBot DBT therapy chatbot, providing AI-powered therapeutic responses.

## Architecture
- **Framework**: FastAPI with HuggingFace Inference API
- **Model**: Fine-tuned Llama 3.1 8B Instruct with LoRA
- **Deployment**: HuggingFace Spaces (free hosting with GPU support)

## Key Features
- **DBT Specialization**: Trained on 173,145+ therapy dialogue exchanges
- **Crisis Detection**: Automatic identification of self-harm/suicide mentions
- **Memory Management**: Context-aware conversation flow
- **Safety Systems**: Built-in crisis intervention and resource provision

## API Endpoints
- `POST /chat` - Main conversation endpoint
- `GET /health` - Service health check
- `POST /feedback` - User feedback collection
- `GET /resources` - Crisis resources and information

## Technology Stack
- **Model**: Llama 3.1 8B Instruct (fine-tuned with LoRA)
- **Inference**: HuggingFace Transformers + PEFT
- **API**: FastAPI with async support
- **Memory**: External session storage with privacy controls

## Development Status
🚧 **In Development** - Backend implementation planned for Phase 4 of THERABOT_PLAN

## References
- [HuggingFace LLM Course Chapter 3](https://huggingface.co/learn/llm-course/chapter3/1)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PEFT Documentation](https://huggingface.co/docs/peft/)
