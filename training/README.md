# Training Scripts

## Overview
Comprehensive fine-tuning pipeline for TheraBot DBT therapy chatbot using LoRA (Parameter Efficient Fine-Tuning).

## Architecture
- **Model**: Llama 3.1 8B Instruct
- **Fine-tuning**: LoRA with 4-bit quantization
- **Training**: HuggingFace Trainer API + Custom training loops
- **Platform**: Google Colab Pro (optional) or local GPU

## Key Features
- **LoRA Configuration**: Optimized for therapy conversation fine-tuning
- **Memory Optimization**: 4-bit quantization, gradient checkpointing
- **Evaluation**: Comprehensive metrics and DBT-specific criteria
- **Monitoring**: Learning curves, validation tracking, early stopping

## Training Pipeline
1. **Data Preparation**: Process 173,145+ therapy dialogue exchanges
2. **LoRA Setup**: Configure parameter-efficient fine-tuning
3. **Training Loop**: Implement with modern optimizations
4. **Evaluation**: DBT-specific metrics and safety assessment
5. **Model Saving**: Upload to HuggingFace Hub

## Configuration Options
- **LoRA Rank**: 8-32 (higher = more parameters, better performance)
- **Quantization**: 4-bit for memory efficiency
- **Batch Size**: 1-4 (depending on GPU memory)
- **Learning Rate**: 1e-4 to 5e-4 with warmup
- **Epochs**: 2-5 (monitor for overfitting)

## Technology Stack
- **Framework**: HuggingFace Transformers + PEFT
- **Optimization**: LoRA, 4-bit quantization, mixed precision
- **Monitoring**: Weights & Biases, TensorBoard
- **Evaluation**: BLEU, ROUGE, BERTScore, DBT-specific metrics

## Development Status
🚧 **In Development** - Training implementation planned for Phase 3 of THERABOT_PLAN

## References
- [HuggingFace LLM Course Chapter 3](https://huggingface.co/learn/llm-course/chapter3/1)
- [PEFT Documentation](https://huggingface.co/docs/peft/)
- [Performance Optimization](https://huggingface.co/docs/transformers/main/en/performance)
