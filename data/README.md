# Training Data Documentation

This directory contains the training data for fine-tuning TheraBot.

## Files

- `dbt_knowledge.md` - Extracted DBT concepts, skills, and protocols from workbook
- `dbt_conversations.json` - Formatted training dataset for fine-tuning
- `data_generation_helper.py` - (Optional) Helper script to structure workbook content

## Data Format

Training data follows the instruction-tuning format:

```json
[
  {
    "instruction": "System prompt defining the bot's role",
    "input": "User's message expressing distress or asking for help",
    "output": "Empathetic, DBT-informed response"
  }
]
```

## Data Collection Guidelines

### Quality over Quantity
- Aim for 100-300 high-quality examples
- Focus on realistic scenarios
- Ensure empathetic and accurate DBT responses

### Categories to Cover

1. **Emotional Distress** (30-40%)
   - Anxiety, depression, anger
   - Feeling overwhelmed
   - Emotional dysregulation

2. **DBT Skills Teaching** (25-30%)
   - TIPP (Temperature, Intense exercise, Paced breathing, Paired muscle relaxation)
   - PLEASE (Physical health, eating, avoiding substances, sleep, exercise)
   - DEAR MAN (interpersonal effectiveness)
   - Mindfulness techniques

3. **Validation & Empathy** (20-25%)
   - Active listening
   - Emotional validation
   - Supportive responses

4. **Crisis Situations** (10-15%)
   - Self-harm thoughts (with disclaimers)
   - Intense distress
   - Always include crisis resources

5. **Daily Support** (10%)
   - Check-ins
   - Mood tracking
   - Positive reinforcement

## Privacy & Ethics

- Use synthetic/anonymized scenarios only
- Never include real patient data
- Ensure responses include appropriate disclaimers
- Always provide crisis resources for serious situations

## Data Sources

- DBT Skills Training Workbook
- DBT worksheets and exercises
- Clinical DBT resources (with proper attribution)
- Structured scenarios based on common BPD experiences

