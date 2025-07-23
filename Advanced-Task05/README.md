# 🏷️ Support Ticket Tagging Pipeline – Task 5

This project implements a modular tagging pipeline for support tickets using three approaches:
- **Zero-Shot Prompting**
- **Fine-Tuned Classification**
- **Few-Shot Prompt Engineering**

It compares each method’s performance on a representative sample of 25 real-world support messages.

---

## 🔧 Methodology Overview

### 1️⃣ Zero-Shot Tagging
- Used GPT-4o via OpenRouter
- Applied plain prompt: `"Ticket: {instruction}\nTags:"`
- Tags inferred without examples or tuning
- Logged in `zero_shot_tags` column

### 2️⃣ Fine-Tuned Model
- Model: `distilbert-base-uncased` from Hugging Face
- Fine-tuned on sample dataset: `instruction → intent`
- Used Trainer API with 5 epochs on ~20 training examples
- Predictions stored in `fine_tuned_pred`

### 3️⃣ Few-Shot Prompting
- Embedded 3–5 examples of labeled tickets in the prompt
- Improved generalization on rare and nuanced intents
- Predictions stored in `few_shot_tags`

---

## 📊 Performance Summary (Test Subset)

| Method        | Accuracy |
|---------------|----------|
| Zero-Shot     | 4%       |
| Fine-Tuned    | 0%       |
| Few-Shot      | 28%      |

Few-Shot Prompting dramatically outperformed others, especially on low-frequency intents like `registration_problems` and `check_payment_methods`.

---

## 📁 Data Columns (in `sample_df`)
- `instruction`: User input / support ticket text
- `intent`: True label for ticket category
- `zero_shot_tags`: Tags predicted via zero-shot LLM prompt
- `fine_tuned_pred`: Tags from custom fine-tuned classifier
- `few_shot_tags`: Tags from few-shot LLM prompt
- `*_match`: Binary score indicating prediction match to `intent`

---

## 📦 Evaluation Script

Match scoring and comparison logic is implemented in:
```python
def match_score(pred, true):
    return int(true.lower() in str(pred).lower())
