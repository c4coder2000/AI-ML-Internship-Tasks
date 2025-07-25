# 📰 Advanced Task 01 — BERT News Headline Classifier

Fine-tuned BERT (`bert-base-uncased`) on AG News dataset for news headline classification. This pipeline is modular, reproducible, and designed for clean evaluation and optional deployment.

---

## 🏆 Model Performance (Epoch 3)

| Metric                | Score      |
|-----------------------|------------|
| **Eval Loss**         | 0.1323     |
| **Accuracy**          | 97.54%     |
| **F1 Score**          | 97.54%     |
| **Runtime**           | 88.9 sec   |
| **Samples/sec**       | 64.10      |
| **Steps/sec**         | 4.01       |

> 📈 These results were achieved after 3 epochs using the Hugging Face `Trainer` API with standard hyperparameters.

---

## 🔧 Workflow Overview

### Phase 1: Data Preprocessing

- Load AG News CSV
- Clean and map labels to numeric form
- Tokenize using `bert-base-uncased`

### Phase 2: Model Fine-Tuning

- Load pre-trained BERT
- Fine-tune using `Trainer` with accuracy & F1 as metrics
- Save trained weights locally (optional)

### Phase 3: Evaluation

- Compute Accuracy, Precision, Recall, F1
- Generate confusion matrix

### Phase 4: Inference

Use `predict.py` to classify custom headlines from CLI or integrated UI.

---

## 🚀 Setup & Usage

### 1. Environment Setup

```bash
cd Advanced-Task01
pip install -r requirements.txt
