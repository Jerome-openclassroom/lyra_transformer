# Lyra_Transformer_Training

🧠 Fine-tuning and evaluation of a DistilBERT transformer model on a custom news classification task (tech, science, business, sports). This repository is part of the Lyra series — open, scientific, and pedagogical IA projects focused on human–AI synergy.

---

## 🌐 Overview

This project demonstrates the fine-tuning of a lightweight Transformer (DistilBERT) on a small labeled dataset (CSV format) using Hugging Face's `transformers` and `datasets` libraries. It includes:
- Training in both **Jupyter Notebook** and **Google Colab**
- Label encoding and tokenizer preprocessing
- Saving and reloading the fine-tuned model
- Evaluation on custom test examples
- GitHub integration and reproducible setup

---

## 🧾 Classes and Labels

The classification task involves 4 categories:

| Label ID | Class      |
|----------|------------|
| 0        | business   |
| 1        | science    |
| 2        | sports     |
| 3        | tech       |

---

## 🧪 Model

- 🧠 Base: `distilbert-base-uncased`
- Optimizer: `AdamW` (managed by `Trainer`)
- Learning rate: `2e-5`
- Weight decay: `0.01`
- Batch size: `16`
- Max sequence length: `128`
- Epochs: `1` (Colab), `3` (Jupyter)
- Tokenizer padding: `max_length` with truncation
- Regularization: L2 on weights only (biases excluded)

---

## 📊 Results

### ✔️ Colab Training (1 epoch, 2000 train / 500 eval)
- Validation loss: **0.478**
- Training loss: **0.703**

### ✔️ Jupyter Training (3 epochs, 114 steps total)
- Validation loss: **0.009**
- Training loss: **0.173**
- Accuracy on custom examples: **100%**

---

## 💬 Example Predictions

Input:  
`"Quantum computing makes another leap forward."`  
→ **Predicted class: tech**

Input:  
`"The championship game ended in a thrilling overtime."`  
→ **Predicted class: sports**

Input:  
`"Researchers discover a new species of dinosaur."`  
→ **Predicted class: science**

---

## 💾 Folder Structure

```
lyra_transformer/                       
├── README.md                          # Project overview and instructions
├── code/                              # Training & inference scripts
│   ├── transformer_Google_collab.ipynb  # Colab notebook for quick prototyping
│   ├── transformer_Google_collab.py      # Python script version of the Colab workflow
│   └── transformer_jupyter_notebook.md   # Jupyter-friendly markdown summary
└── datasets/                          # Local CSV datasets for fine-tuning
    ├── train.csv                     # Training set (labeled examples)
    └── validation.csv                # Validation set (held-out for evaluation)

```

---

## 🚀 Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("MODEL_TRANSFORMER")
model = AutoModelForSequenceClassification.from_pretrained("MODEL_TRANSFORMER")

text = "Apple launches its next-gen AI chip."
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)

with torch.no_grad():
    logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=1).item()

label_mapping = {0: "business", 1: "science", 2: "sports", 3: "tech"}
print(label_mapping[pred])
```

---

## 🧭 Notes

- This project is part of the **Lyra IA training and ecological automation** corpus.
- Designed for interpretability, reproducibility, and post-AGI agent readiness.

---


---

🧬 *Designed by Jérôme — post-AGI ready IA engineering.*
