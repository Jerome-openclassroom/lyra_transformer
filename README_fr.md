# Lyra_Transformer_Training

🧠 Fine-tuning et évaluation d’un modèle Transformer DistilBERT sur une tâche personnalisée de classification de nouvelles (tech, science, business, sports).  
Ce dépôt fait partie de la série Lyra — projets IA ouverts, scientifiques et pédagogiques, centrés sur la synergie humain–IA.

---

## 🌐 Aperçu

Ce projet démontre le fine-tuning d’un Transformer léger (DistilBERT) sur un petit jeu de données étiqueté (format CSV) en utilisant les bibliothèques `transformers` et `datasets` de Hugging Face.  
Il inclut :  
- L’entraînement à la fois en **Jupyter Notebook** et **Google Colab**  
- L’encodage des labels et le prétraitement avec tokenizer  
- La sauvegarde et le rechargement du modèle fine-tuné  
- L’évaluation sur des exemples de test personnalisés  
- L’intégration GitHub et une configuration reproductible  

---

## 🧾 Classes et Labels

La tâche de classification couvre 4 catégories :

| ID Label | Classe    |
|----------|-----------|
| 0        | business  |
| 1        | science   |
| 2        | sports    |
| 3        | tech      |

---

## 🧪 Modèle

👉 Le modèle entraîné n’est pas inclus dans ce dépôt pour des raisons de taille.

- 🧠 Base : `distilbert-base-uncased`  
- Optimiseur : `AdamW` (géré par `Trainer`)  
- Learning rate : `2e-5`  
- Weight decay : `0.01`  
- Batch size : `16`  
- Longueur max. des séquences : `128`  
- Époques : `1` (Colab), `3` (Jupyter)  
- Tokenizer padding : `max_length` avec troncature  
- Régularisation : L2 sur les poids uniquement (biais exclus)  

---

## 📊 Résultats

### ✔️ Entraînement Colab (1 époque, 2000 train / 500 eval)
- Perte validation : **0.478**  
- Perte entraînement : **0.703**  

### ✔️ Entraînement Jupyter (3 époques, 114 steps au total)
- Perte validation : **0.009**  
- Perte entraînement : **0.173**  
- Précision sur exemples personnalisés : **100%**  

---

## 💬 Exemples de prédictions

Entrée :  
`"Quantum computing makes another leap forward."`  
→ **Classe prédite : tech**  

Entrée :  
`"The championship game ended in a thrilling overtime."`  
→ **Classe prédite : sports**  

Entrée :  
`"Researchers discover a new species of dinosaur."`  
→ **Classe prédite : science**  

---

## 💾 Arborescence du projet

```
lyra_transformer/                       
├── README.md                          # Présentation et instructions du projet - version en anglais
├── README_fr.md                       # Présentation et instructions du projet - version en français (ce fichier)
├── code/                              # Scripts d’entraînement et d’inférence
│   ├── transformer_Google_collab.ipynb  # Notebook Colab pour prototypage rapide
│   ├── transformer_Google_collab.py      # Script Python équivalent au workflow Colab
│   └── transformer_jupyter_notebook.md   # Résumé markdown compatible Jupyter
└── datasets/                          # Jeux de données CSV locaux pour le fine-tuning
    ├── train.csv                     # Jeu d’entraînement (exemples étiquetés)
    └── validation.csv                # Jeu de validation (pour évaluation)
```

---

## 🚀 Utilisation

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

- Ce projet fait partie du corpus **Lyra — IA appliquée à la formation et à l’automatisation écologique**.  
- Conçu pour l’interprétabilité, la reproductibilité et une préparation au contexte post-AGI.  

---

🧬 *Conçu par Jérôme — ingénierie IA prête pour l’ère post-AGI.*
