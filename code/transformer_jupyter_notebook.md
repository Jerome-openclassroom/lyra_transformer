```python
!pip install torch transformers datasets scikit-learn --quiet

```


```python
!pip install tf-keras
```



```python
!pip install accelerate
```


   

```python
import transformers
print(transformers.__version__)

```

    4.54.0
    

```python
# --- 1. Imports ---
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.preprocessing import LabelEncoder
import torch

# --- 2. Chemins des fichiers CSV ---
train_path = "datasets/train.csv"
val_path = "datasets/validation.csv"

# --- 3. Chargement et encodage des labels ---
train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)

if train_df["label"].dtype == object:
    le = LabelEncoder()
    train_df["label"] = le.fit_transform(train_df["label"])
    val_df["label"] = le.transform(val_df["label"])
    id2label = {i: l for i, l in enumerate(le.classes_)}
    label2id = {l: i for i, l in enumerate(le.classes_)}
else:
    id2label = {i: str(i) for i in sorted(train_df["label"].unique())}
    label2id = {v: k for k, v in id2label.items()}

# --- 4. Conversion pandas → Dataset Hugging Face ---
train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)

# --- 5. Tokenization ---
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

# --- 6. Chargement du modèle ---
model = AutoModelForSequenceClassification.from_pretrained(
    model_ckpt,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# --- 7. Entraînement sur CPU ---
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="no",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_steps=10,
    report_to="none"  # désactive wandb et cie
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer
)

# --- 8. Lancement de l'entraînement ---
trainer.train()

```

    WARNING:tensorflow:From C:\Users\jerom\anaconda3\Lib\site-packages\tf_keras\src\losses.py:2976: The name tf.losses.sparse_softmax_cross_entropy is deprecated. Please use tf.compat.v1.losses.sparse_softmax_cross_entropy instead.
    
    


    Map:   0%|          | 0/300 [00:00<?, ? examples/s]



    Map:   0%|          | 0/100 [00:00<?, ? examples/s]





    <div>

      <progress value='114' max='114' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [114/114 08:56, Epoch 3/3]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Epoch</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>0.174500</td>
      <td>0.040063</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.019100</td>
      <td>0.011913</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.012300</td>
      <td>0.009580</td>
    </tr>
  </tbody>
</table><p>





    TrainOutput(global_step=114, training_loss=0.17327756970597988, metrics={'train_runtime': 540.6489, 'train_samples_per_second': 1.665, 'train_steps_per_second': 0.211, 'total_flos': 29806227763200.0, 'train_loss': 0.17327756970597988, 'epoch': 3.0})




```python

```


```python
import os
os.getcwd()

```




    'C:\\Users\\jerom'




```python
# Exemple à tester
text = "Quantum computing makes another leap forward."

# Prétraitement (tokenisation)
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)

# Prédiction sans calcul de gradient
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

# Décodage du label (si tu connais l’ordre)
label_mapping = {
    0: "business",
    1: "science",
    2: "sports",
    3: "tech"
}

print(f"Texte : {text}")
print(f"Classe prédite : {predicted_class} → {label_mapping[predicted_class]}")
```

    Texte : Quantum computing makes another leap forward.
    Classe prédite : 3 → tech
    


```python
examples = [
    "The stock market sees significant gains today.",
    "Researchers discover a new species of dinosaur.",
    "The championship game ended in a thrilling overtime.",
    "Apple announces the next generation of its AI chip.",
    "Scientists are concerned about rising sea levels.",
    "The football team secured their third win this season.",
    "Meta launches a virtual reality workspace."
]

for text in examples:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
        print(f"{text}\n→ Prédit : {label_mapping[pred]}\n")

```

    The stock market sees significant gains today.
    → Prédit : business
    
    Researchers discover a new species of dinosaur.
    → Prédit : science
    
    The championship game ended in a thrilling overtime.
    → Prédit : sports
    
    Apple announces the next generation of its AI chip.
    → Prédit : tech
    
    Scientists are concerned about rising sea levels.
    → Prédit : science
    
    The football team secured their third win this season.
    → Prédit : sports
    
    Meta launches a virtual reality workspace.
    → Prédit : tech
    
    


```python
# Sauvegarde du modèle fine-tuné
model.save_pretrained("./MODEL_TRANSFORMER")
tokenizer.save_pretrained("./MODEL_TRANSFORMER")

```




    ('./MODEL_TRANSFORMER\\tokenizer_config.json',
     './MODEL_TRANSFORMER\\special_tokens_map.json',
     './MODEL_TRANSFORMER\\vocab.txt',
     './MODEL_TRANSFORMER\\added_tokens.json',
     './MODEL_TRANSFORMER\\tokenizer.json')




```python

```
