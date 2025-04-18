from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
from tqdm import tqdm
import json

# Load tokenizer and models
tokenizer = AutoTokenizer.from_pretrained("../autodl-fs/fine_tuned_twhin_bert", local_files_only=True)

model1 = AutoModelForSequenceClassification.from_pretrained(
    "../autodl-fs/fine_tuned_twhin_bert_4emo",
    local_files_only=True
)

model2 = AutoModelForSequenceClassification.from_pretrained(
    "../autodl-fs/fine_tuned_twhin_bert",
    local_files_only=True
)

# Load dataset
dataset_address = "../autodl-fs/analyzed_comments.csv"
df = pd.read_csv(dataset_address, encoding="latin-1")
text_name = 'cleaned_text'

# Load label mappings
with open('label_mapping4emo.json', 'r') as f:
    label_mapping1 = json.load(f)
with open('label_mapping27.json', 'r') as f:
    label_mapping2 = json.load(f)

# Reverse mappings: index → label
reverse_mapping1 = {code: label for label, code in label_mapping1.items()}
reverse_mapping2 = {code: label for label, code in label_mapping2.items()}

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1.to(device)
model2.to(device)

# Tokenization
texts = df[text_name].fillna("").tolist()
inputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

input_ids = inputs['input_ids'].to(device)
attention_mask = inputs['attention_mask'].to(device)

# Batched inference
batch_size = 64
num_batches = (len(texts) + batch_size - 1) // batch_size

model1_labels = []
model2_labels = []

model1.eval()
model2.eval()

with torch.no_grad():
    for i in tqdm(range(num_batches), desc="Running inference"):
        start = i * batch_size
        end = min(start + batch_size, len(texts))

        input_ids_batch = input_ids[start:end]
        attention_mask_batch = attention_mask[start:end]

        out1 = model1(input_ids=input_ids_batch, attention_mask=attention_mask_batch)
        out2 = model2(input_ids=input_ids_batch, attention_mask=attention_mask_batch)

        pred1 = torch.argmax(out1.logits, dim=1).cpu().numpy()
        pred2 = torch.argmax(out2.logits, dim=1).cpu().numpy()

        model1_labels.extend([reverse_mapping1[p] for p in pred1])
        model2_labels.extend([reverse_mapping2[p] for p in pred2])

# Add only label predictions to DataFrame
df['model1_predictions'] = model1_labels
df['model2_predictions'] = model2_labels

# Save results
df.to_csv('predictions_results.csv', index=False)
