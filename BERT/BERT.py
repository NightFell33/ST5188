import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from scipy.special import softmax
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train a model with different cases")
parser.add_argument("--case", type=int, required=True, choices=[1, 2, 3, 4])
args = parser.parse_args()

BATCH_SIZE = 64

# Load dataset
dataset_address = "../autodl-fs/preprocessed_balanced_classified_goemotions.csv"
df = pd.read_csv(dataset_address, encoding="latin-1")

# label_name = 'emotion'
label_name = 'emotion_category'
df = df[['text', label_name]].rename(columns={label_name: 'label'})

# Remove rows where label is NaN or "#N/A"
df = df.dropna(subset=["label"])
df = df[df["label"] != "#N/A"]
df = df[df["label"] != " "]
# df = df[df["label"] != "neutral"]

# Convert label to categorical codes
label_mapping = {label: code for code, label in enumerate(df['label'].unique())}
print(label_mapping)

with open('label_mapping4emo.json', 'w') as f:
    json.dump(label_mapping, f)
df['label'] = df['label'].map(label_mapping)
num_labels = df["label"].nunique()
print(f"Number of unique labels: {num_labels}")
print(f"Label mapping: {label_mapping}")

'''
1: Twitter/twhin-bert-base
2: bert-base-uncased
3: vinai/bertweet-base
4: answerdotai/ModernBERT-base
'''
case = args.case
if case == 1:
    tokenizer = AutoTokenizer.from_pretrained("Twitter/twhin-bert-base", local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained("Twitter/twhin-bert-base", num_labels = num_labels, ignore_mismatched_sizes=True, local_files_only=True)
elif case == 2:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels = num_labels, ignore_mismatched_sizes=True, local_files_only=True)
elif case == 3:
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained("vinai/bertweet-base", num_labels = num_labels, ignore_mismatched_sizes=True, local_files_only=True)
elif case == 4:
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base", local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained("answerdotai/ModernBERT-base", num_labels = num_labels, ignore_mismatched_sizes=True, local_files_only=True)



# Define dataset class
class GoEmotionsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = str(self.data.iloc[index]["text"])
        label = int(self.data.iloc[index]["label"])  # Ensure integer labels

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.int64),  # Ensure correct dtype
        }

# Create dataset
train_size = int(0.8 * len(df))
val_size = len(df) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(
    GoEmotionsDataset(df, tokenizer), [train_size, val_size]
)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


def compute_metrics(p):
    predictions, labels = p
    probs = softmax(predictions, axis=1)
    preds = np.argmax(probs, axis=1)

    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    auc = roc_auc_score(labels, probs, multi_class='ovr')

    # Compute and save confusion matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_mapping.keys(), yticklabels=label_mapping.keys())
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    plt.tight_layout()
    plt.savefig(f"4emo/confusion_matrix_epoch_{trainer.state.epoch:.0f}.png")  # saves plot with epoch number
    plt.close()

    return {
        "accuracy": accuracy,
        "f1": f1,
        "auc_roc": auc
    }


# Training arguments
training_args = TrainingArguments(
    output_dir="../autodl-tmp/results",
    num_train_epochs=5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="../autodl-fs/logs",
    logging_steps=10,
    evaluation_strategy="epoch",  # Evaluate after each epoch
    save_strategy="no",       # Save model after each epoch
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,  # Specify the evaluation metric
)

# Train model
trainer.train()

# Save model
if case == 1:
    model.save_pretrained("../autodl-fs/fine_tuned_twhin_bert_4emo")
    tokenizer.save_pretrained("../autodl-fs/fine_tuned_twhin_bert_4emo")
elif case == 2:
    model.save_pretrained("../autodl-fs/fine_tuned_bert")
    tokenizer.save_pretrained("../autodl-fs/fine_tuned_bert")
elif case == 3:
    model.save_pretrained("../autodl-fs/fine_tuned_bertweet")
    tokenizer.save_pretrained("../autodl-fs/fine_tuned_bertweet")
elif case == 4:
    model.save_pretrained("../autodl-fs/fine_tuned_modern_bert")
    tokenizer.save_pretrained("../autodl-fs/fine_tuned_modern_bert")