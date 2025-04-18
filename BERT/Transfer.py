from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_from_disk
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
from scipy.special import softmax


BATCH_SIZE = 64
Case = 1

if Case == 1:
    # Load TweetEval (for reference/analysis)
    print("Loading TweetEval dataset...")
    ds = load_from_disk("../autodl-fs/tweet_eval_sentiment")
    ds = ds.rename_column("text", "sentence")
    num_labels = 3

elif Case == 2:
    # or Load SST (Stanford Sentiment Treebank)
    print("Loading SST dataset...")
    ds = load_from_disk("../autodl-fs/stanford_sentiment_treebank")
    ds = ds.rename_column("cleaned_text", "sentence")
    num_labels = 5 

# Load tokenizer and model (fine-tuned on TweetEval sentiment)
tokenizer = AutoTokenizer.from_pretrained("../autodl-fs/fine_tuned_twhin_bert", local_files_only=True)

model = AutoModelForSequenceClassification.from_pretrained(
    # "../autodl-fs/fine_tuned_twhin_bert_4emo",
    "../autodl-fs/fine_tuned_twhin_bert",
    num_labels=num_labels,
    ignore_mismatched_sizes=True,
    local_files_only=True
)

for name, param in model.bert.named_parameters():
    if name.startswith("encoder.layer.0") or name.startswith("encoder.layer.1"):
        param.requires_grad = False

# Preprocessing
def preprocess(examples):
    return tokenizer(
        examples["sentence"],
        truncation=True,
        padding='max_length',  # pad all sequences to the same length
        max_length=128,        # or any number that fits your model and data
    )

encoded_ds = ds.map(preprocess, batched=True)


def compute_metrics(p):
    predictions, labels = p
    probs = softmax(predictions, axis=1)  # Convert logits to probabilities
    preds = np.argmax(probs, axis=1)      # Predicted class from probabilities

    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    auc = roc_auc_score(labels, probs, multi_class='ovr')

    return {
        "accuracy": accuracy,
        "f1": f1,
        "auc_roc": auc
    }

# Training arguments
training_args = TrainingArguments(
    output_dir="../autodl-tmp/results/transfer",
    num_train_epochs=5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=500,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="../autodl-fs/logs/transfer",
    logging_steps=10,
    evaluation_strategy="epoch",  # Evaluate after each epoch
    save_strategy="no",       # Save model after each epoch
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_ds["train"],
    eval_dataset=encoded_ds["validation"],
    compute_metrics=compute_metrics,  # Specify the evaluation metric
)

# Fine-tune
trainer.train()
