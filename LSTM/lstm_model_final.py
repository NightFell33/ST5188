import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, SpatialDropout1D
from tensorflow.keras.optimizers import Adam
import time
import os
import tensorflow as tf
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

df = pd.read_csv('preprocessed_balanced_goemotions.csv')

df_noneutral = df[~df['emotion'].isin(['neutral'])]

df_cleaned = df_noneutral.dropna(subset=['emotion'])

df = df_cleaned

print(len(df))

# Tokenize and convert texts to sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])
word_index = tokenizer.word_index

# Prepare embedding matrix
embedding_dim = 200
vocab_size = len(word_index) + 1

# Pad sequences
max_len = 200
X = pad_sequences(sequences, maxlen=max_len)

# Extract labels
labels = df[['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity',
             'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement',
             'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
             'relief', 'remorse', 'sadness', 'surprise']].values
num_classes = labels.shape[1]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Build the LSTM model with improved architecture
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                    input_length=max_len, trainable=True))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(27, activation='softmax'))  # Adjust according to number of classes
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

# Display the model summary
print(model.summary())

start_time = time.time()

# Train the model
history = model.fit(X_train, y_train, epochs=3, batch_size=256, validation_data=(X_test, y_test))

end_time = time.time()
print(f'模型运行时间：{end_time-start_time}')

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")


model.save('lstm_model.h5')

# model = tf.keras.models.load_model('lstm_model.h5')

# 预测概率和类别
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

y_test_argmax = np.argmax(y_test, axis=1)

# 计算 F1
f1_macro = f1_score(y_test_argmax, y_pred, average='macro')
f1_micro = f1_score(y_test_argmax, y_pred, average='micro')
print(f"Macro F1: {f1_macro:.4f}, Micro F1: {f1_micro:.4f}")

# 计算 AUC-ROC
y_test_binarized = label_binarize(y_test, classes=np.arange(27))
auc_roc = roc_auc_score(
    y_test_binarized,
    y_pred_prob,
    multi_class='ovr',
    average='macro'
)
print(f"AUC-ROC (Macro OvR): {auc_roc:.4f}")

# # Predict classes for the test set
# y_pred_probs = model.predict(X_test)
# y_pred = (y_pred_probs > 0.5).astype(int)

# List of actual class names without 'Text'
target_names = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity',
             'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement',
             'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
             'relief', 'remorse', 'sadness', 'surprise']

# Classification Report
print("Classification Report:")
print(classification_report(y_test_argmax, y_pred, target_names=target_names))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred)

# Plot Confusion Matrix
plt.figure(figsize=(20, 16))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
