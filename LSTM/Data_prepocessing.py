import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter
import nltk
from nltk.corpus import stopwords
import string
import re
import os
import nlpaug.augmenter.word as naw
import random
import emoji

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Part one, this part is for data augmentation
# Load the dataset
combined_df = pd.read_csv("goemotions.csv", encoding='ISO-8859-1')

df1=pd.read_csv('preprocessed_balanced_goemotions.csv')

print(combined_df.isnull().sum())
# Print the initial class distribution
class_counts = combined_df['emotion'].value_counts()
print("Initial class distribution:")
print(class_counts)

# Maximum size to match
max_size = 17131

# Initialize augmenters
synonym_augmenter = naw.SynonymAug(aug_src='wordnet')
spelling_augmenter = naw.SpellingAug()

# Function to augment data
def augment_data(group, label, synonym_augmenter, spelling_augmenter, samples_needed):
    augmented_texts = []
    while len(augmented_texts) < samples_needed:
        text = random.choice(group['text'].values)  # Randomly choose a text to augment
        if random.random() > 0.5:  # Randomly choose the type of augmentation
            augmented_text = synonym_augmenter.augment(text)
        else:
            augmented_text = spelling_augmenter.augment(text)
        augmented_texts.append(augmented_text)
    new_data = pd.DataFrame(augmented_texts, columns=['text'])
    new_data['emotion'] = label
    return new_data

# Augment each class if needed
augmented_data = pd.DataFrame()
emotion_cols = combined_df[['text','emotion']]

for emotion, group in emotion_cols.groupby('emotion'):
    count = len(group)
    if count < max_size:
        augmented_data = pd.concat([augmented_data, augment_data(group, emotion, synonym_augmenter, spelling_augmenter, max_size - count)])


# Combine original and augmented data
balanced_df = pd.concat([combined_df, augmented_data], ignore_index=True)

# Define the processing range
START_ROW = 211225
EMOTION_COLUMNS = balanced_df.columns[9:37]

def fill_emotion_labels(row):
    """Processing logic for a single row"""
    emotion = str(row['emotion']).strip()

    # Initialize all emotion columns to 0
    row[EMOTION_COLUMNS] = 0

    # Find the matching column
    for col in EMOTION_COLUMNS:
        if emotion.lower() == col.lower():
            row[col] = 1
            break  # Because each emotion corresponds to only one emotion

    return row

# Process only the data after the specified row
balanced_df.iloc[START_ROW:, :] = balanced_df.iloc[START_ROW:].apply(
    fill_emotion_labels,
    axis=1
)

# Check new class distribution
new_class_counts = balanced_df['emotion'].value_counts()
print("New class distribution after augmentation:")
print(new_class_counts)


# Save the balanced dataset to a CSV file
balanced_df.to_csv("balanced_goemotions.csv", index=False)
print("Balanced dataset saved as 'balanced_goemotions.csv'.")


# This is part two, preprocessing the augmented data
# Text Preprocessing

# Preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert emojis to text (if any)
    text = emoji.demojize(text)
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Tokenize
    words = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in words]
    return tokens

# Preprocess the text data
preprocessed_balanced_goemotions =balanced_df

preprocessed_balanced_goemotions['text'] = balanced_df['text'].apply(preprocess_text)

preprocessed_balanced_goemotions.to_csv("preprocessed_balanced_goemotions.csv", index=False)
