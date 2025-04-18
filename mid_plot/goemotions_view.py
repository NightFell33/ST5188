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

# Download NLTK stopwords data
nltk.download('stopwords')

# General text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Tokenize
    words = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return words

# Get current working directory
current_work_dir = os.getcwd()
print(current_work_dir)

# Import data
goemotions = pd.read_csv('goemotions.csv', encoding='ISO-8859-1')

# Handle missing values
print("Missing value statistics:")
print(goemotions.isnull().sum())

# goemotions = goemotions.dropna(subset=['body'])

# Count the number of emotions labeled per comment (single-label vs multi-label)
goemotions['label_number'] = goemotions.iloc[:, 9:37].sum(axis=1)

# Plot histogram
counts, bins, patches = plt.hist(goemotions['label_number'], bins=10, edgecolor='black', range=[0,3], orientation= "horizontal")

# Set plot title and labels
plt.title('Number of emotions labelled per comment')
plt.xlabel('Frequency')
plt.ylabel('Value')

# Show plot
plt.show()

# Group by comment ID and calculate the total annotation count for each emotion
grouped_sum = goemotions.groupby('id').sum()

# Check if each comment has at least one emotion with a total annotation count >=2
agreement = grouped_sum.iloc[:,8:36].ge(2).any(axis=1)
percentage_agreement = agreement.mean() * 100

print(f"Percentage of samples with at least two annotators agreeing on the same label: {percentage_agreement:.2f}%")

# Co-occurrence statistics at the annotator level
co_occur_annotations = goemotions[(goemotions['excitement'] == 1) & (goemotions['amusement'] == 1)].shape[0]
total_excitement = goemotions['excitement'].sum()
co_occur_ratio = co_occur_annotations / total_excitement * 100

print(f"Percentage of times amusement is annotated when excitement is annotated: {co_occur_ratio:.2f}%")

# # Co-occurrence statistics at the comment level
# co_occur_comments = goemotions.apply(lambda x: ((x['excitement'] == 1) & (x['amusement'] == 1)).any())
# co_occur_percent = co_occur_comments.sum() / len(goemotions) * 100
#
# print(f"Percentage of comments with at least one annotator annotating both excitement and amusement: {co_occur_percent:.2f}%")

# # Correlation analysis
# correlation = grouped_sum[['excitement', 'amusement']].corr().iloc[0, 1]
# print(f"Correlation between excitement and amusement: {correlation:.4f}")

# Extract all emotion column names (assuming emotions are consecutively distributed in the data)
emotion_columns = goemotions.columns[9:37]  # Adjust column range based on actual data (exclude id and annotator_id etc. non-emotion columns)

# Count the total frequency of each emotion
emotion_counts = goemotions[emotion_columns].sum().sort_values(ascending=False)
emotion_counts.head(28)

# Extract top 10 most frequent emotions
top10_emotions = emotion_counts.head(10)

# Plot bar chart
plt.figure(figsize=(12, 6))
sns.barplot(x=top10_emotions.values, y=top10_emotions.index, palette="viridis")
plt.title("Top 10 Most Frequent Emotions")
plt.xlabel("Frequency")
plt.ylabel("Emotion Category")
plt.show()
