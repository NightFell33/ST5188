import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# load the data
df = pd.read_csv('Goemotions.csv', encoding='ISO-8859-1')

# define the emotions columns
emotions = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
            'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
            'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
            'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
            'relief', 'remorse', 'sadness', 'surprise', 'neutral']

# preprocess function
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

emotion_data = {}
all_words = set()

# get frequency for different emotions
for emotion in emotions:
    texts = df[df[emotion] == 1]['text'].astype(str).tolist()
    tokens = []
    for text in texts:
        tokens.extend(preprocess_text(text))
    freq = Counter(tokens)
    emotion_data[emotion] = {
        'freq': freq,
        'words_set': set(freq.keys())
    }
    all_words.update(freq.keys())

# Count how many emotions each word appears in
word_df = {word: 0 for word in all_words}
for word in all_words:
    for emotion in emotions:
        if word in emotion_data[emotion]['words_set']:
            word_df[word] += 1



N = len(emotions)

for emotion in emotions:
    freq = emotion_data[emotion]['freq']
    total_words = sum(freq.values())
    tfidf = {}

    for word, count in freq.items():
        tf = count / total_words
        df = word_df.get(word, 0)
        idf = np.log(N / (df + 1))
        tfidf[word] = tf * idf

    # generate wordcloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=100,
        collocations=False
    ).generate_from_frequencies(tfidf)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'TF-IDF Word Cloud for {emotion}')
    plt.savefig(f'{emotion}_wordcloud.png', bbox_inches='tight')
    plt.close()