import pandas as pd
import re
import nltk
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('concat_comments_3_11.csv')

# Basic text cleaning
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower().strip()         # Convert to lowercase
    return text

# Apply text cleaning
df['text'] = df['body'].astype(str)
df['cleaned_text'] = df['text'].apply(clean_text)

# Delete empty lines and deleted comments
print(df.isnull().sum())
df = df.dropna(subset=['cleaned_text'])
row2delete = df[df['cleaned_text'] == 'deleted'].index
df = df.drop(row2delete)

##
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize VADER
nltk.download('vader_lexicon')
vader = SentimentIntensityAnalyzer()

# Define analysis function
def vader_sentiment(text):
    scores = vader.polarity_scores(text)
    return scores['compound']  # Return compound score

# Apply analysis
df['vader_score'] = df['cleaned_text'].apply(vader_sentiment)

# Convert to three categories
df['vader_label'] = pd.cut(df['vader_score'],
                           bins=[-1, -0.05, 0.05, 1],
                           labels=['negative', 'neutral', 'positive'])

# Analysis
# Save results
df.to_csv('analyzed_comments.csv', index=False)

# Basic analysis
print("VADER distribution:")
print(df['vader_label'].value_counts())

# If the type is object, force conversion
df['datetime_time'] = pd.to_datetime(df['standard_time'])

# Sort by time order
df = df.sort_values(by='datetime_time', ascending=False)

# Set time index permanently during preprocessing
df = df.set_index('datetime_time')

# Weekly statistics of emotional trends
df1 = df.resample('W')['vader_score'].mean()

plt.plot(df1)
plt.title('Average of weekly comment emotional tendency scores')
plt.show()

from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

def process_text(text):
    # Load stop words library
    stop_words = set(stopwords.words('english'))

    # Add custom stop words
    custom_stopwords = {
        'www', 'com', 'http', 'https', 'just', 'dont'
    }
    stop_words = stop_words.union(custom_stopwords)

    # Tokenize and filter stop words
    tokens = word_tokenize(text)
    filtered = [word for word in tokens
                if word.lower() not in stop_words
                and len(word) > 2  # Filter short words
                and word.isalpha()]  # Retain alphabetic words only

    return ' '.join(filtered)

# Generate word cloud for positive comments
positive_text = ' '.join(df[df['vader_label']=='positive']['cleaned_text'])
processed_text = process_text(positive_text)

wordcloud1 = (WordCloud(
    width=800,
    height=400,
    background_color='white',
    max_words=100,
    collocations=False
).generate(processed_text).to_image())

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud1, interpolation='bilinear')
plt.axis('off')
plt.title('Wordcloud for positive comments')
plt.savefig('positive_wordcloud.png', bbox_inches='tight')
plt.close()

negative_text = ' '.join(df[df['vader_label']=='negative']['cleaned_text'])
processed_text2 = process_text(negative_text)

wordcloud1 = (WordCloud(
    width=800,
    height=400,
    background_color='white',
    max_words=100,
    collocations=False
).generate(processed_text2).to_image())

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud1, interpolation='bilinear')
plt.axis('off')
plt.title('Wordcloud for negative comments')
plt.savefig('negative_wordcloud.png', bbox_inches='tight')
plt.close()
