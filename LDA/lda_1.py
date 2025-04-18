import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
import pyLDAvis.gensim_models
import time
import numpy
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download(['stopwords', 'wordnet', 'omw-1.4'])

# Custom stopword list
custom_stopwords = set(stopwords.words('english') + [
    'apple', 'iphone', 'ipad', 'macbook', 'mac', 'ios', 'phone',
    'device', 'product', 'just', 'get', 'also',
    'reddit', 'subreddit', 'www', 'http', 'com', 'know',
    'yet', 'would'
])

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+', '', text)  # Remove URLs

    # Remove special characters (keep numbers)
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)

    # Tokenize
    words = text.split()

    # Remove stopwords and short words (length < 2)
    words = [word for word in words if word not in custom_stopwords and len(word) > 2]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return words

# Load data
df = pd.read_csv('D:/file/predictions_results_vader+BERT.csv')

def train_lda_per_sentiment(df, sentiment_col='model2_predictions'):
    # Store all models and visualization results
    models_dict = {}

    for sentiment in df[sentiment_col].unique():
        # Filter comments of the current sentiment type
        subset = df[df[sentiment_col] == sentiment]

        # Preprocess text
        processed_docs = subset['body'].apply(preprocess_text)

        # Create dictionary and corpus
        dictionary = corpora.Dictionary(processed_docs)
        corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

        # Train LDA model
        lda_model = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=5,  # Default 5 topics per sentiment type, adjustable
            random_state=42,
            passes=10,
            alpha='auto'
        )

        # Save results
        models_dict[sentiment] = {
            'model': lda_model,
            'dictionary': dictionary,
            'corpus': corpus,
            'log_perplexity': lda_model.log_perplexity(corpus)
        }
        vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
        pyLDAvis.save_html(vis, f'./lda_single_sentiment_view/{sentiment}_vis.html')
    return models_dict

start = time.time()
# Execute modeling per sentiment type
sentiment_models = train_lda_per_sentiment(df)

end = time.time()
print(f'Sentiment classification LDA runtime: {end-start:.4f} seconds')

start1 = time.time()
# Preprocess all data
all_processed_docs = df['body'].apply(preprocess_text)

# Create global dictionary and corpus
global_dictionary = corpora.Dictionary(all_processed_docs)
global_corpus = [global_dictionary.doc2bow(doc) for doc in all_processed_docs]

# Train global LDA model
global_lda = models.LdaModel(
    corpus=global_corpus,
    id2word=global_dictionary,
    num_topics=20,  # Global number of topics can be appropriately increased
    random_state=42,
    passes=15,
    alpha='auto'
)

end1 = time.time()
print(f'Global LDA runtime: {end1-start1:.4f} seconds')

# Generate visualization
global_vis = pyLDAvis.gensim_models.prepare(global_lda, global_corpus, global_dictionary)
pyLDAvis.save_html(global_vis, f'./lda/global_lda_view.html')

def show_sentiment_topics(sentiment_type, num_words=5):
    model_info = sentiment_models[sentiment_type]
    topics = model_info['model'].print_topics(num_words=num_words)
    print(f"Sentiment: {sentiment_type}")
    for topic in topics:
        print(f"Topic {topic[0]}: {topic[1]}\n")
    print(f"log_perplexity: {sentiment_models[sentiment_type]['log_perplexity']}")

# Example: View topics for sentiment type 0
show_sentiment_topics(0)

# Display all topics and perplexity
for sentiment in df['model2_predictions'].unique():
    show_sentiment_topics(sentiment)

# Display visualization in Jupyter
pyLDAvis.display(global_vis)

# Save results to HTML file
pyLDAvis.save_html(global_vis, 'global_vis.html')

# Calculate perplexity
topics1 = global_lda.print_topics(num_words=20)
print(f"For All Comments:")
for topic in topics1:
    print(f"Topic {topic[0]}: {topic[1]}\n")
print(f"Global Model Perplexity: {global_lda.log_perplexity(global_corpus)}")

def get_dominant_topic(lda_model, corpus):
    """Get the dominant topic for each document"""
    topic_distributions = [lda_model.get_document_topics(doc) for doc in corpus]
    dominant_topics = []
    for doc_topics in topic_distributions:
        if not doc_topics:  # Handle empty documents
            dominant_topics.append(-1)
            continue
        # Get the topic with the highest probability
        dominant_topic = sorted(doc_topics, key=lambda x: x[1], reverse=True)[0][0]
        dominant_topics.append(dominant_topic)
    return dominant_topics

# Add dominant topic column to original data
df['dominant_topic'] = get_dominant_topic(global_lda, global_corpus)

# Filter out invalid topics (optional)
df = df[df['dominant_topic'] != -1]

# Create topic-sentiment distribution matrix
topic_sentiment_matrix = pd.crosstab(
    index=df['dominant_topic'],
    columns=df['model2_predictions'],
    normalize='index'  # Calculate row proportions (sentiment distribution within each topic)
)

# Set visualization parameters
plt.figure(figsize=(20, 12))
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

# Create heatmap
heatmap = sns.heatmap(
    topic_sentiment_matrix.T,  # Transpose matrix: rows represent sentiments, columns represent topics
    annot=True,
    fmt=".1%",
    cmap="YlGnBu",
    cbar_kws={'label': 'Percentage'},
    linewidths=0.5
)

# Set axis labels
plt.title("Topic-Sentiment Distribution", fontsize=18)
plt.xlabel("LDA Topic ID", fontsize=14)
plt.ylabel("Sentiment Type", fontsize=14)
heatmap.set_xticklabels([f"Topic {i}" for i in topic_sentiment_matrix.index], rotation=45)
heatmap.set_yticklabels([f"Sentiment {i}" for i in topic_sentiment_matrix.columns], rotation=0)

plt.tight_layout()
plt.show(block=True)

# Save distribution matrix to CSV
topic_sentiment_matrix.to_csv("topic_sentiment_distribution.csv")

# Print statistical summary
print("\nTopic-Sentiment Distribution Statistical Summary:")
print(topic_sentiment_matrix.describe())
