import pandas as pd
import re
from tqdm import tqdm  # Progress bar support

# Assume the original data structure includes: comment text, sentiment label, date
df = pd.read_csv('D:/file/predictions_results_vader+BERT.csv', parse_dates=['standard_time'])

# Example vocabulary table definition (replace with actual 25 vocabulary tables)
vocab_groups = {
    # Multi-word groups (inside parentheses)
    "app_group": ["app", "apps", "store", "software"],
    "music_group": ["music", "sound", "voice"],
    "version_group": ["version", "update"],
    "chip_group": ["chip", "core"],
    "photo_group": ["photo", "camera", "lens"],
    "car_group": ["carplay", "car"],
    "AI_group": ["AI", "chatgpt", "bot", "revolutionary"],
    "company_group": ["company", "business"],
    "price_group": ["price", "pay", "money", "dollar"],
    "market_group": ["market", "discount"],  # Note to retain original spelling

    # Single-word groups
    "siri": ["siri"],
    "display": ["display"],
    "battery": ["battery"],
    "video": ["video"],
    "game": ["game"],
    "magsafe": ["magsafe"],
    "looking": ["looking"],
    # "sound_single": ["sound"],  # Standalone sound
    "glass": ["glass"],
    "notification": ["notification"],
    "trump": ["trump"],
    "government": ["government"],
    "support": ["support"],
    "ecosystem": ["ecosystem"],
    "platform": ["platform"],
    "firmware": ["firmware"]
}

# Build regex pattern (match whole words)
def build_regex_pattern(words):
    return r'\b(' + '|'.join([re.escape(word) for word in words]) + r')\b'

# Create vocabulary matching column
for group_name, keywords in tqdm(vocab_groups.items()):
    pattern = build_regex_pattern(keywords)
    df[group_name] = df['text'].str.contains(
        pattern,
        case=False,  # Case insensitive
        regex=True,
        na=False
    )

# Define positive and negative emotions according to GoEmotions official labels (adjust according to actual labels)
positive_emotions = [
    'admiration', 'amusement', 'approval', 'caring', 'desire',
    'excitement', 'gratitude', 'joy', 'love', 'optimism',
    'pride', 'relief'
]

negative_emotions = [
    'anger', 'annoyance', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'fear', 'grief', 'nervousness',
    'remorse', 'sadness'
]

# Create mapping dictionary
emotion_mapping = {emotion: 'positive' for emotion in positive_emotions}
emotion_mapping.update({emotion: 'negative' for emotion in negative_emotions})

# Merge sentiment labels (according to previous definition)
df['sentiment'] = df['model2_predictions'].map(emotion_mapping)
df = df.dropna(subset=['sentiment'])  # Filter neutral sentiments

# Convert time format
df['year_week'] = df['standard_time'].dt.strftime('%Y-%W')

# Create empty data container
result_dfs = []

# Traverse each vocabulary table for statistics
for vocab in tqdm(vocab_groups.keys()):
    # Filter comments containing the current vocabulary
    temp_df = df[df[vocab]]

    # Aggregate by time and sentiment
    grouped = temp_df.groupby(['year_week', 'sentiment']).size().unstack(fill_value=0)
    grouped['vocab_group'] = vocab  # Add vocabulary table identifier

    result_dfs.append(grouped)

# Merge results
final_df = pd.concat(result_dfs).reset_index()

# Data reshaping
melted_df = final_df.melt(
    id_vars=['year_week', 'vocab_group'],
    value_vars=['positive', 'negative'],
    var_name='sentiment',
    value_name='count'
)

# Add time sorting basis
melted_df['sort_date'] = pd.to_datetime(melted_df['year_week'] + '-1', format='%Y-%W-%w')

import plotly.express as px

# Generate interactive faceted graph
fig = px.line(
    melted_df,
    x='sort_date',
    y='count',
    color='sentiment',
    facet_col='vocab_group',
    facet_col_wrap=5,  # Display 5 vocabulary tables per row
    height=2000,  # Adjust according to actual needs
    title='Temporal Trends in Positive and Negative Sentiment by Vocabulary Group',
    labels={'sort_date': 'time', 'count': 'frequency'},
    color_discrete_map={'positive':'#1f77b4', 'negative':'#ff7f0e'}
)

# Optimize layout
fig.update_layout(
    hovermode='x unified',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

# Adjust axis display
fig.update_xaxes(
    tickformat="%Y\n%W week",  # Display year + week number
    dtick="M1",  # Show a major tick every month
    showticklabels=True,
    matches=None  # Allow different subplots to have different scales
)

# Optimize subplot spacing
fig.update_layout(
    margin=dict(l=50, r=50, b=100),
    font=dict(size=10)
)

# Add scrollbar
fig.update_layout(
    xaxis=dict(rangeslider=dict(visible=True)),
    xaxis2=dict(rangeslider=dict(visible=True)),
    # ... need to set for each subplot separately
)

fig.show()
fig.write_html('Dual Sentiment Word Frequency Statistics.html')
