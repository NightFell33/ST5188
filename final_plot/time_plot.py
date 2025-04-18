import pandas as pd
import plotly.express as px
from sympy.physics.units import frequency

df = pd.read_csv('D:/file/predictions_results_vader+BERT.csv')

# Ensure the time column is correctly parsed
df['standard_time'] = pd.to_datetime(df['standard_time'])

# Create year-week combination column (ISO week number, Monday as the start of the week)
df['year'] = df['standard_time'].dt.isocalendar().year
df['week'] = df['standard_time'].dt.isocalendar().week

# Group by week and sentiment to perform statistical aggregation
weekly_counts = df.groupby(['year', 'week', 'model2_predictions']).size().reset_index(name='amount')

# Create continuous time labels
weekly_counts['start_time'] = weekly_counts.apply(
    lambda x: pd.to_datetime(f"{int(x['year'])}-W{int(x['week'])}-1", format='%Y-W%W-%w'), axis=1)

# Create animated bar chart
fig = px.bar(
    weekly_counts,
    x='model2_predictions',
    y='amount',
    color='model2_predictions',
    animation_frame='start_time',  # Use formatted date as animation frame
    category_orders={'start_time': sorted(weekly_counts['start_time'].unique())},
    labels={'amount': 'amount', 'model2_predictions': 'emotions'},
    title='Dynamic chart of the distribution of the number of emotional comments per week',
    range_y=[0, weekly_counts['amount'].max() + 10]  # Keep y-axis range stable
)

# Optimize animation settings
fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 500
fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 300

# Adjust layout
fig.update_layout(
    showlegend=True,
    xaxis_tickangle=45,
    height=600,
    width=1200,
    coloraxis_showscale=False  # Turn off color scale due to too many colors
)

# Add week number annotation
fig.layout.sliders[0].currentvalue = {
    'prefix': 'Current week：',
    'font': {'size': 14}
}

# Display chart
fig.show()

# Save as HTML file
fig.write_html("Emotional Comment Dynamic Distribution.html")

import pandas as pd
import re
from collections import defaultdict

vocab_dict = {
    # Multi-word phrases (inside parentheses)
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

    # Single-word phrases
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

# Verify the number of entries
print(f"Vocabulary Amount: {len(vocab_dict)}")  # Output: Total vocabulary amount: 25

# Preprocessing function
def create_vocab_patterns(vocab_dict):
    """Create regex matching pattern dictionary"""
    patterns = {}
    for name, words in vocab_dict.items():
        # Generate word boundary matching pattern (case-insensitive)
        pattern = r'\b(' + '|'.join([re.escape(word) for word in words]) + r')\b'
        patterns[name] = re.compile(pattern, flags=re.IGNORECASE)
    return patterns

# Create regex pattern dictionary
vocab_patterns = create_vocab_patterns(vocab_dict)

# Check each comment for matches
for vocab_name, pattern in vocab_patterns.items():
    df[vocab_name] = df['body'].str.contains(pattern).astype(int)

# Group by week to count frequencies
weekly_freq = df.groupby(['year', 'week'])[list(vocab_dict.keys())].sum().reset_index()

# Melt frequency data
freq_melted = pd.melt(weekly_freq,
                     id_vars=['year', 'week'],
                     value_vars=vocab_dict.keys(),
                     var_name='vocab',
                     value_name='amount')

# Add time labels
freq_melted['start_time'] = freq_melted.apply(
    lambda x: pd.to_datetime(f"{int(x['year'])}-W{int(x['week'])}-1",
                            format='%Y-W%W-%w'), axis=1)

# Create animated bar chart
fig = px.bar(
    freq_melted,
    x='vocab',
    y='amount',
    color='vocab',
    animation_frame='start_time',
    category_orders={'start_time': sorted(freq_melted['start_time'].unique())},
    labels={'amount': 'amount'},
    title='Weekly Glossary Frequency Chart',
    range_y=[0, freq_melted['amount'].max() + 5],
    color_discrete_sequence=px.colors.qualitative.Alphabet  # Use extended color palette
)

# Optimize display settings
fig.update_layout(
    xaxis_tickangle=45,
    height=700,
    width=1400,
    showlegend=False,  # Hide legend due to too many colors
    hovermode='x unified'
)

# Add interactive tooltips
fig.update_traces(
    hovertemplate="<b>%{x}</b><br>frequency：%{y}"
)

# Adjust animation parameters
fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 800
fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 500

# Display and save
fig.show()
fig.write_html("Vocabulary Frequency Dynamic Distribution.html")
