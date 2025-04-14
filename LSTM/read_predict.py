import pandas as pd

lda = pd.read_csv('D:/file/predictions_results_vader+BERT.csv')

lstm_data = pd.read_csv('D:/file/preprocessed_balanced_classified_goemotions.csv')

lstm_data1= lstm_data.drop(['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity',
             'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement',
             'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
             'relief', 'remorse', 'sadness', 'surprise'], axis=1)

lstm_data1 = lstm_data1.dropna(subset=['emotion'])

df = lstm_data1
df['negative'] = (df['emotion_category'] == 'negative').astype(int)
df['positive'] = (df['emotion_category'] == 'positive').astype(int)
df['ambiguous'] = (df['emotion_category'] == 'ambiguous').astype(int)

df.to_csv('lstm_data2.csv')