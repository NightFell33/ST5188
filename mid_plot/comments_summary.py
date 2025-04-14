import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('analyzed_comments.csv')

df['datetime_time']=pd.to_datetime(df['standard_time'])

df.set_index('datetime_time', inplace=True)

weekly_counts = df.resample('D').count()

plt.plot(weekly_counts['created_utc'])
plt.title('Distribution of the number of comments per day')
plt.show(block = True)

weekly_counts = df.resample('W').count()
plt.plot(weekly_counts['created_utc'])
plt.title('Distribution of the number of comments per week')
plt.show(block = True)