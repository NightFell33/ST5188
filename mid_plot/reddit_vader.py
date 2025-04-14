import pandas as pd
import re
import nltk
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

# 加载数据（假设是CSV格式）
df = pd.read_csv('concat_comments_3_11.csv')

# 基础文本清洗
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # 去除URL
    text = re.sub(r'[^\w\s]', '', text)  # 去除标点
    text = text.lower().strip()         # 转为小写
    return text

#应用文本清洗
df['text'] = df['body'].astype(str)
df['cleaned_text'] = df['text'].apply(clean_text)

#删除空行和已删除的评论
print(df.isnull().sum())
df = df.dropna(subset=['cleaned_text'])
row2delete = df[df['cleaned_text'] == 'deleted'].index
df = df.drop(row2delete)


##
from nltk.sentiment import SentimentIntensityAnalyzer

# 初始化VADER
nltk.download('vader_lexicon')
vader = SentimentIntensityAnalyzer()

# 定义分析函数
def vader_sentiment(text):
    scores = vader.polarity_scores(text)
    return scores['compound']  # 返回综合得分

# 应用分析
df['vader_score'] = df['cleaned_text'].apply(vader_sentiment)

# 转换为三分类
df['vader_label'] = pd.cut(df['vader_score'],
                           bins=[-1, -0.05, 0.05, 1],
                           labels=['negative', 'neutral', 'positive'])




# 分析
# 保存结果
df.to_csv('analyzed_comments.csv', index=False)

# 基本分析
print("VADER分布:")
print(df['vader_label'].value_counts())


# 如果显示object类型需要强制转换
df['datetime_time'] = pd.to_datetime(df['standard_time'])

# 按时间顺序排序
df = df.sort_values(by='datetime_time', ascending=False)

# 预处理时永久设置时间索引
df = df.set_index('datetime_time')

# 按周统计情感趋势
df1 = df.resample('W')['vader_score'].mean()

plt.plot(df1)
plt.title('Average of weekly comment emotional tendency scores')
plt.show()

from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# 下载NLTK资源（首次运行需要执行）
nltk.download('stopwords')
nltk.download('punkt')

def process_text(text):
    # 加载停用词库
    stop_words = set(stopwords.words('english'))

    # 添加自定义停用词（根据实际情况调整）
    custom_stopwords = {
        'www', 'com', 'http', 'https', 'just', 'dont'
    }
    stop_words = stop_words.union(custom_stopwords)

    # 分词并过滤停用词
    tokens = word_tokenize(text)
    filtered = [word for word in tokens
                if word.lower() not in stop_words
                and len(word) > 2  # 过滤短词
                and word.isalpha()]  # 保留纯字母词

    return ' '.join(filtered)

# 生成正面评论词云
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