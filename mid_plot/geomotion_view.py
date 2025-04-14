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

# 下载nltk停用词数据
nltk.download('stopwords')

# 通用预处理函数
def preprocess_text(text):
    # 转换为小写
    text = text.lower()
    # 移除URL
    text = re.sub(r'http\S+', '', text)
    # 移除标点符号
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 移除非字母字符
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # 分词
    words = text.split()
    # 移除停用词
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return words

# 获取当前工作目录
current_work_dir = os.getcwd()
print(current_work_dir)

# 导入数据
goemotions = pd.read_csv('goemotions.csv', encoding='ISO-8859-1')

# 处理缺失值
print("缺失值统计:")
print(goemotions.isnull().sum())

# goemotions = goemotions.dropna(subset=['body'])

# 统计每条评论标注的情绪数量（单标签vs多标签）
goemotions['label_number'] = goemotions.iloc[:, 9:37].sum(axis=1)

# 绘制直方图
counts, bins, patches = plt.hist(goemotions['label_number'], bins=10, edgecolor='black', range=[0,3], orientation= "horizontal")

# 设置图形标题和标签
plt.title('Number of emotions labelled per comment')
plt.xlabel('Frequency')
plt.ylabel('Value')

# 显示图形
plt.show()



# 按评论ID分组，计算每个情感的总标注次数
grouped_sum = goemotions.groupby('id').sum()

# 检查每个评论是否有至少一个情感的总标注次数 >=2
agreement = grouped_sum.iloc[:,8:36].ge(2).any(axis=1)
percentage_agreement = agreement.mean() * 100

print(f"至少有两个标注者同意同一标签的样本比例: {percentage_agreement:.2f}%")


# 标注者层面的共现统计
co_occur_annotations = goemotions[(goemotions['excitement'] == 1) & (goemotions['amusement'] == 1)].shape[0]
total_excitement = goemotions['excitement'].sum()
co_occur_ratio = co_occur_annotations / total_excitement * 100

print(f"当标注者标记excitement时，同时标记amusement的比例: {co_occur_ratio:.2f}%")

# # 评论层面的共现统计
# co_occur_comments = goemotions.apply(lambda x: ((x['excitement'] == 1) & (x['amusement'] == 1)).any())
# co_occur_percent = co_occur_comments.sum() / len(goemotions) * 100
#
# print(f"至少有一个标注者同时标记两者的评论比例: {co_occur_percent:.2f}%")

# # 相关系数分析
# correlation = grouped_sum[['excitement', 'amusement']].corr().iloc[0, 1]
# print(f"Excitement和amusement的相关系数: {correlation:.4f}"



# 提取所有情感列名（假设情感列在数据中连续分布）
emotion_columns = goemotions.columns[9:37]  # 根据实际数据调整列范围（排除id和annotator_id等非情感列）

# 统计每个情感的总频次
emotion_counts = goemotions[emotion_columns].sum().sort_values(ascending=False)
emotion_counts.head(28)

# 提取Top10高频情绪
top10_emotions = emotion_counts.head(10)

# 绘制条形图
plt.figure(figsize=(12, 6))
sns.barplot(x=top10_emotions.values, y=top10_emotions.index, palette="viridis")
plt.title("Top 10 高频情绪分布")
plt.xlabel("频次")
plt.ylabel("情绪类别")
plt.show()