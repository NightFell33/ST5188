import pandas as pd
import re
from tqdm import tqdm  # 进度条支持

# 假设原始数据结构包含：评论文本、情感标签、日期
df = pd.read_csv('D:/file/predictions_results_vader+BERT.csv', parse_dates=['standard_time'])

# 示例词汇表定义（需替换为实际25个词汇表）
vocab_groups = {
    # 多词组（括号内部分）
    "app_group": ["app", "apps", "store", "software"],
    "music_group": ["music", "sound", "voice"],
    "version_group": ["version", "update"],
    "chip_group": ["chip", "core"],
    "photo_group": ["photo", "camera", "lens"],
    "car_group": ["carplay", "car"],
    "AI_group": ["AI", "chatgpt", "bot", "revolutionary"],
    "company_group": ["company", "business"],
    "price_group": ["price", "pay", "money", "dollar"],
    "market_group": ["market", "discount"],  # 注意保留原始拼写

    # 单词组
    "siri": ["siri"],
    "display": ["display"],
    "battery": ["battery"],
    "video": ["video"],
    "game": ["game"],
    "magsafe": ["magsafe"],
    "looking": ["looking"],
    # "sound_single": ["sound"],  # 单独存在的sound
    "glass": ["glass"],
    "notification": ["notification"],
    "trump": ["trump"],
    "government": ["government"],
    "support": ["support"],
    "ecosystem": ["ecosystem"],
    "platform": ["platform"],
    "firmware": ["firmware"]
}


# 构建正则表达式模式（匹配完整单词）
def build_regex_pattern(words):
    return r'\b(' + '|'.join([re.escape(word) for word in words]) + r')\b'

# 创建词汇匹配列
for group_name, keywords in tqdm(vocab_groups.items()):
    pattern = build_regex_pattern(keywords)
    df[group_name] = df['text'].str.contains(
        pattern,
        case=False,  # 不区分大小写
        regex=True,
        na=False
    )

# 根据GoEmotions官方标签定义正负情绪（需根据实际标签调整）
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

# 创建映射字典
emotion_mapping = {emotion: 'positive' for emotion in positive_emotions}
emotion_mapping.update({emotion: 'negative' for emotion in negative_emotions})

# 情感标签合并（根据之前定义）
df['sentiment'] = df['model2_predictions'].map(emotion_mapping)
df = df.dropna(subset=['sentiment'])  # 过滤中性情绪

# 转换时间格式
df['year_week'] = df['standard_time'].dt.strftime('%Y-%W')

# 创建空数据容器
result_dfs = []

# 遍历每个词汇表进行统计
for vocab in tqdm(vocab_groups.keys()):
    # 筛选包含当前词汇的评论
    temp_df = df[df[vocab]]

    # 按时间和情绪聚合
    grouped = temp_df.groupby(['year_week', 'sentiment']).size().unstack(fill_value=0)
    grouped['vocab_group'] = vocab  # 添加词汇表标识

    result_dfs.append(grouped)

# 合并结果
final_df = pd.concat(result_dfs).reset_index()

# 数据重塑
melted_df = final_df.melt(
    id_vars=['year_week', 'vocab_group'],
    value_vars=['positive', 'negative'],
    var_name='sentiment',
    value_name='count'
)

# 添加时间排序依据
melted_df['sort_date'] = pd.to_datetime(melted_df['year_week'] + '-1', format='%Y-%W-%w')




import plotly.express as px

# 生成交互式分面图
fig = px.line(
    melted_df,
    x='sort_date',
    y='count',
    color='sentiment',
    facet_col='vocab_group',
    facet_col_wrap=5,  # 每行显示5个词汇表
    height=2000,  # 根据实际需要调整
    title='Temporal Trends in Positive and Negative Sentiment by Vocabulary Group',
    labels={'sort_date': 'time', 'count': 'frequency'},
    color_discrete_map={'positive':'#1f77b4', 'negative':'#ff7f0e'}
)

# 优化布局
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

# 调整坐标轴显示
fig.update_xaxes(
    tickformat="%Y\n%W周",  # 显示年+周数
    dtick="M1",  # 每月显示一个主刻度
    showticklabels=True,
    matches=None  # 允许不同子图有不同刻度
)

# 优化子图间距
fig.update_layout(
    margin=dict(l=50, r=50, b=100),
    font=dict(size=10)
)

# 添加滚动条
fig.update_layout(
    xaxis=dict(rangeslider=dict(visible=True)),
    xaxis2=dict(rangeslider=dict(visible=True)),
    # ... 需要为每个子图单独设置
)

fig.show()
fig.write_html('双情感词频统计.html')