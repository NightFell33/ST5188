import pandas as pd
import plotly.express as px
from sympy.physics.units import frequency

df =  pd.read_csv('D:/file/predictions_results_vader+BERT.csv')

# 确保时间列正确解析
df['standard_time'] = pd.to_datetime(df['standard_time'])



# 创建年周组合列（ISO周数，周一作为周开始）
df['year'] = df['standard_time'].dt.isocalendar().year
df['week'] = df['standard_time'].dt.isocalendar().week

# 按周和情感进行分组统计
weekly_counts = df.groupby(['year', 'week', 'model2_predictions']).size().reset_index(name='amount')

# # 创建完整的时间情感组合（填充缺失周）
# all_combinations = pd.MultiIndex.from_product(
#     [weekly_counts[['year', 'week']].drop_duplicates().values,
#      weekly_counts['model2_predictions'].unique()],
#     names=['year', 'week', 'model2_predictions']
# ).to_frame(index=False)
#
# weekly_counts = pd.merge(all_combinations, weekly_counts, how='left').fillna(0)

# 创建连续的时间标签
weekly_counts['start_time'] = weekly_counts.apply(
    lambda x: pd.to_datetime(f"{int(x['year'])}-W{int(x['week'])}-1", format='%Y-W%W-%w'), axis=1)

# 创建动态条形图
fig = px.bar(
    weekly_counts,
    x='model2_predictions',
    y='amount',
    color='model2_predictions',
    animation_frame='start_time',  # 使用格式化后的日期作为动画帧
    category_orders={'start_time': sorted(weekly_counts['start_time'].unique())},
    labels={'amount': 'amount', 'model2_predictions': 'emotions'},
    title='Dynamic chart of the distribution of the number of emotional comments per week',
    range_y=[0, weekly_counts['amount'].max() + 10]  # 保持y轴范围稳定
)

# 优化动画设置
fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 500
fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 300

# 调整布局
fig.update_layout(
    showlegend=True,
    xaxis_tickangle=45,
    height=600,
    width=1200,
    coloraxis_showscale=False  # 由于颜色太多，关闭色标
)

# 添加周数标注
fig.layout.sliders[0].currentvalue = {
    'prefix': 'Current week：',
    'font': {'size': 14}
}

# 显示图表
fig.show()

# 保存为HTML文件
fig.write_html("情感评论动态分布.html")





import pandas as pd
import re
from collections import defaultdict

vocab_dict = {
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
    # "sound_single": ["sound"], 
    "glass": ["glass"],
    "notification": ["notification"],
    "trump": ["trump"],
    "government": ["government"],
    "support": ["support"],
    "ecosystem": ["ecosystem"],
    "platform": ["platform"],
    "firmware": ["firmware"]
}

# 验证数量
print(f"Vocabulary Amount：{len(vocab_dict)}")  # 输出：总词汇表数量：25

# 预处理函数
def create_vocab_patterns(vocab_dict):
    """创建正则表达式匹配模式字典"""
    patterns = {}
    for name, words in vocab_dict.items():
        # 生成单词边界匹配模式（区分大小写）
        pattern = r'\b(' + '|'.join([re.escape(word) for word in words]) + r')\b'
        patterns[name] = re.compile(pattern, flags=re.IGNORECASE)
    return patterns

# 创建正则表达式模式字典
vocab_patterns = create_vocab_patterns(vocab_dict)

# 检查每个评论匹配情况
for vocab_name, pattern in vocab_patterns.items():
    df[vocab_name] = df['body'].str.contains(pattern).astype(int)

# # 时间处理（与之前相同）
# df['日期'] = pd.to_datetime(df['日期列名'])
# df['年'] = df['日期'].dt.isocalendar().year
# df['周'] = df['日期'].dt.isocalendar().week

# 按周聚合计数
weekly_freq = df.groupby(['year', 'week'])[list(vocab_dict.keys())].sum().reset_index()

# freq_melted = weekly_freq.melt(id_vars=['year', 'week'],var_name='vocab',value_name='amount')

# # 按周和情感进行分组统计
# weekly_counts = df.groupby(['year', 'week', 'model2_predictions']).size().reset_index(name='amount')

# # 创建完整时间序列
# all_weeks = df[['year', 'week']].drop_duplicates()
# full_index = pd.MultiIndex.from_product(
#     [all_weeks.values, vocab_dict.keys()],
#     names=['year', 'week', 'vocab']
# ).to_frame(index=False)

# # 合并填充缺失值
freq_melted = pd.melt(weekly_freq,
                     id_vars=['year', 'week'],
                     value_vars=vocab_dict.keys(),
                     var_name='vocab',
                     value_name='amount')
#
# weekly_full = pd.merge(full_index, weekly_long,
#                       how='left',
#                       on=['年', '周', '词汇表']).fillna(0)

# 添加时间标签
freq_melted['start_time'] = freq_melted.apply(
    lambda x: pd.to_datetime(f"{int(x['year'])}-W{int(x['week'])}-1",
                            format='%Y-W%W-%w'), axis=1)
#
# import plotly.express as px

# 创建动画条形图
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
    color_discrete_sequence=px.colors.qualitative.Alphabet  # 使用扩展色板
)

# 优化显示设置
fig.update_layout(
    xaxis_tickangle=45,
    height=700,
    width=1400,
    showlegend=False,  # 因颜色过多建议隐藏图例
    hovermode='x unified'
)

# 添加交互提示
fig.update_traces(
    hovertemplate="<b>%{x}</b><br>frequency：%{y}"
)

# 调整动画参数
fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 800
fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 500

# 显示并保存
fig.show()
fig.write_html("词汇表频率动态分布.html")
