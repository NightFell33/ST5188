import praw
import pandas as pd
import time
from tqdm import tqdm  # 可选，用于进度条

reddit = praw.Reddit(
    client_id='pi5KqsKi1X_m3czgdSFRVw',
    client_secret='jmgplRBFxA6ZqTO89uvQm6t72SzlcA',
    user_agent='Myapp by NightFell33 (2630008256@qq.com)'
)


# 获取按时间排序的帖子（调整为new/hot/top等）
def get_posts_by_time(subreddit_name, limit=2000, sort='new'):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []

    # 根据排序类型获取帖子
    if sort == 'new':
        submission_generator = subreddit.new(limit=limit)
    elif sort == 'hot':
        submission_generator = subreddit.hot(limit=limit)
    else:
        raise ValueError("Unsupported sort type")

    for post in submission_generator:
        posts.append({
            'id': post.id,
            'created_utc': post.created_utc,
            'title': post.title,
            'url': post.url
        })

    # 按发布时间排序（PRAW的new本身是按时间排序，此处可省略）
    return pd.DataFrame(posts).sort_values('created_utc', ascending=False)


# 获取所有帖子的ID
new_posts_df1 = get_posts_by_time('apple', limit=2000, sort='new')
# post_ids = new_posts_df1['id'].tolist()  # 直接使用PRAW的post.id更可靠

new_posts_df1['standared_time'] = pd.to_datetime(new_posts_df1['created_utc'], unit='s', utc=True)

new_posts_df1_cut = new_posts_df1[0:232]

old_posts = pd.read_csv('new_posts.csv')

old_posts['standared_time'] = pd.to_datetime(old_posts['created_utc'], unit = 's', utc=True)

concat_posts_3_11 = pd.concat([new_posts_df1_cut, old_posts], axis = 0)

concat_posts_3_11.to_csv('concat_posts_3_11.csv', index = False)

new_posts_df1_cut.to_csv('cut_posts_3_11.csv', index = False)

post_ids = new_posts_df1_cut['id'].tolist()
# 获取完整评论数据
def get_all_comments(post_id):
    post = reddit.submission(id=post_id)
    post.comments.replace_more(limit=None)  # 展开所有嵌套评论
    comments = []

    for comment in post.comments.list():  # 获取全部评论
        comments.append({
            'post_id': post_id,
            'comment_id': comment.id,
            'author': comment.author.name if comment.author else 'N/A',
            'score': comment.score,
            'body': comment.body,
            'created_utc': comment.created_utc
        })

    return pd.DataFrame(comments)


# 循环获取所有评论并遵守速率限制
all_comments = []
request_count = 0
DELAY = 1.1  # 根据Reddit API限制调整（1.1秒/请求）

for pid in tqdm(post_ids):
    try:
        comments_df = get_all_comments(pid)
        all_comments.append(comments_df)
        request_count += 1

        # 遵守速率限制
        if request_count % 60 == 0:
            time.sleep(60)  # 每分钟不超过60次请求
        else:
            time.sleep(DELAY)

    except Exception as e:
        print(f"Error fetching {pid}: {str(e)}")
        time.sleep(10)  # 出错时延长等待

# 合并并保存结果
final_df = pd.concat(all_comments, ignore_index=True)


final_df.to_csv('new_reddit_comments_apple_3_11.csv', index=False)
print(f"共爬取 {len(final_df)} 条评论")
