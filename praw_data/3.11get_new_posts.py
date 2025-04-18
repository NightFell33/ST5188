import praw
import pandas as pd
import time
from tqdm import tqdm  # 可选，用于进度条

reddit = praw.Reddit(
    client_id='pi5KqsKi1X_m3czgdSFRVw',
    client_secret='jmgplRBFxA6ZqTO89uvQm6t72SzlcA',
    user_agent='Myapp by NightFell33 (2630008256@qq.com)'
)


# get the posts by time(sort as new,hot,top, etc.)
def get_posts_by_time(subreddit_name, limit=2000, sort='new'):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []

    # get the posts with specified sort type
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

    # order by time
    return pd.DataFrame(posts).sort_values('created_utc', ascending=False)


# get id for all posts
new_posts_df1 = get_posts_by_time('apple', limit=2000, sort='new')
# post_ids = new_posts_df1['id'].tolist()

new_posts_df1['standared_time'] = pd.to_datetime(new_posts_df1['created_utc'], unit='s', utc=True)

new_posts_df1_cut = new_posts_df1[0:232]

old_posts = pd.read_csv('new_posts.csv')

old_posts['standared_time'] = pd.to_datetime(old_posts['created_utc'], unit = 's', utc=True)

concat_posts_3_11 = pd.concat([new_posts_df1_cut, old_posts], axis = 0)

concat_posts_3_11.to_csv('concat_posts_3_11.csv', index = False)

new_posts_df1_cut.to_csv('cut_posts_3_11.csv', index = False)

post_ids = new_posts_df1_cut['id'].tolist()

# get all comments for a post
def get_all_comments(post_id):
    post = reddit.submission(id=post_id)
    post.comments.replace_more(limit=None)
    comments = []

    for comment in post.comments.list():
        comments.append({
            'post_id': post_id,
            'comment_id': comment.id,
            'author': comment.author.name if comment.author else 'N/A',
            'score': comment.score,
            'body': comment.body,
            'created_utc': comment.created_utc
        })

    return pd.DataFrame(comments)


# Loop through all comments and respect rate limits
all_comments = []
request_count = 0
DELAY = 1.1  # Adjust based on Reddit API limits (1.1 seconds/request)

for pid in tqdm(post_ids):
    try:
        comments_df = get_all_comments(pid)
        all_comments.append(comments_df)
        request_count += 1

        # Respect rate limits
        if request_count % 60 == 0:
            time.sleep(60)  # No more than 60 requests per minute
        else:
            time.sleep(DELAY)

    except Exception as e:
        print(f"Error fetching {pid}: {str(e)}")
        time.sleep(10)  # Extend wait time on error

# Merge and save results
final_df = pd.concat(all_comments, ignore_index=True)


final_df.to_csv('new_reddit_comments_apple_3_11.csv', index=False)
print(f"get {len(final_df)} comments")
