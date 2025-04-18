import praw
import pandas as pd
import time
from tqdm import tqdm

reddit = praw.Reddit(
    client_id='',
    client_secret='',
    user_agent=''
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
new_posts_df = get_posts_by_time('Iphone', limit=2000, sort='new')
post_ids = new_posts_df['id'].tolist()


# get all comments for a post
def get_all_comments(post_id):
    post = reddit.submission(id=post_id)
    post.comments.replace_more(limit=None)   # Expand all nested comments
    comments = []

    for comment in post.comments.list():  # get all comments
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
DELAY = 1.1   # Adjust based on Reddit API limits (1.1 seconds/request)

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
        time.sleep(10) # Extend wait time on error

# Merge and save results
final_df = pd.concat(all_comments, ignore_index=True)
final_df.to_csv('reddit_comments_Iphone.csv', index=False)
print(f"get {len(final_df)} comments")
new_posts_df.to_csv('hot_posts_Iphone.csv', index=False)
print(f"get {len(new_posts_df)} posts")
