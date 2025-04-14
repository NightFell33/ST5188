import pandas as pd

old_posts = pd.read_csv('reddit_comments.csv')
new_posts = pd.read_csv('new_reddit_comments_apple_3_11.csv')

concat_posts = pd.concat([new_posts, old_posts], axis = 0)

concat_posts['standard_time'] = pd.to_datetime(concat_posts['created_utc'], unit='s', utc = True)

concat_posts.to_csv('concat_comments_3_11.csv', index = False)

concat_posts_id = pd.read_csv('concat_posts_3_11.csv')
