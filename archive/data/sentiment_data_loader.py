import praw
import pandas as pd

# Set up Reddit API credentials
reddit = praw.Reddit(
    client_id="hWYH3nOo_HgpMkDQXU-7gg",
    client_secret="YyWEaUgNUPFR3yMHxJvJIDaEPR01og",
    user_agent="crypto_sentiment_collector"
)

# Function to fetch posts
def fetch_reddit_posts(subreddit_name, limit=100):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    for post in subreddit.hot(limit=limit):
        posts.append({
            "title": post.title,
            "created_utc": post.created_utc,
            "text": post.selftext
        })
    return pd.DataFrame(posts)

# Example: Fetch posts from r/CryptoCurrency
crypto_posts = fetch_reddit_posts("CryptoCurrency", limit=200)
print(crypto_posts.head())
crypto_posts.to_csv("reddit_posts.csv", index=False)
