


import praw
import datetime
import json 
from dotenv import load_dotenv
import os

load_dotenv()

# Reddit API credentials
reddit = praw.Reddit(
    client_id=os.getenv("client_id"),
    client_secret=os.getenv("client_secret"),
    user_agent=os.getenv("user_agent")
)

def fetch_reddit_data(subreddit_name, post_limit=3):
    subreddit = reddit.subreddit(subreddit_name)
    posts_data = []

    for post in subreddit.new(limit=post_limit):
        post_info = {}

        # Basic post info
        post_info["subreddit"] = subreddit_name
        post_info["id"] = post.id
        post_info["title"] = post.title
        post_info["selftext"] = post.selftext
        post_info["url"] = post.url
        post_info["created_utc"] = datetime.datetime.utcfromtimestamp(post.created_utc).isoformat()
        post_info["upvotes"] = post.score
        post_info["num_comments"] = post.num_comments

        # Author Info
        if post.author:
            try:
                author = post.author
                post_info["author"] = {
                    "username": str(author.name),
                    "link_karma": author.link_karma,
                    "comment_karma": author.comment_karma,
                    "total_karma": author.link_karma + author.comment_karma,
                    "account_created": datetime.datetime.utcfromtimestamp(author.created_utc).isoformat()
                }
            except Exception as e:
                post_info["author"] = {"error": str(e)}
        else:
            post_info["author"] = {"username": "deleted"}

        # Top-level comments
        post.comments.replace_more(limit=0)
        top_comments = [comment.body for comment in post.comments.list()[:5]]
        post_info["top_comments"] = top_comments

        posts_data.append(post_info)

    return posts_data

# Subreddits of interest
subreddits = ["worldnews", "geopolitics", "IndiaNews", "entertainment"]

if __name__ == "__main__":
    all_posts = []

    for sub in subreddits:
        print(f"Fetching from r/{sub}")
        try:
            posts = fetch_reddit_data(subreddit_name=sub, post_limit=3)
            all_posts.extend(posts)
        except Exception as e:
            print(f"Error fetching from r/{sub}: {e}")

    # Save combined data
    with open("reddit_posts.json", "w", encoding='utf-8') as f:
        json.dump(all_posts, f, indent=4, ensure_ascii=False)

    print(f" Saved {len(all_posts)} posts from {len(subreddits)} subreddits to reddit_posts.json")
