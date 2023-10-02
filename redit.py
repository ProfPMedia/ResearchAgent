from datetime import datetime
import pandas as pd
import praw
import os


def getHotPosts(subreddit: str) -> str:
    # How many posts
    posts = 10

    # Authenticate
    reddit = praw.Reddit(
        client_id=os.getenv("REDIT_CLIENT_ID"),
        client_secret=os.getenv("REDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDIT_USER_AGENT")
    )

    submissions = []
    comments = []
    result = ""
    subreddit = reddit.subreddit(subreddit)
    # Read each post and its comments and generate posts and comments dataframes
    for submission in subreddit.hot(limit=posts):
        submissions.append({'date':datetime.fromtimestamp(submission.created_utc),'subredit':subreddit,'title':submission.title,'score':submission.score,'comments':submission.num_comments,'id':submission.id,'url':submission.url,'post':submission.selftext})
        submission.comments.replace_more(limit=0)
        for comment in submission.comments.list():
            comments.append({'id':submission.id,'comment':comment.body})
    pdf = pd.DataFrame(submissions)
    cdf = pd.DataFrame(comments)
    if len(pdf)>0:
        for i in range(len(pdf)):
            result = result + f"SUBREDDIT: {subreddit}\n TITLE: {pdf.iloc[i].title}\n POST: {pdf.iloc[i].post}\n COMMENTS: "
            if len(cdf)>0:
                for c in cdf[cdf.id==pdf.iloc[i].id].comment:
                    result = result + c + "\n"
    return result