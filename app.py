
from fastapi import Depends, FastAPI, Response
from fastapi.security.api_key import APIKey
from pydantic import BaseModel, Field
import auth
import reddit 
import research


# Set this as an API endpoint via FastAPI
app = FastAPI()

# Pydantic model for the input query
class Query(BaseModel):
    query: str

# Pydantic model for the subreddit
class RedditObj(BaseModel):
    subreddit: str
    numposts: int

# Pydantic model for the subreddit summary
class RedditSummaryObj(BaseModel):
    subredditId: str

@app.post("/research")
async def researchAgent(query: Query, api_key: APIKey = Depends(auth.get_api_key)):
    query = query.query
    print("Doing Research on: ",query)
    content = research.agent({"input": query})
    actual_content = content['output']
    return actual_content

 
@app.post("/reddit/category")
async def redditCategoryAgent(reddit_obj: RedditObj, api_key: APIKey = Depends(auth.get_api_key)):
    subreddit = reddit_obj.subreddit
    numposts = reddit_obj.numposts
    print("Pulling hot/top posts on: ",subreddit,numposts)
    content = await reddit.getRedditPosts(subreddit,numposts)
    return content

@app.post("/reddit/summary")
async def redditCategoryAgent(reddit_obj: RedditSummaryObj, api_key: APIKey = Depends(auth.get_api_key)):
    id = reddit_obj.subredditId
    print("Summarizing: ",id)
    # Summarize the content
    summary = await reddit.redditSummary(id)
    return summary


@app.get("/health")
def health(response: Response):
    response.status_code = 200
    return "The Agent is still ticking...\n"
