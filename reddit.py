import asyncpraw
import os
import time
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter


# How many posts
NUM_POSTS = 10

def getAPIref():
    return asyncpraw.Reddit(
        client_id=os.getenv("REDIT_CLIENT_ID"),
        client_secret=os.getenv("REDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDIT_USER_AGENT")
    )

"""
Routine to get a specified number of the hot and top posts
from a subreddit.
"""
async def getRedditPosts(subreddit_topic: str, num_posts: int = NUM_POSTS) -> list:
    # Authenticate
    reddit = getAPIref()

    # Grab the posts from the specified subreddit category
    subreddit = await reddit.subreddit(subreddit_topic, fetch=True)

    # Iterate through the posts in this subredit
    results=[]
    async for submission in subreddit.hot(limit=num_posts):
        results.append({"subreddit": subreddit_topic,
               "id": submission.id,
               "url":submission.url,
               "title": submission.title,
               "post": submission.selftext,
               "c_cnt": submission.num_comments,           
               })
    async for submission in subreddit.top(limit=num_posts):
        results.append({"subreddit": subreddit_topic,
               "id": submission.id,
               "url":submission.url,
               "title": submission.title,
               "post": submission.selftext,
               "c_cnt": submission.num_comments,           
               })   
    return results



"""
Routine to Summarize a specific reddit post
including its comments.
"""
async def redditSummary(url: str) -> str:

    # Authenticate
    reddit = getAPIref()

    start_time = time.time()
    # Iterate through the comments
    submission = await reddit.submission(url=url)
    print(f"Number of comments in {url}: ",submission.num_comments)
    result = ""
    comments = await submission.comments()
    await comments.replace_more(limit=None)
    all_comments = comments.list()
    for comment in all_comments:
        result = result + f"{comment.body}\n"

    comment_time = time.time()        

    # Summarize the Redit Post.
    pre_prompt = f"""
    Your task is to generate a summary of this reddit post: {submission.selftext} with topic: {submission.title}. 
    and the comments delimited by the triple backticks.\ 
    """
    prompt_template = pre_prompt + """
    Structure the summary should be in markdown with the following sections: \
    * Topic: <list the topic> \
    ** Subreddit: <list the subreddit> \
    ** Post: <list the post> \
    ** Comment Summary: <list the major topics and posts in the comments> \
    ** References: <list any urls provided in the post or comments one per line> \
    ```{text}``` Summary:
    """

    # Define LLM chain
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")

    # Define a Map Reduce Chain
    map_reduce_chain = load_summarize_chain(llm, chain_type="map_reduce")

    # Set the summary prompt
    map_reduce_chain.combine_document_chain.llm_chain.prompt.template = prompt_template
    
    # Split text to assure it fits in the context window
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(result)

    # Create multiple documents
    docs = [Document(page_content=t) for t in texts]

    # Generate the summary and append it to the results
    summary_of_post = map_reduce_chain.run(docs)
    
    summary_time = time.time()


    elapsed_time = comment_time - start_time  # Calculate the elapsed time
    print(f"Comment time: {elapsed_time}:4.2f seconds")
    elapsed_time = summary_time - comment_time  # Calculate the elapsed time
    print(f"LLM Summarization time: {elapsed_time}:4.2f seconds")

    return summary_of_post