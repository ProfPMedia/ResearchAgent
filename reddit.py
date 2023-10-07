import asyncpraw
import os
import openai
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

# How many posts
NUM_HOT_POSTS = 10
MAX_CHAR_LENGTH = 50000

def getAPIref():
    return asyncpraw.Reddit(
        client_id=os.getenv("REDIT_CLIENT_ID"),
        client_secret=os.getenv("REDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDIT_USER_AGENT")
    )

async def getRedditPosts(subreddit_topic: str, num_posts: int = NUM_HOT_POSTS) -> list:
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

async def getHotPosts(subreddit_topic: str, num_posts: int = NUM_HOT_POSTS) -> str:
    
    # Authenticate
    reddit = getAPIref()

    # Iterate through the top hot posts in this subredit
    results=[]
    subreddit = await reddit.subreddit(subreddit_topic, fetch=True)
    async for submission in subreddit.hot(limit=num_posts):
        print(f"Number of comments in {submission.url}: ",submission.num_comments)
        result = ""
        result = result + f"RefUrl: {submission.url}\n"
        result = result + f"Title: {submission.title}\n"
        result = result + f"Post: {submission.selftext}\n"
        comments = await submission.comments()
        await comments.replace_more(limit=None)
        all_comments = comments.list()
        result = result + "Comments: "
        for comment in all_comments:
            result = result + f"{comment.body}\n"
        
        #print(result)

        # Summarize the Redit Post and add it to the results.
        pre_prompt = f"""
        Your task is to generate a short summary of this reddit {subreddit_topic} subreddit \ 
        post id {submission.id}. 
        """
        prompt_template = pre_prompt + """
        delimited by triple \
        backticks by including the subreddit, the main topic or question, points or answers given, \
        and list any urls given under a distinct reference section one URL per line. \
        Take extra care to not repeat yourself. In particular only list a given URL once. \
        Output the summary in a JSON object with a key id for the post id, key subreddit for the given subreddit, a key topic \
        for the main topic which holds a string, \
        a key points for main points or answers given in the comments which is an array of strings and a key for the references \
        which is an arrary of strings.
        Post: ```{text}```
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
        results.append(summary_of_post)
        
        print(summary_of_post)

    return results