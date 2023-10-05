import asyncpraw
import os
import openai
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

# How many posts
NUM_HOT_POSTS = 10
MAX_CHAR_LENGTH = 50000

async def getHotPosts(subreddit_topic: str, num_posts: int = NUM_HOT_POSTS) -> str:
    
    # Authenticate
    reddit = asyncpraw.Reddit(
        client_id=os.getenv("REDIT_CLIENT_ID"),
        client_secret=os.getenv("REDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDIT_USER_AGENT")
    )

    # Iterate throgu the top hot posts in this subredit
    results=[]
    subreddit = await reddit.subreddit(subreddit_topic, fetch=True)
    async for submission in subreddit.hot(limit=num_posts):
        result = ""
        result = result + f"RefUrl: {submission.url}\n"
        result = result + f"Title: {submission.title}\n"
        result = result + f"Post: {submission.selftext}\n"
        comments = await submission.comments()
        await comments.replace_more(limit=None)
        all_comments = comments.list()
        for comment in all_comments:
            result = result + f"Comment: {comment.body}\n"

        # Summarize the Redit Post and add it to the results.
        pre_prompt = f"""
        Your task is to generate a short summary of the reddit {subreddit_topic} subreddit \ 
        post id {submission.id}. 
        """
        prompt_template = pre_prompt + """
        Summarize the post below, delimited by triple \
        backticks by including the subreddit, the main topic or question, points or answers given, \
        and list any urls given under a distinct reference section one URL per line. \
        Take extra care to not repeat yourself. In particular only list a given URL once. \
        Output the summary in a JSON object with a key id for the post id, key subreddit for the given subreddit, a key topic \
        for the main topic which holds a string, \
        a key points for main points or answers which is an array of strings no more than 20 elements long and a key for the references \
        which is an arrary of strings.
        Post: ```{text}```
        """
    
        prompt = PromptTemplate.from_template(prompt_template)
        
        # Define LLM chain
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        
        # Define StuffDocumentsChain
        stuff_chain = StuffDocumentsChain(
            llm_chain=llm_chain, document_variable_name="text"
        )
        
        # Split text
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
    
        if len(result) < MAX_CHAR_LENGTH:
            texts = text_splitter.split_text(result)
        else:
            print(f"truncated post of length {len(result)}")
            texts = text_splitter.split_text(result[0:MAX_CHAR_LENGTH])

        # Create multiple documents
        docs = [Document(page_content=t) for t in texts]
        results.append(stuff_chain.run(docs))
        
    return results