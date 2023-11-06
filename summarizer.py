import sys
import os
import time
import pandas as pd
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import textwrap
import nbformat as nbf

def read_transcript(id):
    txt = ""
    for x in YouTubeTranscriptApi.get_transcript(id):
        txt += f"{x['text']}\n "
    return txt

"""
Routine to Summarize a specific youtube video transcript
"""
def youtubeSummary(id: str, method='map-reduce') -> str:
     # Define LLM chain
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")

    # Split text to assure it fits in the context window
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=4000, chunk_overlap=500)

    start_time = time.time()
    texts = text_splitter.split_text(read_transcript(id))
    transcript_time = time.time()        
    elapsed_time = transcript_time - start_time  # Calculate the elapsed time
    print(f"read the transcript in: {elapsed_time:4.2f} seconds")

    # Create multiple documents
    docs = [Document(page_content=t) for t in texts]
    print(f"Video transcript split into {len(docs)} chunks (docs)")

    prompt_template = """You are a professor in the school of Data Science. Write a narrative summary in the second person for each of the main topics of the following youtube lecture transcript:

    {text}

    output the lecture summary in markdown format with lecture topic headings preceeded by ## and the summaries of each topic in a paragraph:"""

    if method == 'map-reduce':
        print(f"performning a map reduce summary of {id} video")
        # Define a Map Reduce Chain
        map_reduce_chain = load_summarize_chain(llm, chain_type="map_reduce")

        # Set the summary prompt
        map_reduce_chain.combine_document_chain.llm_chain.prompt.template = prompt_template
        
        # Generate the summary and append it to the results
        summary_of_video = map_reduce_chain.run(docs)
    else:
        print(f"performning a refined summary of {id} video")
        PROMPT = PromptTemplate(template=prompt_template, 
                                input_variables=["text"])

        refine_template = (
            "Your job is to produce a final summary\n"
            "We have provided an existing summary up to a certain point: {existing_answer}\n"
            "We have the opportunity to add topics to the existing summary"
            "(only if needed) with some more context below.\n"
            "------------\n"
            "{text}\n"
            "------------\n"
            "Given the new context, add topics to the original summary"
            "If the context isn't useful, return the original summary."
        )
        refine_prompt = PromptTemplate(
            input_variables=["existing_answer", "text"],
            template=refine_template,
        )
        chain = load_summarize_chain(llm, 
                                    chain_type="refine", 
                                    return_intermediate_steps=True, 
                                    question_prompt=PROMPT, 
                                    refine_prompt=refine_prompt)

        summary_of_video = chain({"input_documents": docs}, return_only_outputs=True)['output_text']

    
    summary_time = time.time()

    elapsed_time = summary_time - transcript_time  # Calculate the elapsed time
    print(f"LLM Summarization time: {elapsed_time:4.2f} seconds")

    return summary_of_video

# Read in the list of videos from the csv file given on the command line
# no header is expected and each line is https://youtu.be/<VIDEO ID>
def read_csv_into_dataframe(file_path):
    df = pd.read_csv(file_path, header=0)
    df['video']=[x.split('/')[-1] for x in df['video'].values]
    return df

def read_csv_into_dict(file_path):
    df = pd.read_csv(file_path, header=0)
    categories = df.category.unique()
    result = {}
    for c in categories:
        result[c]=[x.split('/')[-1] for x in df[df.category==c]['video'].values] 
    return result

def main():
    load_dotenv()  # This will load variables from .env into the environment

    # Check if any environment variable is needed
    some_env_variable = os.getenv('OPENAI_API_KEY')
    if some_env_variable:
        print(f"Environment variable OPENAI_API_KEY is set.")
    else:
        print("OPENAI_API_KEY not found in .env")

    if len(sys.argv) < 2:
        print("Please provide a CSV file path as an argument.")
        sys.exit(1)


    csv_file_path = sys.argv[1]
    video_dict = read_csv_into_dict(csv_file_path)
    # check if we need to create the output directory
    if not os.path.exists("output"):
        os.makedirs("output")

    for lecture_section in video_dict.keys():
        print(f"Starting to process the {lecture_section} section that containt {len(video_dict[lecture_section])} videos to process.")
        # Create a new Jupyter notebook and put the raw header on it.
        nb = nbf.v4.new_notebook()
        cell = nbf.v4.new_raw_cell("""---\ntitle: "TITLE"\ndate: "xxxx-xx-xx"\ncategories: []\n---""")
        nb.cells.append(cell)

        for id in video_dict[lecture_section]:
            method = "refine"
            result = youtubeSummary(id, method)
            cell = nbf.v4.new_markdown_cell(f"[youTube lecture video](https://www.youtube.com/watch?v={id})")
            nb.cells.append(cell)
            cell = nbf.v4.new_markdown_cell(result)
            nb.cells.append(cell)

        # Write the notebook to a file
        notebook_path = os.path.join("output/",f"{lecture_section}-refine.ipynb")
        with open(notebook_path, 'w') as f:
            nbf.write(nb, f)

if __name__ == "__main__":
    main()

"""
if method=="refine":
    wrapped_text = textwrap.fill(result['output_text'], 
                            width=100,
                            break_long_words=False,
                            replace_whitespace=False)
else:
    wrapped_text = textwrap.fill(result, 
                            width=100,
                            break_long_words=False,
                            replace_whitespace=False)        
print(wrapped_text)
"""