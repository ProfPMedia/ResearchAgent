import requests
import json

subreddits = ["datascience",
"dataanalysis",
"dataengineering",
"machinelearning",
"businessanalysis",
"businessintelligence",
"learnpython",
"learnmachinelearning",
"dataisbeautiful",
"datasets",
"deeplearning",
"visualization",
"applyingtocollege"]


def getRedditCategory(reddit, numposts):
    url = "http://0.0.0.0:10000/reddit/category"
    payload = {
            "subreddit": f"{reddit}",
            "numposts": numposts,
        }
    headers = {
        'Content-Type': 'application/json',
        'accept': 'application/json',
        'access_token': "I Paint The Town Red!"
        }
    res = requests.post(url, json=payload, headers=headers).json()
    return res  

def getRedditSummary(url_to_summarize):
    url = "http://0.0.0.0:10000/reddit/summary"
    payload = {
            "url": f"{url_to_summarize}",
        }
    headers = {
        'Content-Type': 'application/json',
        'accept': 'application/json',
        'access_token': "I Paint The Town Red!"
        }
    res = requests.post(url, json=payload, headers=headers).json()
    return res  
 
def getResearch(query):
    url = "http://0.0.0.0:10000/research"
    payload = {
            "query": f"{query}",
        }
    headers = {
        'Content-Type': 'application/json',
        'accept': 'application/json',
        'access_token': "I Paint The Town Red!"
        }
    res = requests.post(url, json=payload, headers=headers).json()
    return res  

f = open("reddits_summary.txt", "w")

tmp = getResearch("Why did things escalate in Israel recently?")
print("RESEARCH:\n",tmp)
f.write("RESEARCH:\n")
f.write(tmp)
f.write("\n\n")

tmp = getRedditCategory("datasets",10)
print("HOT/TOP Reddit Posts:\n",tmp)
f.write("HOT/TOP Reddit Posts:\n")
f.write( json.dumps(tmp))
f.write("\n\n")

tmp = getRedditSummary("https://www.reddit.com/r/datasets/comments/173rxw1/are_there_any_cool_geology_datasets/")
print("Reddit Summary:\n",tmp)
f.write("Reddit Summary:\n")
f.write(tmp)
f.write("\n\n")

f.close()