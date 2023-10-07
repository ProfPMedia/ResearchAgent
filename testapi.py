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

def get_the_hot_reddits(reddit, numposts=2):

    url = "http://0.0.0.0:10000/reddit"
    payload = {
            "subreddit": f"{reddit}",
            "numposts": numposts
        }
    headers = {
        'Content-Type': 'application/json',
        'accept': 'application/json',
        'access_token': "I Paint The Town Red!"
        }

    res = requests.post(url, json=payload, headers=headers).json()
    return res

#f = open("reddits_summary.txt", "a")


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


for r in subreddits:
    res = getRedditCategory(r,10)
    print(r,"\n",res)
    """
    f.write(r+"\n")
    for i in res:
        output = json.dumps(json.loads(i), indent=2)
        print(output,"\n")
        f.write(output+"\n")
    """
#f.close()