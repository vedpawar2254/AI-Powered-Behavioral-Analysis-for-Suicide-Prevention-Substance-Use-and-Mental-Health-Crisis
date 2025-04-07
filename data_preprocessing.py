import praw
import pandas as pd
import re
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import json
import os

from dotenv import load_dotenv


load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
USER_AGENT = os.getenv("USER_AGENT")


nltk.download('punkt')
nltk.download('stopwords')




subreddits = ["depression", "SuicideWatch", "addiction", "Anxiety","traumatoolbox","MMFB","offmychest","stopdrinking"]


keywords = [
    "depressed", "depression", "anxiety", "suicidal", "addiction help", 
    "overwhelmed", "panic attack", "self-harm", "can't cope", "mental breakdown", 
    "need help", "drug abuse", "feeling hopeless", "relapse", "want to die"
]



reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
)


stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  
    text = re.sub(r'@\w+', '', text)     
    text = re.sub(r'#\w+', '', text)    
    text = re.sub(r'[^\w\s]', '', text) 
    text = re.sub(r'\d+', '', text)      
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(tokens)


posts = []


print("Fetching posts....")

for sub in subreddits:
    subreddit = reddit.subreddit(sub)
    for submission in subreddit.new(limit=50):  
        content = submission.title + " " + submission.selftext
        if any(kw.lower() in content.lower() for kw in keywords):
            cleaned = clean_text(content)
            posts.append({
                "post_id": submission.id,
                "timestamp": datetime.utcfromtimestamp(submission.created_utc).isoformat(),
                "subreddit": sub,
                "content": content,
                "cleaned_content": cleaned,
                "likes": submission.score,
                "comments": submission.num_comments
            })

print(f"Collected {len(posts)} relevant posts.")



df = pd.DataFrame(posts)
df.to_csv("mental_health_posts.csv", index=False)
df.to_json("mental_health_posts.json", orient="records", indent=2)

print("Data saved as 'mental_health_posts.csv' and 'mental_health_posts.json'")
