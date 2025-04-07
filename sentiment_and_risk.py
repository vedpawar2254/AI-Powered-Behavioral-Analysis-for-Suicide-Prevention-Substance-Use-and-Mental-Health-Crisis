import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_json("./mental_health_posts.json") 
df['text'] = df['subreddit'] + " " + df['cleaned_content']

def clean_text(text):
    text = re.sub(r"http\S+", "", text) 
    text = re.sub(r"[^a-zA-Z\s]", "", text)  
    return text.lower().strip()

df['clean_text'] = df['text'].apply(clean_text)




analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = analyzer.polarity_scores(text)
    compound = score['compound']
    if compound >= 0.05:
        return 'Positive'
    elif compound <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment'] = df['clean_text'].apply(get_sentiment)



high_risk_terms = ["i want to die", "i don't want to be here", "end it all", "suicide", "kill myself"]
moderate_risk_terms = ["i feel lost", "i need help", "i'm struggling", "i'm tired of life"]
low_risk_terms = ["mental health is important", "i'm just venting"]

def get_risk_level(text):
    text = text.lower()
    if any(phrase in text for phrase in high_risk_terms):
        return 'High-Risk'
    elif any(phrase in text for phrase in moderate_risk_terms):
        return 'Moderate Concern'
    elif any(phrase in text for phrase in low_risk_terms):
        return 'Low Concern'
    else:
        return 'Uncategorized'

df['risk_level'] = df['clean_text'].apply(get_risk_level)





pivot = df.pivot_table(index='sentiment', columns='risk_level', aggfunc='size', fill_value=0)
pivot.plot(kind='bar', stacked=True)
plt.title("Reddit Posts by Sentiment and Risk Level")
plt.ylabel("Number of Posts")
plt.show()