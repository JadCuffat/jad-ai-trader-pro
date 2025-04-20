import requests
import pandas as pd
from datetime import datetime, timedelta
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time

analyzer = SentimentIntensityAnalyzer()

def fetch_reddit_sentiment(subreddit="CryptoCurrency", days=60):
    all_sentiment = []

    for i in range(days):
        day = datetime.utcnow() - timedelta(days=i)
        after = int(datetime.combine(day, datetime.min.time()).timestamp())
        before = int(datetime.combine(day + timedelta(days=1), datetime.min.time()).timestamp())

        url = f"https://api.pushshift.io/reddit/search/submission/?subreddit={subreddit}&after={after}&before={before}&size=100"
        response = requests.get(url)
        data = response.json().get('data', [])

        daily_scores = []
        for post in data:
            text = post.get("title", "")
            if text:
                score = analyzer.polarity_scores(text)["compound"]
                daily_scores.append(score)

        avg_score = sum(daily_scores) / len(daily_scores) if daily_scores else 0
        all_sentiment.append({"date": day.date(), "reddit_sentiment": avg_score})
        time.sleep(1.2)

    df = pd.DataFrame(all_sentiment)
    df.sort_values("date", inplace=True)
    df.to_csv("reddit_sentiment.csv", index=False)
    print("✅ Saved as reddit_sentiment.csv")

fetch_reddit_sentiment()
