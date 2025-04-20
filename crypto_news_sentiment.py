import requests
import pandas as pd
import time

API_KEY = '71f66c58f3217e3ea0204b03968c543ba24ec448'  # Your CryptoPanic key

def fetch_all_news_sentiment(pages=20):
    all_articles = []

    for page in range(1, pages + 1):
        url = f'https://cryptopanic.com/api/v1/posts/?auth_token={API_KEY}&kind=news&filter=rising&page={page}'
        response = requests.get(url)

        if response.status_code != 200:
            print(f"Failed on page {page}, status code: {response.status_code}")
            break

        data = response.json()
        for item in data.get('results', []):
            all_articles.append({
                'published_at': item['published_at'],
                'title': item['title'],
                'sentiment': item.get('sentiment') or 'neutral'
            })

        time.sleep(1.5)

    df = pd.DataFrame(all_articles)
    df['published_at'] = pd.to_datetime(df['published_at']).dt.date
    df['sentiment_score'] = df['sentiment'].map({'positive': 1, 'negative': -1, 'neutral': 0})
    daily_sentiment = df.groupby('published_at')['sentiment_score'].mean().reset_index()
    daily_sentiment.columns = ['date', 'news_sentiment']

    daily_sentiment.to_csv("crypto_news_sentiment.csv", index=False)
    print("✅ Saved as crypto_news_sentiment.csv")

# Run it
fetch_all_news_sentiment()
