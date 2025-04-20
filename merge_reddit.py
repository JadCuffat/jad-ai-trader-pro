import pandas as pd

# Load existing merged dataset
df = pd.read_csv("merged_dataset.csv", parse_dates=["date"])

# Load Reddit sentiment
reddit = pd.read_csv("reddit_sentiment.csv", parse_dates=["date"])

# Merge on date
df = df.merge(reddit, on="date", how="left")

# Fill any missing sentiment with 0
df["reddit_sentiment"].fillna(0, inplace=True)

# Save updated version
df.to_csv("final_dataset.csv", index=False)
print("✅ Final dataset with Reddit sentiment saved as final_dataset.csv")
