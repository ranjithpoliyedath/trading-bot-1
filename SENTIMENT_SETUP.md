# Sentiment setup guide

## 1. Install dependencies
pip install praw transformers torch nltk

## 2. Add Reddit credentials to .env
Go to https://www.reddit.com/prefs/apps → create app (script type)

REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=trading-bot/0.1 by YourUsername

## 3. Run the sentiment pipeline
python -m bot.sentiment.sentiment_pipeline

## 4. Re-run the main data pipeline to merge features
python -m bot.pipeline

## 5. Retrain your ML model with new features
python models/train.py

## Sentiment features added to ML model (6 base + 3 momentum = 9 new features)
- news_sentiment_mean    Avg FinBERT score from Alpaca news
- news_sentiment_std     Volatility of news sentiment
- news_count             Number of articles that day
- reddit_sentiment_mean  Avg score from Reddit posts
- reddit_sentiment_std   Volatility of Reddit sentiment
- reddit_score_sum       Total upvotes (retail attention proxy)
- combined_sentiment     Weighted avg: 65% news + 35% Reddit
- sentiment_momentum     3-day rolling avg of combined_sentiment
- sentiment_change       Day-over-day sentiment change
- sentiment_accel        Acceleration of sentiment change
