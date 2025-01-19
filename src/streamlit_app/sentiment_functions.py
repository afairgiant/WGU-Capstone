import os
import time

import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def fetch_news(api_key, query, language="en", page_size=10):
    """
    Fetch news articles using NewsAPI.

    Parameters:
        api_key (str): Your NewsAPI key.
        query (str): Query for the news search (e.g., "Bitcoin").
        language (str): Language of the articles (default: "en").
        page_size (int): Number of articles to fetch per request (default: 10).

    Returns:
        list: A list of dictionaries containing article details.
    """
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": language,
        "pageSize": page_size,
        "apiKey": api_key,
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        articles = response.json().get("articles", [])
        return articles
    else:
        print(f"Error fetching news: {response.status_code} {response.text}")
        return []


def analyze_sentiment_vader(text):
    """
    Analyze sentiment of the given text using VADER.

    Parameters:
        text (str): The text to analyze.

    Returns:
        dict: Sentiment scores (negative, neutral, positive, compound).
    """
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)


def save_sentiment_data(df, cache_file):
    """
    Save sentiment data to a CSV file.

    Parameters:
        df (pd.DataFrame): DataFrame containing sentiment data.
        cache_file (str): Path to the cache file.
    """
    df.to_csv(cache_file, index=False)


def load_sentiment_data(cache_file):
    """
    Load sentiment data from a CSV file.

    Parameters:
        cache_file (str): Path to the cache file.

    Returns:
        pd.DataFrame: Loaded DataFrame or None if the file is empty.
    """
    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file)
            if df.empty:
                print("Cache file is empty.")
                return None
            return df
        except pd.errors.EmptyDataError:
            print("Cache file is empty or invalid.")
            return None
    return None


def is_cache_valid(cache_file, max_age_hours):
    """
    Check if the cache file is still valid.

    Parameters:
        cache_file (str): Path to the cache file.
        max_age_hours (int): Maximum age of the cache in hours.

    Returns:
        bool: True if the cache is valid, False otherwise.
    """
    if os.path.exists(cache_file):
        file_age = time.time() - os.path.getmtime(cache_file)
        return file_age <= max_age_hours * 3600
    return False


def process_news_sentiment(
    api_key,
    query,
    language="en",
    page_size=10,
    cache_file="news_sentiment_cache.csv",
    max_age_hours=6,
):
    """
    Fetch news and perform sentiment analysis on each article. Use cache if available and valid.

    Parameters:
        api_key (str): Your NewsAPI key.
        query (str): Query for the news search (e.g., "Bitcoin").
        language (str): Language of the articles (default: "en").
        page_size (int): Number of articles to fetch per request (default: 10).
        cache_file (str): Path to the cache file (default: "news_sentiment_cache.csv").
        max_age_hours (int): Maximum age of the cache in hours (default: 6).

    Returns:
        pd.DataFrame: DataFrame containing article details and sentiment scores.
    """
    if is_cache_valid(cache_file, max_age_hours):
        print("Loading data from cache...")
        cached_data = load_sentiment_data(cache_file)
        if cached_data is not None:
            return cached_data

    print("Fetching fresh data...")
    articles = fetch_news(api_key, query, language, page_size)
    results = []

    for article in articles:
        sentiment = analyze_sentiment_vader(article["description"] or "")
        results.append(
            {
                "title": article["title"],
                "description": article["description"],
                "url": article["url"],
                "published_at": article["publishedAt"],
                "sentiment_neg": sentiment["neg"],
                "sentiment_neu": sentiment["neu"],
                "sentiment_pos": sentiment["pos"],
                "sentiment_compound": sentiment["compound"],
            }
        )

    df = pd.DataFrame(results)
    save_sentiment_data(df, cache_file)
    return df


# Example usage
if __name__ == "__main__":
    NEWS_API_KEY = "d6fd5c5f1f6d423eaf6ba10dd1f197ac"
    QUERY = "Bitcoin"

    # Fetch news and analyze sentiment with caching
    sentiment_df = process_news_sentiment(NEWS_API_KEY, QUERY)

    # Display the DataFrame
    print(sentiment_df)

    # Save to CSV if needed
    sentiment_df.to_csv("news_sentiment.csv", index=False)
