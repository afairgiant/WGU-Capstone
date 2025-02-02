import logging
import os
import time

import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import requests
from utils import setup_logging
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud

# Set up logging
setup_logging()


def fetch_news(api_key, query, max_pages=1, language="en", page_size=100):
    """
    Fetch news articles using NewsAPI with pagination and improved error handling.

    Parameters:
        api_key (str): Your NewsAPI key.
        query (str): Query for the news search (e.g., "Bitcoin").
        max_pages (int): Maximum number of pages to fetch (default: 1).
        language (str): Language of the articles (default: "en").
        page_size (int): Number of articles to fetch per request (default: 100, max: 100).

    Returns:
        list: A list of dictionaries containing article details.
    """
    logging.info(f"Fetching news articles for query: {query}")
    if not query:
        logging.error("Query parameter cannot be empty.")
        raise ValueError("Query parameter cannot be empty.")

    if page_size > 100:
        logging.error("Page size cannot exceed 100.")
        raise ValueError("Page size cannot exceed 100.")

    max_results = 100
    total_fetched = 0
    url = "https://newsapi.org/v2/everything"
    all_articles = []
    max_retries = 3

    for page in range(1, max_pages + 1):
        params = {
            "q": query,
            "language": language,
            "pageSize": min(page_size, max_results - total_fetched),
            "page": page,
            "apiKey": api_key,
        }

        if total_fetched >= max_results:
            logging.info("Reached maximum limit of 100 articles.")
            break

        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params)
                logging.info(
                    f"API Request Status Code: {response.status_code} for page {page}"
                )

                if response.status_code == 200:
                    response_json = response.json()
                    articles = response_json.get("articles", [])
                    if not articles:
                        logging.info(f"No articles found on page {page}.")
                        break

                    all_articles.extend(articles)
                    total_fetched += len(articles)
                    break

                elif response.status_code == 401:
                    logging.error("Invalid API Key.")
                    raise ValueError("Invalid API Key.")

                elif response.status_code == 403:
                    logging.error("API Key does not have the required permissions.")
                    raise ValueError("API Key does not have the required permissions.")

                elif response.status_code == 429:
                    logging.warning("Rate limit exceeded. Retrying after a delay...")
                    time.sleep(5)

                elif response.status_code >= 500:
                    logging.error("Server error. Retrying...")
                    time.sleep(2)

                else:
                    response.raise_for_status()

            except requests.exceptions.RequestException as e:
                logging.error(f"Request failed on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise

    logging.info(f"Total articles fetched: {len(all_articles)}")
    return all_articles


def analyze_sentiment_vader(text):
    """
    Analyze sentiment of the given text using VADER.

    Parameters:
        text (str): The text to analyze.

    Returns:
        dict: Sentiment scores (negative, neutral, positive, compound).
    """
    logging.info("Analyzing sentiment using VADER...")
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)


def save_sentiment_data(df, save_file):
    """
    Save or update sentiment data in a CSV file, ensuring no duplicates.

    Parameters:
        df (pd.DataFrame): DataFrame containing new sentiment data.
        save_file (str): Path to the save file.
    """
    logging.info(f"Saving sentiment data to {save_file}...")
    if os.path.exists(save_file):
        logging.info("Save file exists. Loading existing data...")
        try:
            # Load existing data
            existing_df = pd.read_csv(save_file)
            logging.info(f"Loaded {len(existing_df)} rows from save file.")
        except Exception as e:
            logging.error(f"Error loading save file: {e}")
            existing_df = pd.DataFrame()

        # Combine new and existing data
        combined_df = pd.concat([existing_df, df])
        logging.info(
            f"Combining data. Total rows before deduplication: {len(combined_df)}"
        )

        # Remove duplicates
        combined_df = combined_df.drop_duplicates(subset=["title", "url"])
        logging.info(f"Total rows after deduplication: {len(combined_df)}")
    else:
        logging.info("No save file found. Creating new save file.")
        combined_df = df

    # Save the updated data to the save file
    try:
        combined_df.to_csv(save_file, index=False)
        logging.info(
            f"Data saved successfully. Total rows in save file: {len(combined_df)}"
        )
    except Exception as e:
        logging.error(f"Error saving data to save file: {e}")


def load_sentiment_data(save_file):
    """
    Load sentiment data from a CSV file.

    Parameters:
        save_file (str): Path to the save file.

    Returns:
        pd.DataFrame: Loaded DataFrame or an empty DataFrame if the file doesn't exist.
    """
    logging.info(f"Loading sentiment data from {save_file}...")
    if os.path.exists(save_file):
        try:
            return pd.read_csv(save_file)
        except pd.errors.EmptyDataError:
            logging.warning("Save file is empty.")
            return pd.DataFrame()
    return pd.DataFrame()


def process_news_sentiment(api_key, query, language="en", page_size=10, max_pages=1):
    """
    Fetch news and perform sentiment analysis on each article.

    Parameters:
        api_key (str): Your NewsAPI key.
        query (str): Query for the news search (e.g., "Bitcoin").
        language (str): Language of the articles (default: "en").
        page_size (int): Number of articles to fetch per request (default: 10).
        max_pages (int): Number of pages to fetch (default: 1).

    Returns:
        pd.DataFrame: DataFrame containing article details and sentiment scores.
    """
    logging.info("Fetching and processing news sentiment...")
    articles = fetch_news(api_key, query, max_pages, language, page_size)
    if not articles:
        logging.error("No articles fetched from NewsAPI. Check query or API limits.")
        raise ValueError("No articles fetched from NewsAPI. Check query or API limits.")

    results = []
    for article in articles:
        sentiment = analyze_sentiment_vader(article.get("description", "") or "")
        results.append(
            {
                "title": article.get("title", ""),
                "description": article.get("description", ""),
                "url": article.get("url", ""),
                "published_at": article.get("publishedAt", None),
                "sentiment_neg": sentiment["neg"],
                "sentiment_neu": sentiment["neu"],
                "sentiment_pos": sentiment["pos"],
                "sentiment_compound": sentiment["compound"],
            }
        )

    df = pd.DataFrame(results)
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    df = df.dropna(subset=["published_at"])

    if df.empty:
        logging.error("No valid articles with 'published_at' timestamps available.")
        raise ValueError("No valid articles with 'published_at' timestamps available.")

    logging.info("News sentiment processing complete.")
    return df


def process_and_save_sentiment(
    api_key,
    query,
    save_file="sentiment_data.csv",
    language="en",
    page_size=10,
    max_pages=1,
):
    """
    Fetch news, perform sentiment analysis, and update the save file with new results.

    Parameters:
        api_key (str): Your NewsAPI key.
        query (str): Query for the news search (e.g., "Bitcoin").
        save_file (str): Path to the save file for storing sentiment data.
        language (str): Language of the articles (default: "en").
        page_size (int): Number of articles to fetch per request (default: 10).
        max_pages (int): Number of pages to fetch (default: 1).

    Returns:
        pd.DataFrame: Updated DataFrame with all sentiment data.
    """
    logging.info("Processing and saving sentiment data...")
    cached_data = load_sentiment_data(save_file)
    new_data = process_news_sentiment(api_key, query, language, page_size, max_pages)
    save_sentiment_data(new_data, save_file)
    return load_sentiment_data(save_file)


def plot_sentiment_over_time(df):
    """
    Create a line chart for sentiment compound scores over time.

    Parameters:
        df (pd.DataFrame): DataFrame containing sentiment data.

    Returns:
        Altair Chart: Line chart of sentiment compound scores.
    """
    logging.info("Plotting sentiment over time...")
    if df.empty or "published_at" not in df.columns or df["published_at"].isna().all():
        logging.error("No valid published_at data available for visualization.")
        raise ValueError("No valid published_at data available for visualization.")

    line_chart = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x="published_at:T",
            y="sentiment_compound:Q",
            tooltip=["title", "sentiment_compound"],
        )
        .properties(title="Sentiment Compound Score Over Time")
    )
    logging.info("Sentiment over time plot generated.")
    return line_chart


def plot_sentiment_distribution(df):
    """
    Create a bar chart for sentiment distribution.

    Parameters:
        df (pd.DataFrame): DataFrame containing sentiment data.

    Returns:
        Altair Chart: Bar chart of sentiment categories.
    """
    logging.info("Plotting sentiment distribution...")
    sentiment_categories = df["sentiment_compound"].apply(
        lambda x: "Positive" if x > 0 else ("Negative" if x < 0 else "Neutral")
    )
    sentiment_counts = sentiment_categories.value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]

    bar_chart = (
        alt.Chart(sentiment_counts)
        .mark_bar()
        .encode(
            x=alt.X("Sentiment:N", sort=["Positive", "Neutral", "Negative"]),
            y="Count:Q",
            color="Sentiment:N",
        )
        .properties(title="Sentiment Distribution")
    )
    logging.info("Sentiment distribution plot generated.")
    return bar_chart


def generate_word_cloud(df):
    """
    Generate a word cloud from article descriptions.

    Parameters:
        df (pd.DataFrame): DataFrame containing sentiment data.

    Returns:
        matplotlib Figure: Word cloud figure.
    """
    logging.info("Generating word cloud...")
    text = " ".join(df["description"].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
        text
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    logging.info("Word cloud generated.")
    return fig


if __name__ == "__main__":
    logging.info("Starting sentiment_functions script...")
    NEWS_API_KEY = "d6fd5c5f1f6d423eaf6ba10dd1f197ac"
    QUERY = "Bitcoin"
    SAVE_FILE = "src/streamlit_app/data/sentiment_data.csv"

    updated_sentiment_df = process_and_save_sentiment(
        NEWS_API_KEY, QUERY, save_file=SAVE_FILE
    )

    logging.info("Updated sentiment data:")
    logging.info(updated_sentiment_df.head())
