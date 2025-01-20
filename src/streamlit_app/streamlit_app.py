import os
import sys
from http import server

from utils import load_api_key

# Add the project root directory to Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
print("Current working directory:", os.getcwd())

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from data_downloader_2 import download_and_save_ohlc_data
from financial_functions import (
    calculate_daily_average,
    calculate_moving_averages,
    lstm_crypto_forecast,
    run_ohlc_prediction,
)
from matplotlib import markers
from sentiment_functions import (
    generate_word_cloud,
    plot_sentiment_distribution,
    plot_sentiment_over_time,
    process_news_sentiment,
)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

# Update any missing data from past 30 days.
COIN_ID = "bitcoin"
download_and_save_ohlc_data(COIN_ID, 30)

st.set_page_config(page_title="Crypto Data Visualizer", layout="wide")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Home", "Historical Charts", "LSTM Prediction", "About", "Testing"]
)

# Tab 1: Home
with tab1:
    st.title("Welcome to the App")
    st.write("This is the home page. Use the tabs above to navigate.")


# Tab 2: Charts
with tab2:
    # Path to the CSV file
    server_csv_path = "src/streamlit_app/data/ohlc_data.csv"

    # Streamlit Tab for Historical Data
    st.title("Historical Data")

    try:
        # Calculate moving averages
        moving_averages = calculate_moving_averages(server_csv_path)

        # Display the DataFrame
        st.subheader("Processed Data")
        st.dataframe(moving_averages)

        # Plot the data
        st.subheader("Moving Averages Chart")
        plt.figure(figsize=(10, 6))
        plt.plot(
            moving_averages["Time"],
            moving_averages["Daily Average Price"],
            label="Daily Average Price",
            color="blue",
        )
        plt.plot(
            moving_averages["Time"],
            moving_averages["7-Day Moving Average"],
            label="7-Day Moving Average",
            color="orange",
        )
        plt.plot(
            moving_averages["Time"],
            moving_averages["30-Day Moving Average"],
            label="30-Day Moving Average",
            color="green",
        )
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.title("Daily Average and Moving Averages")
        plt.legend(loc="upper left")  # Add legend to the chart
        plt.grid(True)

        # Display the plot in Streamlit
        st.pyplot(plt)

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Tab 3: Futures
with tab3:
    st.title("Bitcoin Price Prediction")

    # Path to the CSV file
    server_csv_path = "src/streamlit_app/data/ohlc_data.csv"

    # Number of days for predictions
    days = st.number_input(
        "Enter the number of future days to predict",
        min_value=1,
        max_value=365,
        value=30,
    )

    if server_csv_path is not None:
        try:
            # Read the uploaded file as a DataFrame
            data = pd.read_csv(server_csv_path)

            # Calculate the daily averages
            daily_averages = calculate_daily_average(data)

            # Run the prediction function
            predictions = run_ohlc_prediction(data, days)

            # Display the predictions in a table
            st.subheader("Predicted Prices")
            st.dataframe(predictions)

            # Plot the predictions and daily averages
            st.subheader(f"Price Predictions and Daily Averages ({days} Days)")
            fig, ax = plt.subplots()

            # Plot daily averages
            ax.plot(
                daily_averages["Date"],
                daily_averages["Average Price"],
                label="Daily Average",
                color="blue",
                marker="o",
            )

            # Plot predictions
            ax.plot(
                predictions["Date"],
                predictions["Predicted Price"],
                label="Predicted Price",
                color="red",
                marker="o",
                markersize=4,
            )

            ax.set_xlabel("Date")
            ax.set_ylabel("Price (USD)")
            ax.set_title(f"Predicted Prices and Daily Averages ({days} Days)")
            ax.legend()
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # Run the LSTM prediction function
            lstm_predictions = lstm_crypto_forecast(data, days)

            # Plot LSTM predictions and actual data
            st.subheader("LSTM Predicted Prices with Historical Data")
            fig_lstm, ax_lstm = plt.subplots()

            # Plot historical close prices
            ax_lstm.plot(
                data.index, data["close"], label="Historical Close Prices", color="blue"
            )

            # Plot LSTM predictions
            ax_lstm.plot(
                lstm_predictions["Date"],
                lstm_predictions["Predicted Price"],
                label="LSTM Predicted Price",
                color="green",
                marker="o",
                markersize=3,
            )

            ax_lstm.set_xlabel("Date")
            ax_lstm.set_ylabel("Price (USD)")
            ax_lstm.set_title(
                f"LSTM Predicted Prices with Historical Data ({days} Days)"
            )
            ax_lstm.legend()
            plt.xticks(rotation=90)
            st.pyplot(fig_lstm)

            # Calculate the daily change
            lstm_predictions["Daily Change"] = lstm_predictions[
                "Predicted Price"
            ].diff()

            # Optionally, calculate the percentage change (uncomment if needed)
            lstm_predictions["Daily % Change"] = (
                lstm_predictions["Predicted Price"].pct_change() * 100
            )

            # Display the LSTM predictions in a table
            st.subheader("LSTM Predicted Prices with Daily Change")
            st.dataframe(lstm_predictions)

        except Exception as e:
            st.error(f"An error occurred: {e}")

# Tab 4: About
with tab4:
    st.title("About This App")
    st.write(
        """
        This application demonstrates how to use tabs in Streamlit to display
        multiple views. Use the "Charts" tab to upload your CSV and visualize
        the data.
        """
    )

# Tab 5: Testing
with tab5:
    st.title("Testing")
    st.write("This is a test tab to try new things.")
    # User inputs
    # NEWS_API_KEY = st.text_input("Enter your NewsAPI Key:", type="password")
    NEWS_API_KEY = load_api_key("configs/api_keys.json", "apiKey_newsapi")
    QUERY = st.text_input("Enter the keyword to search for:", value="Bitcoin")

    if st.button("Run Sentiment Analysis"):
        if NEWS_API_KEY and QUERY:
            try:
                # Fetch sentiment data
                sentiment_df = process_news_sentiment(NEWS_API_KEY, QUERY)

                # Display sentiment data
                st.subheader("Sentiment Data")
                st.dataframe(sentiment_df)

                # Plot sentiment over time
                st.subheader("Sentiment Over Time")
                line_chart = plot_sentiment_over_time(sentiment_df)
                st.altair_chart(line_chart, use_container_width=True)

                # Plot sentiment distribution
                st.subheader("Sentiment Distribution")
                bar_chart = plot_sentiment_distribution(sentiment_df)
                st.altair_chart(bar_chart, use_container_width=True)

                # Generate and display word cloud
                st.subheader("Word Cloud of Article Descriptions")
                word_cloud_fig = generate_word_cloud(sentiment_df)
                st.pyplot(word_cloud_fig)

            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please provide both the API key and a keyword.")
