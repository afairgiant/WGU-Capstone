import logging
import os
import sys

import plotly.graph_objects as go
from utils import load_api_key, setup_logging

# Add the project root directory to Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from data_downloader_2 import download_blockchain_metrics, download_ohlc_data
from matplotlib import markers
from ohlc_functions import (
    analyze_prices_by_day,
    calculate_daily_average,
    calculate_moving_averages,
    lstm_crypto_forecast,
    run_ohlc_prediction,
)
from sentiment_functions import (
    generate_word_cloud,
    plot_sentiment_distribution,
    plot_sentiment_over_time,
    process_news_sentiment,
)

from src.streamlit_app.ohlc_functions import (
    analyze_prices_by_day,
    calculate_daily_average,
    calculate_moving_averages,
    lstm_crypto_forecast,
    run_ohlc_prediction,
)

# Setup Logging
setup_logging()

# Log the current working directory
logging.info(f"Current working directory: {os.getcwd()}")

# Configuration
DATA_PATH = "src/streamlit_app/data"
COIN_ID = "bitcoin"

# Update any missing data from past 30 days at program start.
logging.info("Downloading OHLC data for the past 30 days...")
download_ohlc_data(COIN_ID, 30)

# Streamlit page configuration
st.set_page_config(page_title="Crypto Data Visualizer", layout="wide")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Home", "Historical Charts", "LSTM Prediction", "Sentiment Analysis", "About"]
)

# Tab 1: Home
with tab1:
    st.title("Welcome to the App")
    st.write("This is the home page. Use the tabs above to navigate.")
    logging.info("Home tab loaded.")

# Tab 2: Historical Charts
with tab2:
    # Path to the CSV file
    server_csv_path = f"{DATA_PATH}/ohlc_data.csv"

    # Streamlit Tab for Historical Data
    st.title("Historical Data")
    st.write("This tab is used to visualize historical data for bitcoin.")
    try:
        # Calculate moving averages
        logging.info("Calculating moving averages...")
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
            color="red",
        )
        plt.plot(
            moving_averages["Time"],
            moving_averages["30-Day Moving Average"],
            label="30-Day Moving Average",
            color="green",
        )
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.title("Daily Average and Moving Averages")
        plt.legend(loc="upper left")  # Add legend to the chart
        plt.grid(True)

        # Display the plot in Streamlit
        st.pyplot(plt)

        # New: Calculate and display average prices by day of the week
        st.subheader("Average Prices by Day of the Week")
        logging.info("Calculating average prices by day of the week...")
        day_avg = analyze_prices_by_day(server_csv_path)

        # Plot the new chart
        plt.figure(figsize=(10, 6))
        day_avg.plot(kind="bar", color="skyblue", edgecolor="black")
        plt.title("Average Prices by Day of the Week", fontsize=16)
        plt.xlabel("Day of the Week", fontsize=12)
        plt.ylabel("Average Price", fontsize=12)
        plt.xticks(rotation=45, fontsize=10)
        plt.tight_layout()

        # Display the new plot in Streamlit
        st.pyplot(plt)

        # Download latest market data
        logging.info("Downloading blockchain metrics...")
        download_blockchain_metrics("configs/api_keys.json", "apiKey_gecko")

        # Load market data
        market_data = pd.read_csv(f"{DATA_PATH}//market_metrics_data.csv")

        # Convert the 'time' column to datetime if not already
        market_data["time"] = pd.to_datetime(market_data["time"])

        # Calculate the start and end dates for the past 30 days
        end_date = market_data["time"].max()  # Most recent date in the data
        start_date = end_date - pd.Timedelta(
            days=30
        )  # 30 days before the most recent date

        # Create a Plotly figure
        fig = go.Figure()

        # Add bars for Total Volume on the primary y-axis
        fig.add_trace(
            go.Bar(
                x=market_data["time"],
                y=market_data["total_volume"],
                name="Total Volume",
                marker=dict(opacity=0.6),
            )
        )

        # Add a line for Market Cap on the secondary y-axis
        fig.add_trace(
            go.Scatter(
                x=market_data["time"],
                y=market_data["market_cap"],
                mode="lines",
                name="Market Cap",
                yaxis="y2",
                line=dict(color="green"),
            )
        )

        # Customize layout
        fig.update_layout(
            title="Market Trends Over Time",
            xaxis=dict(
                title="Time",
                rangeslider=dict(visible=True),  # Enable range slider for zooming
                type="date",
            ),
            yaxis=dict(
                title="Volume",
                title_font=dict(color="blue"),
            ),
            yaxis2=dict(
                title="Market Cap",
                title_font=dict(color="green"),
                overlaying="y",  # Overlay on the same plot
                side="right",  # Align the second axis on the right
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        # Set the initial x-axis range to the past 30 days
        fig.update_xaxes(
            range=[start_date, end_date],  # Focus on the past 30 days
            showspikes=True,
            spikemode="across",
            spikecolor="grey",
        )

        # Add interactivity: zoom/pan will auto-scale the x-axis labels
        fig.update_layout(hovermode="x unified")

        # Display the chart in Streamlit
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        logging.error(f"An error occurred in the Historical Charts tab: {e}")
        st.error(f"An error occurred: {e}")

# Tab 3: Futures
with tab3:
    st.title("Bitcoin Price Prediction")
    st.write(
        "This tab is used to predict future prices of bitcoin using linear regression and a LSTM prediction model."
    )
    # Path to the CSV file
    server_csv_path = f"{DATA_PATH}//ohlc_data.csv"

    # Number of days for predictions
    days = st.number_input(
        "Enter the number of future days to predict",
        min_value=1,
        max_value=365,
        value=30,
    )
    # Button to trigger predictions
    if st.button("Run Price Prediction"):
        if server_csv_path is not None:
            try:
                # Update any missing data from past 30 days before future predictions.
                logging.info("Downloading OHLC data for the past 30 days...")
                download_ohlc_data(COIN_ID, 30)

                # Read the uploaded file as a DataFrame
                data = pd.read_csv(server_csv_path)

                # Calculate the daily averages
                logging.info("Calculating daily averages...")
                daily_averages = calculate_daily_average(data)

                # Run the prediction function
                logging.info("Running OHLC prediction...")
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
                plt.xticks(rotation=90)
                st.pyplot(fig)

                # Run the LSTM prediction function
                logging.info("Running LSTM prediction...")
                lstm_predictions = lstm_crypto_forecast(data, days)

                # Plot LSTM predictions and actual data
                st.subheader("LSTM Predicted Prices with Daily Price Historical Data")
                fig_lstm, ax_lstm = plt.subplots()

                # Plot historical close prices
                ax_lstm.plot(
                    data.index,
                    data["close"],
                    label="Historical Close Prices",
                    color="blue",
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

                lstm_predictions["Daily % Change"] = (
                    lstm_predictions["Predicted Price"].pct_change() * 100
                )

                # Display the LSTM predictions in a table
                st.subheader("LSTM Predicted Prices with Daily Change")
                st.dataframe(lstm_predictions)

            except Exception as e:
                logging.error(f"An error occurred in the Futures tab: {e}")
                st.error(f"An error occurred: {e}")

# Tab 4: Sentiment Analysis
with tab4:
    st.title("Sentiment Analysis of News Articles")
    st.write(
        "This analysis uses the NewsAPI to fetch news articles and analyze public sentiment."
    )

    NEWS_API_KEY = load_api_key("configs/api_keys.json", "apiKey_newsapi")
    QUERY = st.text_input("Enter the keyword to search for:", value="Bitcoin")

    if st.button("Run Sentiment Analysis"):
        if NEWS_API_KEY and QUERY:
            try:
                # Fetch sentiment data
                logging.info("Fetching sentiment data...")
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
                logging.error(f"An error occurred in the Sentiment Analysis tab: {e}")
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please provide both the API key and a keyword.")

# Tab 5: About
with tab5:
    st.title("About This App")
    st.write(
        """
        This application is used to predict bitcoin prices and sentiment analysis of news articles.
        """
    )
    logging.info("About tab loaded.")
