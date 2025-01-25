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
    analyze_prices_by_day,
    calculate_daily_average,
    calculate_moving_averages,
    evaluate_linear_regression_model,
    evaluate_lstm_model,
    evaluate_model,
    lstm_crypto_forecast,
    plot_correlation_heatmap,
    run_ohlc_prediction,
    train_lstm_model,
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

# Update any missing data from past 30 days at program start.
COIN_ID = "bitcoin"
download_and_save_ohlc_data(COIN_ID, 30)

st.set_page_config(page_title="Crypto Data Visualizer", layout="wide")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Home", "Historical Charts", "LSTM Prediction", "Sentiment Analysis", "About"]
)

# Tab 1: Home
with tab1:
    st.title("Welcome to the App")
    st.write("This is the home page. Use the tabs above to navigate.")

    # # Load data
    # server_csv_path = "src/streamlit_app/data/ohlc_data.csv"
    # data = pd.read_csv(server_csv_path)
    # # Run linear regression predictions and get metrics
    # predictions, metrics = run_ohlc_prediction(data, 30)

    # # Display Metrics and Predictions for Linear Regression
    # st.subheader("Linear Regression Model Evaluation Metrics")
    # st.write(f"Mean Absolute Error (MAE): {metrics['MAE']:.2f}")
    # st.write(f"Root Mean Squared Error (RMSE): {metrics['RMSE']:.2f}")
    # st.write(f"R-squared (R²): {metrics['R²']:.2f}")
    # st.subheader("Predicted Prices")
    # st.dataframe(predictions)

    # # Run LSTM predictions and get metrics
    # lstm_predictions, metrics = lstm_crypto_forecast(data, 30)

    # # Debugging: Check structure and type
    # print("Type of lstm_predictions:", type(lstm_predictions))
    # print("Content of lstm_predictions:", lstm_predictions)

    # # Convert lstm_predictions to a DataFrame
    # if isinstance(lstm_predictions, dict):
    #     lstm_predictions = pd.DataFrame(lstm_predictions)

    # # Ensure proper formatting
    # lstm_predictions["Date"] = pd.to_datetime(lstm_predictions["Date"], errors="coerce")
    # lstm_predictions["Predicted Price"] = pd.to_numeric(
    #     lstm_predictions["Predicted Price"], errors="coerce"
    # )

    # # Drop rows with invalid data
    # lstm_predictions = lstm_predictions.dropna()

    # # Display metrics
    # st.subheader("LSTM Model Evaluation Metrics")
    # st.write(f"Mean Absolute Error (MAE): {metrics['MAE']:.2f}")
    # st.write(f"Root Mean Squared Error (RMSE): {metrics['RMSE']:.2f}")
    # st.write(f"R-squared (R²): {metrics['R²']:.2f}")

    # # Display predictions
    # st.subheader("LSTM Predicted Prices")
    # st.dataframe(lstm_predictions)

# Tab 2: Charts
with tab2:
    # Path to the CSV file
    server_csv_path = "src/streamlit_app/data/ohlc_data.csv"

    # Streamlit Tab for Historical Data
    st.title("Historical Data")
    st.write("This tab is used to visualize historical data for bitcoin.")
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
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.title("Daily Average and Moving Averages")
        plt.legend(loc="upper left")  # Add legend to the chart
        plt.grid(True)

        # Display the plot in Streamlit
        st.pyplot(plt)
        # New: Calculate and display average prices by day of the week
        st.subheader("Average Prices by Day of the Week")
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

    except Exception as e:
        st.error(f"An error occurred: {e}")

    st.subheader("Correlation Heatmap")
    st.write(
        """
        This correlation heatmap provides insights into the relationships between different numerical features in the cryptocurrency dataset, 
        such as prices, volume, and moving averages. Strong correlations (above 0.8 or below -0.8) are highlighted, as they indicate variables 
        that move together (positively or negatively).

        - **What this means for you**: Use this heatmap to identify key drivers of cryptocurrency price changes. For example:
            - A strong positive correlation between a moving average and the closing price suggests that trends in the moving average 
            could help predict future price movements.
            - A strong negative correlation might highlight features that behave inversely, offering insights into market dynamics.
        
        - **How to use this**: Focus on features that show significant correlations with the target variable (e.g., 'close' or 'price'). 
        These features are likely to be the most predictive and can be prioritized for your machine learning models or trading strategies.
    """
    )

    try:
        # Read the dataset from the predefined path
        data = pd.read_csv(server_csv_path)

        # Allow users to select a target column for highlighting correlations
        target_column = st.selectbox(
            "Select a target column for analysis (optional)",
            ["None"] + list(data.columns),
        )
        if target_column == "None":
            target_column = None

        # Generate the heatmap figure
        heatmap_fig = plot_correlation_heatmap(data, target_column=target_column)

        # Display the heatmap
        st.pyplot(heatmap_fig)

    except FileNotFoundError:
        st.error(
            f"File not found: {server_csv_path}. Please ensure the dataset exists at this location."
        )
    except Exception as e:
        st.error(f"Error generating heatmap: {e}")

# Tab 3: Futures
with tab3:

    st.title("Bitcoin Price Prediction")
    st.write(
        "This tab is used to predict future prices of bitcoin using linear regression and an LSTM prediction model."
    )
    # Path to the CSV file
    server_csv_path = "src/streamlit_app/data/ohlc_data.csv"

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
                download_and_save_ohlc_data(COIN_ID, 30)

                # Read the uploaded file as a DataFrame
                data = pd.read_csv(server_csv_path)

                # Calculate the daily averages
                daily_averages = calculate_daily_average(data)

                # Run the prediction function
                # Predict future prices using Linear Regression
                linear_predictions = run_ohlc_prediction(data, days)

                # Evaluate the Linear Regression model
                linear_metrics = evaluate_linear_regression_model(data)

                # Plot Linear Regression predictions
                st.subheader("Linear Regression Predicted Prices")
                fig_lr, ax_lr = plt.subplots()

                # Plot historical close prices
                ax_lr.plot(
                    data["time"],
                    data["close"],
                    label="Historical Close Prices",
                    color="blue",
                )

                # Plot linear regression predictions
                ax_lr.plot(
                    linear_predictions["Date"],
                    linear_predictions["Predicted Price"],
                    label="Linear Regression Predicted Price",
                    color="red",
                    marker="o",
                    markersize=3,
                )

                ax_lr.set_xlabel("Date")
                ax_lr.set_ylabel("Price (USD)")
                ax_lr.set_title(f"Linear Regression Predicted Prices ({days} Days)")
                ax_lr.legend()
                plt.xticks(rotation=90)
                st.pyplot(fig_lr)

                # Display Linear Regression metrics
                st.subheader("Linear Regression Model Performance Metrics")
                st.write(f"Mean Absolute Error (MAE): {linear_metrics['MAE']:.2f}")
                st.write(
                    f"Root Mean Squared Error (RMSE): {linear_metrics['RMSE']:.2f}"
                )
                st.write(f"R-squared (R²): {linear_metrics['R²']:.2f}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
            try:
                # Train the LSTM model
                model, scaler, X_test, y_test = train_lstm_model(data)
                print("getting past training data")
                # Predict future prices
                lstm_predictions = lstm_crypto_forecast(model, scaler, data, days)
                print("getting past predictions")
                # Evaluate the model
                lstm_metrics = evaluate_model(y_test, model.predict(X_test).flatten())

                # Plot LSTM predictions
                st.subheader("LSTM Predicted Prices with Historical Data")
                fig_lstm, ax_lstm = plt.subplots()

                # Sort the index
                if not data.index.is_monotonic_increasing:
                    data = data.sort_index()
                    st.write("Sorted the index.")

                # Plot historical data
                ax_lstm.plot(
                    data.index,
                    data["close"],
                    label="Historical Close Prices",
                    color="blue",
                )

                # Plot LSTM predictions
                ax_lstm.plot(
                    lstm_predictions["Date"],  # Use 'Date' column instead of 'time'
                    lstm_predictions["Predicted Price"],
                    label="LSTM Predicted Price",
                    color="green",
                    marker="o",
                    markersize=3,
                )

                ax_lstm.set_xlabel("Date")
                ax_lstm.set_ylabel("Price (USD)")
                ax_lstm.set_title("LSTM Predicted Prices with Historical Data")
                ax_lstm.legend()
                plt.xticks(rotation=90)
                st.pyplot(fig_lstm)

                # Display predictions
                st.subheader("LSTM Predicted Prices")
                st.dataframe(lstm_predictions)

                # Display evaluation metrics
                st.subheader("LSTM Model Performance Metrics")
                st.write(f"Mean Absolute Error (MAE): {lstm_metrics['MAE']:.2f}")
                st.write(f"Root Mean Squared Error (RMSE): {lstm_metrics['RMSE']:.2f}")
                st.write(f"R-squared (R²): {lstm_metrics['R²']:.2f}")

            except Exception as e:
                st.error(f"An error occurred: {e}")

# Tab 4: Sentiment Analysis
with tab4:
    st.title("Sentiment Analysis of News Articles")
    st.write(
        "This analysis uses the NewsAPI to fetch news articles and analyze public sentiment."
    )
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

# Tab 5: About
with tab5:
    st.title("About This App")
    st.write(
        """
        This application demonstrates how to use tabs in Streamlit to display
        multiple views. Use the "Charts" tab to upload your CSV and visualize
        the data.
        """
    )
