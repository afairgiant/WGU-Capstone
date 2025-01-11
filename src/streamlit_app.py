import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Crypto Data Visualizer", layout="wide")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Home", "Charts", "Prediction", "About"])

# Tab 1: Home
with tab1:
    st.title("Welcome to the App")
    st.write("This is the home page. Use the tabs above to navigate.")


# def basic_charts():
#     # Load the CSV file
#     uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

#     if uploaded_file is not None:
#         # Read the CSV file
#         data = pd.read_csv(uploaded_file, parse_dates=["timestamp"])

#         # Show the raw data
#         st.write("## Raw Data")
#         st.dataframe(data)

#         # Line chart for price and moving averages
#         st.write("## Price and Moving Averages Over Time")
#         st.line_chart(data.set_index("timestamp")[["price", "7d_ma", "30d_ma"]])

#         # Line chart for price change
#         st.write("## Price Change Over Time")
#         st.line_chart(data.set_index("timestamp")[["price_change"]])

#         # Custom Matplotlib chart
#         st.write("## Custom Chart with Matplotlib")
#         fig, ax = plt.subplots(figsize=(10, 6))
#         ax.plot(data["timestamp"], data["price"], label="Price", color="blue")
#         ax.plot(data["timestamp"], data["7d_ma"], label="7-Day MA", color="orange")
#         ax.plot(data["timestamp"], data["30d_ma"], label="30-Day MA", color="green")
#         ax.set_title("Price and Moving Averages Over Time")
#         ax.set_xlabel("Timestamp")
#         ax.set_ylabel("Value")
#         ax.legend()
#         st.pyplot(fig)
# Tab 2: Charts
with tab2:
    st.title("Historical Data")

    # File uploader
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        # Read the CSV
        data = pd.read_csv(uploaded_file, parse_dates=["timestamp"])

        # Display data
        st.write("### Raw Data")
        st.dataframe(data)

        # Chart for price and moving averages
        st.write("### Price and Moving Averages Over Time")
        st.line_chart(data.set_index("timestamp")[["price", "7d_ma", "30d_ma"]])

        # Custom Matplotlib chart
        st.write("### Custom Matplotlib Chart")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data["timestamp"], data["price"], label="Price", color="blue")
        ax.plot(data["timestamp"], data["7d_ma"], label="7-Day MA", color="orange")
        ax.plot(data["timestamp"], data["30d_ma"], label="30-Day MA", color="green")
        ax.set_title("Price and Moving Averages Over Time")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Value")
        ax.legend()
        st.pyplot(fig)

# Tab 3: Futures
with tab3:
    st.title("Bitcoin Price Prediction")

    # Define the relative path
    relative_path = os.path.join("..", "data", "processed", "bitcoin_final.csv")

    # Resolve the absolute path
    absolute_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), relative_path)
    )

    # Load data
    if not os.path.exists(absolute_path):
        st.error(f"File not found at: {absolute_path}")
    else:
        data = pd.read_csv(absolute_path, parse_dates=["timestamp"])
        st.write("### Historical Data")
        st.dataframe(data)

        # Prepare data for regression
        data["timestamp"] = (
            data["timestamp"].astype("int64") // 10**9
        )  # Convert to seconds
        X = data[["timestamp"]]
        y = data["price"]

        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Fit the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict future prices
        st.write("### Predict Future Prices")
        days_to_predict = st.number_input(
            "Number of days to predict:", min_value=1, max_value=365, value=30
        )
        future_timestamps = np.arange(
            data["timestamp"].max() + 86400,  # Start from the next day
            data["timestamp"].max() + days_to_predict * 86400 + 1,
            86400,
        ).reshape(-1, 1)
        future_prices = model.predict(future_timestamps)

        # Plot results
        st.write("### Predicted Prices")
        future_dates = pd.to_datetime(future_timestamps.flatten(), unit="s")
        predictions = pd.DataFrame(
            {"timestamp": future_dates, "predicted_price": future_prices}
        )
        st.dataframe(predictions)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data["timestamp"], y, label="Historical Prices", color="blue")
        ax.plot(future_timestamps, future_prices, label="Predicted Prices", color="red")
        ax.set_title("Bitcoin Price Prediction")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)


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
# Run the function
# if __name__ == "__main__":
# st.title("Page1: Basic Charts")
# basic_charts()
