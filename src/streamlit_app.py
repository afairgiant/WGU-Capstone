import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

st.set_page_config(page_title="Crypto Data Visualizer", layout="wide")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Home", "Charts", "Prediction", "About"])

# Tab 1: Home
with tab1:
    st.title("Welcome to the App")
    st.write("This is the home page. Use the tabs above to navigate.")


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

    # Resolve the file path
    relative_path = os.path.join("..", "data", "processed", "bitcoin_final.csv")
    absolute_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), relative_path)
    )

    # Load the data
    try:
        data = pd.read_csv(absolute_path, parse_dates=["timestamp"])
        st.write("### Historical Data")
        st.dataframe(data)
    except FileNotFoundError:
        st.error(f"File not found at: {absolute_path}")
        data = None

    if data is not None:
        # Normalize timestamps
        data["normalized_timestamp"] = (
            data["timestamp"].astype("int64") // 10**9
            - data["timestamp"].astype("int64").min() // 10**9
        ) / (
            60 * 60 * 24
        )  # Convert to days
        X = data[["normalized_timestamp"]]
        y = data["price"]

        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Polynomial regression
        degree = st.slider(
            "Polynomial Degree for Trendline:",
            min_value=1,
            max_value=5,
            value=2,
            step=1,
        )
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        # Fit the model
        model = LinearRegression()
        model.fit(X_train_poly, y_train)

        # Predict future prices
        st.write("### Predict Future Prices")
        days_to_predict = st.number_input(
            "Number of days to predict:", min_value=1, max_value=365, value=30
        )
        future_normalized_timestamps = np.arange(
            X["normalized_timestamp"].max() + 1,
            X["normalized_timestamp"].max() + days_to_predict + 1,
        ).reshape(-1, 1)
        future_poly = poly.transform(future_normalized_timestamps)
        future_prices = model.predict(future_poly)

        # Convert normalized timestamps back to real timestamps for plotting
        future_timestamps = (
            future_normalized_timestamps.flatten() * (60 * 60 * 24)
            + data["timestamp"].astype("int64").min() // 10**9
        )

        # Plot results
        st.write("### Predicted Prices")
        future_dates = pd.to_datetime(future_timestamps, unit="s")
        predictions = pd.DataFrame(
            {"timestamp": future_dates, "predicted_price": future_prices}
        )
        st.dataframe(predictions)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            data["timestamp"].astype("int64") // 10**9,
            y,
            label="Historical Prices",
            color="blue",
        )
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
