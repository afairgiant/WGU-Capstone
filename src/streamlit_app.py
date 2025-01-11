import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


def basic_charts():
    # Load the CSV file
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the CSV file
        data = pd.read_csv(uploaded_file, parse_dates=["timestamp"])

        # Show the raw data
        st.write("## Raw Data")
        st.dataframe(data)

        # Line chart for price and moving averages
        st.write("## Price and Moving Averages Over Time")
        st.line_chart(data.set_index("timestamp")[["price", "7d_ma", "30d_ma"]])

        # Line chart for price change
        st.write("## Price Change Over Time")
        st.line_chart(data.set_index("timestamp")[["price_change"]])

        # Custom Matplotlib chart
        st.write("## Custom Chart with Matplotlib")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data["timestamp"], data["price"], label="Price", color="blue")
        ax.plot(data["timestamp"], data["7d_ma"], label="7-Day MA", color="orange")
        ax.plot(data["timestamp"], data["30d_ma"], label="30-Day MA", color="green")
        ax.set_title("Price and Moving Averages Over Time")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Value")
        ax.legend()
        st.pyplot(fig)


# Run the function
if __name__ == "__main__":
    basic_charts()
