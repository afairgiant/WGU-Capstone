# Plot predictions interactively
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st


# Load and validate CSV
@st.cache_data
def load_and_validate_csv(file_path):
    """
    Loads and validates a CSV file for required OHLC (Open, High, Low, Close) columns.

    This function reads a CSV file from the given file path and checks if it contains
    the required columns: "Date", "open", "high", "low", and "close". If any of these
    columns are missing, an error message is displayed using Streamlit and the function
    returns None. If the file is successfully read and validated, the data is returned.

    Args:
        file_path (str): The path to the CSV file to be loaded and validated.

    Returns:
        pandas.DataFrame or None: The loaded data as a pandas DataFrame if validation
        is successful, otherwise None.
    """
    try:
        data = pd.read_csv(file_path)
        required_columns = ["Date", "open", "high", "low", "close"]
        if not all(col in data.columns for col in required_columns):
            st.error(f"CSV file must contain the following columns: {required_columns}")
            return None
        return data
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return None


def plot_predictions_interactive(daily_averages, predictions, title):
    """
    Plots interactive line charts for daily average prices and predicted prices using Plotly.

    Parameters:
    daily_averages (pd.DataFrame): DataFrame containing the daily average prices with columns "Date" and "Average Price".
    predictions (pd.DataFrame): DataFrame containing the predicted prices with columns "Date" and "Predicted Price".
    title (str): The title of the plot.

    Returns:
    None
    """
    fig = px.line(title=title)
    fig.add_scatter(
        x=daily_averages["Date"],
        y=daily_averages["Average Price"],
        name="Daily Average",
        line=dict(color="blue"),
    )
    fig.add_scatter(
        x=predictions["Date"],
        y=predictions["Predicted Price"],
        name="Predicted Price",
        line=dict(color="red"),
    )
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Price (USD)", hovermode="x unified"
    )
    st.plotly_chart(fig)
