# Plot predictions interactively
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st


# Load and validate CSV
@st.cache_data
def load_and_validate_csv(file_path):
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
