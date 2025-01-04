import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib.widgets import RangeSlider

# Configure logging
logger = logging.getLogger(__name__)


def load_config(config_file):
    try:
        with open(config_file, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")


def load_cleaned_data(crypto_id, data_dir):
    file_path = os.path.join(data_dir, f"{crypto_id}_cleaned.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Error loading data for {crypto_id}: {e}")


def plot_price_trend(data, crypto_id):
    try:
        # Ensure timestamps are sorted
        data = data.sort_values("timestamp")

        # Convert timestamp to numerical values for initial regression
        x_full = (data["timestamp"] - data["timestamp"].min()).dt.total_seconds()
        y_full = data["price"]

        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(10, 5))
        plt.subplots_adjust(bottom=0.25)  # Leave space for slider
        (line,) = ax.plot(
            data["timestamp"],
            data["price"],
            label=f"{crypto_id.capitalize()} Price",
            color="blue",
        )
        (trend,) = ax.plot(
            data["timestamp"],
            np.polyval(np.polyfit(x_full, y_full, 1), x_full),
            label="Trend (Linear Regression)",
            color="red",
            linestyle="--",
        )
        ax.set_title(f"{crypto_id.capitalize()} Price Trend")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        ax.grid(True)

        # Add slider for date range
        ax_slider = plt.axes(
            [0.15, 0.1, 0.7, 0.03]
        )  # Position [left, bottom, width, height]
        date_slider = RangeSlider(
            ax_slider,
            "Date Range",
            data["timestamp"].min().toordinal(),
            data["timestamp"].max().toordinal(),
            valinit=(
                data["timestamp"].min().toordinal(),
                data["timestamp"].max().toordinal(),
            ),
        )

        # Update function for the slider
        def update(val):
            # Get the new date range from the slider
            start_date = pd.Timestamp.fromordinal(int(date_slider.val[0]))
            end_date = pd.Timestamp.fromordinal(int(date_slider.val[1]))

            # Filter data for the selected date range
            mask = (data["timestamp"] >= start_date) & (data["timestamp"] <= end_date)
            x_filtered = (
                data["timestamp"][mask] - data["timestamp"].min()
            ).dt.total_seconds()
            y_filtered = data["price"][mask]

            # Recompute the regression for the filtered range
            if len(x_filtered) > 1:  # Ensure there's enough data for regression
                coeffs = np.polyfit(x_filtered, y_filtered, 1)
                trend_line_filtered = np.polyval(coeffs, x_filtered)

                # Update the plot with filtered data
                line.set_data(data["timestamp"][mask], y_filtered)
                trend.set_data(data["timestamp"][mask], trend_line_filtered)
                ax.set_xlim(start_date, end_date)
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw_idle()

        date_slider.on_changed(update)

        plt.show()
    except Exception as e:
        raise ValueError(f"Error plotting price trend for {crypto_id}: {e}")


def plot_price_distribution(data, crypto_id):
    try:
        plt.figure(figsize=(8, 5))
        sns.histplot(data["price"], bins=30, kde=True)
        plt.title(f"{crypto_id.capitalize()} Price Distribution")
        plt.xlabel("Price (USD)")
        plt.ylabel("Frequency")
        plt.show()
    except Exception as e:
        raise ValueError(f"Error plotting price distribution for {crypto_id}: {e}")


def explore_data(crypto_id, data):
    """
    Perform a consolidated exploration of the cryptocurrency data
    using the existing plotting functions.
    """
    logger.info(f"Exploring data for {crypto_id}...")
    try:
        # Ensure the timestamp column is in datetime format
        data["timestamp"] = pd.to_datetime(data["timestamp"])

        # Plot price trend
        plot_price_trend(data, crypto_id)

        # Plot price distribution
        plot_price_distribution(data, crypto_id)

        # Additional analysis (if required in the future) can go here
    except Exception as e:
        logger.error(f"Error during exploration for {crypto_id}: {e}", exc_info=True)


if __name__ == "__main__":
    config_file = "configs/config.yaml"
    try:
        config = load_config(config_file)
    except Exception as e:
        raise SystemExit(f"Failed to load configuration: {e}")

    cryptocurrencies = config.get("cryptocurrencies", [])
    data_dir = "data/processed"

    if not cryptocurrencies:
        raise ValueError("No cryptocurrencies specified in the configuration.")

    for crypto_id in cryptocurrencies:
        print(f"Exploring data for {crypto_id}...")
        try:
            data = load_cleaned_data(crypto_id, data_dir)
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            plot_price_trend(data, crypto_id)
            plot_price_distribution(data, crypto_id)
        except Exception as e:
            print(f"Error processing data for {crypto_id}: {e}")

    print("Data exploration completed!")
