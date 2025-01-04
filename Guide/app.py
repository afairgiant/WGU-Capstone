import json
import time
from datetime import datetime, timedelta

import numpy as np
import requests
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

API_KEY = "CG-jMJRBmwSsWizV5LXEExWkn9K "


@app.route("/", methods=["GET", "POST"])
def index():
    # Get the list of available coins
    response = requests.get(
        f"https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&api_key={API_KEY}"
    )
    coins_list = response.json()
    coins = [coin["id"] for coin in coins_list]

    if request.method == "POST":
        coin_id = request.form.get("coin_id")
        days = request.form.get("days")
        if coin_id in coins:
            predictions = get_market_chart(coin_id, days)
            return render_template(
                "index.html",
                predictions=predictions,
                coins=coins,
                coin_id=coin_id,
                days=days,
            )
        else:
            return render_template("index.html", error="Invalid coin ID", coins=coins)

    return render_template("index.html", coins=coins)


def get_market_chart(coin_id, days):
    # Get the current timestamp in seconds
    end_timestamp = int(time.time())
    # Calculate the start timestamp based on the number of days
    start_timestamp = end_timestamp - int(days) * 24 * 60 * 60

    # Get OHLC data
    response = requests.get(
        f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days={days}&api_key={API_KEY}"
    )
    ohlc_data = response.json()

    # Get historical data for a specific date
    date = time.strftime("%d-%m-%Y", time.gmtime(start_timestamp))
    response = requests.get(
        f"https://api.coingecko.com/api/v3/coins/{coin_id}/history?date={date}&api_key={API_KEY}"
    )
    history_data = response.json()

    # Get current market data
    response = requests.get(
        f"https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&ids={coin_id}&api_key={API_KEY}"
    )
    market_data = response.json()[0]

    # Assuming 'ohlc_data' is a list of [time, open, high, low, close] lists
    ohlc = np.array(ohlc_data)

    # Use only the 'time' and 'close' columns
    X = ohlc[:, [0, 4]]  # Time and Close price
    y = ohlc[:, 4]  # Close price

    # Add historical price and current market data as features
    X = np.concatenate(
        (
            X,
            np.full((len(X), 1), history_data["market_data"]["current_price"]["usd"]),
            np.full((len(X), 1), market_data["current_price"]),
        ),
        axis=1,
    )

    # Scale the data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Train the model
    model = LinearRegression()
    model.fit(X, y)

    # Make a prediction
    y_pred = model.predict(X)

    # Create a list of dates for the predictions
    start_date = datetime.now()
    dates = [
        (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(len(y_pred))
    ]

    # Combine the dates and predictions into a dictionary
    predictions = {date: pred for date, pred in zip(dates, y_pred)}

    return predictions


if __name__ == "__main__":
    app.run(debug=True)
