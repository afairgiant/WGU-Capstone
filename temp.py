import csv

import requests

# Fetch data
url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
params = {"vs_currency": "usd", "days": "365", "precision": "full"}
headers = {"accept": "application/json"}

response = requests.get(url, headers=headers, params=params)
data = response.json()

# Extract prices
prices = data.get("prices", [])

# Save to CSV
with open("output.csv", "w", newline="") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Timestamp", "Price"])  # Header
    csvwriter.writerows(prices)

print("Data saved to output.csv")
