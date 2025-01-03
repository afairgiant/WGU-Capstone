import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the CSV file
file_path = "data/processed/bitcoin_cleaned.csv"
data = pd.read_csv(file_path)

# Preview the first few rows
print(data.head())
print("Number of rows and columns:", data.shape)
print(data.info())
print("Missing values per column:\n", data.isnull().sum())


sns.heatmap(data.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Data Visualization")
plt.show()
