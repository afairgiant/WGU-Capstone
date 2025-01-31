import unittest

import numpy as np
import pandas as pd

from src.streamlit_app.ohlc_functions import (
    analyze_prices_by_day,
    calculate_daily_average,
    calculate_moving_averages,
    lstm_crypto_forecast,
    run_ohlc_prediction,
)


class TestFinancialFunctions(unittest.TestCase):
    def setUp(self):
        # Generate consistent-length sample data for testing
        num_rows = 50  # Ensure all arrays have the same number of elements
        self.sample_data = pd.DataFrame(
            {
                "time": pd.date_range(start="2023-01-01", periods=num_rows, freq="D"),
                "open": np.linspace(10, 20, num_rows),
                "high": np.linspace(11, 21, num_rows),
                "low": np.linspace(9, 19, num_rows),
                "close": np.linspace(10.5, 20.5, num_rows),
            }
        )

    def test_run_ohlc_prediction(self):
        result = run_ohlc_prediction(self.sample_data, days=5)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 5)

    def test_calculate_daily_average(self):
        result = calculate_daily_average(self.sample_data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("Average Price", result.columns)
        self.assertEqual(len(result), len(self.sample_data))

    def test_lstm_crypto_forecast(self):
        result = lstm_crypto_forecast(self.sample_data, days=5)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 5)

    def test_calculate_moving_averages(self):
        # Write sample data to CSV for testing
        self.sample_data.to_csv("sample.csv", index=False)
        result = calculate_moving_averages("sample.csv")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("7-Day Moving Average", result.columns)

    def test_analyze_prices_by_day(self):
        # Write sample data to CSV for testing
        self.sample_data.to_csv("sample.csv", index=False)
        result = analyze_prices_by_day("sample.csv")
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), 7)  # 7 days of the week


if __name__ == "__main__":
    unittest.main()
