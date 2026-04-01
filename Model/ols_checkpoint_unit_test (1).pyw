import unittest
import pandas as pd
import numpy as np

from ols_daily_factor_model import (
    robust_to_datetime,
    rolling_ols_prediction,
    FACTOR_COLS
)


class TestOLSModel(unittest.TestCase):

    def setUp(self):

        np.random.seed(42)

        dates = pd.date_range("2020-01-01", periods=300)

        data = {
            "company": ["AAPL"] * 300,
            "date": dates,
            "ret_next": np.random.randn(300)
        }

        # add factor columns
        for col in FACTOR_COLS:
            data[col] = np.random.randn(300)

        self.df = pd.DataFrame(data)


    def test_date_conversion(self):
        """
        Test robust_to_datetime works
        """
        df = self.df.copy()
        df["date"] = robust_to_datetime(df["date"])

        self.assertTrue(
            pd.api.types.is_datetime64_any_dtype(df["date"])
        )


    def test_rolling_ols_runs(self):
        """
        Test rolling OLS runs
        """
        pred, coef = rolling_ols_prediction(
            self.df,
            FACTOR_COLS,
            window_days=50
        )

        self.assertTrue(len(pred) > 0)
        self.assertTrue(len(coef) > 0)


    def test_prediction_column(self):
        """
        Test prediction column exists
        """
        pred, _ = rolling_ols_prediction(
            self.df,
            FACTOR_COLS,
            window_days=50
        )

        self.assertIn("pred_ret_ols", pred.columns)


    def test_prediction_not_nan(self):
        """
        Test predictions valid
        """
        pred, _ = rolling_ols_prediction(
            self.df,
            FACTOR_COLS,
            window_days=50
        )

        self.assertTrue(
            pred["pred_ret_ols"].notna().sum() > 0
        )



if __name__ == "__main__":
    unittest.main()