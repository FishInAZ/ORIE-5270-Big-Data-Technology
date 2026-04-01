import unittest
import pandas as pd
import numpy as np

from ols_daily_factor_model import rolling_ols_prediction


class TestOLSDailyFactorModel(unittest.TestCase):

    def setUp(self):
        """
        Create small synthetic dataset
        """
        np.random.seed(42)

        dates = pd.date_range("2020-01-01", periods=300)

        self.df = pd.DataFrame({
            "company": ["AAPL"] * 300,
            "date": dates,
            "ret_next": np.random.randn(300),

            "liquidity_alpha": np.random.randn(300),
            "size_alpha": np.random.randn(300),
            "signal_2": np.random.randn(300),
            "sentiment_change_alpha": np.random.randn(300),
            "sentiment_exp_decay_alpha": np.random.randn(300),
            "sent_macd_alpha": np.random.randn(300),

            "signal1": np.random.randn(300),
            "signal2": np.random.randn(300),
            "signal3": np.random.randn(300),
            "signal4": np.random.randn(300),
            "signal5": np.random.randn(300),
            "signal6": np.random.randn(300),
            "signal7": np.random.randn(300),
            "signal8": np.random.randn(300),
        })

        self.factor_cols = [
            "liquidity_alpha",
            "size_alpha",
            "signal_2",
            "sentiment_change_alpha",
            "sentiment_exp_decay_alpha",
            "sent_macd_alpha",
            "signal1",
            "signal2",
            "signal3",
            "signal4",
            "signal5",
            "signal6",
            "signal7",
            "signal8"
        ]


    def test_rolling_ols_runs(self):
        """
        Test model runs without crashing
        """
        pred, coef = rolling_ols_prediction(
            self.df,
            self.factor_cols,
            window_days=50
        )

        self.assertTrue(len(pred) > 0)
        self.assertTrue(len(coef) > 0)


    def test_prediction_column_exists(self):
        """
        Test prediction column exists
        """
        pred, _ = rolling_ols_prediction(
            self.df,
            self.factor_cols,
            window_days=50
        )

        self.assertIn("pred_ret_ols", pred.columns)


    def test_prediction_not_all_nan(self):
        """
        Test predictions not all NaN
        """
        pred, _ = rolling_ols_prediction(
            self.df,
            self.factor_cols,
            window_days=50
        )

        self.assertTrue(pred["pred_ret_ols"].notna().sum() > 0)


    def test_coef_output(self):
        """
        Test coefficient output exists
        """
        _, coef = rolling_ols_prediction(
            self.df,
            self.factor_cols,
            window_days=50
        )

        self.assertTrue(len(coef.columns) > 2)


if __name__ == "__main__":
    unittest.main()