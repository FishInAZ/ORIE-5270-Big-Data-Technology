import unittest
import pandas as pd
import numpy as np

from backtest import (
    assign_quantiles_by_date,
    compute_daily_ic,
    backtest_one_model
)


class TestBacktestSystem(unittest.TestCase):

    def setUp(self):
        """
        Create synthetic dataset
        """
        np.random.seed(42)

        n = 500

        self.df = pd.DataFrame({
            "date": np.repeat(
                pd.date_range("2020-01-01", periods=100),
                5
            ),
            "company": ["AAPL","MSFT","TSLA","NVDA","AMZN"] * 100,
            "ret_next": np.random.randn(500),
            "pred_ret": np.random.randn(500)
        })


    def test_quantile_assignment(self):
        """
        Test quantile assignment
        """
        result = self.df.groupby("date", group_keys=False).apply(
            assign_quantiles_by_date,
            pred_col="pred_ret",
            n_quantiles=4
        )

        self.assertTrue("quantile" in result.columns)
        self.assertTrue(result["quantile"].notna().sum() > 0)


    def test_ic_calculation(self):
        """
        Test IC calculation
        """
        sample = self.df.groupby("date").apply(
            compute_daily_ic,
            pred_col="pred_ret",
            ret_col="ret_next"
        )

        self.assertTrue("IC" in sample.columns)
        self.assertTrue("RankIC" in sample.columns)


    def test_backtest_runs(self):
        """
        Test backtest runs
        """
        result = backtest_one_model(
            self.df,
            model_name="Test",
            pred_col="pred_ret",
            date_col="date",
            asset_col="company",
            ret_col="ret_next",
            n_quantiles=4
        )

        self.assertTrue("quantile_ret" in result)
        self.assertTrue("summary_df" in result)


    def test_long_short_exists(self):
        """
        Test long-short return exists
        """
        result = backtest_one_model(
            self.df,
            model_name="Test",
            pred_col="pred_ret",
            date_col="date",
            asset_col="company",
            ret_col="ret_next",
            n_quantiles=4
        )

        quantile_ret = result["quantile_ret"]

        self.assertTrue("long_short" in quantile_ret.columns)


if __name__ == "__main__":
    unittest.main()