import pandas as pd

from orie5270_bdt import BacktestConfig, quantile_backtest


def test_quantile_backtest_runs():
    df = pd.DataFrame(
        {
            "date": ["2024-01-01"] * 6 + ["2024-01-02"] * 6,
            "ticker": list("ABCDEF") * 2,
            "prediction": [1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1],
            "target_return": [0.01, 0.02, 0.00, 0.03, 0.04, 0.05, 0.01, -0.01, 0.02, 0.00, -0.02, -0.03],
        }
    )
    result = quantile_backtest(df, BacktestConfig(n_quantiles=3, long_quantile=3, short_quantile=1))
    assert not result["daily_returns"].empty
    assert not result["summary"].empty
