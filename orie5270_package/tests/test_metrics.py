import pandas as pd

from orie5270_bdt.metrics import max_drawdown, performance_summary


def test_max_drawdown_negative_after_peak():
    returns = pd.Series([0.10, -0.20, 0.05])
    assert max_drawdown(returns) < 0


def test_performance_summary_has_expected_keys():
    summary = performance_summary(pd.Series([0.01, -0.005, 0.02]), annualization=252)
    assert "annualized_return" in summary
    assert "sharpe_ratio" in summary
    assert "max_drawdown" in summary
