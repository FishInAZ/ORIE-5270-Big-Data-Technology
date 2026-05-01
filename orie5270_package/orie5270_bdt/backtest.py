from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .metrics import information_coefficient, performance_summary


def _assign_quantiles(g: pd.DataFrame, prediction_col: str, n_quantiles: int) -> pd.Series:
    ranked = g[prediction_col].rank(method="first")
    try:
        return pd.qcut(ranked, q=n_quantiles, labels=range(1, n_quantiles + 1)).astype(int)
    except ValueError:
        return pd.Series(np.nan, index=g.index, dtype="float")


def quantile_backtest(predictions: pd.DataFrame, config: BacktestConfig) -> dict[str, pd.DataFrame]:
    """Run an equal-weight quantile long-short backtest.

    Returns a dictionary with daily returns, quantile returns, IC series, and summary tables.
    """
    config.validate()
    required = {
        config.date_col,
        config.asset_col,
        config.prediction_col,
        config.realized_return_col,
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    df = predictions.copy()
    df[config.date_col] = pd.to_datetime(df[config.date_col])
    df = df.dropna(subset=[config.prediction_col, config.realized_return_col])

    df["quantile"] = df.groupby(config.date_col, group_keys=False).apply(
        lambda g: _assign_quantiles(g, config.prediction_col, config.n_quantiles)
    )
    df = df.dropna(subset=["quantile"])
    df["quantile"] = df["quantile"].astype(int)

    quantile_returns = (
        df.groupby([config.date_col, "quantile"])[config.realized_return_col]
        .mean()
        .reset_index(name="return")
    )
    wide = quantile_returns.pivot(index=config.date_col, columns="quantile", values="return")
    wide.columns = [f"q{int(c)}" for c in wide.columns]
    wide = wide.sort_index()

    long_col = f"q{config.long_quantile}"
    short_col = f"q{config.short_quantile}"
    daily = pd.DataFrame(index=wide.index)
    daily["long_return"] = wide.get(long_col)
    daily["short_return"] = wide.get(short_col)
    daily["long_short_return_gross"] = daily["long_return"] - daily["short_return"]

    # Simple round-trip cost approximation for one long book and one short book.
    cost = 2.0 * config.transaction_cost_bps / 10_000.0
    daily["long_short_return"] = daily["long_short_return_gross"] - cost
    daily["cumulative_return"] = (1.0 + daily["long_short_return"].fillna(0.0)).cumprod() - 1.0
    daily = daily.reset_index()

    ic = information_coefficient(
        df,
        date_col=config.date_col,
        prediction_col=config.prediction_col,
        realized_return_col=config.realized_return_col,
    )

    summary = pd.DataFrame(
        [performance_summary(daily["long_short_return"], annualization=config.annualization)]
    )
    if not ic.empty:
        summary["mean_ic"] = ic["ic"].mean()
        summary["mean_rank_ic"] = ic["rank_ic"].mean()

    return {
        "scored_assets": df.sort_values([config.date_col, "quantile", config.asset_col]),
        "quantile_returns": quantile_returns,
        "daily_returns": daily,
        "ic": ic,
        "summary": summary,
    }
