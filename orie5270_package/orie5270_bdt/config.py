from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for rolling daily factor models.

    Parameters
    ----------
    date_col:
        Column containing the trading date.
    asset_col:
        Column containing the stock identifier, ticker, or permno.
    price_col:
        Optional price column. If target_col is not supplied, next-period returns are computed
        from this column by asset.
    return_col:
        Optional existing return column. If target_col is not supplied and return_col exists,
        target is next-period return_col by asset.
    target_col:
        Existing next-period return target. If supplied, no target construction is performed.
    feature_cols:
        Factor columns used to predict next-period returns.
    model_type:
        One of {'ols', 'ridge', 'lasso'}.
    window:
        Rolling estimation window length in rows per asset, usually 252 trading days.
    min_periods:
        Minimum observations needed to fit a rolling model.
    alpha:
        Regularization strength for ridge/lasso. Ignored for OLS.
    standardize:
        Whether to standardize features inside each rolling fit.
    """

    date_col: str = "date"
    asset_col: str = "ticker"
    price_col: str | None = "close"
    return_col: str | None = None
    target_col: str | None = None
    feature_cols: Sequence[str] = field(default_factory=tuple)
    model_type: str = "ols"
    window: int = 252
    min_periods: int = 126
    alpha: float = 1.0
    standardize: bool = True

    def validate(self) -> None:
        if self.model_type not in {"ols", "ridge", "lasso"}:
            raise ValueError("model_type must be one of {'ols', 'ridge', 'lasso'}")
        if self.window <= 1:
            raise ValueError("window must be greater than 1")
        if self.min_periods <= 1 or self.min_periods > self.window:
            raise ValueError("min_periods must be in [2, window]")
        if not self.feature_cols:
            raise ValueError("feature_cols cannot be empty")


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for quantile long-short backtesting."""

    date_col: str = "date"
    asset_col: str = "ticker"
    prediction_col: str = "prediction"
    realized_return_col: str = "target_return"
    n_quantiles: int = 5
    long_quantile: int = 5
    short_quantile: int = 1
    annualization: int = 252
    transaction_cost_bps: float = 0.0

    def validate(self) -> None:
        if self.n_quantiles < 2:
            raise ValueError("n_quantiles must be at least 2")
        if not 1 <= self.short_quantile <= self.n_quantiles:
            raise ValueError("short_quantile must be between 1 and n_quantiles")
        if not 1 <= self.long_quantile <= self.n_quantiles:
            raise ValueError("long_quantile must be between 1 and n_quantiles")
        if self.long_quantile == self.short_quantile:
            raise ValueError("long_quantile and short_quantile must be different")
        if self.transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps cannot be negative")
