from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown from a return series."""
    returns = pd.Series(returns).dropna()
    if returns.empty:
        return np.nan
    cumulative = (1.0 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1.0
    return float(drawdown.min())


def annualized_return(returns: pd.Series, annualization: int = 252) -> float:
    returns = pd.Series(returns).dropna()
    if returns.empty:
        return np.nan
    growth = float((1.0 + returns).prod())
    periods = len(returns)
    return growth ** (annualization / periods) - 1.0


def annualized_volatility(returns: pd.Series, annualization: int = 252) -> float:
    returns = pd.Series(returns).dropna()
    if len(returns) < 2:
        return np.nan
    return float(returns.std(ddof=1) * np.sqrt(annualization))


def sharpe_ratio(returns: pd.Series, annualization: int = 252) -> float:
    returns = pd.Series(returns).dropna()
    vol = annualized_volatility(returns, annualization)
    if not np.isfinite(vol) or vol == 0:
        return np.nan
    return float(annualized_return(returns, annualization) / vol)


def information_coefficient(
    df: pd.DataFrame,
    date_col: str,
    prediction_col: str,
    realized_return_col: str,
) -> pd.DataFrame:
    """Compute daily Pearson IC and Spearman rank IC."""
    rows = []
    for date, g in df.groupby(date_col):
        g = g[[prediction_col, realized_return_col]].dropna()
        if len(g) < 2:
            continue
        pearson_ic = g[prediction_col].corr(g[realized_return_col])
        rank_ic = spearmanr(g[prediction_col], g[realized_return_col]).correlation
        rows.append({date_col: date, "ic": pearson_ic, "rank_ic": rank_ic, "n_assets": len(g)})
    return pd.DataFrame(rows)


def performance_summary(returns: pd.Series, annualization: int = 252) -> dict[str, float]:
    """Return common backtest metrics."""
    r = pd.Series(returns).dropna()
    return {
        "n_periods": float(len(r)),
        "mean_return": float(r.mean()) if len(r) else np.nan,
        "annualized_return": annualized_return(r, annualization),
        "annualized_volatility": annualized_volatility(r, annualization),
        "sharpe_ratio": sharpe_ratio(r, annualization),
        "max_drawdown": max_drawdown(r),
        "hit_rate": float((r > 0).mean()) if len(r) else np.nan,
    }
