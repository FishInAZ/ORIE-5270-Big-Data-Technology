from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import ModelConfig


def load_table(path: str | Path, **kwargs) -> pd.DataFrame:
    """Load a CSV, Excel, or Parquet table."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, **kwargs)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, **kwargs)
    if suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    raise ValueError(f"Unsupported file type: {suffix}")


def save_table(df: pd.DataFrame, path: str | Path, index: bool = False, **kwargs) -> None:
    """Save a dataframe as CSV, Excel, or Parquet based on file suffix."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=index, **kwargs)
    elif suffix in {".xlsx", ".xls"}:
        df.to_excel(path, index=index, **kwargs)
    elif suffix == ".parquet":
        df.to_parquet(path, index=index, **kwargs)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def infer_feature_columns(
    df: pd.DataFrame,
    exclude: Iterable[str],
    require_numeric: bool = True,
) -> list[str]:
    """Infer usable factor columns by excluding identifiers and nonnumeric columns."""
    excluded = {c for c in exclude if c is not None}
    candidates = [c for c in df.columns if c not in excluded]
    if require_numeric:
        candidates = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
    return candidates


def prepare_model_frame(df: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    """Clean and align a panel dataframe for rolling prediction.

    The function sorts by asset/date, constructs a next-period target if needed, keeps only
    model columns, converts features to numeric, and drops rows with missing model inputs.
    """
    config.validate()
    required = {config.date_col, config.asset_col, *config.feature_cols}
    if config.target_col is None:
        if config.return_col is None and config.price_col is None:
            raise ValueError("Provide target_col, return_col, or price_col")
        required.add(config.return_col or config.price_col)
    else:
        required.add(config.target_col)

    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    out[config.date_col] = pd.to_datetime(out[config.date_col])
    out = out.sort_values([config.asset_col, config.date_col]).reset_index(drop=True)

    target_name = config.target_col or "target_return"
    if config.target_col is None:
        if config.return_col is not None:
            out[target_name] = out.groupby(config.asset_col)[config.return_col].shift(-1)
        else:
            returns = out.groupby(config.asset_col)[config.price_col].pct_change()
            out["_computed_return"] = returns
            out[target_name] = out.groupby(config.asset_col)["_computed_return"].shift(-1)

    for col in config.feature_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out[target_name] = pd.to_numeric(out[target_name], errors="coerce")

    keep_cols = [config.date_col, config.asset_col, target_name, *config.feature_cols]
    out = out[keep_cols].dropna().reset_index(drop=True)
    return out.rename(columns={target_name: "target_return"})
