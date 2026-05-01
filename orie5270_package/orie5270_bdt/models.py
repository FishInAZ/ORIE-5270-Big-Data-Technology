from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import ModelConfig
from .data import prepare_model_frame


class RollingFactorModel:
    """Rolling OLS/Ridge/Lasso model for stock-level next-period return prediction."""

    def __init__(self, config: ModelConfig):
        config.validate()
        self.config = config

    def _estimator(self):
        if self.config.model_type == "ols":
            model = LinearRegression()
        elif self.config.model_type == "ridge":
            model = Ridge(alpha=self.config.alpha)
        elif self.config.model_type == "lasso":
            model = Lasso(alpha=self.config.alpha, max_iter=20_000)
        else:  # protected by validate
            raise ValueError(f"Unsupported model_type: {self.config.model_type}")

        if self.config.standardize:
            return Pipeline([("scaler", StandardScaler()), ("model", model)])
        return model

    def predict_asset(self, asset_df: pd.DataFrame) -> pd.DataFrame:
        """Generate rolling out-of-sample predictions for one asset."""
        cfg = self.config
        x = asset_df[list(cfg.feature_cols)].to_numpy(dtype=float)
        y = asset_df["target_return"].to_numpy(dtype=float)
        dates = asset_df[cfg.date_col].to_numpy()
        assets = asset_df[cfg.asset_col].to_numpy()

        rows: list[dict] = []
        for i in range(cfg.min_periods, len(asset_df)):
            start = max(0, i - cfg.window)
            train_x = x[start:i]
            train_y = y[start:i]
            valid = np.isfinite(train_x).all(axis=1) & np.isfinite(train_y)
            if valid.sum() < cfg.min_periods:
                continue

            estimator = self._estimator()
            estimator.fit(train_x[valid], train_y[valid])
            pred = float(estimator.predict(x[i : i + 1])[0])
            rows.append(
                {
                    cfg.date_col: dates[i],
                    cfg.asset_col: assets[i],
                    "prediction": pred,
                    "target_return": float(y[i]),
                    "model_type": cfg.model_type,
                    "train_start": dates[start],
                    "train_end": dates[i - 1],
                    "n_train": int(valid.sum()),
                }
            )
        return pd.DataFrame(rows)

    def fit_predict(self, df: pd.DataFrame, already_prepared: bool = False) -> pd.DataFrame:
        """Fit rolling models by asset and return out-of-sample predictions."""
        cfg = self.config
        model_df = df.copy() if already_prepared else prepare_model_frame(df, cfg)
        predictions = []
        for _, asset_df in model_df.groupby(cfg.asset_col, sort=False):
            asset_df = asset_df.sort_values(cfg.date_col).reset_index(drop=True)
            pred_df = self.predict_asset(asset_df)
            if not pred_df.empty:
                predictions.append(pred_df)
        if not predictions:
            return pd.DataFrame(
                columns=[
                    cfg.date_col,
                    cfg.asset_col,
                    "prediction",
                    "target_return",
                    "model_type",
                    "train_start",
                    "train_end",
                    "n_train",
                ]
            )
        out = pd.concat(predictions, ignore_index=True)
        return out.sort_values([cfg.date_col, cfg.asset_col]).reset_index(drop=True)

    def to_dict(self) -> dict:
        return asdict(self.config)


def fit_predict_rolling(df: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    """Convenience wrapper around RollingFactorModel."""
    return RollingFactorModel(config).fit_predict(df)
