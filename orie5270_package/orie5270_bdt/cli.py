from __future__ import annotations

import argparse
from pathlib import Path

from .backtest import quantile_backtest
from .config import BacktestConfig, ModelConfig
from .data import load_table, save_table
from .models import fit_predict_rolling


def _parse_features(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def fit_command() -> None:
    parser = argparse.ArgumentParser(description="Fit rolling OLS/Ridge/Lasso factor models.")
    parser.add_argument("--input", required=True, help="Input CSV/XLSX/Parquet panel file.")
    parser.add_argument("--output", required=True, help="Output prediction CSV/XLSX/Parquet file.")
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--asset-col", default="ticker")
    parser.add_argument("--price-col", default="close")
    parser.add_argument("--return-col", default=None)
    parser.add_argument("--target-col", default=None)
    parser.add_argument("--features", required=True, help="Comma-separated factor columns.")
    parser.add_argument("--model", choices=["ols", "ridge", "lasso"], default="ols")
    parser.add_argument("--window", type=int, default=252)
    parser.add_argument("--min-periods", type=int, default=126)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    df = load_table(args.input)
    config = ModelConfig(
        date_col=args.date_col,
        asset_col=args.asset_col,
        price_col=args.price_col,
        return_col=args.return_col,
        target_col=args.target_col,
        feature_cols=_parse_features(args.features),
        model_type=args.model,
        window=args.window,
        min_periods=args.min_periods,
        alpha=args.alpha,
    )
    predictions = fit_predict_rolling(df, config)
    save_table(predictions, Path(args.output))


def backtest_command() -> None:
    parser = argparse.ArgumentParser(description="Run quantile long-short backtest.")
    parser.add_argument("--input", required=True, help="Prediction file from orie5270-fit.")
    parser.add_argument("--output-dir", required=True, help="Directory for backtest output CSVs.")
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--asset-col", default="ticker")
    parser.add_argument("--prediction-col", default="prediction")
    parser.add_argument("--realized-return-col", default="target_return")
    parser.add_argument("--n-quantiles", type=int, default=5)
    parser.add_argument("--long-quantile", type=int, default=5)
    parser.add_argument("--short-quantile", type=int, default=1)
    parser.add_argument("--transaction-cost-bps", type=float, default=0.0)
    args = parser.parse_args()

    df = load_table(args.input)
    config = BacktestConfig(
        date_col=args.date_col,
        asset_col=args.asset_col,
        prediction_col=args.prediction_col,
        realized_return_col=args.realized_return_col,
        n_quantiles=args.n_quantiles,
        long_quantile=args.long_quantile,
        short_quantile=args.short_quantile,
        transaction_cost_bps=args.transaction_cost_bps,
    )
    result = quantile_backtest(df, config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in result.items():
        save_table(table, output_dir / f"{name}.csv")
