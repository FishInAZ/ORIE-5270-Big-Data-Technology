from orie5270_bdt.config import ModelConfig, BacktestConfig
from orie5270_bdt.data import load_table, prepare_model_frame
from orie5270_bdt.models import fit_predict_rolling
from orie5270_bdt.backtest import quantile_backtest

df = load_table("../Dataset/merged_df.csv")

feature_cols = [
    "liquidity_alpha",
    "size_alpha",
    "sentiment_change_alpha",
    "sentiment_exp_decay_alpha",
    "ema_3",
    "ema_10",
    "sent_macd",
    "sent_macd_signal",
    "sent_macd_alpha",
    "signal1",
    "signal2",
    "signal3",
    "signal4",
    "signal5",
    "signal6",
    "signal7",
    "signal8",
]

model_config = ModelConfig(
    date_col="date",
    asset_col="company",
    price_col="adjclose",
    return_col=None,
    feature_cols=feature_cols,
    target_col=None,
    model_type="ridge",
    window=252,
    min_periods=126,
    alpha=1.0,
)

panel = prepare_model_frame(df, model_config)

print("Prepared panel shape:")
print(panel.shape)

print("Prepared panel preview:")
print(panel[["date", "company", "target_return"] + feature_cols[:5]].head())

predictions = fit_predict_rolling(df, model_config)

print("Predictions shape:")
print(predictions.shape)

print("Number of non-missing predictions:")
print(predictions["prediction"].notna().sum())

backtest_config = BacktestConfig(
    date_col="date",
    asset_col="company",
    prediction_col="prediction",
    realized_return_col="target_return",
    n_quantiles=5,
    long_quantile=5,
    short_quantile=1,
)

result = quantile_backtest(predictions, backtest_config)

print("Backtest summary:")
print(result["summary"])

print("Available result keys:")
print(result.keys())

print("Backtest daily returns preview:")
print(result["daily_returns"].head())

print("Quantile returns preview:")
print(result["quantile_returns"].head())

predictions.to_csv("full_test_predictions.csv", index=False)
result["daily_returns"].to_csv("full_test_daily_returns.csv", index=False)
result["quantile_returns"].to_csv("full_test_quantile_returns.csv", index=False)
result["scored_assets"].to_csv("full_test_scored_assets.csv", index=False)
result["ic"].to_csv("full_test_ic.csv", index=False)
result["summary"].to_csv("full_test_summary.csv", index=False)

print("Saved files:")
print("full_test_predictions.csv")
print("full_test_daily_returns.csv")
print("full_test_quantile_returns.csv")
print("full_test_scored_assets.csv")
print("full_test_ic.csv")
print("full_test_summary.csv")