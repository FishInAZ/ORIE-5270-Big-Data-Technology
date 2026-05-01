# orie5270-bdt

Reusable Python package for the ORIE 5270 Big Data Technology multi-factor trading strategy project.

This package wraps the main project workflow into importable modules:

- data loading and model-frame preparation
- rolling OLS, Ridge, and Lasso next-day return prediction
- equal-weight quantile long-short backtesting
- IC, Rank IC, Sharpe ratio, annualized return, annualized volatility, and max drawdown metrics
- command-line scripts for model fitting and backtesting

## Project location

This package is stored inside the main GitHub project:

```text
ORIE-5270-Big-Data-Technology/
├── Dataset/
├── Factor Analysis/
├── Model/
├── Web Scrapping/
├── orie5270_package/
│   ├── orie5270_bdt/
│   ├── tests/
│   ├── test_outputs/
│   ├── full_project_test.py
│   ├── pyproject.toml
│   └── README.md
└── README.md
```

The outer `README.md` explains the full GitHub project.
This inner `README.md` explains only the Python package.

## Install locally

From the package folder:

```bash
cd orie5270_package
pip install -e .
```

The `-e` flag means editable mode. If you change the package files, Python will immediately use the updated version.

For tests:

```bash
pip install -e ".[dev]"
python -m pytest tests
```

Current unit test result:

```text
3 passed
```

## Actual project data

The package has been tested on:

```text
../Dataset/merged_df.csv
```

The dataset shape is:

```text
(44988, 32)
```

The actual column mapping used in the project is:

```text
date_col  = "date"
asset_col = "company"
price_col = "adjclose"
```

The package creates `target_return` internally as the next-period return based on `adjclose`.

## Feature columns used

The full project test uses the following factor/signal columns:

```python
features = [
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
```

## Example Python usage

```python
from orie5270_bdt.config import ModelConfig, BacktestConfig
from orie5270_bdt.data import load_table
from orie5270_bdt.models import fit_predict_rolling
from orie5270_bdt.backtest import quantile_backtest

df = load_table("../Dataset/merged_df.csv")

features = [
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
    feature_cols=features,
    target_col=None,
    model_type="ridge",
    window=252,
    min_periods=126,
    alpha=1.0,
)

predictions = fit_predict_rolling(df, model_config)

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

print(result["summary"])
print(result["daily_returns"].head())
```

## Full project test

Run the full end-to-end test from inside `orie5270_package`:

```bash
python full_project_test.py
```

This script runs the complete workflow:

```text
load real data
prepare modeling panel
run rolling Ridge regression
generate next-day return predictions
run quantile long-short backtest
save output CSV files
```

Successful test output:

```text
Prepared panel shape: (44123, 20)
Predictions shape: (41099, 8)
Number of non-missing predictions: 41099
```

The backtest summary from the successful run was:

```text
n_periods: 2577
mean_return: 0.000561
annualized_return: 0.101037
annualized_volatility: 0.299849
sharpe_ratio: 0.336961
max_drawdown: -0.590637
hit_rate: 0.506015
mean_ic: 0.01098
mean_rank_ic: 0.011054
```

## Backtest output files

The full test saves output files to:

```text
orie5270_package/test_outputs/
```

Generated files:

```text
full_test_predictions.csv
full_test_daily_returns.csv
full_test_quantile_returns.csv
full_test_scored_assets.csv
full_test_ic.csv
full_test_summary.csv
```

These files are generated test outputs. They do not need to be committed to GitHub unless you want to keep them as example results.

## Returned backtest objects

`quantile_backtest()` returns a dictionary with the following keys:

```python
dict_keys([
    "scored_assets",
    "quantile_returns",
    "daily_returns",
    "ic",
    "summary"
])
```

Important objects:

```python
result["summary"]
result["daily_returns"]
result["quantile_returns"]
result["scored_assets"]
result["ic"]
```

Use `result["daily_returns"]`, not `result["returns"]`.

## Package modules

```text
orie5270_bdt/
├── __init__.py
├── backtest.py
├── cli.py
├── config.py
├── data.py
├── metrics.py
└── models.py
```

Module purpose:

```text
config.py   - ModelConfig and BacktestConfig
data.py     - data loading, saving, and model-frame preparation
models.py   - rolling OLS, Ridge, and Lasso prediction
backtest.py - quantile portfolio backtesting
metrics.py  - performance and information-coefficient metrics
cli.py      - command-line entry points
```

## CLI usage

Example model fitting:

```bash
orie5270-fit \
  --input ../Dataset/merged_df.csv \
  --output ../Model/ridge_predictions.csv \
  --date-col date \
  --asset-col company \
  --price-col adjclose \
  --features liquidity_alpha,size_alpha,sentiment_change_alpha,sentiment_exp_decay_alpha,ema_3,ema_10,sent_macd,sent_macd_signal,sent_macd_alpha,signal1,signal2,signal3,signal4,signal5,signal6,signal7,signal8 \
  --model ridge \
  --window 252 \
  --min-periods 126 \
  --alpha 1.0
```

Example backtest:

```bash
orie5270-backtest \
  --input ../Model/ridge_predictions.csv \
  --output-dir ../Model/backtest_output \
  --date-col date \
  --asset-col company \
  --prediction-col prediction \
  --realized-return-col target_return \
  --n-quantiles 5 \
  --long-quantile 5 \
  --short-quantile 1
```

## Notes

A pandas `FutureWarning` may appear from `groupby.apply()` inside `backtest.py`:

```text
FutureWarning: DataFrameGroupBy.apply operated on the grouping columns
```

This warning does not stop the package from running. The package still installs, imports, passes unit tests, loads real data, generates predictions, runs the backtest, and saves output files successfully.
