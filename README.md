# ORIE-5270 Big Data Technology Project - Multi-Factor Trading Strategy

## Overview

This project develops a data-driven stock return prediction pipeline by combining web-scraped market data, market news, engineered financial signals, rolling predictive models, and portfolio backtesting. The workflow covers the full process from raw data collection to feature engineering, predictive modeling, model validation, and performance evaluation.

The project integrates three main categories of signals:

- technical indicators derived from stock price data
- liquidity and size-related factors
- sentiment-based signals extracted from market news

Using these engineered factors, we build and compare rolling **OLS**, **Ridge**, and **Lasso** models to predict next-day stock returns. The predictions are evaluated through quantile-based long-short portfolio backtesting.

---

## Project Structure

```bash
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

---

## Web Scrapping

The `Web Scrapping/` folder contains the notebooks used to collect the raw data for this project through web scraping. This module is the starting point of the pipeline.

### Main contents

* `stock_data.ipynb`
  Scrapes stock market data used for return computation and factor construction.

* `market_news.ipynb`
  Scrapes market news data used for sentiment analysis and sentiment-based signal generation.

Overall, this folder provides the raw stock and news data that later feed into the `Dataset/` and `Factor Analysis/` modules.

---

## Dataset

The `Dataset/` folder contains the raw, intermediate, and merged datasets used throughout the project. The stock-related data were collected through web scraping, and multiple engineered signals were constructed and merged into a unified dataset for downstream analysis and modeling.

### Main contents

* `stock_data.csv`
  Raw stock-level data collected through web scraping.

* `stock_returns.csv`
  Stock return data derived for analysis and modeling.

* `market_news_data.csv`
  Scraped market news data used for text-based analysis.

* `market_news_with_sentiment.csv`
  Market news data after sentiment analysis, used to construct sentiment-based signals.

* `daily_liquidity_size_factor_panel_company_first.xlsx`
  Dataset used to construct liquidity-related and size-related signals.

* `sentiment_signals.csv`
  Engineered sentiment-based signals.

* `alpha_factors.csv`
  Engineered alpha or technical factor signals.

* `merged_signals.csv`
  Combined signal-level dataset before final merging.

* `merged_df.csv`
  Final merged dataframe containing stock data and engineered signals.

* `merge_data.ipynb`
  Algorithm used to merge different stock signals into a complete dataframe for later factor analysis and modeling.

Overall, the dataset module organizes the raw and engineered data into a single modeling-ready framework.

---

## Factor Analysis

The `Factor Analysis/` folder contains the Jupyter notebooks that document the construction logic of the engineered factors used in this project. These notebooks explain how different categories of signals are defined, calculated, and prepared for later modeling.

### Main contents

* `Technical Indicators.ipynb`
  Constructs technical factors and indicators from stock data.

* `daily liquidity and size factors.ipynb`
  Generates liquidity- and size-related factors.

* `sentiment_analysis.ipynb`
  Processes and analyzes market news text for sentiment-related information.

* `sentiment_alpha.ipynb`
  Transforms sentiment analysis outputs into sentiment-based alpha signals.

This folder focuses on the feature engineering process, including technical, liquidity, size, and sentiment signals.

---

## Model

The `Model/` folder contains the original predictive modeling, testing, and backtesting scripts of the project. Using the engineered factors produced earlier, this module builds daily rolling factor models to predict next-day stock returns and evaluates the resulting signals through portfolio backtesting.

### Predictive Models

Three rolling models are implemented:

* `ols_daily_factor_model.py`
  Builds a rolling OLS factor model using a 252-trading-day estimation window.

* `ridge_daily_factor_model.py`
  Builds a rolling Ridge regression model with L2 regularization.

* `lasso_daily_factor_model.py`
  Builds a rolling Lasso regression model with L1 regularization.

All three models use the same factor set, including liquidity, size, sentiment-based signals, and technical indicators.

### Modeling Workflow

For each stock and date, the scripts:

1. read the final factor dataset,
2. clean and standardize the data,
3. compute daily stock returns,
4. construct the next-day return target,
5. estimate rolling models using the previous 252 trading days,
6. generate out-of-sample next-day return predictions.

### Model Outputs

The model scripts generate outputs such as:

* predicted next-day returns
* daily coefficient estimates
* summary result files
* final factor files used in modeling

Separate subfolders store the corresponding model-specific outputs:

* `ols/`
* `ridge/`
* `lasso/`

These subfolders contain model prediction outputs, coefficient files, and summary files for each method.

### Unit Tests

The `Model/` folder also includes unit test files to validate the original modeling and backtesting scripts.

Main testing files include:

* `ols_unit_test.pyw`
* `ols_checkpoint_unit_test.pyw`
* `ridge_unit_test.pyw`
* `lasso_unit_test.pyw`
* `backtest_unit_test.pyw`

These test scripts are used to check the implementation of the OLS, Ridge, Lasso, and backtesting modules.

### Backtesting

* `backtest.py`
  Compares OLS, Ridge, and Lasso predictions through a quantile-based backtesting framework.

Stocks are sorted into quantiles based on predicted returns for each day, and portfolio performance is evaluated using:

* long-short returns
* cumulative returns
* Information Coefficient (IC)
* Rank IC
* annualized return
* annualized volatility
* Sharpe ratio
* maximum drawdown

The `backtest_output/` folder stores the generated backtest summaries, daily return files, IC results, and comparison plots.

---

## Python Package

The `orie5270_package/` folder contains a reusable Python package version of the main modeling and backtesting workflow. It turns the project logic into importable modules so the pipeline can be tested, reused, and run more cleanly.

### Package contents

```bash
orie5270_package/
├── orie5270_bdt/
│   ├── __init__.py
│   ├── backtest.py
│   ├── cli.py
│   ├── config.py
│   ├── data.py
│   ├── metrics.py
│   └── models.py
├── tests/
│   ├── test_backtest.py
│   └── test_metrics.py
├── test_outputs/
├── full_project_test.py
├── pyproject.toml
└── README.md
```

### Module purpose

* `config.py`
  Defines `ModelConfig` and `BacktestConfig`.

* `data.py`
  Handles data loading, saving, and model-frame preparation.

* `models.py`
  Implements rolling OLS, Ridge, and Lasso prediction workflows.

* `backtest.py`
  Implements quantile-based long-short portfolio backtesting.

* `metrics.py`
  Contains performance metrics such as Sharpe ratio, annualized return, volatility, maximum drawdown, IC, and Rank IC.

* `cli.py`
  Provides command-line entry points for model fitting and backtesting.

* `tests/`
  Contains package-level unit tests.

* `full_project_test.py`
  Runs an end-to-end test on the real project dataset.

### Package test result

The package has been tested on:

```text
Dataset/merged_df.csv
```

The actual dataset column mapping is:

```text
date_col  = "date"
asset_col = "company"
price_col = "adjclose"
```

The full project test successfully produced:

```text
Prepared panel shape: (44123, 20)
Predictions shape: (41099, 8)
Number of non-missing predictions: 41099
```

The unit tests passed:

```text
3 passed
```

To install and test the package locally:

```bash
cd orie5270_package
pip install -e .
python -m pytest tests
python full_project_test.py
```

---

## Methodology

The overall project workflow can be summarized as follows:

1. **Web scrape stock and market news data**
2. **Construct technical, liquidity, size, and sentiment-based factors**
3. **Merge all engineered signals into a unified dataset**
4. **Build rolling OLS, Ridge, and Lasso factor models**
5. **Generate out-of-sample predictions of next-day stock returns**
6. **Validate the model and backtest modules using unit tests**
7. **Evaluate predictive performance through quantile portfolio backtesting**
8. **Package the modeling and backtesting workflow into a reusable Python package**

This structure allows us to compare how different regularized and unregularized linear models perform when applied to engineered multi-source financial signals.

---

## Main Features

* End-to-end data pipeline from web scraping to backtesting
* Multi-source feature engineering using price, liquidity, size, and news sentiment data
* Rolling out-of-sample prediction framework
* Comparison of OLS, Ridge, and Lasso models
* Unit tests for model and backtest validation
* Quantile portfolio backtesting and performance evaluation
* Reusable Python package for model fitting, backtesting, and testing

---

## Technologies Used

* Python
* Jupyter Notebook
* pandas
* numpy
* scikit-learn
* statsmodels
* matplotlib
* pytest

---

## How to Run

A general order for running the original project workflow is:

1. Run the notebooks in `Web Scrapping/` to collect stock and market news data.
2. Use the notebooks in `Factor Analysis/` to construct technical, liquidity, size, and sentiment factors.
3. Use `merge_data.ipynb` in `Dataset/` to combine all signals into the final modeling dataset.
4. Run the model scripts in `Model/`:

   * `ols_daily_factor_model.py`
   * `ridge_daily_factor_model.py`
   * `lasso_daily_factor_model.py`
5. Run the relevant unit test files to verify the model and backtest scripts.
6. Run `backtest.py` to evaluate the model predictions.

Alternatively, run the reusable package workflow:

```bash
cd orie5270_package
pip install -e .
python -m pytest tests
python full_project_test.py
```

The package workflow loads `Dataset/merged_df.csv`, prepares the modeling panel, runs rolling Ridge prediction, runs quantile long-short backtesting, and saves test output files.

---

## Notes

* `.DS_Store` and `.ipynb_checkpoints` are system-generated files or folders and are not part of the main project logic.
* The folder name `Web Scrapping/` is preserved to match the repository structure, although “Web Scraping” is the standard spelling.
* The `orie5270_package/test_outputs/` folder contains generated package test outputs. These files do not need to be committed unless they are kept as example results.
* Depending on the local environment, file paths or input filenames may need to be adjusted before execution.

---

## Authors

ORIE-5270 Big Data Technology Project Team

Boxuan Hu, Chenyu Li, Jialu Xu, Yingjie Wang

---

## Conclusion

This project shows that combining technical, liquidity, size, and sentiment-based signals can produce a usable stock return prediction and backtesting pipeline. Although next-day return prediction remains noisy, the rolling factor models were able to generate out-of-sample signals that could be evaluated through a systematic quantile long-short strategy.

Our results suggest that the engineered factors contain some predictive information, but the signal is relatively weak. In the package-based end-to-end test, the rolling Ridge model generated 41,099 non-missing predictions and produced a positive annualized return of about 10.10% in the long-short backtest. However, the Sharpe ratio was modest at around 0.34 and the maximum drawdown was large at about -59.06%, indicating that the strategy is not yet stable enough for practical trading without further improvement.

A key takeaway is that regularized models such as Ridge and Lasso are natural choices for this setting because the dataset contains many correlated financial and sentiment signals. Regularization can help reduce overfitting risk and improve the stability of rolling predictions, especially when the model is re-estimated over time. The backtesting framework also shows the importance of evaluating models not only by prediction accuracy, but also by portfolio-level performance metrics such as cumulative return, volatility, Sharpe ratio, drawdown, IC, and Rank IC.

Overall, this project demonstrates a complete big-data workflow for financial prediction, from data collection and factor construction to rolling modeling, validation, portfolio backtesting, and package-based reuse. The main takeaway is that multi-source financial signals can provide measurable predictive value, but the strategy requires further refinement in feature selection, model tuning, transaction cost modeling, and risk control before it can be considered a reliable trading strategy.
