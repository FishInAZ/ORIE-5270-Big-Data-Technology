import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# =========================================================
# USER SETTINGS
# =========================================================
OLS_FILE = "factor_model_ols_output_daily.csv"
RIDGE_FILE = "factor_model_ridge_output_daily.csv"
LASSO_FILE = "factor_model_lasso_output_daily.csv"

DATE_COL = "date"
ASSET_COL = "company"
RET_COL = "ret_next"

MODEL_PRED_COLS = {
    "OLS": "pred_ret_ols",
    "Ridge": "pred_ret_ridge",
    "Lasso": "pred_ret_lasso"
}

N_QUANTILES = 4
TRADING_DAYS = 252

OUTPUT_DIR = "backtest_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def print_line():
    print("=" * 100)


def load_model_file(file_path, date_col, asset_col, ret_col, pred_col, model_name):
    """Load one model output file and keep only needed columns."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)

    required_cols = [date_col, asset_col, ret_col, pred_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{model_name}: missing columns {missing}. Actual columns: {df.columns.tolist()}"
        )

    df = df[required_cols].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[ret_col] = pd.to_numeric(df[ret_col], errors="coerce")
    df[pred_col] = pd.to_numeric(df[pred_col], errors="coerce")

    df = df.dropna(subset=[date_col, asset_col, ret_col, pred_col]).copy()
    df = df.sort_values([date_col, asset_col]).reset_index(drop=True)

    return df


def assign_quantiles_by_date(group, pred_col, n_quantiles):
    """
    Assign quantiles cross-sectionally within each date.
    Returns labels 1...n_quantiles
    """
    g = group.copy()

    # if too few unique predictions, qcut may fail
    try:
        g["quantile"] = pd.qcut(
            g[pred_col],
            q=n_quantiles,
            labels=False,
            duplicates="drop"
        )
        g["quantile"] = g["quantile"] + 1
    except Exception:
        g["quantile"] = np.nan

    return g


def compute_daily_ic(group, pred_col, ret_col):
    """Compute daily Pearson IC and Spearman Rank IC."""
    g = group[[pred_col, ret_col]].dropna().copy()

    if len(g) < 3:
        return pd.Series({"IC": np.nan, "RankIC": np.nan})

    if g[pred_col].nunique() < 2 or g[ret_col].nunique() < 2:
        return pd.Series({"IC": np.nan, "RankIC": np.nan})

    ic = g[pred_col].corr(g[ret_col], method="pearson")
    rank_ic = g[pred_col].corr(g[ret_col], method="spearman")

    return pd.Series({"IC": ic, "RankIC": rank_ic})


def annualized_return(daily_ret, trading_days=252):
    daily_ret = pd.Series(daily_ret).dropna()
    if len(daily_ret) == 0:
        return np.nan
    cum = (1 + daily_ret).prod()
    n = len(daily_ret)
    return cum ** (trading_days / n) - 1


def annualized_volatility(daily_ret, trading_days=252):
    daily_ret = pd.Series(daily_ret).dropna()
    if len(daily_ret) == 0:
        return np.nan
    return daily_ret.std() * np.sqrt(trading_days)


def sharpe_ratio(daily_ret, trading_days=252):
    ann_ret = annualized_return(daily_ret, trading_days)
    ann_vol = annualized_volatility(daily_ret, trading_days)
    if pd.isna(ann_vol) or ann_vol == 0:
        return np.nan
    return ann_ret / ann_vol


def max_drawdown(daily_ret):
    daily_ret = pd.Series(daily_ret).fillna(0.0)
    cum = (1 + daily_ret).cumprod()
    running_max = cum.cummax()
    drawdown = cum / running_max - 1
    return drawdown.min()


def backtest_one_model(df, model_name, pred_col, date_col, asset_col, ret_col, n_quantiles=5):
    """
    Full quantile backtest for one model.
    """
    print_line()
    print(f"Running backtest for {model_name}")
    print_line()

    work_df = df.copy()

    # 1) assign quantiles by date
    work_df = work_df.groupby(date_col, group_keys=False).apply(
        assign_quantiles_by_date,
        pred_col=pred_col,
        n_quantiles=n_quantiles
    )

    work_df = work_df.dropna(subset=["quantile"]).copy()
    work_df["quantile"] = work_df["quantile"].astype(int)

    print(f"{model_name} usable rows after quantile assignment: {len(work_df)}")
    print(f"{model_name} unique dates: {work_df[date_col].nunique()}")
    print(f"{model_name} unique assets: {work_df[asset_col].nunique()}")

    # 2) quantile portfolio daily returns
    quantile_ret = (
        work_df.groupby([date_col, "quantile"])[ret_col]
        .mean()
        .reset_index()
        .pivot(index=date_col, columns="quantile", values=ret_col)
        .sort_index()
    )

    # rename columns
    quantile_ret.columns = [f"Q{int(c)}" for c in quantile_ret.columns]

    # 3) long-short daily return
    top_col = f"Q{n_quantiles}"
    bottom_col = "Q1"

    quantile_ret["long_short"] = quantile_ret[top_col] - \
        quantile_ret[bottom_col]
    quantile_ret["long_only_top"] = quantile_ret[top_col]
    quantile_ret["short_only_bottom"] = -quantile_ret[bottom_col]

    # 4) cumulative returns
    cum_ret = (1 + quantile_ret).cumprod() - 1

    # 5) daily IC
    ic_df = (
        work_df.groupby(date_col)
        .apply(compute_daily_ic, pred_col=pred_col, ret_col=ret_col)
        .reset_index()
        .sort_values(date_col)
    )

    ic_df["IC_cumsum"] = ic_df["IC"].fillna(0).cumsum()
    ic_df["RankIC_cumsum"] = ic_df["RankIC"].fillna(0).cumsum()

    # 6) summary stats
    summary = {
        "model": model_name,
        "n_rows": len(work_df),
        "n_dates": work_df[date_col].nunique(),
        "n_assets": work_df[asset_col].nunique(),

        "mean_IC": ic_df["IC"].mean(),
        "std_IC": ic_df["IC"].std(),
        "ICIR": ic_df["IC"].mean() / ic_df["IC"].std() if ic_df["IC"].std() not in [0, np.nan] else np.nan,

        "mean_RankIC": ic_df["RankIC"].mean(),
        "std_RankIC": ic_df["RankIC"].std(),
        "RankICIR": ic_df["RankIC"].mean() / ic_df["RankIC"].std() if ic_df["RankIC"].std() not in [0, np.nan] else np.nan,

        "LS_ann_return": annualized_return(quantile_ret["long_short"]),
        "LS_ann_vol": annualized_volatility(quantile_ret["long_short"]),
        "LS_sharpe": sharpe_ratio(quantile_ret["long_short"]),
        "LS_max_drawdown": max_drawdown(quantile_ret["long_short"]),

        "TopQ_ann_return": annualized_return(quantile_ret[top_col]),
        "TopQ_ann_vol": annualized_volatility(quantile_ret[top_col]),
        "TopQ_sharpe": sharpe_ratio(quantile_ret[top_col]),

        "BottomQ_ann_return": annualized_return(quantile_ret[bottom_col]),
        "BottomQ_ann_vol": annualized_volatility(quantile_ret[bottom_col]),
        "BottomQ_sharpe": sharpe_ratio(quantile_ret[bottom_col]),
    }

    summary_df = pd.DataFrame([summary])

    return {
        "work_df": work_df,
        "quantile_ret": quantile_ret,
        "cum_ret": cum_ret,
        "ic_df": ic_df,
        "summary_df": summary_df
    }


def save_model_outputs(model_name, result_dict, output_dir):
    """Save outputs for one model."""
    result_dict["work_df"].to_csv(
        os.path.join(output_dir, f"{model_name.lower()}_backtest_panel.csv"),
        index=False
    )
    result_dict["quantile_ret"].to_csv(
        os.path.join(output_dir, f"{model_name.lower()}_quantile_returns.csv")
    )
    result_dict["cum_ret"].to_csv(
        os.path.join(
            output_dir, f"{model_name.lower()}_cumulative_returns.csv")
    )
    result_dict["ic_df"].to_csv(
        os.path.join(output_dir, f"{model_name.lower()}_daily_ic.csv"),
        index=False
    )
    result_dict["summary_df"].to_csv(
        os.path.join(output_dir, f"{model_name.lower()}_summary.csv"),
        index=False
    )


def plot_long_short_cumulative(results_dict, output_dir):
    """Plot cumulative long-short returns of all models."""
    plt.figure(figsize=(12, 7))

    for model_name, result in results_dict.items():
        series = result["cum_ret"]["long_short"]
        plt.plot(series.index, series.values, label=model_name)

    plt.title("Long-Short Cumulative Return: OLS vs Ridge vs Lasso")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(
        output_dir, "compare_long_short_cumulative.png"), dpi=200)
    plt.show()


def plot_ic_cumsum(results_dict, output_dir):
    """Plot cumulative IC of all models."""
    plt.figure(figsize=(12, 7))

    for model_name, result in results_dict.items():
        series = result["ic_df"].set_index(DATE_COL)["IC_cumsum"]
        plt.plot(series.index, series.values, label=model_name)

    plt.title("Cumulative IC: OLS vs Ridge vs Lasso")
    plt.xlabel("Date")
    plt.ylabel("Cumulative IC")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "compare_ic_cumsum.png"), dpi=200)
    plt.show()


# =========================================================
# MAIN
# =========================================================
def main():
    print_line()
    print("OLS / Ridge / Lasso Quantile Backtest Started")
    print_line()

    # 1) load model files
    ols_df = load_model_file(
        OLS_FILE, DATE_COL, ASSET_COL, RET_COL,
        MODEL_PRED_COLS["OLS"], "OLS"
    )
    ridge_df = load_model_file(
        RIDGE_FILE, DATE_COL, ASSET_COL, RET_COL,
        MODEL_PRED_COLS["Ridge"], "Ridge"
    )
    lasso_df = load_model_file(
        LASSO_FILE, DATE_COL, ASSET_COL, RET_COL,
        MODEL_PRED_COLS["Lasso"], "Lasso"
    )

    print("Loaded files successfully.")
    print(f"OLS rows: {len(ols_df)}")
    print(f"Ridge rows: {len(ridge_df)}")
    print(f"Lasso rows: {len(lasso_df)}")

    # 2) run backtests
    ols_result = backtest_one_model(
        ols_df, "OLS", MODEL_PRED_COLS["OLS"],
        DATE_COL, ASSET_COL, RET_COL, N_QUANTILES
    )
    ridge_result = backtest_one_model(
        ridge_df, "Ridge", MODEL_PRED_COLS["Ridge"],
        DATE_COL, ASSET_COL, RET_COL, N_QUANTILES
    )
    lasso_result = backtest_one_model(
        lasso_df, "Lasso", MODEL_PRED_COLS["Lasso"],
        DATE_COL, ASSET_COL, RET_COL, N_QUANTILES
    )

    results = {
        "OLS": ols_result,
        "Ridge": ridge_result,
        "Lasso": lasso_result
    }

    # 3) save each model output
    for model_name, result in results.items():
        save_model_outputs(model_name, result, OUTPUT_DIR)

    # 4) build combined summary
    summary_all = pd.concat(
        [results[m]["summary_df"] for m in ["OLS", "Ridge", "Lasso"]],
        axis=0
    ).reset_index(drop=True)

    summary_all.to_csv(
        os.path.join(OUTPUT_DIR, "summary_all_models.csv"),
        index=False
    )

    print_line()
    print("Combined summary:")
    print(summary_all)
    print_line()

    # 5) plot comparisons
    plot_long_short_cumulative(results, OUTPUT_DIR)
    plot_ic_cumsum(results, OUTPUT_DIR)

    # 6) save combined quantile comparison
    ls_compare = pd.DataFrame(index=ols_result["quantile_ret"].index)
    for model_name, result in results.items():
        ls_compare[model_name] = result["quantile_ret"]["long_short"]

    ls_compare.to_csv(
        os.path.join(OUTPUT_DIR, "compare_long_short_daily_returns.csv")
    )

    # 7) save combined IC comparison
    ic_compare = None
    for model_name, result in results.items():
        tmp = result["ic_df"][[DATE_COL, "IC", "RankIC"]].copy()
        tmp = tmp.rename(columns={
            "IC": f"{model_name}_IC",
            "RankIC": f"{model_name}_RankIC"
        })

        if ic_compare is None:
            ic_compare = tmp
        else:
            ic_compare = ic_compare.merge(tmp, on=DATE_COL, how="outer")

    ic_compare = ic_compare.sort_values(DATE_COL)
    ic_compare.to_csv(
        os.path.join(OUTPUT_DIR, "compare_ic_daily.csv"),
        index=False
    )

    print("Files saved to:", OUTPUT_DIR)
    print("Done.")
    print_line()


if __name__ == "__main__":
    main()
