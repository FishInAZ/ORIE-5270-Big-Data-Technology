from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import sys
import os
import warnings
warnings.filterwarnings("ignore")


# =========================================================
# USER SETTINGS
# =========================================================
INPUT_FILE = "final_factors.xlsx"
SHEET_NAME = 0   # use first sheet; change to sheet name string if needed

OUTPUT_FILE = "factor_model_lasso_output_daily.csv"
COEF_FILE = "factor_model_lasso_coefficients_daily.csv"

WINDOW_DAYS = 252
LASSO_ALPHA = 0.001

FACTOR_COLS = [
    "liquidity_alpha",
    "size_alpha",
    "signal_2",
    "sentiment_change_alpha",
    "sentiment_exp_decay_alpha",
    "sent_macd_alpha",
    "signal1",
    "signal2",
    "signal3",
    "signal4",
    "signal5",
    "signal6",
    "signal7",
    "signal8"
]

REQUIRED_COLS = ["company", "date", "adjclose_y"] + FACTOR_COLS


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def print_line():
    print("=" * 80)


def robust_to_datetime(series: pd.Series) -> pd.Series:
    """
    Robust date conversion:
    1. already datetime
    2. Excel serial number
    3. regular date string
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series)

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() >= 0.8 * len(series):
        try:
            return pd.to_datetime(numeric, origin="1899-12-30", unit="D")
        except Exception:
            pass

    return pd.to_datetime(series, errors="coerce")


def check_columns(df: pd.DataFrame, required_cols: list):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print_line()
        print("ERROR: Missing columns in Excel:")
        for c in missing:
            print(f"  - {c}")
        print("\nActual columns in your file:")
        print(df.columns.tolist())
        print_line()
        sys.exit(1)


def rolling_lasso_prediction(work_df: pd.DataFrame, factor_cols: list, window_days: int, alpha: float):
    """
    Rolling Lasso:
    For each day t:
      - use past window_days trading days as training sample
      - estimate Lasso on ret_next ~ factors
      - use factors at day t to predict ret_next
    """
    all_dates = sorted(work_df["date"].dropna().unique())

    pred_list = []
    coef_list = []

    if len(all_dates) <= window_days:
        print(
            f"ERROR: Need more than {window_days} dates, only got {len(all_dates)}.")
        return pd.DataFrame(), pd.DataFrame()

    for i in range(window_days, len(all_dates)):
        train_start = all_dates[i - window_days]
        train_end = all_dates[i - 1]
        pred_date = all_dates[i]

        train_data = work_df[
            (work_df["date"] >= train_start) &
            (work_df["date"] <= train_end)
        ].copy()

        test_data = work_df[
            work_df["date"] == pred_date
        ].copy()

        if len(train_data) == 0 or len(test_data) == 0:
            continue

        X_train = train_data[factor_cols].copy()
        y_train = train_data["ret_next"].copy()
        X_test = test_data[factor_cols].copy()

        # standardize factors using training data only
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        try:
            model = Lasso(
                alpha=alpha,
                fit_intercept=True,
                max_iter=10000,
                tol=1e-4
            )
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            test_data["pred_ret_lasso"] = y_pred

            pred_list.append(
                test_data[["company", "date", "ret_next", "pred_ret_lasso"]]
            )

            coef_row = {
                "date": pred_date,
                "intercept": model.intercept_,
                "train_start": train_start,
                "train_end": train_end,
                "n_train": len(train_data)
            }

            for j, col in enumerate(factor_cols):
                coef_row[col] = model.coef_[j]

            coef_row["n_nonzero_factors"] = int(
                np.sum(np.abs(model.coef_) > 1e-12))
            coef_list.append(coef_row)

        except Exception as e:
            print(
                f"[Lasso warning] date {pred_date} skipped due to error: {e}")
            continue

    pred_df = pd.concat(pred_list, axis=0).reset_index(
        drop=True) if pred_list else pd.DataFrame()
    coef_df = pd.DataFrame(coef_list)

    return pred_df, coef_df


# =========================================================
# MAIN
# =========================================================
def main():
    print_line()
    print("Daily Lasso Factor Model Pipeline Started")
    print_line()

    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: File not found -> {INPUT_FILE}")
        print("Put the Excel file in the same folder as this .py file,")
        print("or modify INPUT_FILE at the top of the script.")
        sys.exit(1)

    # 1. Read file
    print("Reading Excel file...")
    df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)

    print(f"Raw shape: {df.shape}")
    print("Columns found:")
    print(df.columns.tolist())

    # 2. Check columns
    check_columns(df, REQUIRED_COLS)

    # 3. Convert dates
    print_line()
    print("Converting dates...")
    df["date"] = robust_to_datetime(df["date"])

    # 4. Sort
    df = df.sort_values(["company", "date"]).reset_index(drop=True)

    # 5. Convert numeric columns
    print_line()
    print("Converting numeric columns...")
    numeric_cols = ["adjclose_y"] + FACTOR_COLS
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 6. Missing value report
    print_line()
    print("Missing value report:")
    print(df[REQUIRED_COLS].isna().sum())

    # 7. Compute daily returns
    print_line()
    print("Computing daily returns...")
    df["ret"] = df.groupby("company")["adjclose_y"].pct_change()

    # 8. Build next-day return target
    print("Building ret_next...")
    df["ret_next"] = df.groupby("company")["ret"].shift(-1)

    # 9. Show sample
    print_line()
    print("Sample after return construction:")
    print(df[["company", "date", "adjclose_y", "ret", "ret_next"]].head(15))

    # 10. Prepare rolling dataset
    print_line()
    print("Preparing rolling dataset...")
    work_df = df[["company", "date", "ret_next"] + FACTOR_COLS].copy()
    work_df = work_df.dropna(subset=FACTOR_COLS + ["ret_next"]).copy()
    work_df = work_df.sort_values(["date", "company"]).reset_index(drop=True)

    print(f"Rolling dataset shape: {work_df.shape}")
    print(f"Unique dates: {work_df['date'].nunique()}")
    print(f"Unique companies: {work_df['company'].nunique()}")

    # 11. Rolling Lasso
    print_line()
    print(f"Running rolling Lasso with alpha = {LASSO_ALPHA} ...")
    pred_lasso, coef_lasso = rolling_lasso_prediction(
        work_df=work_df,
        factor_cols=FACTOR_COLS,
        window_days=WINDOW_DAYS,
        alpha=LASSO_ALPHA
    )

    print(f"Prediction output shape: {pred_lasso.shape}")
    print(f"Coefficient output shape: {coef_lasso.shape}")

    if len(pred_lasso) > 0:
        print_line()
        print("Prediction preview:")
        print(pred_lasso.head(10))

    if len(coef_lasso) > 0:
        print_line()
        print("Coefficient preview:")
        print(coef_lasso.head(5))

        print_line()
        print("Average number of selected factors:")
        print(coef_lasso["n_nonzero_factors"].describe())

    # 12. Merge back
    print_line()
    print("Merging predictions back to original data...")
    if len(pred_lasso) > 0:
        df_out = df.merge(
            pred_lasso[["company", "date", "pred_ret_lasso"]],
            on=["company", "date"],
            how="left"
        )
    else:
        df_out = df.copy()
        df_out["pred_ret_lasso"] = np.nan

    # 13. Final output
    final_cols = [
        "company", "date", "adjclose_y",
        "ret", "ret_next", "pred_ret_lasso"
    ] + FACTOR_COLS

    final_output = df_out[final_cols].copy()

    print_line()
    print("Final output preview:")
    print(final_output.head(20))

    print_line()
    print("Summary statistics:")
    print(final_output[["ret_next", "pred_ret_lasso"]].describe())

    print_line()
    print("Non-null Lasso predictions:")
    print(final_output["pred_ret_lasso"].notna().sum())

    # 14. Save outputs
    final_output.to_csv(OUTPUT_FILE, index=False)
    coef_lasso.to_csv(COEF_FILE, index=False)

    print_line()
    print(f"Saved prediction file: {OUTPUT_FILE}")
    print(f"Saved coefficient file: {COEF_FILE}")
    print("Done.")
    print_line()


if __name__ == "__main__":
    main()
