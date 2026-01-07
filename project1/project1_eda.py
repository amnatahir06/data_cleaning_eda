"""
Project 2 — Exploratory Data Analysis (EDA)

Input:
- titanic_cleaned.csv (cleaned dataset from Project 1)

Outputs:
- eda_outputs/ folder (plots + CSV reports)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def colname(df, target):
    """Find actual column name in df matching target ignoring case."""
    for c in df.columns:
        if c.lower() == target.lower():
            return c
    return None


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_plot(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print("✅ Saved plot:", path)


def main():
    # ---------------------------
    # 1) Load cleaned dataset
    # ---------------------------
    input_file = "titanic_cleaned.csv"
    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"❌ '{input_file}' not found. Make sure it is in the same folder as this script."
        )

    df = pd.read_csv(input_file)

    out_dir = "eda_outputs"
    ensure_dir(out_dir)

    print("✅ Loaded:", input_file)
    print("Shape (rows, cols):", df.shape)
    print("Columns:", df.columns.tolist())

    # ---------------------------
    # 2) Basic info + missing check
    # ---------------------------
    print("\n--- INFO ---")
    print(df.info())

    missing_counts = df.isna().sum().sort_values(ascending=False)
    missing_percent = (df.isna().mean() * 100).sort_values(ascending=False)

    print("\n--- Missing counts (>0) ---")
    print(missing_counts[missing_counts > 0])

    print("\n--- Missing percent (>0) ---")
    print(missing_percent[missing_percent > 0])

    # ---------------------------
    # 3) Numeric stats (mean, median, min, max)
    # ---------------------------
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) == 0:
        print("\n❌ No numeric columns found.")
    else:
        stats_table = pd.DataFrame({
            "mean": df[num_cols].mean(),
            "median": df[num_cols].median(),
            "min": df[num_cols].min(),
            "max": df[num_cols].max(),
        }).sort_values(by="mean", ascending=False)

        print("\n--- Basic Statistics (Numeric) ---")
        print(stats_table)

        stats_path = os.path.join(out_dir, "basic_stats_numeric.csv")
        stats_table.to_csv(stats_path)
        print("✅ Saved stats:", stats_path)

    # ---------------------------
    # 4) Categorical summary
    # ---------------------------
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    print("\n--- Categorical Value Counts (Top 10 each) ---")
    for col in cat_cols:
        print(f"\n[{col}]")
        print(df[col].value_counts(dropna=False).head(10))

    # ---------------------------
    # 5) Outliers (IQR method)
    # ---------------------------
    outlier_report = {}
    for col in num_cols:
        series = df[col].dropna()
        if series.empty:
            outlier_report[col] = 0
            continue

        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = series[(series < lower) | (series > upper)]
        outlier_report[col] = int(outliers.shape[0])

    outlier_series = pd.Series(outlier_report).sort_values(ascending=False)
    print("\n--- Outlier Report (IQR counts) ---")
    print(outlier_series)

    outlier_path = os.path.join(out_dir, "outlier_report_iqr.csv")
    outlier_series.to_csv(outlier_path, header=["outlier_count"])
    print("✅ Saved outlier report:", outlier_path)

    # ---------------------------
    # 6) Correlation + Heatmap
    # ---------------------------
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        corr_path = os.path.join(out_dir, "correlation_matrix.csv")
        corr.to_csv(corr_path)
        print("\n✅ Saved correlation matrix:", corr_path)

        plt.figure(figsize=(10, 6))
        sns.heatmap(corr, annot=True, fmt=".2f")
        plt.title("Correlation Heatmap")
        save_plot(os.path.join(out_dir, "correlation_heatmap.png"))
    else:
        print("\n⚠️ Not enough numeric columns for correlation.")

    # ---------------------------
    # 7) Histograms (numeric distributions)
    # ---------------------------
    if len(num_cols) > 0:
        df[num_cols].hist(figsize=(12, 8), bins=20)
        plt.suptitle("Histograms of Numeric Features")
        save_plot(os.path.join(out_dir, "histograms_numeric.png"))

    # ---------------------------
    # 8) Bar chart (example): Sex counts
    # (handles Sex vs sex automatically)
    # ---------------------------
    sex_col = colname(df, "sex")
    if sex_col is not None:
        counts = df[sex_col].value_counts(dropna=False)
        plt.figure(figsize=(6, 4))
        plt.bar(counts.index.astype(str), counts.values)
        plt.title("Count by Sex")
        plt.xlabel("Sex")
        plt.ylabel("Count")
        save_plot(os.path.join(out_dir, "bar_sex_counts.png"))
    else:
        print("\n⚠️ 'sex' column not found. Skipping sex bar chart.")

    # ---------------------------
    # 9) Bar chart: Survival rate by Sex (if survived exists)
    # ---------------------------
    surv_col = colname(df, "survived")
    if sex_col is not None and surv_col is not None:
        df[surv_col] = pd.to_numeric(df[surv_col], errors="coerce")
        survival_by_sex = df.groupby(sex_col)[surv_col].mean().sort_values(ascending=False)

        plt.figure(figsize=(6, 4))
        plt.bar(survival_by_sex.index.astype(str), survival_by_sex.values)
        plt.title("Survival Rate by Sex")
        plt.xlabel("Sex")
        plt.ylabel("Survival Rate (0 to 1)")
        save_plot(os.path.join(out_dir, "bar_survival_rate_by_sex.png"))

        survival_by_sex.to_csv(os.path.join(out_dir, "survival_rate_by_sex.csv"))
        print("✅ Saved survival rate by sex CSV.")
    else:
        print("\n⚠️ Need both 'sex' and 'survived' for survival-by-sex plot. Skipping.")

    # ---------------------------
    # 10) Line plot: Survival rate across Age bins
    # ---------------------------
    age_col = colname(df, "age")
    if age_col is not None and surv_col is not None:
        temp = df[[age_col, surv_col]].copy()
        temp[age_col] = pd.to_numeric(temp[age_col], errors="coerce")
        temp[surv_col] = pd.to_numeric(temp[surv_col], errors="coerce")
        temp = temp.dropna()

        if temp.empty:
            print("\n⚠️ No valid rows for age/survived line plot after dropping NaNs.")
        else:
            temp["age_bin"] = pd.cut(temp[age_col], bins=10)
            survival_by_agebin = temp.groupby("age_bin")[surv_col].mean()

            plt.figure(figsize=(10, 4))
            plt.plot(range(len(survival_by_agebin)), survival_by_agebin.values, marker="o")
            plt.title("Survival Rate across Age Bins")
            plt.xlabel("Age Bin (0..9)")
            plt.ylabel("Survival Rate")
            plt.grid(True)
            save_plot(os.path.join(out_dir, "line_survival_rate_by_age_bins.png"))

            survival_by_agebin.to_csv(os.path.join(out_dir, "survival_rate_by_age_bins.csv"))
            print("✅ Saved survival rate by age bins CSV.")
    else:
        print("\n⚠️ Need both 'age' and 'survived' for line plot. Skipping.")

    # ---------------------------
    # DONE
    # ---------------------------
    print("\n✅ EDA completed. Check the 'eda_outputs' folder for plots and reports.")


if __name__ == "__main__":
    main()
