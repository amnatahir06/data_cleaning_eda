"""
Project 1 — Data Collection & Cleaning (Titanic Dataset)

What this script does:
1) Loads Titanic dataset from URL
2) Shows basic info
3) Removes duplicate rows
4) Fixes formatting issues (strip spaces, normalize casing)
5) Handles missing values:
   - Drops columns with >60% missing values
   - Fills numeric missing values with median
   - Fills categorical missing values with mode
6) Saves cleaned dataset as titanic_cleaned.csv
"""

import pandas as pd
import numpy as np


def main():
    # ---------------------------
    # 1) Load Dataset
    # ---------------------------
    input_path = "data\titanic_raw.csv"
    df = pd.read_csv(input_path)

    print("✅ Dataset loaded successfully.")
    print("Initial shape (rows, cols):", df.shape)

    # ---------------------------
    # 2) Quick Preview
    # ---------------------------
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    # ---------------------------
    # 3) Missing Values Report (Before)
    # ---------------------------
    missing_counts = df.isna().sum().sort_values(ascending=False)
    missing_percent = (df.isna().mean() * 100).sort_values(ascending=False)

    print("\nMissing values (count):")
    print(missing_counts[missing_counts > 0])

    print("\nMissing values (%):")
    print(missing_percent[missing_percent > 0])

    # ---------------------------
    # 4) Remove Duplicates
    # ---------------------------
    dupes = df.duplicated().sum()
    print("\nDuplicate rows found:", dupes)

    df = df.drop_duplicates().reset_index(drop=True)
    print("Shape after removing duplicates:", df.shape)

    # ---------------------------
    # 5) Fix Formatting Issues (Text cleanup)
    # ---------------------------
    text_cols = df.select_dtypes(include=["object"]).columns

    for col in text_cols:
        df[col] = df[col].astype("string").str.strip()

    # Normalize common columns if they exist
    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].str.lower()

    if "Ticket" in df.columns:
        df["Ticket"] = df["Ticket"].str.upper()

    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].replace({"": pd.NA})

    # ---------------------------
    # 6) Handle Missing Values
    # ---------------------------
    missing_ratio = df.isna().mean() * 100

    # Drop columns with > 60% missing
    drop_cols = missing_ratio[missing_ratio > 60].index
    if len(drop_cols) > 0:
        print("\nDropping columns with >60% missing:", list(drop_cols))
        df = df.drop(columns=drop_cols)

    # Fill numeric missing with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isna().any():
            med = df[col].median()
            df[col] = df[col].fillna(med)

    # Fill categorical missing with mode
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in cat_cols:
        if df[col].isna().any():
            mode_series = df[col].mode(dropna=True)
            fill_value = mode_series.iloc[0] if len(mode_series) > 0 else "Unknown"
            df[col] = df[col].fillna(fill_value)

    # ---------------------------
    # 7) Missing Values Report (After)
    # ---------------------------
    missing_after = df.isna().sum().sort_values(ascending=False)
    print("\nMissing values after cleaning:")
    print(missing_after[missing_after > 0])

    print("\n✅ Final shape (rows, cols):", df.shape)

    # ---------------------------
    # 8) Save Cleaned Dataset
    # ---------------------------
    output_file = "titanic_cleaned.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✅ Cleaned dataset saved as: {output_file}")


if __name__ == "__main__":
    main()
