import numpy as np
import pandas as pd

RAW, CLEAN = "data/fertilizer.csv", "data/cleaned_dataset.csv"
FEATURES = ["Carbon", "Moisture", "Phosphorous", "Potassium", "Rainfall", "Temperature"]


def main():
    df = pd.read_csv(RAW)
    print(f"Raw shape: {df.shape}")
    print("Missing values:\n", df.isna().sum().to_string())
    print("Duplicate rows:", df.duplicated().sum())

    # 1. ID is only a row counter -> no predictive value
    df = df.drop(columns=["ID"])

    # 2. Remove exact duplicates (none expected, kept for safety)
    df = df.drop_duplicates()

    # 3. Negative nutrient / weather readings are physically impossible -> treat as invalid,
    #    replace with the column median (robust to the skewed Rainfall / Phosphorous columns)
    neg = (df[FEATURES] < 0).sum()
    print("\nNegative values found:\n", neg[neg > 0].to_string())
    df[FEATURES] = df[FEATURES].mask(df[FEATURES] < 0, np.nan)
    df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())

    # 4. Any other missing values -> median
    df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())

    # 5. Enforce types. (Feature scaling is done inside the model pipelines in
    #    model_comparison.py so no information leaks from the test set.)
    df["Crop"] = df["Crop"].astype(int)

    df.to_csv(CLEAN, index=False)
    print(f"\nCleaned shape: {df.shape} -> saved to {CLEAN}")
    print(df.describe().T.round(2).to_string())


if __name__ == "__main__":
    main()
