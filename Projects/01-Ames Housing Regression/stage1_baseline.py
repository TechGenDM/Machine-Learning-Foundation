"""Stage 1 - Ames Housing baselines.

Strictly: load -> split -> EDA on train only -> 3 baselines with 5-fold CV.
X_test / y_test are created by the split and never used afterwards.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.datasets import fetch_openml
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
DATA_HOME = Path(__file__).parent / "data"
FEATURES = ["OverallQual", "GrLivArea"]


def load_and_split():
    X, y = fetch_openml(
        name="house_prices", as_frame=True, return_X_y=True, data_home=DATA_HOME
    )
    y = y.astype(float)
    # Split immediately. X_test / y_test are never referenced again.
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)


def eda_summary(X_train, y_train):
    print("=" * 70)
    print("EDA SUMMARY (train only)")
    print("=" * 70)
    print(f"X_train shape: {X_train.shape}")

    print("\n1) Target distribution (excess kurtosis, Fisher; normal = 0)")
    for label, series in [("y_train", y_train), ("log1p(y_train)", np.log1p(y_train))]:
        print(f"   {label:<16} skew = {series.skew():7.4f}   kurtosis = {series.kurt():8.4f}")

    print("\n2) Top 15 columns by missing value %")
    missing = (X_train.isna().mean() * 100).sort_values(ascending=False).head(15)
    print(missing.round(2).rename("missing_%").to_string())

    # SalePrice is the target, so it lives in y_train, not X_train.
    mask = (X_train["GrLivArea"] > 4000) & (y_train < 300_000)
    print(f"\n3) Houses with GrLivArea > 4000 sq ft AND SalePrice < 300000: {int(mask.sum())}")
    print(X_train.loc[mask, ["GrLivArea"]].assign(SalePrice=y_train[mask]).to_string())


def make_linear():
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("model", LinearRegression()),
        ]
    )


def evaluate_baselines(X_train, y_train):
    models = {
        "A: Dummy (median)": DummyRegressor(strategy="median"),
        "B: LinReg [OverallQual, GrLivArea]": make_linear(),
        "C: LinReg + log1p target": TransformedTargetRegressor(
            regressor=make_linear(), func=np.log1p, inverse_func=np.expm1
        ),
    }
    # B and C use only two columns; A ignores features, but gets the same subset
    # so all three see identical inputs.
    X_sub = X_train[FEATURES]
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"}

    print("\n" + "=" * 70)
    print("BASELINES: 5-fold CV on X_train (errors in dollars, mean +/- std)")
    print("=" * 70)
    print(f"{'Model':<38}{'RMSE':>20}{'MAE':>20}")
    for name, model in models.items():
        res = cross_validate(model, X_sub, y_train, cv=cv, scoring=scoring)
        rmse, mae = -res["test_rmse"], -res["test_mae"]
        print(
            f"{name:<38}"
            f"{f'${rmse.mean():,.0f} +/- {rmse.std():,.0f}':>20}"
            f"{f'${mae.mean():,.0f} +/- {mae.std():,.0f}':>20}"
        )


def main():
    pd.set_option("display.width", 120)
    X_train, X_test, y_train, y_test = load_and_split()
    print(f"Split: train={len(X_train)}  test={len(X_test)} (test held out, untouched)\n")
    eda_summary(X_train, y_train)
    evaluate_baselines(X_train, y_train)


if __name__ == "__main__":
    main()
