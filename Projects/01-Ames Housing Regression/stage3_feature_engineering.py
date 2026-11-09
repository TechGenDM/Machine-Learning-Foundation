"""Stage 3 - Ames Housing: feature engineering + skew handling.

Builds on Stage 2: same 80/20 split and 5-fold shuffled KFold (random_state=42).
  * Drops the 2 outlier rows from X_train / y_train ONLY.
  * Drops "Id" (a row label, not a feature) from X_train and X_test immediately.
  * E : Stage 2 Model E (Ridge(alpha=10), log1p target) without Id  -> clean benchmark.
  * F1: E + engineered features.
  * F2: F1 + median-impute -> clip(0) -> log1p -> scale on skewed area columns.

FeatureEngineer is stateless (no statistics are learned), so it is leakage-free by
construction; it still sits first in the Pipeline so every CV fold re-derives the
features from its own raw rows. X_test / y_test are never used.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline

from stage1_baseline import RANDOM_STATE, load_and_split
from stage2_preprocessing import log_target, make_preprocessor, remove_corrupted_rows

ENGINEERED_COLS = [
    "TotalSF", "TotalBath", "TotalPorchSF", "HouseAge", "YearsSinceRemod",
    "IsRemodeled", "HasPool", "Has2ndFloor", "HasGarage", "Qual_x_TotalSF",
]
LOG_COLS = [
    "LotArea", "LotFrontage", "MasVnrArea", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF",
    "GrLivArea", "GarageArea", "TotalSF", "TotalPorchSF", "Qual_x_TotalSF",
]
RIDGE_ALPHA = 10.0


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Append engineered columns to a DataFrame. Stateless: fit learns nothing.

    NaNs in the source columns are treated as 0 *only inside the sums/flags*
    (e.g. no basement -> 0 sq ft). Raw columns are left untouched so the downstream
    imputers still see their original NaNs.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        z = lambda col: X[col].fillna(0)  # noqa: E731 - NaN -> 0 for arithmetic

        total_sf = z("TotalBsmtSF") + z("1stFlrSF") + z("2ndFlrSF")
        X["TotalSF"] = total_sf
        X["TotalBath"] = (
            z("FullBath") + 0.5 * z("HalfBath") + z("BsmtFullBath") + 0.5 * z("BsmtHalfBath")
        )
        X["TotalPorchSF"] = (
            z("OpenPorchSF") + z("EnclosedPorch") + z("3SsnPorch")
            + z("ScreenPorch") + z("WoodDeckSF")
        )
        X["HouseAge"] = X["YrSold"] - X["YearBuilt"]
        X["YearsSinceRemod"] = X["YrSold"] - X["YearRemodAdd"]
        X["IsRemodeled"] = (X["YearRemodAdd"] != X["YearBuilt"]).astype(int)
        X["HasPool"] = (z("PoolArea") > 0).astype(int)
        X["Has2ndFloor"] = (z("2ndFlrSF") > 0).astype(int)
        X["HasGarage"] = (z("GarageArea") > 0).astype(int)
        X["Qual_x_TotalSF"] = z("OverallQual") * total_sf
        return X


def make_model(X_train, engineer=False, log_cols=()):
    """Full pipeline: [FeatureEngineer ->] preprocessor -> Ridge, on a log1p target.

    Column groups come from the schema of the (stateless) engineered frame, so the
    new numeric columns are routed to the numeric branch - no data statistics used.
    """
    steps = []
    schema_frame = X_train
    if engineer:
        steps.append(("engineer", FeatureEngineer()))
        schema_frame = FeatureEngineer().transform(X_train.head())
    steps += [
        ("prep", make_preprocessor(schema_frame, log_cols=log_cols)),
        ("model", Ridge(alpha=RIDGE_ALPHA)),
    ]
    return log_target(Pipeline(steps))


def sanity_check_features(X_train):
    """Eyeball the engineered columns on raw data (train only)."""
    out = FeatureEngineer().transform(X_train)
    print("\nEngineered feature sanity check (train only):")
    print(f"  NaNs in engineered columns: {int(out[ENGINEERED_COLS].isna().sum().sum())}")
    print(out[ENGINEERED_COLS].describe().loc[["min", "mean", "max"]].round(2).T.to_string())
    row = X_train.iloc[0]
    manual = row["TotalBsmtSF"] + row["1stFlrSF"] + row["2ndFlrSF"]
    assert out["TotalSF"].iloc[0] == manual, "TotalSF mismatch"
    assert (out["HouseAge"] == X_train["YrSold"] - X_train["YearBuilt"]).all()
    assert set(out["HasPool"].unique()) <= {0, 1}
    missing_cols = [c for c in LOG_COLS if c not in out.columns]
    assert not missing_cols, missing_cols


def main():
    pd.set_option("display.width", 120)
    X_train, X_test, y_train, y_test = load_and_split()  # identical to Stages 1-2
    X_train, y_train = remove_corrupted_rows(X_train, y_train)
    X_train = X_train.drop(columns="Id")
    X_test = X_test.drop(columns="Id")  # kept consistent; X_test is never scored
    print(
        f"Train={len(X_train)} x {X_train.shape[1]} cols  "
        f"Test={len(X_test)} x {X_test.shape[1]} cols (held out, untouched)"
    )
    sanity_check_features(X_train)

    models = {
        "E : Stage 2 Ridge, no Id": make_model(X_train),
        "F1: E + engineered features": make_model(X_train, engineer=True),
        "F2: F1 + log1p on skewed cols": make_model(X_train, engineer=True, log_cols=LOG_COLS),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"}
    print("\n" + "=" * 78)
    print(f"5-fold CV on cleaned X_train, Ridge(alpha={RIDGE_ALPHA:g}), log1p target ($ mean +/- std)")
    print("=" * 78)
    print(f"{'Model':<34}{'RMSE':>22}{'MAE':>22}")
    per_fold = {}
    for name, model in models.items():
        res = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
        rmse, mae = -res["test_rmse"], -res["test_mae"]
        per_fold[name] = rmse
        print(
            f"{name:<34}"
            f"{f'${rmse.mean():,.0f} +/- {rmse.std():,.0f}':>22}"
            f"{f'${mae.mean():,.0f} +/- {mae.std():,.0f}':>22}"
        )

    base = per_fold["E : Stage 2 Ridge, no Id"]
    print("\nPer-fold RMSE ($), same folds:")
    for name, rmse in per_fold.items():
        print(f"  {name:<34}" + "  ".join(f"{v:>8,.0f}" for v in rmse))
    for name, rmse in list(per_fold.items())[1:]:
        print(f"  {name.split(':')[0]} beats E in {int((rmse < base).sum())}/5 folds")


if __name__ == "__main__":
    main()
