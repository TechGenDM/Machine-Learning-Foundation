"""Stage 2 - Ames Housing: missing data + preprocessing.

Builds on Stage 1: same 80/20 split and 5-fold shuffled KFold (random_state=42).
  * Drops the 2 corrupted outlier rows from X_train / y_train ONLY.
  * Re-runs Baselines B and C on the cleaned data.
  * Evaluates two full-feature models (LinearRegression, Ridge) behind a
    leakage-free ColumnTransformer, both on a log1p target.

X_test / y_test are created by the split and never used. Every fitted step
(imputers, encoders, scaler) lives inside the estimator, so each CV fold fits
them on its own training portion only.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from stage1_baseline import FEATURES, RANDOM_STATE, load_and_split, make_linear

QUALITY_LEVELS = ["None", "Po", "Fa", "TA", "Gd", "Ex"]
ORDINAL_COLS = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
    "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
]
NONE_FILL_NOMINAL_COLS = [
    "Alley", "Fence", "MiscFeature", "GarageType", "GarageFinish",
    "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "MasVnrType",
]


def cast_mssubclass_to_str(X):
    """MSSubClass is a building-type code, not a quantity (stateless, leak-free)."""
    X = X.copy()
    X["MSSubClass"] = X["MSSubClass"].astype(str)
    return X


def derive_column_groups(X):
    """Split columns by schema (dtype after the MSSubClass cast) - no data statistics."""
    X = cast_mssubclass_to_str(X)
    is_num = {c: pd.api.types.is_numeric_dtype(X[c]) for c in X.columns}
    categorical = [c for c in X.columns if not is_num[c]]
    numeric = [c for c in X.columns if is_num[c]]
    special = set(ORDINAL_COLS) | set(NONE_FILL_NOMINAL_COLS)
    other_nominal = [c for c in categorical if c not in special]
    return numeric, other_nominal


def remove_corrupted_rows(X_train, y_train):
    """Drop the high-leverage rows (GrLivArea > 4000 and SalePrice < 300000)."""
    mask = (X_train["GrLivArea"] > 4000) & (y_train < 300_000)
    print(f"Removing {int(mask.sum())} corrupted rows from train only:")
    print(X_train.loc[mask, ["GrLivArea"]].assign(SalePrice=y_train[mask]).to_string())
    return X_train.loc[~mask], y_train.loc[~mask]


def clip_log1p(X):
    """Clip at 0 then log1p (stateless). Used for skewed area/size columns."""
    return np.log1p(np.clip(X, 0, None))


def make_preprocessor(X_train, log_cols=()):
    """Leakage-free ColumnTransformer.

    log_cols: numeric columns that get median-impute -> clip(0) -> log1p -> scale
    instead of the plain median-impute -> scale. Empty (default) = Stage 2 behaviour.
    """
    numeric, other_nominal = derive_column_groups(X_train)
    missing = set(log_cols) - set(numeric)
    if missing:
        raise ValueError(f"log_cols not numeric columns of X_train: {sorted(missing)}")
    numeric = [c for c in numeric if c not in set(log_cols)]

    ordinal = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="None")),
            (
                "encode",
                OrdinalEncoder(
                    categories=[QUALITY_LEVELS] * len(ORDINAL_COLS),
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )
    nominal_none = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="None")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    nominal_mode = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    transformers = [
        ("ordinal", ordinal, ORDINAL_COLS),
        ("nominal_none", nominal_none, NONE_FILL_NOMINAL_COLS),
        ("nominal_mode", nominal_mode, other_nominal),
        ("numeric", numeric_pipe, numeric),
    ]
    if log_cols:
        numeric_log = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("log", FunctionTransformer(clip_log1p)),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(("numeric_log", numeric_log, list(log_cols)))
    columns = ColumnTransformer(transformers, remainder="drop")
    return Pipeline(
        [
            ("cast", FunctionTransformer(cast_mssubclass_to_str)),
            ("columns", columns),
        ]
    )


def log_target(regressor):
    return TransformedTargetRegressor(
        regressor=regressor, func=np.log1p, inverse_func=np.expm1
    )


def report(models, y_train, title):
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"}
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    print(f"{'Model':<42}{'RMSE':>18}{'MAE':>18}")
    for name, (model, X) in models.items():
        res = cross_validate(model, X, y_train, cv=cv, scoring=scoring)
        rmse, mae = -res["test_rmse"], -res["test_mae"]
        print(
            f"{name:<42}"
            f"{f'${rmse.mean():,.0f} +/- {rmse.std():,.0f}':>18}"
            f"{f'${mae.mean():,.0f} +/- {mae.std():,.0f}':>18}"
        )


def main():
    X_train, X_test, y_train, y_test = load_and_split()  # identical to Stage 1
    print(f"Split: train={len(X_train)}  test={len(X_test)} (test held out, untouched)\n")

    # Uncleaned train, for the before/after comparison (matches Stage 1).
    X_sub_before = X_train[FEATURES]
    before = {
        "B (uncleaned): LinReg [OvQual, GrLivArea]": (make_linear(), X_sub_before),
        "C (uncleaned): B + log1p target": (log_target(make_linear()), X_sub_before),
    }
    report(before, y_train, "Baselines B & C on UNCLEANED X_train (reproduces Stage 1)")


    X_train, y_train = remove_corrupted_rows(X_train, y_train)
    print(f"Cleaned train size: {len(X_train)}")

    numeric, other_nominal = derive_column_groups(X_train)
    print(
        f"Column groups: ordinal={len(ORDINAL_COLS)}  "
        f"nominal('None' fill)={len(NONE_FILL_NOMINAL_COLS)}  "
        f"nominal(most_frequent)={len(other_nominal)}  numeric={len(numeric)}  "
        f"-> total {len(ORDINAL_COLS) + len(NONE_FILL_NOMINAL_COLS) + len(other_nominal) + len(numeric)}"
    )

    X_sub = X_train[FEATURES]
    after = {
        "B (cleaned): LinReg [OvQual, GrLivArea]": (make_linear(), X_sub),
        "C (cleaned): B + log1p target": (log_target(make_linear()), X_sub),
    }
    full = {
        "D: full CT + LinearRegression (log1p)": (
            log_target(Pipeline([("prep", make_preprocessor(X_train)), ("model", LinearRegression())])),
            X_train,
        ),
        "E: full CT + Ridge(alpha=10) (log1p)": (
            log_target(Pipeline([("prep", make_preprocessor(X_train)), ("model", Ridge(alpha=10.0))])),
            X_train,
        ),
    }
    report(after, y_train, "Baselines B & C on CLEANED X_train (5-fold CV, $ mean +/- std)")
    report(full, y_train, "Full-feature models on CLEANED X_train (5-fold CV, $ mean +/- std)")


if __name__ == "__main__":
    main()
