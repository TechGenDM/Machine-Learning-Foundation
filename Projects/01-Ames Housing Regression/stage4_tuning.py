"""Stage 4 - Ames Housing: hyperparameter tuning (Step 8).

Same data handling as Stages 1-3: 80/20 split (random_state=42), the 2 outlier rows
dropped from train only, "Id" dropped, 5-fold shuffled KFold (random_state=42).

  1. F3 = F1 + log1p on 4 strictly positive skewed columns only; compare with F1.
  2. Take the better of F1 / F3 (by CV RMSE) as the feature pipeline.
  3. GridSearchCV (dollar-space neg-RMSE, refit on RMSE, MAE tracked too) for
     Ridge, Lasso and ElasticNet, each wrapped in a log1p TransformedTargetRegressor.
  4. Report best params, CV RMSE/MAE, and for the sparse models how many
     post-preprocessing features were zeroed out + top-10 weights.

X_test / y_test are never used. Caveat: the tuned scores are the grid's best CV
scores on the same folds used to pick them, so they are slightly optimistic.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
from sklearn.pipeline import Pipeline

from stage1_baseline import RANDOM_STATE, load_and_split
from stage2_preprocessing import log_target, make_preprocessor, remove_corrupted_rows
from stage3_feature_engineering import FeatureEngineer

F3_LOG_COLS = ["LotArea", "GrLivArea", "1stFlrSF", "TotalSF"]
SCORING = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"}


def make_pipeline(X_train, model, log_cols=()):
    """FeatureEngineer -> preprocessor -> model, on a log1p target."""
    schema_frame = FeatureEngineer().transform(X_train.head())  # stateless, schema only
    return log_target(
        Pipeline(
            [
                ("engineer", FeatureEngineer()),
                ("prep", make_preprocessor(schema_frame, log_cols=log_cols)),
                ("model", model),
            ]
        )
    )


def fmt(arr):
    return f"${arr.mean():,.0f} +/- {arr.std():,.0f}"


def run_grid(name, estimator, param_grid, X, y, cv):
    grid = GridSearchCV(
        estimator, param_grid, scoring=SCORING, refit="rmse", cv=cv, n_jobs=-1
    )
    grid.fit(X, y)
    i = grid.best_index_
    r = grid.cv_results_
    res = {
        "name": name,
        "grid": grid,
        "params": {k.split("__")[-1]: v for k, v in grid.best_params_.items()},
        "rmse_mean": -r["mean_test_rmse"][i], "rmse_std": r["std_test_rmse"][i],
        "mae_mean": -r["mean_test_mae"][i], "mae_std": r["std_test_mae"][i],
    }
    # Warn if the optimum sits on the edge of the searched range.
    for key, values in param_grid.items():
        v = grid.best_params_[key]
        if len(values) > 1 and (v == min(values) or v == max(values)):
            print(f"  [!] {name}: best {key.split('__')[-1]}={v:.4g} is at the edge of its grid")
    return res


def sparsity_report(res):
    """Zeroed vs total features after preprocessing, plus top-10 |weights|."""
    fitted = res["grid"].best_estimator_.regressor_  # Pipeline fitted on all of X_train
    names = fitted.named_steps["prep"].named_steps["columns"].get_feature_names_out()
    coef = fitted.named_steps["model"].coef_
    zero = int((coef == 0).sum())
    print(f"\n{res['name']}: {len(coef)} features after preprocessing, "
          f"{zero} exactly 0 ({zero / len(coef):.0%}), {len(coef) - zero} kept")
    top = pd.Series(coef, index=names).sort_values(key=np.abs, ascending=False).head(10)
    print("  Top 10 by |weight| (standardised features, log1p-target scale):")
    for feat, w in top.items():
        print(f"    {feat:<40}{w:>10.4f}")


def main():
    pd.set_option("display.width", 120)
    X_train, X_test, y_train, y_test = load_and_split()
    X_train, y_train = remove_corrupted_rows(X_train, y_train)
    X_train, X_test = X_train.drop(columns="Id"), X_test.drop(columns="Id")
    print(f"Train={len(X_train)} x {X_train.shape[1]} cols (test held out, untouched)")

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # --- Step 1: F1 vs F3 at Ridge(alpha=10) -------------------------------
    candidates = {
        "F1": make_pipeline(X_train, Ridge(alpha=10.0)),
        "F3": make_pipeline(X_train, Ridge(alpha=10.0), log_cols=F3_LOG_COLS),
    }
    rows, scores = [], {}
    for name, model in candidates.items():
        res = cross_validate(model, X_train, y_train, cv=cv, scoring=SCORING)
        rmse, mae = -res["test_rmse"], -res["test_mae"]
        scores[name] = rmse.mean()
        rows.append((f"{name} (Ridge a=10)", "alpha=10", rmse, mae))
    best_name = min(scores, key=scores.get)
    log_cols = F3_LOG_COLS if best_name == "F3" else ()
    print(f"\nFeature pipeline chosen for tuning: {best_name} "
          f"(F1 ${scores['F1']:,.0f} vs F3 ${scores['F3']:,.0f} RMSE)")

    # --- Step 2: grid searches ---------------------------------------------
    searches = {
        "Tuned Ridge": (
            Ridge(),
            {"regressor__model__alpha": np.logspace(-1, 3, 25)},
        ),
        "Tuned Lasso": (
            Lasso(max_iter=10_000),
            {"regressor__model__alpha": np.logspace(-5, -1, 25)},
        ),
        "Tuned ElasticNet": (
            ElasticNet(max_iter=10_000),
            {
                "regressor__model__alpha": np.logspace(-4, 0, 15),
                "regressor__model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99],
            },
        ),
    }
    tuned = {}
    for name, (model, grid) in searches.items():
        print(f"Running GridSearchCV: {name} ...")
        tuned[name] = run_grid(name, make_pipeline(X_train, model, log_cols), grid, X_train, y_train, cv)

    # --- Step 3: comparison table ------------------------------------------
    print("\n" + "=" * 104)
    print(f"5-fold CV, cleaned X_train, log1p target ($ mean +/- std); tuned models on {best_name} features")
    print("=" * 104)
    print(f"{'Model':<22}{'Best hyperparameters':<34}{'RMSE':>24}{'MAE':>24}")
    for label, hp, rmse, mae in rows:
        print(f"{label:<22}{hp:<34}{fmt(rmse):>24}{fmt(mae):>24}")
    for name, t in tuned.items():
        hp = ", ".join(f"{k}={v:.5g}" for k, v in t["params"].items())
        print(
            f"{name:<22}{hp:<34}"
            f"{f'${t['rmse_mean']:,.0f} +/- {t['rmse_std']:,.0f}':>24}"
            f"{f'${t['mae_mean']:,.0f} +/- {t['mae_std']:,.0f}':>24}"
        )

    # --- Step 4: sparsity ---------------------------------------------------
    print("\n" + "=" * 104)
    print("Feature sparsity")
    print("=" * 104)
    for name in ("Tuned Lasso", "Tuned ElasticNet"):
        sparsity_report(tuned[name])


if __name__ == "__main__":
    main()
