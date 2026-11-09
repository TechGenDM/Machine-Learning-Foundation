"""Stage 5 - Ames Housing: error analysis + final holdout evaluation (Step 9).

Same data handling as Stages 1-4: 80/20 split (random_state=42), the 2 outlier rows
dropped from X_train / y_train ONLY (X_test keeps all 292 rows), "Id" dropped,
5-fold shuffled KFold (random_state=42).

  1. Out-of-fold (OOF) dollar predictions on cleaned X_train for Baseline B,
     Tuned Lasso and Tuned ElasticNet.
  2. Error analysis on the Tuned ElasticNet OOF predictions (price quartiles,
     SaleCondition, worst 10).
  3. 2x2 diagnostic figure: Baseline B vs Tuned ElasticNet (residuals vs predicted,
     Q-Q of standardised residuals).
  4. Final holdout: every milestone model is fit on cleaned X_train and scored
     ONCE on X_test. No model or hyperparameter is changed after seeing these numbers.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline

from stage1_baseline import FEATURES, RANDOM_STATE, load_and_split, make_linear
from stage2_preprocessing import log_target, make_preprocessor, remove_corrupted_rows
from stage4_tuning import F3_LOG_COLS, make_pipeline

HERE = Path(__file__).parent
FIGURE_PATH = HERE / "diagnostics_baseline_vs_final.png"
# Exact grid values selected in Stage 4 (np.logspace(-5,-1,25)[10], np.logspace(-4,0,15)[3]).
LASSO_ALPHA = 10 ** (-10 / 3)    # 4.6416e-4
ENET_ALPHA = 10 ** (-22 / 7)     # 7.1969e-4
ENET_L1_RATIO = 0.5
SCORING = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"}
QUARTILES = ["Q1 Cheap", "Q2 Lower-Mid", "Q3 Upper-Mid", "Q4 Luxury"]


# --------------------------------------------------------------------------- data
def load_data():
    X_train, X_test, y_train, y_test = load_and_split()
    X_train, y_train = remove_corrupted_rows(X_train, y_train)  # train only
    X_train, X_test = X_train.drop(columns="Id"), X_test.drop(columns="Id")
    assert len(X_test) == 292, "X_test must keep all 292 rows"
    print(f"Train={len(X_train)} x {X_train.shape[1]} cols   Test={len(X_test)} x {X_test.shape[1]} cols (untouched)")
    return X_train, X_test, y_train, y_test


def plain_model(X_train, model):
    """Stage 2 style (no feature engineering): preprocessor -> model, log1p target."""
    return log_target(Pipeline([("prep", make_preprocessor(X_train)), ("model", model)]))


def milestone_models(X_train):
    """name -> (estimator, column subset or None for all columns)."""
    return {
        "Baseline A (Dummy median)": (DummyRegressor(strategy="median"), FEATURES),
        "Baseline B (LinReg 2 feat)": (make_linear(), FEATURES),
        "Baseline C (B + log1p y)": (log_target(make_linear()), FEATURES),
        "Model D (CT + LinReg, log1p)": (plain_model(X_train, LinearRegression()), None),
        "Model E (CT + Ridge a=10)": (plain_model(X_train, Ridge(alpha=10.0)), None),
        "Model F3 (FE+4 log1p, Ridge)": (make_pipeline(X_train, Ridge(alpha=10.0), F3_LOG_COLS), None),
        "Tuned Lasso": (make_pipeline(X_train, Lasso(alpha=LASSO_ALPHA, max_iter=10_000), F3_LOG_COLS), None),
        "Tuned ElasticNet": (
            make_pipeline(
                X_train,
                ElasticNet(alpha=ENET_ALPHA, l1_ratio=ENET_L1_RATIO, max_iter=10_000),
                F3_LOG_COLS,
            ),
            None,
        ),
    }


# --------------------------------------------------------------------------- metrics
def mape(y, p):
    return float(np.mean(np.abs(y - p) / y) * 100)


def slice_metrics(y, p):
    return pd.Series(
        {
            "count": len(y),
            "mean_actual": y.mean(),
            "RMSE": root_mean_squared_error(y, p),
            "MAE": mean_absolute_error(y, p),
            "MAPE%": mape(y, p),
        }
    )


def print_table(df, title, formats):
    print(f"\n{title}")
    print("-" * 78)
    out = df.copy()
    for col, f in formats.items():
        out[col] = out[col].map(f.format)
    print(out.to_string())


# --------------------------------------------------------------------------- analysis
def error_analysis(X_train, y_train, oof):
    df = X_train.assign(Actual=y_train, Predicted=oof)
    df["Residual"] = df["Actual"] - df["Predicted"]
    df["AbsError"] = df["Residual"].abs()

    print("\n" + "=" * 78)
    print("ERROR ANALYSIS REPORT - Tuned ElasticNet, out-of-fold predictions (5-fold)")
    print("=" * 78)
    overall = slice_metrics(df["Actual"], df["Predicted"])
    print(f"Overall: RMSE ${overall['RMSE']:,.0f}  MAE ${overall['MAE']:,.0f}  MAPE {overall['MAPE%']:.2f}%")

    # NOTE: quartiles of the *actual* price - a diagnostic slice, never a model input.
    df["Quartile"] = pd.qcut(df["Actual"], 4, labels=QUARTILES)
    q = df.groupby("Quartile", observed=True).apply(
        lambda g: slice_metrics(g["Actual"], g["Predicted"]), include_groups=False
    )
    print_table(
        q, "Subgroup slice: SalePrice quartile",
        {"count": "{:.0f}", "mean_actual": "${:,.0f}", "RMSE": "${:,.0f}", "MAE": "${:,.0f}", "MAPE%": "{:.2f}%"},
    )

    sc = (
        df.groupby("SaleCondition", observed=True)
        .apply(lambda g: slice_metrics(g["Actual"], g["Predicted"]), include_groups=False)
        .drop(columns="mean_actual")
        .sort_values("count", ascending=False)
    )
    print_table(
        sc, "Slice: SaleCondition",
        {"count": "{:.0f}", "RMSE": "${:,.0f}", "MAE": "${:,.0f}", "MAPE%": "{:.2f}%"},
    )

    worst = df.nlargest(10, "AbsError")[
        ["Actual", "Predicted", "Residual", "AbsError", "OverallQual", "GrLivArea", "Neighborhood", "SaleCondition"]
    ]
    print_table(
        worst, "Top 10 worst predictions (OOF)",
        {"Actual": "${:,.0f}", "Predicted": "${:,.0f}", "Residual": "${:+,.0f}", "AbsError": "${:,.0f}"},
    )
    share = worst["AbsError"].pow(2).sum() / df["AbsError"].pow(2).sum()
    print(f"\nThese 10 rows ({10 / len(df):.1%} of train) contribute {share:.1%} of total squared error.")
    return df


def residual_diagnostics(name, y, p):
    r = y - p
    z = (r - r.mean()) / r.std(ddof=1)
    rho = stats.spearmanr(p, np.abs(r)).statistic
    print(
        f"  {name:<18} mean resid ${r.mean():>8,.0f}  std ${r.std():>8,.0f}  "
        f"skew {stats.skew(r):6.2f}  excess kurt {stats.kurtosis(r):7.2f}  "
        f"Spearman(pred,|resid|) {rho:5.2f}"
    )
    return r, z


def make_figure(y, oof_b, oof_f):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    rows = [("Baseline B (2-feature LinReg)", oof_b), ("Tuned ElasticNet (F3 pipeline)", oof_f)]
    print("\nResidual diagnostics (OOF, residual = actual - predicted):")
    for (title, pred), (ax_res, ax_qq) in zip(rows, axes):
        r, z = residual_diagnostics(title.split(" (")[0], y, pred)
        ax_res.scatter(pred, r, s=12, alpha=0.5)
        ax_res.axhline(0, color="crimson", lw=1)
        ax_res.set_title(f"{title}\nOOF residuals vs predicted")
        ax_res.set_xlabel("Predicted price ($)")
        ax_res.set_ylabel("Residual ($)")
        ax_res.ticklabel_format(style="plain")
        stats.probplot(z, dist="norm", plot=ax_qq)
        ax_qq.set_title(f"{title}\nNormal Q-Q (standardised residuals)")
        ax_qq.get_lines()[0].set_markersize(3)
        ax_qq.get_lines()[0].set_alpha(0.6)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=130)
    plt.close(fig)
    print(f"\nSaved figure -> {FIGURE_PATH.name}")


def final_holdout(X_train, X_test, y_train, y_test, cv):
    print("\n" + "=" * 100)
    print("FINAL HOLDOUT EVALUATION - fit on cleaned X_train (1166), scored once on X_test (292)")
    print("=" * 100)
    print(f"{'Model':<32}{'CV RMSE':>20}{'Test RMSE':>12}{'CV MAE':>20}{'Test MAE':>11}")
    rows = []
    for name, (model, cols) in milestone_models(X_train).items():
        Xtr = X_train if cols is None else X_train[cols]
        Xte = X_test if cols is None else X_test[cols]
        res = cross_validate(model, Xtr, y_train, cv=cv, scoring=SCORING)
        rmse, mae = -res["test_rmse"], -res["test_mae"]
        pred = model.fit(Xtr, y_train).predict(Xte)  # single test evaluation
        t_rmse, t_mae = root_mean_squared_error(y_test, pred), mean_absolute_error(y_test, pred)
        rows.append((name, rmse.mean(), rmse.std(), t_rmse, mae.mean(), mae.std(), t_mae))
        print(
            f"{name:<32}{f'${rmse.mean():,.0f} +/- {rmse.std():,.0f}':>20}{f'${t_rmse:,.0f}':>12}"
            f"{f'${mae.mean():,.0f} +/- {mae.std():,.0f}':>20}{f'${t_mae:,.0f}':>11}"
        )
    return rows


def main():
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)
    X_train, X_test, y_train, y_test = load_data()
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    models = milestone_models(X_train)
    base_model, base_cols = models["Baseline B (LinReg 2 feat)"]
    oof_b = pd.Series(cross_val_predict(base_model, X_train[base_cols], y_train, cv=cv), index=y_train.index)
    oof_l = pd.Series(cross_val_predict(models["Tuned Lasso"][0], X_train, y_train, cv=cv), index=y_train.index)
    oof_e = pd.Series(cross_val_predict(models["Tuned ElasticNet"][0], X_train, y_train, cv=cv), index=y_train.index)
    print("\nOOF RMSE / MAE ($): " + "   ".join(
        f"{n}: {root_mean_squared_error(y_train, p):,.0f} / {mean_absolute_error(y_train, p):,.0f}"
        for n, p in [("Baseline B", oof_b), ("Lasso", oof_l), ("ElasticNet", oof_e)]
    ))

    error_analysis(X_train, y_train, oof_e)
    make_figure(y_train, oof_b, oof_e)
    final_holdout(X_train, X_test, y_train, y_test, cv)


if __name__ == "__main__":
    main()
