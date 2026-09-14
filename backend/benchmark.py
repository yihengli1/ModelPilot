"""
benchmark.py

Compares ModelPilot's model-selection output against scikit-learn baselines
(both untuned defaults and grid-searched/tuned versions) on a set of
classification/regression datasets.

HOW TO USE
----------
1. Download the Kaggle datasets below and place the CSVs in a `data/` folder
   next to this script:

     data/titanic.csv           (Kaggle: c/titanic  -> use train.csv)
     data/heart_failure.csv     (Kaggle: fedesoriano/heart-failure-prediction)
     data/wine_quality.csv      (Kaggle: yasserh/wine-quality-dataset)
     data/house_prices.csv      (Kaggle: c/house-prices-advanced-regression-techniques -> train.csv)
     data/insurance.csv         (Kaggle: mirichoi0218/insurance)

2. Fill in DATASETS below with the target column for each file (already done
   for the datasets listed above -- adjust if your column names differ).

3. Run:
     python benchmark_modelpilot.py

4. For each dataset, the script:
   - Splits into train/test
   - Trains scikit-learn baselines (default hyperparameters, zero tuning)
   - Trains scikit-learn baselines with GridSearchCV (a fair "manually tuned"
     comparison point)
   - Prints accuracy (classification) or R^2 / RMSE (regression) for each
   - Leaves clearly marked slots for you to paste in ModelPilot's own results
     (whatever pipeline/hyperparameters it selected + its resulting score)
   - Prints a final summary table so you can compute:
       - avg accuracy delta vs. untuned baseline
       - avg accuracy delta vs. tuned baseline
       - iteration count comparison (manual grid search trials vs.
         ModelPilot's automated iterations to reach its final config)

This gives you real, defensible numbers instead of a vague "~10%" claim --
you'll know exactly which baseline it's measured against.
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. DATASET CONFIG -- edit paths/target columns to match your downloaded CSVs
# ---------------------------------------------------------------------------
DATASETS = [
    {
        "name": "Titanic",
        "path": "data/titanic.csv",
        "target": "Survived",
        "task": "classification",
        "drop_cols": ["PassengerId", "Name", "Ticket", "Cabin"],
    },
    {
        "name": "Heart Failure",
        "path": "data/heart_failure.csv",
        "target": "HeartDisease",
        "task": "classification",
        "drop_cols": [],
    },
    {
        "name": "Wine Quality",
        "path": "data/wine_quality.csv",
        "target": "quality",
        "task": "classification",  # treat as multi-class; switch to "regression" if you prefer
        "drop_cols": [],
    },
    {
        "name": "House Prices",
        "path": "data/house_prices.csv",
        "target": "SalePrice",
        "task": "regression",
        "drop_cols": ["Id"],
    },
    {
        "name": "Insurance Cost",
        "path": "data/insurance.csv",
        "target": "charges",
        "task": "regression",
        "drop_cols": [],
    },
]

# ---------------------------------------------------------------------------
# 2. MODEL GRIDS -- one classifier set, one regressor set
#    (mirrors the "manual grid search" a person would run by hand)
# ---------------------------------------------------------------------------
CLASSIFIERS = {
    "DecisionTree": (
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        {"max_depth": [3, 5, 10, None], "criterion": ["gini", "entropy"]},
    ),
    "NaiveBayes": (
        GaussianNB(),
        {"var_smoothing": [1e-9, 1e-8, 1e-7]},
    ),
    "kNN": (
        KNeighborsClassifier(),
        {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]},
    ),
    "LogisticRegression": (
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        {"C": [0.01, 0.1, 1, 10]},
    ),
    "SVM": (
        SVC(random_state=RANDOM_STATE),
        {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
    ),
    "MLP": (
        MLPClassifier(max_iter=1000, random_state=RANDOM_STATE),
        {"hidden_layer_sizes": [(32,), (64, 32)], "alpha": [0.0001, 0.001]},
    ),
}

REGRESSORS = {
    "Ridge": (
        Ridge(random_state=RANDOM_STATE),
        {"alpha": [0.1, 1.0, 10.0, 100.0]},
    ),
    "Lasso": (
        Lasso(random_state=RANDOM_STATE, max_iter=5000),
        {"alpha": [0.01, 0.1, 1.0, 10.0]},
    ),
    "SVR": (
        SVR(),
        {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
    ),
    "DecisionTreeReg": (
        DecisionTreeRegressor(random_state=RANDOM_STATE),
        {"max_depth": [3, 5, 10, None]},
    ),
    "MLPReg": (
        MLPRegressor(max_iter=1000, random_state=RANDOM_STATE),
        {"hidden_layer_sizes": [(32,), (64, 32)], "alpha": [0.0001, 0.001]},
    ),
}


def load_and_clean(cfg):
    """Basic, generic cleaning: drop id-like cols, impute, encode categoricals."""
    df = pd.read_csv(cfg["path"])
    df = df.drop(columns=[c for c in cfg["drop_cols"] if c in df.columns])
    df = df.dropna(subset=[cfg["target"]])

    y = df[cfg["target"]]
    X = df.drop(columns=[cfg["target"]])

    # Encode categoricals
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = X[col].astype(str)
        X[col] = LabelEncoder().fit_transform(X[col])

    # Impute numeric NaNs
    num_cols = X.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        X[num_cols] = SimpleImputer(
            strategy="median").fit_transform(X[num_cols])

    if cfg["task"] == "classification" and y.dtype == object:
        y = LabelEncoder().fit_transform(y.astype(str))

    return X, y


def run_untuned_baselines(X_train, X_test, y_train, y_test, task):
    """Default hyperparameters, no tuning at all -- the 'zero effort' baseline."""
    models = CLASSIFIERS if task == "classification" else REGRESSORS
    results = {}
    for name, (model, _grid) in models.items():
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            if task == "classification":
                score = accuracy_score(y_test, preds)
            else:
                score = r2_score(y_test, preds)
            results[name] = score
        except Exception as e:
            results[name] = f"ERROR: {e}"
    return results


def run_tuned_baselines(X_train, X_test, y_train, y_test, task):
    """GridSearchCV over each model's grid -- the 'manually tuned' baseline.
    Also tracks total number of fit iterations (candidate configs x folds),
    which is your point of comparison for ModelPilot's iteration count."""
    models = CLASSIFIERS if task == "classification" else REGRESSORS
    scoring = "accuracy" if task == "classification" else "r2"
    results = {}
    total_iterations = 0
    best_name, best_score = None, -np.inf

    for name, (model, grid) in models.items():
        try:
            n_candidates = int(np.prod([len(v) for v in grid.values()]))
            cv_folds = 3
            total_iterations += n_candidates * cv_folds

            gs = GridSearchCV(model, grid, scoring=scoring,
                              cv=cv_folds, n_jobs=-1)
            gs.fit(X_train, y_train)
            preds = gs.best_estimator_.predict(X_test)
            score = (
                accuracy_score(y_test, preds)
                if task == "classification"
                else r2_score(y_test, preds)
            )
            results[name] = score
            if score > best_score:
                best_score, best_name = score, name
        except Exception as e:
            results[name] = f"ERROR: {e}"

    return results, total_iterations, best_name, best_score


def main():
    summary_rows = []

    for cfg in DATASETS:
        print(f"\n{'='*70}\n{cfg['name']}  ({cfg['task']})\n{'='*70}")
        try:
            X, y = load_and_clean(cfg)
        except FileNotFoundError:
            print(f"  [skipped] File not found: {cfg['path']}")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE,
            stratify=y if cfg["task"] == "classification" else None,
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # --- Untuned baseline ---
        untuned = run_untuned_baselines(
            X_train, X_test, y_train, y_test, cfg["task"])
        best_untuned_name = max(
            (k for k, v in untuned.items() if isinstance(v, float)),
            key=lambda k: untuned[k],
            default=None,
        )
        best_untuned_score = untuned.get(best_untuned_name, None)

        # --- Tuned (grid search) baseline ---
        start = time.time()
        tuned, grid_iterations, best_tuned_name, best_tuned_score = run_tuned_baselines(
            X_train, X_test, y_train, y_test, cfg["task"]
        )
        tuned_time = time.time() - start

        print(f"\n  Untuned (defaults) results:")
        for k, v in untuned.items():
            print(f"    {k:20s}: {v}")
        print(
            f"  --> best untuned: {best_untuned_name} = {best_untuned_score:.4f}")

        print(
            f"\n  Tuned (GridSearchCV) results  [{grid_iterations} total fits, {tuned_time:.1f}s]:")
        for k, v in tuned.items():
            print(f"    {k:20s}: {v}")
        print(f"  --> best tuned: {best_tuned_name} = {best_tuned_score:.4f}")

        # ------------------------------------------------------------------
        # PASTE MODELPILOT'S RESULT HERE for this dataset:
        #   - which model/hyperparams it picked
        #   - its test-set accuracy / R^2
        #   - number of automated iterations it took to converge
        # ------------------------------------------------------------------
        modelpilot_score = None          # e.g. 0.83
        modelpilot_iterations = None     # e.g. 6
        modelpilot_model_name = None     # e.g. "kNN (k=7, distance-weighted)"

        summary_rows.append({
            "dataset": cfg["name"],
            "task": cfg["task"],
            "best_untuned_model": best_untuned_name,
            "best_untuned_score": best_untuned_score,
            "best_tuned_model": best_tuned_name,
            "best_tuned_score": best_tuned_score,
            "grid_search_iterations": grid_iterations,
            "modelpilot_model": modelpilot_model_name,
            "modelpilot_score": modelpilot_score,
            "modelpilot_iterations": modelpilot_iterations,
        })

    # -----------------------------------------------------------------
    # Final summary + the deltas you'd actually cite on a resume
    # -----------------------------------------------------------------
    print(f"\n\n{'='*70}\nSUMMARY\n{'='*70}")
    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_string(index=False))

    valid = df_summary.dropna(subset=["modelpilot_score"])
    if len(valid) > 0:
        acc_vs_untuned = (
            (valid["modelpilot_score"] - valid["best_untuned_score"])
            / valid["best_untuned_score"] * 100
        ).mean()
        acc_vs_tuned = (
            (valid["modelpilot_score"] - valid["best_tuned_score"])
            / valid["best_tuned_score"] * 100
        ).mean()
        iter_reduction = (
            (valid["grid_search_iterations"] - valid["modelpilot_iterations"])
            / valid["grid_search_iterations"] * 100
        ).mean()

        print(
            f"\nAvg accuracy/R^2 change vs. UNTUNED baseline: {acc_vs_untuned:+.1f}%")
        print(
            f"Avg accuracy/R^2 change vs. TUNED (grid search) baseline: {acc_vs_tuned:+.1f}%")
        print(
            f"Avg iteration reduction vs. manual grid search: {iter_reduction:.1f}%")
    else:
        print(
            "\n[No ModelPilot results filled in yet. Fill in `modelpilot_score` "
            "and `modelpilot_iterations` in the loop above for each dataset, "
            "then re-run to get your summary deltas.]"
        )


if __name__ == "__main__":
    main()
