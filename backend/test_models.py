#!/usr/bin/env python3
"""
Quick model smoke-test runner for ModelPilot backend.

- Loads ../test_datasets/logisticData.csv
- Builds X/y (auto-detects target column)
- Runs your existing model_control / serialize_artifact (and optionally training_models if you import it)
- Prints a sorted leaderboard
"""

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _repo_relative_path(rel_path):
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(base, rel_path))


def setup_django_if_needed():
    try:
        import django
        django.setup()
    except Exception:
        pass


def load_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    headers = df.columns.tolist()
    dataset = df.to_numpy()
    return headers, dataset


def build_model_plans(include_clustering: bool, include_pca) -> List[Dict[str, Any]]:
    """
    Builds model_plans compatible with execute_training_cycle().
    The hyperparameters can be scalar or lists (ParameterGrid is used inside execute_training_cycle).
    """
    plans: List[Dict[str, Any]] = []

    # Supervised models
    plans.append({
        "model": "naive_bayes",
        "hyperparameters": {},
        "reasoning": "smoke test"
    })
    plans.append({
        "model": "decision_tree",
        "hyperparameters": {
            "max_depth": [3, 5, None],
            "min_samples_split": [2, 5, 10],
        },
        "reasoning": "smoke test"
    })
    plans.append({
        "model": "knn",
        "hyperparameters": {
            "n_neighbors": [3, 5, 9],
            "weights": ["uniform", "distance"],
        },
        "reasoning": "smoke test"
    })

    plans.append({
        "model": "linear_regression",
        "hyperparameters": {
            "loss": ["l2", "l1", "huber"],
            "optimizer": ["sgd", "adam"],
            "learning_rate": [0.001],
            "epochs": [2000],
            "batch_size": [64],
            "regularization": ["l1", "l2"],
            "alpha": [0.3],
        },
        "reasoning": "smoke test"
    })

    plans.append({
        "model": "kernel_polynomial",
        "hyperparameters": {
            "degree": [2, 3],
            "lam": [1e-3, 1],
        },
        "reasoning": "smoke test"
    })

    plans.append({
        "model": "linear_classifier",
        "hyperparameters": {
            "loss": ["hinge", "logistic"],
            "optimizer": ["sgd", "adam"],
            "learning_rate": [1e-3, 1e-2],
            "epochs": [300, 800],
            "batch_size": [280, 64],
            "regularization": ["none", "l2"],
            "alpha": [0.0, 1e-2],
        },
        "reasoning": "Smoke test linear classifier across hinge vs logistic with basic optimizer/reg sweeps."
    })

    if include_pca:
        plans.append({
            "model": "pca",
            "hyperparameters": {
                "n_components": [2, 0.90, 0.95],
                "svd_solver": ["auto"],
                "whiten": [False],
                "random_state": [42],
            },
            "reasoning": "smoke test PCA (dimension reduction) across fixed k vs variance-retention"
        })

    if include_clustering:
        plans.append({
            "model": "kmeans",
            "hyperparameters": {"n_clusters": [2, 3, 4]},
            "reasoning": "smoke test"
        })
        plans.append({
            "model": "dbscan",
            "hyperparameters": {"eps": [0.3, 0.5], "min_samples": [5, 10]},
            "reasoning": "smoke test"
        })
        plans.append({
            "model": "hierarchical",
            "hyperparameters": {"n_clusters": [2, 3, 4], "linkage": ["ward", "average"]},
            "reasoning": "smoke test"
        })

    return plans


def print_leaderboard(results, top_k):
    ok = [r for r in results if not r.get("error")]
    bad = [r for r in results if r.get("error")]

    def sort_key(r):
        m = r.get("metrics", {})
        a = r.get("artifact", {}) or {}
        if "val_score" in m:
            return m["val_score"]

        if "variance_explained" in m and m["variance_explained"] is not None:
            return m["variance_explained"]
        if "variance_explained" in a and a["variance_explained"] is not None:
            return a["variance_explained"]

        if "train_silhouette" in m and m["train_silhouette"] is not None:
            return m["train_silhouette"]

        return m.get("train_silhouette", -1e30)

    ok_sorted = sorted(ok, key=sort_key, reverse=True)

    print("\n=== Leaderboard (sorted by val_score / silhouette) ===")
    for i, r in enumerate(ok_sorted[:top_k], start=1):
        model = r.get("model", "?")
        params = r.get("hyperparameters", {})
        m = r.get("metrics", {})
        metric_name = m.get("primary_metric_name")

        if "val_metric" in m:
            print(
                f"{i:2d}. {model:16s} {metric_name} val={m.get('val_metric')} test={m.get('test_metric')} params={params}")
            continue
        var_exp = m.get("variance_explained", a.get(
            "variance_explained", None))
        k_comp = m.get("k_components", a.get("k_components", None))
        if var_exp is not None:
            print(
                f"{i:2d}. {model:16s} variance_explained={var_exp:.4f} k_components={k_comp} params={params}")
            continue

        if "train_silhouette" in m:
            print(
                f"{i:2d}. {model:16s} silhouette={m.get('train_silhouette')} params={params}")
            continue

    if bad:
        print("\n=== Errors ===")
        for r in bad[:top_k]:
            print(
                f"- {r.get('model', '?'):16s} params={r.get('hyperparameters', {})} error={r.get('error')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", default=_repo_relative_path("../test_datasets/basisData.csv"))
    parser.add_argument("--target", default=None)
    parser.add_argument("--problem-type", default="regression",
                        choices=["classification", "regression", "clustering", "dimension_reduction"])
    parser.add_argument(
        "--models", default="naive_bayes,decision_tree,knn,linear_regression,kmeans,dbscan,hierarchical")
    parser.add_argument("--include-clustering", action="store_true")
    parser.add_argument("--include-pca", action="store_true")
    args = parser.parse_args()

    setup_django_if_needed()

    from api.pipeline import execute_training_cycle, prepare_datasets

    headers, dataset = load_csv(args.csv)

    data_split = {
        "method": "train_val_test",
        "train_val_test": [0.7, 0.15, 0.15],
        "stratify": True,
        "grouping_column": None,
    }

    needs_target = args.problem_type in ("classification", "regression")

    X_train, y_train, X_val, y_val, X_test, y_test, classes = prepare_datasets(
        dataset=dataset,
        target_column=args.target if needs_target else None,
        data_split=data_split,
        problem_type=args.problem_type,
        headers=headers,
    )

    model_plans = build_model_plans(
        include_clustering=args.include_clustering or args.problem_type == "clustering",
        include_pca=args.include_pca or args.problem_type == "dimension_reduction",
    )

    import api.modelling as m
    if args.problem_type == "classification":
        model_plans = [p for p in model_plans if p["model"]
                       in (
            #    "naive_bayes",
            #    "decision_tree",
            #    "knn",
            "linear_classifier",)]
    elif args.problem_type == "regression":
        model_plans = [p for p in model_plans if p["model"]
                       in (
            "linear_regression",
            # "kernel_polynomial",
        )]
    elif args.problem_type == "clustering":
        model_plans = [p for p in model_plans if p["model"]
                       in (
            "kmeans",
            "dbscan",
            "hierarchical")]
    elif args.problem_type == "dimension_reduction":
        model_plans = [p for p in model_plans if p["model"] in ("pca",)]

    results, evaluated = execute_training_cycle(
        X_train, y_train, X_val, y_val, X_test, y_test, classes,
        model_plans, args.problem_type
    )
    print_leaderboard(results, top_k=10)


if __name__ == "__main__":
    main()
