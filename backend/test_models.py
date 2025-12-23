#!/usr/bin/env python3
"""
Quick model smoke-test runner for ModelPilot backend.

- Loads ../test_datasets/logisticData.csv
- Builds X/y (auto-detects target column)
- Runs your existing model_control / serialize_artifact (and optionally training_models if you import it)
- Prints a sorted leaderboard
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, silhouette_score

from api.modelling import model_control, serialize_artifact
from api.pipeline import execute_training_cycle


@dataclass
class RunResult:
    model: str
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, Any]
    artifact: Dict[str, Any]
    error: Optional[str] = None


def _repo_relative_path(rel_path):
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(base, rel_path))


def load_csv_dataset(
    csv_path,
    target,
    problem_type,
):
    df = pd.read_csv(csv_path)

    # Auto-pick target if not provided
    if target is None:
        candidates = [c for c in df.columns if c.lower() in (
            "y", "label", "target", "class")]
        target = candidates[0] if candidates else df.columns[-1]

    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found in CSV columns: {list(df.columns)}")

    y_raw = df[target].to_numpy()
    X_df = df.drop(columns=[target])
    X_num = X_df.apply(pd.to_numeric, errors="coerce")
    # drop only if all na
    X_num = X_num.dropna(axis=1, how="all")

    mask = np.isfinite(X_num.to_numpy(dtype=float)).all(axis=1)

    if problem_type.lower() == "regression":
        y_float = pd.to_numeric(
            pd.Series(y_raw), errors="coerce").to_numpy(dtype=float)
        mask = mask & np.isfinite(y_float)
        y = y_float[mask]
    else:
        y_series = pd.Series(y_raw)
        mask = mask & y_series.notna().to_numpy()
        y_series = y_series[mask]
        classes, y = np.unique(y_series.to_numpy(), return_inverse=True)

    X = X_num.to_numpy(dtype=float)[mask]
    feature_names = list(X_num.columns)

    return X, y, feature_names


def split_data(
    X,
    y,
    train_ratio,
    val_ratio,
    test_ratio,
    problem_type,
):
    ratios = np.array([train_ratio, val_ratio, test_ratio], dtype=float)
    random_state = 42

    if y is None:
        # Train
        X_train, X_tmp = train_test_split(
            X, test_size=(1-ratios[0]), random_state=random_state)
        # Val, Test
        X_val, X_test = train_test_split(X_tmp, test_size=(
            ratios[2] / (ratios[1] + ratios[2])), random_state=random_state)
        return X_train, None, X_val, None, X_test, None

    # Supervised
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=(1.0 - ratios[0]),
        random_state=random_state,
        stratify=y if problem_type.lower() != "regression" else None,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=(ratios[2] / (ratios[1] + ratios[2])),
        random_state=random_state,
        stratify=y_tmp if problem_type.lower() != "regression" else None,
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def run_supervised_model(
    model_key: str,
    hyperparams: Dict[str, Any],
    problem_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> RunResult:
    try:
        # Ensure linear regression has a default loss if user forgets it
        if model_key == "linear_regression":
            hyperparams = dict(hyperparams)
            hyperparams.setdefault("loss", "l2")
            hyperparams.setdefault("learning_rate", 0.001)

        model, is_supervised = model_control(model_key, hyperparams)

        if not is_supervised:
            return RunResult(model_key, hyperparams, {}, {}, error="Requested supervised runner for unsupervised model")

        model.fit(X_train, y_train)

        metrics: Dict[str, Any] = {"supervised": True}

        if problem_type.lower() == "regression":
            # Use loss as negative score (bigger is better)
            val_pred = np.asarray(model.predict(
                X_val), dtype=float).reshape(-1)
            test_pred = np.asarray(model.predict(
                X_test), dtype=float).reshape(-1)
            yv = np.asarray(y_val, dtype=float).reshape(-1)
            yt = np.asarray(y_test, dtype=float).reshape(-1)

            # Guard against non-finite
            if np.any(~np.isfinite(val_pred)) or np.any(~np.isfinite(yv)):
                val_loss = 1e30
            else:
                diff = val_pred - yv
                mse = float(np.mean(diff ** 2))
                mae = float(np.mean(np.abs(diff)))
                loss_name = (getattr(model, "loss", "l2") or "l2").lower()
                if loss_name in ("l1", "mae", "absolute"):
                    val_loss = mae
                else:
                    val_loss = mse

            if np.any(~np.isfinite(test_pred)) or np.any(~np.isfinite(yt)):
                test_loss = 1e30
            else:
                diff = test_pred - yt
                mse = float(np.mean(diff ** 2))
                mae = float(np.mean(np.abs(diff)))
                loss_name = (getattr(model, "loss", "l2") or "l2").lower()
                if loss_name in ("l1", "mae", "absolute"):
                    test_loss = mae
                else:
                    test_loss = mse

            metrics["val_accuracy"] = -float(val_loss)
            metrics["test_accuracy"] = -float(test_loss)
        else:
            val_acc = float(accuracy_score(y_val, model.predict(X_val)))
            test_acc = float(accuracy_score(y_test, model.predict(X_test)))
            metrics["val_accuracy"] = val_acc
            metrics["test_accuracy"] = test_acc

        artifact = serialize_artifact(model, model_key, metrics)
        return RunResult(model_key, hyperparams, metrics, artifact)

    except Exception as exc:
        return RunResult(model_key, hyperparams, {}, {}, error=str(exc))


def run_clustering_model(
    model_key: str,
    hyperparams: Dict[str, Any],
    X_train: np.ndarray,
) -> RunResult:
    try:
        model, is_supervised = model_control(model_key, hyperparams)
        if is_supervised:
            return RunResult(model_key, hyperparams, {}, {}, error="Requested clustering runner for supervised model")

        if hasattr(model, "fit_predict"):
            labels = model.fit_predict(X_train)
        else:
            model.fit(X_train)
            labels = getattr(model, "labels_", None)
            if labels is None:
                labels = getattr(model, "labels", None)

        metrics: Dict[str, Any] = {"supervised": False}

        n_labels = len(set(labels)) if labels is not None else 0
        if labels is not None and 1 < n_labels < len(X_train):
            score = float(silhouette_score(X_train, labels))
        else:
            score = -1.0

        metrics["train_silhouette"] = score
        metrics["val_accuracy"] = score  # proxy sort key

        artifact = serialize_artifact(model, model_key, metrics)
        return RunResult(model_key, hyperparams, metrics, artifact)

    except Exception as exc:
        return RunResult(model_key, hyperparams, {}, {}, error=str(exc))


def print_leaderboard(results: List[RunResult], top_k: int = 10):
    ok = [r for r in results if not r.error]
    bad = [r for r in results if r.error]

    ok_sorted = sorted(ok, key=lambda r: r.metrics.get(
        "val_accuracy", -1e30), reverse=True)

    print("\n=== Leaderboard (sorted by val_accuracy) ===")
    for i, r in enumerate(ok_sorted[:top_k], start=1):
        va = r.metrics.get("val_accuracy")
        ta = r.metrics.get("test_accuracy", None)
        print(
            f"{i:2d}. {r.model:16s} val={va:.6f}  test={ta:.6f}  params={r.hyperparameters}")

    if bad:
        print("\n=== Errors ===")
        for r in bad:
            print(f"- {r.model:16s} params={r.hyperparameters}  error={r.error}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", default=_repo_relative_path("../test_datasets/basisData.csv"))
    parser.add_argument("--target", default=None)
    parser.add_argument("--problem-type", default="regression",
                        choices=["classification", "regression"])
    parser.add_argument(
        "--models", default="naive_bayes,decision_tree,knn,linear_regression,kmeans,dbscan,hierarchical")
    parser.add_argument("--include-clustering", action="store_true")
    args = parser.parse_args()

    X, y, feats = load_csv_dataset(
        args.csv, target=args.target, problem_type=args.problem_type)

    print(
        f"X shape={X.shape}, y shape={None if y is None else y.shape}, #features={len(feats)}")
    print(f"Features: {feats[:10]}{'...' if len(feats) > 10 else ''}")

    X_train, y_train, X_val, y_val, X_test, y_test = split_data(
        X, y,
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        problem_type=args.problem_type
    )

    # Model hyperparam smoke configs
    default_grids: Dict[str, Dict[str, Any]] = {
        "naive_bayes": {},
        "decision_tree": {"max_depth": 5},
        "knn": {"n_neighbors": 5},
        "linear_regression": {"loss": "l2", "learning_rate": 0.001, "epochs": 2000, "batch_size": 64},
        "kmeans": {"n_clusters": 3},
        "dbscan": {"eps": 0.5, "min_samples": 5},
        "hierarchical": {"n_clusters": 3},
    }

    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    results = []

    for m in requested:
        hp = default_grids.get(m)
        if m in ("kmeans", "dbscan", "hierarchical"):
            if args.include_clustering:
                results.append(run_clustering_model(m, hp, X_train))
        else:
            # supervised
            if y_train is None:
                continue
            results.append(run_supervised_model(
                m, hp, args.problem_type,
                X_train, y_train, X_val, y_val, X_test, y_test,
            ))

    print(results)

    print_leaderboard(results, top_k=20)


if __name__ == "__main__":
    main()
