"""
benchmark_modelpilot_llm.py

Runs ModelPilot's LLM-guided classification pipeline (api.pipeline.training_pipeline)
on the classification datasets defined in benchmark.py, and compares its test-set
accuracy against the untuned / grid-search-tuned scikit-learn baselines from that
same file.

Usage:
    cd backend
    ./.venv/bin/python benchmark_modelpilot_llm.py

Requires OPENAI_API_KEY in backend/.env (loaded automatically).
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd

from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, ".env"))
sys.path.insert(0, HERE)

# Reuse the exact dataset config + cleaning + baselines from benchmark.py
from benchmark import (
    DATASETS,
    RANDOM_STATE,
    load_and_clean,
    run_untuned_baselines,
    run_tuned_baselines,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from api.pipeline import training_pipeline

CLASSIFICATION_DATASETS = [d for d in DATASETS if d["task"] == "classification"]


def run_modelpilot(cfg):
    """Feed a cleaned, fully-numeric matrix (features + target as last column) to
    ModelPilot and return its best model + test accuracy + iteration/token counts."""
    X, y = load_and_clean(cfg)

    headers = list(X.columns) + [cfg["target"]]
    matrix = np.column_stack(
        [X.to_numpy(dtype=float), np.asarray(y, dtype=float)]
    ).astype(float)

    prompt = (
        f"This is a supervised classification task. "
        f"The target column is named exactly '{cfg['target']}'. "
        f"Predict '{cfg['target']}' from the remaining feature columns. "
        f"Optimise for classification accuracy."
    )

    # deterministic internal train/val/test split
    np.random.seed(RANDOM_STATE)

    t0 = time.time()
    out = training_pipeline(prompt, matrix, headers=headers)
    elapsed = time.time() - t0

    plan = out["plan"]
    results = out["results"]  # already sorted best-first by val_score

    # pick best by held-out test score among non-errored results
    scored = [r for r in results if "metrics" in r and "test_score" in r["metrics"]]
    scored.sort(key=lambda r: r["metrics"]["test_score"], reverse=True)
    best = scored[0] if scored else results[0]

    return {
        "model": best.get("model"),
        "hyperparameters": best.get("hyperparameters"),
        "test_score": float(best["metrics"]["test_score"]),
        "val_score": float(best["metrics"]["val_score"]),
        "problem_type": plan.get("problem_type"),
        "target_column": plan.get("target_column"),
        "recommended_models": [m.get("model") for m in plan.get("recommended_models", [])],
        "iterations": plan.get("total_models"),
        "tokens": plan.get("total_tokens"),
        "elapsed_s": elapsed,
        "all_results": [
            {
                "model": r.get("model"),
                "hyperparameters": r.get("hyperparameters"),
                "test_score": r.get("metrics", {}).get("test_score"),
                "val_score": r.get("metrics", {}).get("val_score"),
                "error": r.get("error"),
            }
            for r in results
        ],
    }


def main():
    rows = []

    for cfg in CLASSIFICATION_DATASETS:
        print(f"\n{'='*72}\n{cfg['name']}  ({cfg['task']})\n{'='*72}")
        try:
            X, y = load_and_clean(cfg)
        except FileNotFoundError:
            print(f"  [skipped] file not found: {cfg['path']}")
            continue

        # ---- scikit-learn baselines (same split/scaling as benchmark.py) ----
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)

        untuned = run_untuned_baselines(Xtr, Xte, ytr, yte, "classification")
        best_untuned = max(
            (k for k, v in untuned.items() if isinstance(v, float)),
            key=lambda k: untuned[k], default=None,
        )
        best_untuned_score = untuned.get(best_untuned)

        t0 = time.time()
        tuned, grid_iters, best_tuned, best_tuned_score = run_tuned_baselines(
            Xtr, Xte, ytr, yte, "classification"
        )
        tuned_time = time.time() - t0

        print("\n  Untuned baselines:")
        for k, v in untuned.items():
            print(f"    {k:20s}: {v}")
        print(f"  --> best untuned: {best_untuned} = {best_untuned_score:.4f}")

        print(f"\n  Tuned baselines  [{grid_iters} fits, {tuned_time:.1f}s]:")
        for k, v in tuned.items():
            print(f"    {k:20s}: {v}")
        print(f"  --> best tuned: {best_tuned} = {best_tuned_score:.4f}")

        # ---- ModelPilot LLM pipeline ----
        print("\n  Running ModelPilot LLM pipeline ...")
        try:
            mp = run_modelpilot(cfg)
        except Exception as exc:
            print(f"  [ModelPilot ERROR] {exc}")
            mp = None

        if mp:
            print(f"  --> ModelPilot picked: {mp['model']}  {mp['hyperparameters']}")
            print(f"      recommended_models: {mp['recommended_models']}")
            print(f"      problem_type={mp['problem_type']}  target={mp['target_column']}")
            print(f"      test accuracy = {mp['test_score']:.4f}  (val {mp['val_score']:.4f})")
            print(f"      configs evaluated = {mp['iterations']}   LLM tokens = {mp['tokens']}   {mp['elapsed_s']:.1f}s")

        rows.append({
            "dataset": cfg["name"],
            "best_untuned_model": best_untuned,
            "best_untuned_acc": round(best_untuned_score, 4) if best_untuned_score is not None else None,
            "best_tuned_model": best_tuned,
            "best_tuned_acc": round(best_tuned_score, 4) if best_tuned_score is not None else None,
            "grid_iters": grid_iters,
            "modelpilot_model": (mp["model"] if mp else None),
            "modelpilot_acc": (round(mp["test_score"], 4) if mp else None),
            "modelpilot_iters": (mp["iterations"] if mp else None),
            "modelpilot_tokens": (mp["tokens"] if mp else None),
        })

    print(f"\n\n{'='*72}\nSUMMARY\n{'='*72}")
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    valid = df.dropna(subset=["modelpilot_acc"])
    if len(valid):
        vs_untuned = ((valid["modelpilot_acc"] - valid["best_untuned_acc"]) /
                      valid["best_untuned_acc"] * 100).mean()
        vs_tuned = ((valid["modelpilot_acc"] - valid["best_tuned_acc"]) /
                    valid["best_tuned_acc"] * 100).mean()
        iter_red = ((valid["grid_iters"] - valid["modelpilot_iters"]) /
                    valid["grid_iters"] * 100).mean()
        print(f"\nAvg accuracy change vs UNTUNED baseline: {vs_untuned:+.1f}%")
        print(f"Avg accuracy change vs TUNED   baseline: {vs_tuned:+.1f}%")
        print(f"Avg iteration reduction vs manual grid search: {iter_red:.1f}%")

    # ready-to-paste block for benchmark.py
    print("\n--- paste into benchmark.py MODELPILOT_RESULTS ---")
    print("MODELPILOT_RESULTS = " + json.dumps(
        {r["dataset"]: {
            "model": r["modelpilot_model"],
            "score": r["modelpilot_acc"],
            "iterations": r["modelpilot_iters"],
        } for r in rows}, indent=4))


if __name__ == "__main__":
    main()
