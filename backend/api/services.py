"""Lightweight heuristics for turning a dataset + prompt into a model plan."""

from __future__ import annotations

import csv
import io
import random
from pathlib import Path
from typing import Dict, List

from django.conf import settings


def analyze_dataset(
    dataset: str, prompt: str, context: str = "", target_column: str | None = None
) -> Dict:
    """Return a stubbed analysis result for a dataset/prompt pair."""
    columns = _extract_columns(dataset)
    features = _select_features(columns, target_column)
    model, hyperparameters, metrics = _suggest_model(target_column, len(features))

    return {
        "selected_features": features,
        "model": model,
        "hyperparameters": hyperparameters,
        "metrics": metrics,
        "notes": _craft_notes(prompt, context, target_column, features),
    }


def _extract_columns(dataset: str) -> List[str]:
    try:
        reader = csv.reader(io.StringIO(dataset.strip()))
        header = next(reader, [])
    except Exception:
        header = []
    return [col.strip() for col in header if col.strip()]


def _select_features(columns: List[str], target_column: str | None) -> List[str]:
    if not columns:
        return []

    targets = {target_column} if target_column else set()
    candidates = [col for col in columns if col not in targets]
    max_features = min(8, len(candidates))

    random.seed(42)
    random.shuffle(candidates)
    return sorted(candidates[:max_features])


def _suggest_model(target_column: str | None, feature_count: int):
    is_classification = bool(target_column)
    if is_classification:
        model = "RandomForestClassifier"
        hyperparameters = {"n_estimators": 200, "max_depth": 8, "min_samples_split": 4}
        metrics = {"accuracy": 0.84, "f1": 0.81}
    else:
        model = "GradientBoostingRegressor"
        hyperparameters = {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 3}
        metrics = {"rmse": 3.4, "r2": 0.78}

    hyperparameters["feature_count"] = feature_count
    return model, hyperparameters, metrics


def _craft_notes(prompt: str, context: str, target_column: str | None, features: List[str]) -> str:
    parts = [
        f"Prompt length: {len(prompt)} chars",
        f"Context length: {len(context)} chars",
        f"Target column: {target_column or 'not specified'}",
        f"Selected {len(features)} feature(s)",
    ]
    return " | ".join(parts)


def load_sample_dataset() -> str:
    path = Path(settings.SAMPLE_DATA_DIR) / "example_dataset.csv"
    return path.read_text()


def load_sample_prompt() -> str:
    path = Path(settings.SAMPLE_DATA_DIR) / "example_prompt.txt"
    return path.read_text()
