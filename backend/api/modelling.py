import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from rest_framework import status
from rest_framework.response import Response

from .services import generate_plan_gpt, generate_target_gpt, summarize_and_select_features


def _select_target_index(target_column: Any, headers: Optional[List[str]] = None) -> int:
    # -1 Unsupervised
    # >= 0 Supervised
    if target_column == "" or target_column is None:
        return -1
    if isinstance(target_column, str):
        if headers:
            lowered = [h.lower() for h in headers]
            return lowered.index(target_column.lower())
    if isinstance(target_column, int):
        return target_column
    return Response("Target Column is not defined on a Supervised Learning Task")


def _split_dataset(
    dataset: np.ndarray,
    target_column: Any,
    data_split: Dict[str, Any],
    headers: Optional[List[str]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    target_idx = _select_target_index(target_column, headers)

    if data_split is not None:
        ratios = data_split.get("train_val_test")
    if ratios is None:
        ratios = [0.7, 0.15, 0.15]
    ratios = np.array(ratios, dtype=float)
    ratios = ratios / ratios.sum()  # safety

    # Supervised
    if target_idx != -1:
        y = dataset[:, target_idx]
        X = np.delete(dataset, target_idx, axis=1)
        classes, y_encoded = np.unique(y, return_inverse=True)
    # Unsupervised
    else:
        X = dataset
        y_encoded = None
        classes = np.array([])
    try:
        X_float = np.asarray(X, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Features could not be converted to float for training.") from exc

    n, _ = X_float.shape
    perm = torch.randperm(n).numpy()
    train_end = int(ratios[0] * n)
    val_end = train_end + int(ratios[1] * n)

    train_idx = perm[:train_end] if train_end > 0 else perm[:0]
    val_idx = perm[train_end:val_end] if val_end > train_end else perm[:0]
    test_idx = perm[val_end:] if val_end < n else perm[:0]

    X_train = torch.tensor(X_float[train_idx], dtype=torch.float32)
    X_val = torch.tensor(X_float[val_idx], dtype=torch.float32)
    X_test = torch.tensor(X_float[test_idx], dtype=torch.float32)

    if target_idx != -1:
        y_train = torch.tensor(y_encoded[train_idx], dtype=torch.long)
        y_val = torch.tensor(y_encoded[val_idx], dtype=torch.long)
        y_test = torch.tensor(y_encoded[test_idx], dtype=torch.long)
    else:
        y_train = torch.empty(0, dtype=torch.long)
        y_val = torch.empty(0, dtype=torch.long)
        y_test = torch.empty(0, dtype=torch.long)

    return X_train, y_train, X_val, y_val, X_test, y_test, classes


def _accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    if targets.numel() == 0:
        return float("nan")
    correct = (preds == targets).sum().item()
    return correct / targets.numel()


def _train_gaussian_naive_bayes(X: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
    classes = torch.unique(y)
    means = []
    vars_ = []
    priors = []
    for cls in classes:
        cls_mask = y == cls
        X_cls = X[cls_mask]
        means.append(X_cls.mean(dim=0))
        vars_.append(X_cls.var(dim=0, unbiased=False) + 1e-6)
        priors.append(cls_mask.float().mean())
    return {
        "classes": classes,
        "means": torch.stack(means),
        "vars": torch.stack(vars_),
        "log_priors": torch.log(torch.stack(priors)),
    }


def _predict_gaussian_naive_bayes(model: Dict[str, torch.Tensor], X: torch.Tensor) -> torch.Tensor:
    means = model["means"]
    vars_ = model["vars"]
    log_priors = model["log_priors"]
    classes = model["classes"]

    log_likelihood = -0.5 * \
        torch.sum(torch.log(2 * math.pi * vars_) +
                  ((X.unsqueeze(1) - means) ** 2) / vars_, dim=2)
    log_post = log_priors + log_likelihood
    return classes[torch.argmax(log_post, dim=1)]


def _train_decision_stump(X: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
    n_features = X.shape[1]
    best_gini = float("inf")
    best_feature = 0
    best_threshold = 0.0
    best_left_class = 0
    best_right_class = 0

    for feat in range(n_features):
        values = torch.unique(X[:, feat])
        if values.numel() == 1:
            continue
        thresholds = (values[:-1] + values[1:]) / 2
        for thr in thresholds:
            left_mask = X[:, feat] <= thr
            right_mask = ~left_mask
            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue

            def gini(mask: torch.Tensor) -> float:
                subset = y[mask]
                if subset.numel() == 0:
                    return 0.0
                _, counts = torch.unique(subset, return_counts=True)
                probs = counts.float() / subset.numel()
                return 1.0 - torch.sum(probs ** 2).item()

            gini_left = gini(left_mask)
            gini_right = gini(right_mask)
            total = y.numel()
            weighted_gini = (left_mask.sum().item() / total) * \
                gini_left + (right_mask.sum().item() / total) * gini_right

            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature = feat
                best_threshold = thr.item()
                left_labels = y[left_mask]
                right_labels = y[right_mask]
                best_left_class = torch.mode(left_labels).values.item()
                best_right_class = torch.mode(right_labels).values.item()

    return {
        "feature": best_feature,
        "threshold": best_threshold,
        "left_class": best_left_class,
        "right_class": best_right_class,
    }


def _predict_decision_stump(model: Dict[str, Any], X: torch.Tensor) -> torch.Tensor:
    feat = model["feature"]
    thr = model["threshold"]
    left_class = model["left_class"]
    right_class = model["right_class"]
    preds = torch.where(X[:, feat] <= thr, left_class, right_class)
    return preds


def train_and_evaluate_models(
    dataset: np.ndarray,
    target_column: Any,
    model_plans: List[Dict[str, Any]],
    data_split: Dict[str, Any],
    headers: Optional[List[str]] = None,
):
    X_train, y_train, X_val, y_val, X_test, y_test, classes = _split_dataset(
        dataset, target_column, data_split, headers=headers)
    results = []

    def serialize_artifact(model: Dict[str, Any], switch: str) -> Dict[str, Any]:
        if switch == "naive_bayes":
            return {
                "classes": model["classes"].tolist(),
                "means": model["means"].tolist(),
                "vars": model["vars"].tolist(),
                "log_priors": model["log_priors"].tolist(),
            }
        return model

    for plan in model_plans:
        switch = (plan.get("switch") or plan.get("model") or "").lower()
        try:
            # Supervised
            if switch == "naive_bayes":
                model = _train_gaussian_naive_bayes(X_train, y_train)
                val_preds = _predict_gaussian_naive_bayes(
                    model, X_val) if y_val.numel() else torch.tensor([])
                test_preds = _predict_gaussian_naive_bayes(
                    model, X_test) if y_test.numel() else torch.tensor([])
            elif switch == "decision_tree":
                model = _train_decision_stump(X_train, y_train)
                val_preds = _predict_decision_stump(
                    model, X_val) if y_val.numel() else torch.tensor([])
                test_preds = _predict_decision_stump(
                    model, X_test) if y_test.numel() else torch.tensor([])
            else:
                results.append(
                    {
                        "model": plan.get("model"),
                        "error": f"Model type '{switch}' is unsupported.",
                    }
                )
                continue

            val_acc = _accuracy(val_preds, y_val)
            test_acc = _accuracy(test_preds, y_test)
            results.append(
                {
                    "model": plan.get("model"),
                    "switch": switch,
                    "hyperparameters": plan.get("hyperparameters"),
                    "reasoning": plan.get("reasoning"),
                    "val_accuracy": val_acc,
                    "test_accuracy": test_acc,
                    "classes": classes.tolist(),
                    "artifact": serialize_artifact(model, switch),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "model": plan.get("model"),
                    "switch": switch,
                    "error": str(exc),
                }
            )

    return results


def training_pipeline(prompt, dataset: np.ndarray, headers: Optional[List[str]] = None):

    # Sharp Feature Selection for LLM/Target Column identification

    target_name, selected_summaries, aggregated_stats = reduce_features(
        headers, dataset, prompt)

    # Initialization
    llm_result = generate_plan_gpt(
        prompt=prompt,
        summaries=selected_summaries,
        target_name=target_name,
    )

    print("finish")

    # problem_type, target_column, data_split, model_plans = _parsing_initialization(
    #     llm_result)

    # Feature Selection

    # Split Model, PyTorch training

    # model_results = train_and_evaluate_models(
    #     dataset, target_column, model_plans, data_split, headers=headers)

    # Based on results 2 call

    # iterate over range of models/hyperparams/

    # best validation error

    # send back result

    return llm_result

    # return {
    #     "problem_type": problem_type,
    #     "target_column": target_column,
    #     "data_split": data_split,
    #     "models": model_plans,
    #     # "model_results": model_results,
    #     "raw_llm_result": llm_result,
    # }


def reduce_features(headers, dataset, prompt):
    target_name = generate_target_gpt(user_prompt=prompt, headers=headers)

    selected_summaries, aggregated_stats = summarize_and_select_features(
        headers,
        dataset,
        target_name=target_name
    )

    return target_name, selected_summaries, aggregated_stats

# Output Example
# {
#   "problem_type": "...",
#   "target_column": "...",
#   "recommended_models": [
#     {
#       "model": "...",
#       "reasoning": "...",
#       "initial_hyperparameters": {
#           "param1": ...,
#           "param2": ...,
#           "etc": ...
#        }
#     }
#   ],
#   "data_split": {
#     "method": "...",
#     "train_val_test": [ ..., ..., ... ],
#     "stratify": "...",
#     "grouping_column": "..."
#   }
# }


def _parsing_initialization(llm_result):
    problem_type = llm_result.get("problem_type")
    target_column = llm_result.get("target_column")
    data_split = llm_result.get("data_split", {})
    recommended_models = llm_result.get("recommended_models", [])

    model_plans = []
    for model in recommended_models:
        model_name = model.get("model")
        reasoning = model.get("reasoning")
        model_key = model_name.lower().replace(" ", "_")
        hyperparams = model.get("initial_hyperparameters")

        model_plans.append(
            {
                "model": model_key,
                "hyperparameters": hyperparams,
                "reasoning": reasoning
            }
        )

    return problem_type, target_column, data_split, model_plans
