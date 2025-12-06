import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from rest_framework import status
from rest_framework.response import Response

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

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


def to_numpy(x):
    if hasattr(x, 'numpy'):
        return x.numpy()
    return np.array(x)


def training_pipeline(prompt, dataset: np.ndarray, headers: Optional[List[str]] = None):

    # Sharp Feature Selection for LLM/Target Column identification
    target_name, selected_summaries, aggregated_stats, target_tokens = reduce_features(
        headers, dataset, prompt)

    # Initialization
    print("Generating Initial Plan")
    llm_result, plan_tokens = generate_plan_gpt(
        prompt=prompt,
        summaries=selected_summaries,
        target_name=target_name,
    )

    problem_type, target_column, data_split, model_plans = _parsing_initialization(
        llm_result)

    # Feature Selection
    # TODO:

    # Data Prep
    print("Preparing Datasets")
    X_train, y_train, X_val, y_val, X_test, y_test, classes = prepare_datasets(
        dataset, target_column, data_split, headers)

    # # Split Model, PyTorch training
    print("Running Initial Model")
    initial_results = execute_training_cycle(
        X_train, y_train, X_val, y_val, X_test, y_test, classes,
        model_plans
    )

    # Based on results 2 call

    # iterate over range of models/hyperparams/

    # best validation error

    # send back result

    # calculate token use
    total_tokens = target_tokens + plan_tokens
    llm_result["total_tokens"] = total_tokens

    return llm_result, initial_results

    # return {
    #     "problem_type": problem_type,
    #     "target_column": target_column,
    #     "data_split": data_split,
    #     "models": model_plans,
    #     # "model_results": model_results,
    #     "raw_llm_result": llm_result,
    # }


def reduce_features(headers, dataset, prompt):
    target_name, target_tokens = generate_target_gpt(
        user_prompt=prompt, headers=headers)

    selected_summaries, aggregated_stats = summarize_and_select_features(
        headers,
        dataset,
        target_name=target_name
    )

    return target_name, selected_summaries, aggregated_stats, target_tokens

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


def prepare_datasets(
    dataset,
    target_column,
    data_split,
    headers,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
    X_train, y_train, X_val, y_val, X_test, y_test, classes = _split_dataset(
        dataset, target_column, data_split, headers=headers
    )

    return (
        to_numpy(X_train), to_numpy(y_train),
        to_numpy(X_val),   to_numpy(y_val),
        to_numpy(X_test),  to_numpy(y_test),
        classes
    )


def execute_training_cycle(
    X_train, y_train, X_val, y_val, X_test, y_test, classes,
    model_plans
) -> List[Dict[str, Any]]:
    results = []

    for plan in model_plans:
        model = plan.get("model")
        params = plan.get("hyperparameters", {})
        try:
            clf = None
            if model == "naive_bayes":
                clf = GaussianNB(**params)
            elif model == "decision_tree":
                clf = DecisionTreeClassifier(**params, random_state=42)
            else:
                results.append({"model": model, "error": "Unsupported model"})
                continue

            clf.fit(X_train, y_train)

            val_acc = accuracy_score(y_val, clf.predict(
                X_val))
            test_acc = accuracy_score(y_test, clf.predict(
                X_test))

            artifact = serialize_artifact(clf, model)

            results.append({
                "model": model,
                "hyperparameters": params,
                "val_accuracy": val_acc,
                "test_accuracy": test_acc,
                "artifact": artifact
            })

        except Exception as exc:
            results.append({"model": model, "error": str(exc)})

    return results


def serialize_artifact(classifier, model):
    try:
        if model == "naive_bayes":
            return {
                "classes": classifier.classes_.tolist(),
                "means": classifier.theta_.tolist(),
                "vars": classifier.var_.tolist(),
            }
        elif model == "decision_tree":
            return {
                "n_features": classifier.n_features_in_,
                "depth": classifier.get_depth(),
                "n_leaves": classifier.get_n_leaves(),
            }
        else:
            return {}
    except Exception:
        return {"error": "Could not serialize model artifact"}
