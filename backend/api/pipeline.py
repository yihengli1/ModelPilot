import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from rest_framework import status
from rest_framework.response import Response

from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import ParameterGrid


from .services import generate_plan_gpt, generate_target_gpt, summarize_and_select_features, generate_refined_plan_gpt
from .modelling import model_control


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
    print("Choosing Target Column and Reducing Features")
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
    print("Generating Refinement Plan")
    refined_models, refined_tokens = generate_refined_plan_gpt(prompt,
                                                               initial_results,
                                                               target_name,)

    # iterate over range of models/hyperparams/
    print("Refining model")
    refined_results = refineModel(
        refined_models, X_train, y_train, X_val, y_val, X_test, y_test, classes)

    # best validation error
    all_results = initial_results + refined_results
    top_3_models = _select_top_models(all_results, top_k=3)

    # calculate token use
    total_tokens = target_tokens + plan_tokens + refined_tokens
    llm_result["total_tokens"] = total_tokens
    llm_result["total_models"] = len(all_results)

    return {
        "plan": llm_result,
        "results": top_3_models
    }


def _select_top_models(results: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
    valid_results = [r for r in results if not r.get("error")]
    sorted_results = sorted(
        valid_results,
        key=lambda x: x.get("val_accuracy", 0.0),
        reverse=True
    )
    return sorted_results[:top_k]


def refineModel(refined_models, X_train, y_train, X_val, y_val, X_test, y_test, classes):

    refined_model_configs = refined_models.get("refined_models", [])
    refined_plans = []
    for model in refined_model_configs:
        refined_plans.append({
            "model": model.get("model"),
            "hyperparameters": model.get("initial_hyperparameters"),
            "reasoning": model.get("reasoning")
        })

    return execute_training_cycle(
        X_train, y_train, X_val, y_val, X_test, y_test, classes,
        refined_plans
    )


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
        model_type = plan.get("model")
        params = plan.get("hyperparameters", {})

        # list of prams
        grid_params = {k: (v if isinstance(v, list) else [
                           v]) for k, v in params.items()}

        for single_param_set in ParameterGrid(grid_params):
            try:
                model, is_supervised = model_control(
                    model_type, single_param_set)

                metrics = training_models(
                    model, is_supervised, X_train, X_val, X_test, y_train, y_val, y_test)

                artifact = serialize_artifact(model, model_type, metrics)

                results.append({
                    "model": model_type,
                    "hyperparameters": single_param_set,
                    "metrics": metrics,
                    "artifact": artifact
                })

            except Exception as exc:
                results.append({"model": model_type, "error": str(exc)})

    return results


def training_models(model, is_supervised, X_train, X_val, X_test, y_train, y_val, y_test):
    metrics = {}
    metrics["supervised"] = is_supervised
    if is_supervised:
        model.fit(X_train, y_train)
        val_acc = accuracy_score(y_val, model.predict(X_val))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        metrics["val_accuracy"] = val_acc
        metrics["test_accuracy"] = test_acc
    else:
        if hasattr(model, "fit_predict"):
            train_preds = model.fit_predict(X_train)
        else:
            model.fit(X_train)
            train_preds = model.labels_

        n_labels_train = len(set(train_preds))
        if 1 < n_labels_train < len(X_train):
            metrics["train_silhouette"] = silhouette_score(
                X_train, train_preds)
        else:
            metrics["train_silhouette"] = -1.0

        if 1 < n_labels_train < len(X_train):
            score = silhouette_score(X_train, train_preds)
            metrics["train_silhouette"] = score
            # Use Silhouette as the "Accuracy" proxy for sorting later
            metrics["val_accuracy"] = score
            metrics["train_accuracy"] = score
        else:
            metrics["train_silhouette"] = -1.0
            metrics["val_accuracy"] = -1.0
            metrics["train_accuracy"] = -1.0
    return metrics


def serialize_artifact(classifier, model, metrics):
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
        elif model == "knn":
            return {
                "n_samples_fit": classifier.n_samples_fit_,
                "n_features": classifier.n_features_in_,
                "effective_metric": classifier.effective_metric_,
            }
        elif model == "kmeans":
            return {
                "n_clusters": classifier.n_clusters,
                "inertia": float(classifier.inertia_),
                "silhouette_score": metrics.get("train_silhouette", -1),
            }
        elif model == "dbscan":
            return {
                "n_samples_fit": classifier.n_samples_fit_,
                "classes_found": len(set(classifier.classes_)),
                "silhouette_score": metrics.get("train_silhouette", -1),
            }
        elif model == "hierarchical":
            return {
                "n_clusters": classifier.n_clusters_,
                "labels": classifier.labels_.tolist(),
                "n_leaves": classifier.n_leaves_,
                "children": classifier.children_.tolist() if hasattr(classifier, 'children_') else [],
                "silhouette_score": metrics.get("train_silhouette", -1),
            }
        else:
            return {}
    except Exception:
        return {"error": "Could not serialize model artifact"}
