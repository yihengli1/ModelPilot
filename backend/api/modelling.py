from .complexmodels.regression import LinearRegressionTorchNN, KernelPolynomialTorch
from .complexmodels.linear_classifier import LinearClassifierTorchNN
from .complexmodels.pca import PCAModel as WrappedPCA
from .complexmodels.mlp import MLPClassifierTorchNN, MLPRegressorTorchNN


MODEL_TASK = {
    "linear_regression": "regression",
    "kernel_polynomial": "regression",
    "linear_classifier": "classification",
    "decision_tree": "classification",
    "naive_bayes": "classification",
    "knn": "classification",
    "kmeans": "clustering",
    "dbscan": "clustering",
    "hierarchical": "clustering",
    "pca": "dimension_reduction",
    "mlp_classifier": "classification",
    "mlp_regressor": "regression",
}


def model_control(model_type, single_param_set):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier

    is_supervised = True
    if model_type == "naive_bayes":
        model_type = GaussianNB(**single_param_set)
        is_supervised = True
    elif model_type == "decision_tree":
        model_type = DecisionTreeClassifier(
            **single_param_set, random_state=42)
        is_supervised = True
    elif model_type == "knn":
        model_type = KNeighborsClassifier(**single_param_set)
        is_supervised = True
    elif model_type == "linear_classifier":
        model_type = LinearClassifierTorchNN(**single_param_set)
        is_supervised = True
    elif model_type == "linear_regression":
        model_type = LinearRegressionTorchNN(**single_param_set)
        is_supervised = True
    elif model_type == "kernel_polynomial":
        model_type = KernelPolynomialTorch(**single_param_set)
        is_supervised = True
    elif model_type == "kmeans":
        model_type = KMeans(**single_param_set, random_state=42)
        is_supervised = False
    elif model_type == "dbscan":
        model_type = DBSCAN(**single_param_set)
        is_supervised = False
    elif model_type == "hierarchical":
        model_type = AgglomerativeClustering(**single_param_set)
        is_supervised = False
    elif model_type == "pca":
        model_type = WrappedPCA(**single_param_set)
        is_supervised = False
    elif model_type == "mlp_classifier":
        model_type = MLPClassifierTorchNN(**single_param_set)
        is_supervised = True
    elif model_type == "mlp_regressor":
        model_type = MLPRegressorTorchNN(**single_param_set)
        is_supervised = True
    else:
        raise ValueError(
            f"Model type '{model_type}' is not supported or recognized.")

    return model_type, is_supervised


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
        elif model == "linear_regression":
            return {
                "weight": (classifier.coef_.tolist() if hasattr(classifier, "coef_") and classifier.coef_ is not None else []),
                "intercept": float(getattr(classifier, "intercept_", 0.0)),
                "loss": getattr(classifier, "loss"),
            }
        elif model == "linear_classifier":
            return {
                "classes": getattr(classifier, "classes_", None).tolist(),
                "weight": (classifier.coef_.tolist() if hasattr(classifier, "coef_") and classifier.coef_ is not None else []),
                "intercept": float(getattr(classifier, "intercept_", 0.0)),
                "loss": getattr(classifier, "loss"),
            }
        elif model == "kernel_polynomial":
            return {
                "degree": int(getattr(classifier, "degree")),
                "lam": float(getattr(classifier, "lam")),
            }
        elif model == "kmeans":
            return {
                "n_clusters": classifier.n_clusters,
                "inertia": float(classifier.inertia_)
            }
        elif model == "dbscan":
            labels = getattr(classifier, "labels_", None)
            return {
                "n_clusters": int(len(set(labels.tolist())) - (1 if -1 in labels else 0)),
                "n_noise": int((labels == -1).sum())
            }
        elif model == "hierarchical":
            return {
                "n_clusters": classifier.n_clusters_,
                "labels": classifier.labels_.tolist(),
                "n_leaves": classifier.n_leaves_,
                "children": classifier.children_.tolist() if hasattr(classifier, 'children_') else []
            }
        elif model == "pca":
            pca = getattr(classifier, "pca", classifier)
            return {
                "n_components": int(getattr(pca, "n_components_", getattr(pca, "n_components", 0)) or 0),
                "variance_explained": getattr(classifier, "variance_explained_", None)
            }
        elif model == "mlp_classifier":
            return {
                "input_dim": int(getattr(classifier, "input_dim_", 0) or 0),
                "output_dim": int(getattr(classifier, "output_dim_", 0) or 0),
                "hidden_layers": list(getattr(classifier, "hidden_layers", []) or []),
                "activation": getattr(classifier, "activation", "relu"),
                "dropout": float(getattr(classifier, "dropout", 0.0) or 0.0),
                "optimizer": getattr(classifier, "optimizer", "adam"),
                "learning_rate": float(getattr(classifier, "learning_rate", 0.0) or 0.0),
                "epochs_trained": int(getattr(getattr(classifier, "fit_stats_", None), "epochs_trained", 0) or 0),
                "best_val_loss": getattr(getattr(classifier, "fit_stats_", None), "best_val_loss", None),
                "n_params": int(getattr(classifier, "n_params_", 0) or 0),
                "n_classes": int(getattr(classifier, "n_classes_", 0) or 0),
                "loss": getattr(classifier, "loss", "cross_entropy"),
            }

        elif model == "mlp_regressor":
            return {
                "input_dim": int(getattr(classifier, "input_dim_", 0) or 0),
                "output_dim": int(getattr(classifier, "output_dim_", 0) or 0),
                "hidden_layers": list(getattr(classifier, "hidden_layers", []) or []),
                "activation": getattr(classifier, "activation", "relu"),
                "dropout": float(getattr(classifier, "dropout", 0.0) or 0.0),
                "optimizer": getattr(classifier, "optimizer", "adam"),
                "learning_rate": float(getattr(classifier, "learning_rate", 0.0) or 0.0),
                "epochs_trained": int(getattr(getattr(classifier, "fit_stats_", None), "epochs_trained", 0) or 0),
                "best_val_loss": getattr(getattr(classifier, "fit_stats_", None), "best_val_loss", None),
                "n_params": int(getattr(classifier, "n_params_", 0) or 0),
                "loss": getattr(classifier, "loss", "l2"),
            }

        else:
            return {}
    except Exception:
        return {"error": "Could not serialize model artifact"}
