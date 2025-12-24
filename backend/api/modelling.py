from .complexmodels.linear_regression import LinearRegressionTorchNN


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
    elif model_type == "linear_regression":
        model_type = LinearRegressionTorchNN(**single_param_set)
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
                "loss": getattr(classifier, "loss", "l2"),
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
