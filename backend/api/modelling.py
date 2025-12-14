
def model_control(model_type, single_param_set):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier

    is_supervised = True
    if model_type == "naive_bayes":
        model_type = GaussianNB(**single_param_set)
    elif model_type == "decision_tree":
        model_type = DecisionTreeClassifier(
            **single_param_set, random_state=42)
    elif model_type == "knn":
        model_type = KNeighborsClassifier(**single_param_set)
    elif model_type == "kmeans":
        model_type = KMeans(**single_param_set, random_state=42)
        is_supervised = False
    elif model_type == "dbscan":
        model_type = DBSCAN(**single_param_set)
        is_supervised = False
    elif model_type == "hierarchical":
        model_type = AgglomerativeClustering(**single_param_set)
    else:
        raise

    return model_type, is_supervised
