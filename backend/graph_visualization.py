# plot_dataset_and_model.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from api.complexmodels.regression import LinearRegressionTorchNN, KernelPolynomialTorch
from api.complexmodels.linear_classifier import LinearClassifierTorchNN


def plot_original_and_pred(csv_path, model, x_cols=None, y_col="y", task="auto"):
    df = pd.read_csv(csv_path)

    if x_cols is None:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) < 2:
            raise ValueError(
                f"Need at least 2 numeric columns. Found: {df.columns.tolist()}")
        x_cols = (num_cols[0], num_cols[1])

    if y_col in df.columns:
        y = df[y_col].to_numpy()
    else:
        y = None

    X = df.loc[:, list(x_cols)].to_numpy()

    if task == "auto":
        task = "clustering" if y is None else "classification"

    m = clone(model)

    if task == "clustering":
        y_pred = m.fit_predict(X) if hasattr(
            m, "fit_predict") else (m.fit(X) or m.predict(X))
    else:
        if y is None:
            raise ValueError(
                f"Missing target column '{y_col}'. Pass y_col=... or use task='clustering'.")
        m.fit(X, y)
        y_pred = m.predict(X)

    plt.figure()
    if y is None:
        plt.scatter(X[:, 0], X[:, 1], s=18, alpha=0.8)
        plt.title(f"Original (unlabeled) | X={x_cols}")
    else:
        plt.scatter(X[:, 0], X[:, 1], c=y, s=18, alpha=0.8)
        plt.title(f"Original (colored by {y_col}) | X={x_cols}")
        plt.colorbar()
    plt.xlabel(x_cols[0])
    plt.ylabel(x_cols[1])

    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=18, alpha=0.8)
    plt.title(f"Model output | task={task}")
    plt.xlabel(x_cols[0])
    plt.ylabel(x_cols[1])
    plt.colorbar()

    plt.show()


if __name__ == "__main__":
    CSV = "./test_datasets/smallCities.csv"

    # Classification
    clf = DecisionTreeClassifier(
        criterion="gini", max_depth=None, min_samples_leaf=4, random_state=42)

    # Regression
    # reg = LinearRegressionTorchNN()

    # Clustering
    # clu = KMeans(n_clusters=3, random_state=0)

    plot_original_and_pred(CSV, model=clf, task="classification")
    # plot_original_and_pred(CSV, model=reg, task="regression")
    # plot_original_and_pred(CSV, model=clu, task="clustering")
    # plot_original_and_pred(CSV, model=clf, task="auto")  # tries to infer from y
