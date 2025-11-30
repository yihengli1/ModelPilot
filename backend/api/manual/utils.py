import argparse
from pathlib import Path
import pickle

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.optimize import approx_fprime


"""
Manual classes from school assignment lol
"""


def load_dataset(filename):
    with open(Path("..", "data", filename), "rb") as f:
        return pickle.load(f)


def savefig(fname, fig=None, verbose=True):
    path = Path("..", "figs", fname)
    (plt if fig is None else fig).savefig(
        path, bbox_inches="tight", pad_inches=0)
    if verbose:
        print(f"Figure saved as '{path}'")


def shortest_dist(G, i=None, j=None):
    """Computes shortest distance between all pairs of nodes given an adjacency matrix G,
    where G[i,j]=0 implies there is no edge from i to j.

    Parameters
    ----------
    G : an N by N numpy array

    """
    dist = scipy.sparse.csgraph.dijkstra(G, directed=False)
    if i is not None and j is not None:
        return dist[i, j]
    else:
        return dist


def ensure_1d(x):
    if x.ndim == 1:
        return x
    elif x.ndim == 2:
        return x.squeeze(axis=1)
    elif x.ndim == 0:
        return x[np.newaxis]
    else:
        raise ValueError(f"invalid shape {x.shape} for ensure_1d")


def standardize_cols(X, mu=None, sigma=None):
    # Standardize each column with mean 0 and variance 1
    n_rows, n_cols = X.shape

    if mu is None:
        mu = np.mean(X, axis=0)

    if sigma is None:
        sigma = np.std(X, axis=0)
        sigma[sigma < 1e-8] = 1.0

    return (X - mu) / sigma, mu, sigma


def euclidean_dist_squared(X, Xtest):
    """Computes the Euclidean distance between rows of 'X' and rows of 'Xtest'

    Parameters
    ----------
    X : an N by D numpy array
    Xtest: an T by D numpy array

    Returns: an array of size N by T containing the pairwise squared Euclidean distances.

    Python/Numpy (and other numerical languages like Matlab and R)
    can be slow at executing operations in `for' loops, but allows extremely-fast
    hardware-dependent vector and matrix operations. By taking advantage of SIMD registers and
    multiple cores (and faster matrix-multiplication algorithms), vector and matrix operations in
    Numpy will often be several times faster than if you implemented them yourself in a fast
    language like C. The following code will form a matrix containing the squared Euclidean
    distances between all training and test points. If the output is stored in D, then
    element D[i,j] gives the squared Euclidean distance between training point
    i and testing point j. It exploits the identity (a-b)^2 = a^2 + b^2 - 2ab.
    The right-hand-side of the above is more amenable to vector/matrix operations.
    """

    # add extra dimensions so that the function still works for X and/or Xtest are 1-D arrays.
    if X.ndim == 1:
        X = X[None]
    if Xtest.ndim == 1:
        Xtest = Xtest[None]

    return (
        np.sum(X ** 2, axis=1)[:, None]
        + np.sum(Xtest ** 2, axis=1)[None]
        - 2 * np.dot(X, Xtest.T)
    )


def check_gradient(model, X, y, dimensionality, verbose=True, epsilon=1e-6):
    # This checks that the gradient implementation is correct
    w = np.random.randn(dimensionality)
    f, g = model.funObj(w, X, y)

    # Check the gradient
    estimated_gradient = approx_fprime(
        w, lambda w: model.funObj(w, X, y)[0], epsilon=epsilon
    )

    implemented_gradient = model.funObj(w, X, y)[1]

    if (
        np.max(np.abs(estimated_gradient - implemented_gradient))
        / np.linalg.norm(estimated_gradient)
        > 1e-6
    ):
        raise Exception(
            "User and numerical derivatives differ:\n%s\n%s"
            % (estimated_gradient[:5], implemented_gradient[:5])
        )
    else:
        if verbose:
            print("User and numerical derivatives agree.")


def plot_classifier(model, X, y, need_argmax=False, ax=None):
    """plots the decision boundary of the model and the scatterpoints
       of the target values 'y'.

    Assumptions
    -----------
    y : it should contain two classes: '1' and '2'

    Parameters
    ----------
    model : the trained model which has the predict function

    X : the N by D feature array

    y : the N element vector corresponding to the target values

    """
    x1 = X[:, 0]
    x2 = X[:, 1]

    x1_min, x1_max = int(x1.min()) - 1, int(x1.max()) + 1
    x2_min, x2_max = int(x2.min()) - 1, int(x2.max()) + 1

    x1_line = np.linspace(x1_min, x1_max, 200)
    x2_line = np.linspace(x2_min, x2_max, 200)

    x1_mesh, x2_mesh = np.meshgrid(x1_line, x2_line)

    mesh_data = np.c_[x1_mesh.ravel(), x2_mesh.ravel()]

    y_pred = model.predict(mesh_data)
    if need_argmax:
        y_pred = np.argmax(y_pred, axis=1)

    y_pred = np.reshape(y_pred, x1_mesh.shape)

    if ax is None:
        ax = plt.gca()
    ax.set_xlim([x1_mesh.min(), x1_mesh.max()])
    ax.set_ylim([x2_mesh.min(), x2_mesh.max()])

    ax.contourf(
        x1_mesh,
        x2_mesh,
        # unsigned int causes problems with negative sign... o_O
        -y_pred.astype(int),
        cmap=plt.cm.RdBu,
        alpha=0.6,
    )

    y_vals = np.unique(y)
    for c, color in zip(y_vals, "br"):
        in_c = y == c
        ax.scatter(x1[in_c], x2[in_c], color=color, label=f"class {c:+d}")
    ax.legend()


def create_rating_matrix(
    ratings, n, d, user_key="user", item_key="item", valid_frac=0.2
):
    # shuffle the order
    ratings = ratings.sample(frac=1, random_state=123)

    user_mapper = dict(zip(np.unique(ratings[user_key]), list(range(n))))
    item_mapper = dict(zip(np.unique(ratings[item_key]), list(range(d))))

    user_inverse_mapper = dict(
        zip(list(range(n)), np.unique(ratings[user_key])))
    item_inverse_mapper = dict(
        zip(list(range(d)), np.unique(ratings[item_key])))

    # Y = scipy.sparse.csr_matrix((ratings["rating"], (user_ind, item_ind)), shape=(n,d))

    n_train = int(len(ratings) * (1 - valid_frac))
    ratings_train = ratings[:n_train]
    ratings_valid = ratings[n_train:]

    user_ind_train = [user_mapper[i] for i in ratings_train[user_key]]
    item_ind_train = [item_mapper[i] for i in ratings_train[item_key]]

    user_ind_valid = [user_mapper[i] for i in ratings_valid[user_key]]
    item_ind_valid = [item_mapper[i] for i in ratings_valid[item_key]]

    Y_train = np.full((n, d), np.nan)
    Y_train[user_ind_train, item_ind_train] = ratings_train["rating"]

    Y_valid = np.full((n, d), np.nan)
    Y_valid[user_ind_valid, item_ind_valid] = ratings_valid["rating"]

    return Y_train, Y_valid

    # implementation note after 1+ hours wasted debugging::
    # if we're not bothing with a sparse matrix anyway, it's easier
    # to store the missing ratings as NaN instead of 0, because nansum/nanmean is very convenient.
    # there were some horrible bugs using zeros, namely that the centering changes the zeros
    # and then the masing didn't work because those entries weren't 0 anymore, so the masking
    # simply did not happen. disaster.
    # yeah I should have known to do the centering and then put the zeros, but then I couldn't use the nice
    # existing code. So that would have been fine to. blegh.
    # ¯\_(ツ)_/¯


################################################################################
# Helpers for setting up the command-line interface

_funcs = {}


def handle(number):
    def register(func):
        _funcs[number] = func
        return func

    return register


def run(question):
    if question not in _funcs:
        raise ValueError(f"unknown question {question}")
    return _funcs[question]()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", choices=sorted(_funcs.keys()) + ["all"])
    args = parser.parse_args()
    if args.question == "all":
        for q in sorted(_funcs.keys()):
            start = f"== {q} "
            print("\n" + start + "=" * (80 - len(start)))
            run(q)
    else:
        return run(args.question)
