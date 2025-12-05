import numpy as np
from decision_stump import DecisionStumpErrorRate


class DecisionTree:

    stump_model = None
    submodel_yes = None
    submodel_no = None

    def __init__(self, max_depth, stump_class=DecisionStumpErrorRate):
        self.max_depth = max_depth
        self.stump_class = stump_class

    def fit(self, X, y):
        # Fits a decision tree using greedy recursive splitting

        # Learn a decision stump
        stump_model = self.stump_class()
        stump_model.fit(X, y)

        if self.max_depth <= 1 or stump_model.j_best is None:
            # If we have reached the maximum depth or the decision stump does
            # nothing, use the decision stump

            self.stump_model = stump_model
            self.submodel_yes = None
            self.submodel_no = None
            return

        # Fit a decision tree to each split, decreasing maximum depth by 1
        j = stump_model.j_best
        value = stump_model.t_best

        # Find indices of examples in each split
        yes = X[:, j] > value
        no = X[:, j] <= value

        # Fit decision tree to each split
        self.stump_model = stump_model
        self.submodel_yes = DecisionTree(
            self.max_depth - 1, stump_class=self.stump_class
        )
        self.submodel_yes.fit(X[yes], y[yes])
        self.submodel_no = DecisionTree(
            self.max_depth - 1, stump_class=self.stump_class
        )
        self.submodel_no.fit(X[no], y[no])

    def predict(self, X):
        n, d = X.shape
        y = np.zeros(n)

        # GET VALUES FROM MODEL
        j_best = self.stump_model.j_best
        t_best = self.stump_model.t_best
        y_hat_yes = self.stump_model.y_hat_yes

        if j_best is None:
            # If no further splitting, return the majority label
            y = y_hat_yes * np.ones(n)

        # the case with depth=1, just a single stump.
        elif self.submodel_yes is None:
            return self.stump_model.predict(X)

        else:
            # Recurse on both sub-models
            j = j_best
            value = t_best

            yes = X[:, j] > value
            no = X[:, j] <= value

            y[yes] = self.submodel_yes.predict(X[yes])
            y[no] = self.submodel_no.predict(X[no])

        return y
