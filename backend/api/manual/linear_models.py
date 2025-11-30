import numpy as np

"""
Manual classes from school assignment lol
"""


class LinearModel:
    """
    Generic linear model, supporting generic loss functions (FunObj subclasses)
    and optimizers.

    See optimizers.py for optimizers.
    See fun_obj.py for loss function objects, which must implement evaluate()
    and return f and g values corresponding to current parameters.
    """

    def __init__(self, loss_fn, optimizer, check_correctness=False):
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.bias_yes = True
        self.check_correctness = check_correctness

        # For debugging and making learning curves
        self.fs = []
        self.nonzeros = []
        self.ws = []

    def optimize(self, w_init, X, y):
        "Perform gradient descent using the optimizer."
        n, d = X.shape

        # Initial guess
        w = np.copy(w_init)
        f, g = self.loss_fn.evaluate(w, X, y)

        # Reset the optimizer state and tie it to the new parameters.
        # See optimizers.py for why reset() is useful here.
        self.optimizer.reset()
        self.optimizer.set_fun_obj(self.loss_fn)
        self.optimizer.set_parameters(w)
        self.optimizer.set_fun_obj_args(X, y)

        # Collect training information for debugging
        fs = [f]
        gs = [g]
        ws = []

        # Use gradient descent to optimize w
        while True:
            f, g, w, break_yes = self.optimizer.step()
            fs.append(f)
            gs.append(g)
            ws.append(w)
            if break_yes:
                break

        return w, fs, gs, ws

    def fit(self, X, y):
        """
        Generic fitting subroutine:
        1. Make initial guess
        2. Check correctness of function object
        3. Use gradient descent to optimize
        """
        n, d = X.shape

        # Correctness check
        if self.check_correctness:
            w = np.random.rand(d)
            self.loss_fn.check_correctness(w, X, y)

        # Initial guess
        w = np.zeros(d)

        # Optimize
        self.w, self.fs, self.gs, self.ws = self.optimize(w, X, y)

    def predict(self, X):
        """
        By default, implement linear regression prediction
        """
        return X @ self.w


class LinearClassifier(LinearModel):
    def predict(self, X_pred):
        return np.sign(X_pred @ self.w)


class MulticlassLinearClassifier(LinearClassifier):
    """
    LinearClassifier's extention for multiclass classification.
    The constructor method and optimize() are inherited
    """

    def fit(self, X, y):
        n, d = X.shape
        k = len(np.unique(y))
        w_init = np.zeros(k * d)

        if self.check_correctness:
            self.loss_fn.check_correctness(np.random.randn(k * d), X, y)

        w, fs, gs, ws = self.optimize(w_init, X, y)
        W = w.reshape([k, d])
        self.W = W

    def predict(self, X_hat):
        return np.argmax(X_hat @ self.W.T, axis=1)


class LinearModelMultiOutput:
    """
    To be used and orchestrated by a multi-layer perceptron.
    """

    def __init__(self, input_dim, output_dim, scale=1e-1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = scale * np.random.randn(
            output_dim, input_dim
        )  # by convention we will right-multiply by W.T
        self.b = scale * np.random.randn(output_dim)

    @property
    def size(self):
        return self.get_parameters_flattened().size

    def set_weights_and_biases(self, w):
        """
        Take a flattened vector w and take it as weights and biases
        """
        assert w.size == self.W.size + self.b.size
        weights_flat = w[: -self.b.size]
        biases_flat = w[-self.b.size:]
        self.W = weights_flat.reshape(self.W.shape)
        self.b = biases_flat.reshape(self.b.shape)

    def get_parameters_flattened(self):
        return np.concatenate([self.W.reshape(-1), self.b.reshape(-1)])

    def predict(self, X):
        return X @ self.W.T + self.b
