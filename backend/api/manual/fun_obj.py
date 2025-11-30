import numpy as np
from numpy.linalg.linalg import norm
from scipy.optimize.optimize import approx_fprime
from scipy.special import logsumexp

from utils import ensure_1d, euclidean_dist_squared, shortest_dist

"""
Manual classes from school assignment lol
"""

# Neural network helpers


def flatten_weights(weights):
    return np.concatenate([w.flatten() for w in sum(weights, ())])


def unflatten_weights(weights_flat, layer_sizes):
    weights = list()
    counter = 0
    for i in range(len(layer_sizes) - 1):
        W_size = layer_sizes[i + 1] * layer_sizes[i]
        b_size = layer_sizes[i + 1]

        W = np.reshape(
            weights_flat[counter: counter + W_size],
            (layer_sizes[i + 1], layer_sizes[i]),
        )
        counter += W_size

        b = weights_flat[counter: counter + b_size][None]
        counter += b_size

        weights.append((W, b))
    return weights


def log_sum_exp(Z):
    Z_max = np.max(Z, axis=1)
    # per-colmumn max
    return Z_max + np.log(np.sum(np.exp(Z - Z_max[:, None]), axis=1))


class FunObj:
    """
    Function object for encapsulating evaluations of functions and gradients
    """

    def evaluate(self, w, X, y):
        """
        Evaluates the function AND its gradient w.r.t. w.
        Returns the numerical values based on the input.
        IMPORTANT: w is assumed to be a 1d-array, hence shaping will have to be handled.
        """
        raise NotImplementedError

    def check_correctness(self, w, *args, epsilon=1e-6):
        # *args usually X,y but not always
        estimated_gradient = approx_fprime(
            w, lambda w: self.evaluate(w, *args)[0], epsilon=epsilon
        )
        _, implemented_gradient = self.evaluate(w, *args)
        difference = estimated_gradient - implemented_gradient

        # Check the gradient
        if np.max(np.abs(difference)) / np.linalg.norm(estimated_gradient) > 1e-6:
            raise Exception(
                "User and numerical derivatives differ:\n%s\n%s"
                % (estimated_gradient[:], implemented_gradient[:])
            )
        else:
            print("User and numerical derivatives agree.")


class LeastSquaresLoss(FunObj):
    def evaluate(self, w, X, y):
        """
        Evaluates the function and gradient of least squares objective.
        Least squares objective is half the sum of squared residuals.
        """
        # help avoid mistakes by potentially reshaping our arguments
        w = ensure_1d(w)
        y = ensure_1d(y)

        y_hat = X @ w
        m_residuals = y_hat - y  # minus residuals, slightly more convenient here

        # Loss is sum of squared residuals
        f = 0.5 * np.sum(m_residuals ** 2)

        # The gradient, derived mathematically then implemented here
        g = X.T @ m_residuals  # X^T X w - X^T y

        return f, g


class LeastSquaresLossL2(LeastSquaresLoss):
    def __init__(self, lammy):
        self.lammy = lammy

    def evaluate(self, w, X, y):
        f_base, g_base = super().evaluate(w, X, y)
        f = f_base + self.lammy / 2 * (w @ w)
        g = g_base + self.lammy * w
        return f, g


class MultiOutputLeastSquaresLoss(FunObj):
    def evaluate(self, w, X, Y):
        """
        Generalizes least squares error using Frobenius norm.
        Y is now an n-by-k matrix. Hence, W is a k-by-d matrix.
        """
        n, d = X.shape
        n2, k = Y.shape
        W = w.reshape(k, d)

        y_hat = X @ W.T
        m_residuals = y_hat - Y

        f = 0.5 * np.sum(m_residuals ** 2)
        g = (X.T @ m_residuals).T  # k by d
        return f, g.flatten()


class MultiOutputLeastSquaresLossL2(MultiOutputLeastSquaresLoss):
    def __init__(self, lammy):
        self.lammy = lammy

    def evaluate(self, w, X, Y):
        f_base, g_base = super().evaluate(w, X, Y)
        f = f_base + self.lammy / 2 * (w @ w)
        g = g_base + self.lammy * w
        return f, g


class RobustRegressionLoss(FunObj):
    def evaluate(self, w, X, y):
        """
        Evaluates the function and gradient of ROBUST least squares objective.
        """
        # help avoid mistakes by potentially reshaping our arguments
        w = ensure_1d(w)
        y = ensure_1d(y)

        y_hat = X @ w
        residuals = y - y_hat
        exp_residuals = np.exp(residuals)
        exp_minuses = np.exp(-residuals)

        f = np.sum(np.log(exp_minuses + exp_residuals))

        # s is the negative of the "soft sign"
        s = (exp_minuses - exp_residuals) / (exp_minuses + exp_residuals)
        g = X.T @ s

        return f, g


class LogisticRegressionLoss(FunObj):
    def evaluate(self, w, X, y):
        """
        Evaluates the function and gradient of logistics regression objective.
        """
        # help avoid mistakes by potentially reshaping our arguments
        w = ensure_1d(w)
        y = ensure_1d(y)

        Xw = X @ w
        yXw = y * Xw  # element-wise multiply; the y_i are in {-1, 1}

        # Calculate the function value
        # logaddexp(a, b) = log(exp(a) + exp(b)), but more numerically stable
        f = np.logaddexp(0, -yXw).sum()

        # Calculate the gradient value
        with np.errstate(over="ignore"):  # overflowing here is okay: we get 0
            g_bits = -y / (1 + np.exp(yXw))
        g = X.T @ g_bits

        return f, g


class LogisticRegressionLossL2(LogisticRegressionLoss):
    def __init__(self, lammy):
        super().__init__()
        self.lammy = lammy

    def evaluate(self, w, X, y):
        # we sure did copy-paste this a bunch of times;
        # it'd be better to define some kind of generic L2 regularization
        # class and allow for adding function objects together,
        # but oh well.
        w = ensure_1d(w)
        y = ensure_1d(y)

        f_base, g_base = super().evaluate(w, X, y)
        f = f_base + (self.lammy / 2) * (w @ w)
        g = g_base + self.lammy * w
        return f, g


class KernelLogisticRegressionLoss(FunObj):
    def __init__(self, lammy):
        self.lammy = lammy

    def evaluate(self, u, K, y):
        """
        Here u is the length-n vector defining our linear combination in
        the (potentially infinite-dimensional) Z space,
        and K is the Gram matrix K[i, i'] = k(x_i, x_{i'}).

        Note the L2 regularizer is in the transformed space too, not on u.
        """
        u = ensure_1d(u)
        y = ensure_1d(y)

        yKu = y * (K @ u)

        f = np.logaddexp(0, -yKu).sum() + (self.lammy / 2) * u @ K @ u

        with np.errstate(over="ignore"):  # overflowing here is okay: we get 0
            g_bits = -y / (1 + np.exp(yKu))
        g = K @ g_bits + self.lammy * K @ u

        return f, g


class LogisticRegressionLossL0(FunObj):
    def __init__(self, lammy):
        self.lammy = lammy

    def evaluate(self, w, X, y):
        """
        Evaluates the function value of of L0-regularized logistics regression
        objective.
        """
        w = ensure_1d(w)
        y = ensure_1d(y)

        Xw = X @ w
        yXw = y * Xw  # element-wise multiply; the y should be in {-1, 1}

        # Calculate the function value
        f = np.logaddexp(0, -yXw).sum() + self.lammy * np.sum(w != 0)

        # We cannot differentiate the "length" function
        g = None
        return f, g


class SoftmaxLoss(FunObj):
    def evaluate(self, w, X, y):
        w = ensure_1d(w)
        y = ensure_1d(y)

        n, d = X.shape
        k = len(np.unique(y))

        W = w.reshape(k, d)

        # n by k; XW[i, c] is the dot product between example i and class c weights
        XW = X @ W.T

        # n by k: logsumexp_XW[i] = log( sum_c exp(x_i^T w_c) )
        logsumexp_XW = logsumexp(XW, axis=1)

        # n by k; p[i, c] = p(y_i = c | W, x_i)
        # we compute this "in log space" for numerical stability:
        # we never do an exp() of something positive,
        # and the only very-negative things we exp() are the ones that don't matter
        p = np.exp(XW - logsumexp_XW[:, np.newaxis])

        # n by k; one_hots[i, c] is 1 if y_i = c, 0 otherwise (appears in gradient)
        one_hots = np.eye(k)[y]

        # now to compute the function value:
        # n; confusingly, : would NOT work in first index
        Xw_of_y = XW[np.arange(n), y]
        f = -Xw_of_y.sum() + logsumexp_XW.sum()  # scalar

        # einsum, "einstein summation", is a handy function to know:
        # this one gives
        #   G[c, j] = \sum_i X[i, j] (p - one_hots)[j, c]
        G = np.einsum('ij, ic -> cj', X, p - one_hots)
        g = G.reshape(-1)  # kd

        return f, g


class PCAFeaturesLoss(FunObj):
    """
    Evaluates PCA objective function and its gradient with Z, the learned features
    """

    def evaluate(self, z, W, X):
        n, d = X.shape
        k, _ = W.shape
        Z = z.reshape(n, k)

        R = Z @ W - X
        f = np.sum(R ** 2) / 2
        g = R @ W.T
        return f, g.flatten()


class PCAFactorsLoss(FunObj):
    """
    Evaluates PCA objective function and its gradient with W, the learned features
    """

    def evaluate(self, w, Z, X):
        n, d = X.shape
        _, k = Z.shape
        W = w.reshape(k, d)

        R = Z @ W - X
        f = np.sum(R ** 2) / 2
        g = Z.T @ R
        return f, g.flatten()


class CollaborativeFilteringZLoss(FunObj):
    def __init__(self, lammyZ=1, lammyW=1):
        self.lammyZ = lammyZ
        self.lammyW = lammyW

    def evaluate(self, z, W, Y):
        k, d = W.shape
        n, _ = Y.shape
        Z = z.reshape(n, k)

        mask = ~np.isnan(Y)
        Ynew = np.where(mask, Y, 0)

        R = Z @ W - Ynew
        R[~mask] = 0

        zReg = (self.lammyZ / 2) * np.sum(Z ** 2)
        print("F Norm of Z: ",  np.sum(Z ** 2))
        f = 0.5 * np.sum(R ** 2) + zReg

        g = R @ W.T + self.lammyZ * Z

        return f, g.flatten()


class CollaborativeFilteringWLoss(FunObj):
    def __init__(self, lammyZ=1, lammyW=1):
        self.lammyZ = lammyZ
        self.lammyW = lammyW

    def evaluate(self, w, Z, Y):
        n, k = Z.shape
        _, d = Y.shape
        W = w.reshape(k, d)

        mask = ~np.isnan(Y)
        Ynew = np.where(mask, Y, 0)

        R = Z @ W - Ynew
        R[~mask] = 0

        wReg = (self.lammyW / 2) * np.sum(W ** 2)
        print("F Norm of W: ", np.sum(W ** 2))

        f = 0.5 * np.sum(R ** 2) + wReg

        g = Z.T @ R + self.lammyW * W

        return f, g.flatten()


class RobustPCAFeaturesLoss(FunObj):
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def evaluate(self, z, W, X):
        n, d = X.shape
        k, _ = W.shape
        Z = z.reshape(n, k)

        R = Z @ W - X
        f = np.sum(np.sqrt(R ** 2 + self.epsilon))
        g = (R / (R ** 2 + self.epsilon) ** 0.5) @ W.T
        return f, g.flatten()


class RobustPCAFactorsLoss(FunObj):
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def evaluate(self, w, Z, X):
        n, d = X.shape
        _, k = Z.shape
        W = w.reshape(k, d)

        R = Z @ W - X
        f = np.sum(np.sqrt(R ** 2 + self.epsilon))
        g = Z.T @ (R / (R ** 2 + self.epsilon) ** 0.5)
        return f, g.flatten()


class MLPLoss(FunObj):  # (friendship is magic)
    """
    Function object for generic multi-layer perceptron
    (aka fully-connected artificial neural networks)
    Without automatic differentiation, function objects tend to get hairy because
    there's no straightfoward separation of hierarchy in terms of model behaviour
    and function/gradient calculation.
    """

    def __init__(self, encoder, predictor):
        self.encoder = encoder
        self.predictor = predictor

    def add_regularization(self, f, g):
        return f, g

    def get_final_layer_f_and_g(self, Z, y):
        raise NotImplementedError()

    def evaluate(self, w, X, y):
        n, d = X.shape
        _, k = y.shape

        # Parse weights and biases for the encoder and the predictor
        encoder_size = self.encoder.size
        w_encoder = w[:encoder_size]
        w_predictor = w[encoder_size:]

        self.encoder.set_weights_and_biases(w_encoder)
        self.predictor.set_weights_and_biases(w_predictor)

        # Use the encoder weights to produce Z
        Z, activations = self.encoder.encode(X)

        f, grad = self.get_final_layer_f_and_g(Z, y)

        # Backpropagate by recursion
        # Predictor phase
        grad_W = grad.T @ Z  # w.r.t predictor weights
        grad_b = np.sum(grad, axis=0)  # w.r.t. predictor biases

        g = [(grad_W, grad_b)]

        # Last encoder activation
        grad = grad @ self.predictor.W
        grad = grad * (Z * (1 - Z))  # gradient of sigmoid
        # grad = grad * (Z >= 0) # gradient of ReLU

        # Last encoder gradients
        grad_W = grad.T @ activations[-1]
        grad_b = np.sum(grad, axis=0)

        g = [(grad_W, grad_b)] + g  # insert to start of list

        # Penultimate encoder to first
        # goes till i=1.
        for i in range(len(self.encoder.encoders) - 1, 0, -1):
            encoder = self.encoder.encoders[i]
            grad = grad @ encoder.W
            # gradient of sigmoid
            grad = grad * (activations[i] * (1 - activations[i]))
            # grad = grad * (activations[i] >= 0) # gradient of ReLU
            grad_W = grad.T @ activations[i - 1]
            grad_b = np.sum(grad, axis=0)

            g = [(grad_W, grad_b)] + g  # insert to start of list

        g = flatten_weights(g)

        # add L2 regularization
        f += 0.5 * self.lammy * np.sum(w ** 2)
        g += self.lammy * w

        return f, g


class MLPLeastSquaresLossL2(MLPLoss):
    def __init__(self, encoder, predictor, lammy=1.0):
        super().__init__(encoder, predictor)
        self.lammy = lammy

    def get_final_layer_f_and_g(self, Z, y):
        # Use the predictor weights with bias to get prediction
        y_hat = self.predictor.predict(Z)

        f = 0.5 * np.sum((y_hat - y) ** 2)
        grad = y_hat - y  # gradient for L2 loss
        return f, grad


class MLPLogisticRegressionLossL2(MLPLoss):
    def __init__(self, encoder, predictor, lammy=1.0):
        super().__init__(encoder, predictor)
        self.lammy = lammy

    def get_final_layer_f_and_g(self, Z, y):
        y_hat = self.predictor.predict(Z)
        tmp = np.sum(np.exp(y_hat), axis=1)
        # f = -np.sum(yhat[y.astype(bool)] - np.log(tmp))
        f = -np.sum(y_hat[y.astype(bool)] - log_sum_exp(y_hat))
        grad = np.exp(y_hat) / tmp[:, None] - y
        return f, grad
