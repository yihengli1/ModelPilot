import numpy as np
from utils import euclidean_dist_squared

"""
Manual classes from school assignment lol
"""


class Kernel:
    def __init__(self, multiplier=1e-1):
        """
        NumPy's multipliciations can give numerical instability, so
        I introduce a multiplier to make the numbers a few orders of magnitude smaller.
        The effect of the multiplier can be reversed in the linear model by increasing the weights.
        """
        self.multiplier = multiplier

    def evaluate(self, X1, X2):
        """
        Evaluate Gram's kernel matrix based on the two input matrices.
        Shape of X1 is (n1, d) and shape of X2 is (n2, d).
        That is, both matrices should have the same number of columns.
        Will return a n2-by-n1 matrix, e.g. X1 @ X2.T
        """
        raise NotImplementedError()


class KernelLinear(Kernel):
    def evaluate(self, X1, X2):
        return self.multiplier * (X1 @ X2.T)


class KernelPolynomial(Kernel):
    def __init__(self, p, multiplier=1e-1):
        """
        p is the degree of the polynomial
        """
        super().__init__(multiplier=multiplier)
        self.p = p

    def evaluate(self, X1, X2):
        """
        Evaluate the polynomial kernel.
        A naive implementation will use change of basis.
        A "kernel trick" implementation bypasses change of basis.
        """

        """YOUR CODE HERE FOR Q1.1"""
        # raise NotImplementedError()
        return self.multiplier * (1.0 + (X1 @ X2.T)) ** self.p


class KernelGaussianRBF(Kernel):
    def __init__(self, sigma, multiplier=1e-1):
        """
        sigma is the curve width hyperparameter.
        """
        super().__init__(multiplier=multiplier)
        self.sigma = sigma

    def evaluate(self, X1, X2):
        """
        Evaluate Gaussian RBF basis kernel.
        """

        """YOUR CODE HERE FOR Q1.1"""
        # raise NotImplementedError()
        n1, d = X1.shape
        n2, _ = X2.shape
        D = euclidean_dist_squared(X1, X2)
        return np.exp(-D / (2 * self.sigma ** 2))
