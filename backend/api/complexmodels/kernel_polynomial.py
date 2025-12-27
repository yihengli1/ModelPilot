import numpy as np
import torch


class KernelPolynomialTorch:
    """
    Kernel Ridge Regression with polynomial kernel:
        K(x,z) = (gamma * x^T z + coef0) ^ degree

    For small n, we do a closed-form solve:
        (K + lam I) alpha = y
    """

    def __init__(self, degree=3, gamma=None, coef0=1.0, lam=1e-3, dtype=torch.float32):
        self.degree = int(degree)
        self.gamma = gamma  # None => 1/d
        self.coef0 = float(coef0)
        self.lam = float(lam)
        self.dtype = dtype

        self.x_mean_ = None
        self.x_std_ = None
        self.X_train_ = None
        self.alpha_ = None

    def _as_torch(self, X, y=None):
        X_t = torch.tensor(np.asarray(X, dtype=float), dtype=self.dtype)
        if y is None:
            return X_t
        y_t = torch.tensor(np.asarray(y, dtype=float),
                           dtype=self.dtype).reshape(-1, 1)
        return X_t, y_t

    def _standardize_fit(self, X):
        self.x_mean_ = X.mean(dim=0, keepdim=True)
        self.x_std_ = X.std(dim=0, keepdim=True)
        self.x_std_[self.x_std_ == 0] = 1.0
        return (X - self.x_mean_) / self.x_std_

    def _standardize_apply(self, X):
        return (X - self.x_mean_) / self.x_std_

    def _poly_kernel(self, A, B):
        d = A.shape[1]
        gamma = (1.0 / d) if self.gamma is None else float(self.gamma)
        return (gamma * (A @ B.T) + self.coef0) ** self.degree

    def fit(self, X, y):
        X_t, y_t = self._as_torch(X, y)
        n, d = X_t.shape

        if not torch.isfinite(X_t).all() or not torch.isfinite(y_t).all():
            raise ValueError("X or y contains NaN/Inf.")
        if self.lam < 0:
            raise ValueError("lam must be >= 0.")

        Xs = self._standardize_fit(X_t)
        K = self._poly_kernel(Xs, Xs)  # (n, n)

        A = K + self.lam * torch.eye(n, dtype=K.dtype, device=K.device)

        # solve (K + lam I) alpha = y
        try:
            alpha = torch.linalg.solve(A, y_t)  # (n, 1)
        except RuntimeError:
            jitter = 1e-10
            alpha = torch.linalg.solve(
                A + jitter * torch.eye(n, dtype=K.dtype, device=K.device), y_t)

        self.X_train_ = Xs
        self.alpha_ = alpha
        return self

    def predict(self, X):
        if self.alpha_ is None or self.X_train_ is None:
            raise ValueError("Model is not fit yet.")

        X_t = self._as_torch(X)
        if not torch.isfinite(X_t).all():
            raise ValueError("X contains NaN/Inf.")

        Xs = self._standardize_apply(X_t)
        K_test = self._poly_kernel(Xs, self.X_train_)  # (n_test, n_train)
        yhat = K_test @ self.alpha_  # (n_test, 1)
        return yhat.detach().cpu().numpy().reshape(-1)
