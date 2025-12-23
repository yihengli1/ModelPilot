import torch


class LinearRegressionGD:
    """Simple linear regression trained with (mini-)batch gradient descent.

    Supports pure loss functions (no regularization):
    - l2: mean squared error
    - l1: mean absolute error
    - huber: smooth L1 with delta
    """

    def __init__(
        self,
        loss,
        batch_size: int | None = None,
        huber_delta: float = 1.0,
        fit_intercept: bool = True,
        random_state: int = 42,
        learning_rate: float = 0.01,
        epochs: int = 500,
    ):
        self.loss = loss
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = None if batch_size in (None, 0) else int(batch_size)
        self.huber_delta = float(huber_delta)
        self.fit_intercept = bool(fit_intercept)
        self.random_state = int(random_state)

        # w
        self.coef_ = None
        # b
        self.intercept_ = 0.0

    def _as_torch(self, X, y=None):
        import numpy as np
        if isinstance(X, torch.Tensor):
            X_t = X.detach().float()
        else:
            X_t = torch.tensor(np.asarray(
                X, dtype=float), dtype=torch.float32)

        if y is None:
            return X_t

        if isinstance(y, torch.Tensor):
            y_t = y.detach().float()
        else:
            y_t = torch.tensor(np.asarray(
                y, dtype=float), dtype=torch.float32)

        if y_t.ndim == 1:
            y_t = y_t.unsqueeze(1)
        return X_t, y_t

    def _loss_fn(self, preds: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = preds - y
        if self.loss in ("l2", "mse", "squared", "squared_error"):
            return (diff ** 2).mean()
        if self.loss in ("l1", "mae", "absolute"):
            return diff.abs().mean()
        if self.loss in ("huber", "smooth_l1"):
            d = self.huber_delta
            abs_diff = diff.abs()
            quad = torch.minimum(
                abs_diff, torch.tensor(d, device=diff.device))
            lin = abs_diff - quad
            return (0.5 * quad ** 2 + d * lin).mean()
        raise ValueError("Loss function not supported: ", self.loss)

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        X_t, y_t = self._as_torch(X, y)
        n, d = X_t.shape

        w = torch.zeros((d, 1), dtype=torch.float32,
                        requires_grad=True)
        b = torch.zeros((1,), dtype=torch.float32,
                        requires_grad=True) if self.fit_intercept else None

        bs = self.batch_size or n
        lr = self.learning_rate

        for _ in range(max(self.epochs, 1)):
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                idx = perm[i:i+bs]
                Xb = X_t[idx]
                yb = y_t[idx]

                preds = Xb @ w
                if b is not None:
                    preds = preds + b

                loss = self._loss_fn(preds, yb)
                loss.backward()

                # untrack
                with torch.no_grad():
                    w -= lr * w.grad
                    w.grad.zero_()
                    if b is not None:
                        b -= lr * b.grad
                        b.grad.zero_()

            self.coef_ = w.detach().cpu().numpy().reshape(-1)
            self.intercept_ = float(b.detach().cpu().numpy()[
                                    0]) if b is not None else 0.0
        return self

    def predict(self, X):
        X_t = self._as_torch(X)
        w = torch.tensor(self.coef_, dtype=torch.float32).unsqueeze(1)
        preds = X_t @ w
        if self.fit_intercept:
            preds = preds + float(self.intercept_)
        return preds.detach().cpu().numpy().reshape(-1)
