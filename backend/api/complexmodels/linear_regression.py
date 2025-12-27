import numpy as np
import torch

from .regularizers import reg_penalty_linear


class LinearRegressionTorchNN:
    """
    Linear regression via torch.nn.Linear + torch.optim.

    loss: "l2"(MSE) | "l1"(MAE) | "huber"
    optimizer: "sgd" | "adam"
    regularization: "none" | "l2" | "l1"
    """

    def __init__(
        self,
        loss,
        optimizer: str = "sgd",
        learning_rate: float = 1e-3,
        epochs: int = 500,
        batch_size: int | None = 1,   # SGD=1, GD=n, minibatch=oteher
        huber_delta: float = 1.0,

        regularization: str = "none",
        alpha: float = 0.0,
    ):
        self.loss = (loss or "l2").lower()
        self.optimizer_name = (optimizer or "sgd").lower()
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = None if batch_size in (None, 0) else int(batch_size)
        self.huber_delta = float(huber_delta)
        self.random_state = 42  # Change for debugging

        self.regularization = (regularization or "none").lower()
        self.alpha = float(alpha)

        self.coef_ = None
        self.intercept_ = 0.0

        self.x_mean_ = None
        self.x_std_ = None

        self.layer: torch.nn.Linear | None = None

    def _as_torch(self, X, y=None):
        X_t = torch.tensor(np.asarray(X, dtype=float), dtype=torch.float32)
        if y is None:
            return X_t
        y_t = torch.tensor(np.asarray(y, dtype=float), dtype=torch.float32)
        if y_t.ndim == 1:
            y_t = y_t.unsqueeze(1)
        return X_t, y_t

    def _data_loss(self, preds: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = preds - y
        if self.loss in ("l2"):
            return (diff ** 2).mean()
        if self.loss in ("l1"):
            return diff.abs().mean()
        if self.loss in ("huber"):
            d = self.huber_delta
            ad = diff.abs()
            q = torch.minimum(ad, torch.tensor(d, device=diff.device))
            lin = ad - q
            return (0.5 * q**2 + d * lin).mean()
        raise ValueError(f"Unsupported loss: {self.loss}")

    def fit(self, X, y):
        torch.manual_seed(self.random_state)

        X_t, y_t = self._as_torch(X, y)
        n, d = X_t.shape

        if not torch.isfinite(X_t).all() or not torch.isfinite(y_t).all():
            raise ValueError("X or y contains NaN/Inf.")

        # standardize features
        self.x_mean_ = X_t.mean(dim=0, keepdim=True)
        self.x_std_ = X_t.std(dim=0, keepdim=True)
        self.x_std_[self.x_std_ == 0] = 1.0
        X_t = (X_t - self.x_mean_) / self.x_std_

        # nn.Linear
        self.layer = torch.nn.Linear(d, 1, bias=True)

        if self.optimizer_name == "sgd":
            opt = torch.optim.SGD(self.layer.parameters(
            ), lr=self.learning_rate)
        elif self.optimizer_name == "adam":
            opt = torch.optim.Adam(self.layer.parameters(
            ), lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

        bs = self.batch_size or n
        for _ in range(max(self.epochs, 1)):
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                Xb = X_t[idx]
                yb = y_t[idx]

                preds = self.layer(Xb)
                loss = self._data_loss(preds, yb)

                if self.regularization in ("l1", "l2"):
                    loss = loss + reg_penalty_linear(
                        self.layer, self.regularization, self.alpha)

                if not torch.isfinite(loss):
                    raise ValueError(
                        "Diverged: loss became NaN/Inf. Try scaling or smaller learning_rate.")

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        w = self.layer.weight.detach().cpu().numpy().reshape(-1)
        self.coef_ = w
        self.intercept_ = float(self.layer.bias.detach().cpu().numpy()[0])

        return self

    def predict(self, X):
        if self.layer is None:
            raise ValueError("Model is not fit yet.")

        X_t = self._as_torch(X)
        X_t = (X_t - self.x_mean_) / self.x_std_

        with torch.no_grad():
            preds = self.layer(X_t).detach().cpu().numpy().reshape(-1)
        return preds
