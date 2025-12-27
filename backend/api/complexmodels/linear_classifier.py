import numpy as np
import torch

from .regularizers import reg_penalty_linear


class LinearClassifierTorchNN:
    """
    Unified linear classifier via torch.nn.Linear + torch.optim.

    loss:
      - "logistic"  (binary + multiclass)
      - "hinge"     (binary only; margin=1)

    optimizer: "sgd" | "adam"
    regularization: "none" | "l2" | "l1"
    """

    def __init__(
        self,
        loss,
        optimizer: str = "sgd",
        learning_rate: float = 1e-3,
        epochs: int = 200,
        batch_size: int | None = None,   # SGD=1, GD=n, minibatch=oteher

        regularization: str = "none",
        alpha: float = 0.0,
    ):
        self.loss = (loss or "logistic").lower()
        if self.loss not in ("logistic", "hinge"):
            raise ValueError("loss must be one of ['logistic','hinge']")

        self.optimizer_name = optimizer.lower()
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = None if batch_size in (None, 0) else int(batch_size)

        self.regularization = (regularization or "none").lower()
        self.alpha = float(alpha)

        self.random_state = 42

        self.x_mean_ = None
        self.x_std_ = None

        self.layer: torch.nn.Linear | None = None

        self._n_classes = None

        # ARTIFCATS
        self.classes_ = None
        self.coef_ = None
        self.intercept_ = None

    def _as_torch(self, X):
        return torch.tensor(np.asarray(X, dtype=float), dtype=torch.float32)

    def fit(self, X, y):
        torch.manual_seed(self.random_state)

        X_t = self._as_torch(X)
        y_np = np.asarray(y)

        if X_t.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features).")
        n, d = X_t.shape

        if not torch.isfinite(X_t).all():
            raise ValueError("X contains NaN/Inf.")
        if not np.isfinite(y_np).all() and y_np.dtype.kind in "fc":
            raise ValueError("y contains NaN/Inf.")

        classes, y_idx = np.unique(y_np, return_inverse=True)
        self.classes_ = classes
        self._n_classes = int(classes.shape[0])

        if self.loss == "hinge" and self._n_classes != 2:
            raise ValueError(
                "hinge loss currently supports binary classification only.")

        # standardize
        self.x_mean_ = X_t.mean(dim=0, keepdim=True)
        self.x_std_ = X_t.std(dim=0, keepdim=True)
        self.x_std_[self.x_std_ == 0] = 1.0
        Xs = (X_t - self.x_mean_) / self.x_std_

        if self.loss == "hinge":
            out_dim = 1
        else:
            out_dim = 1 if self._n_classes == 2 else self._n_classes

        self.layer = torch.nn.Linear(d, out_dim, bias=True)

        if self.optimizer_name == "sgd":
            opt = torch.optim.SGD(self.layer.parameters(),
                                  lr=self.learning_rate)
        elif self.optimizer_name == "adam":
            opt = torch.optim.Adam(
                self.layer.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

        bs = self.batch_size or n

        y_idx_t = torch.tensor(y_idx, dtype=torch.long)

        for _ in range(max(self.epochs, 1)):
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                Xb = Xs[idx]
                logits = self.layer(Xb)

                if self.loss == "logistic":
                    if self._n_classes == 2:
                        yb = y_idx_t[idx].float().unsqueeze(1)
                        data_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            logits, yb)
                    else:
                        yb = y_idx_t[idx]
                        data_loss = torch.nn.functional.cross_entropy(
                            logits, yb)

                else:  # hinge (binary)
                    # map {0,1} -> {-1,+1}
                    yb = y_idx_t[idx].float().unsqueeze(1)
                    yb = 2.0 * yb - 1.0
                    # hinge: max(0, 1 - y * f(x))
                    data_loss = torch.relu(1.0 - yb * logits).mean()

                loss = data_loss

                if self.regularization in ("l1", "l2"):
                    loss = loss + \
                        reg_penalty_linear(
                            self.layer, self.regularization, self.alpha)

                if not torch.isfinite(loss):
                    raise ValueError(
                        "Diverged: loss became NaN/Inf. Try smaller learning_rate.")

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        # export params
        w = self.layer.weight.detach().cpu().numpy()
        b = self.layer.bias.detach().cpu().numpy()

        if self._n_classes == 2:
            self.coef_ = w.reshape(-1)
            self.intercept_ = float(b.reshape(-1)[0])
        else:
            self.coef_ = w
            self.intercept_ = b
        return self

    def decision_function(self, X):
        if self.layer is None:
            raise ValueError("Model is not fit yet.")
        X_t = self._as_torch(X)
        Xs = (X_t - self.x_mean_) / self.x_std_
        with torch.no_grad():
            logits = self.layer(Xs).detach().cpu().numpy()
        return logits.reshape(-1) if logits.shape[1] == 1 else logits

    def predict_proba(self, X):
        if self.loss != "logistic":
            raise ValueError(
                "predict_proba is only available for logistic loss.")
        logits = self.decision_function(X)
        if self._n_classes == 2:
            p1 = 1.0 / (1.0 + np.exp(-logits))
            p0 = 1.0 - p1
            return np.stack([p0, p1], axis=1)
        ex = np.exp(logits - logits.max(axis=1, keepdims=True))
        return ex / ex.sum(axis=1, keepdims=True)

    def predict(self, X):
        logits = self.decision_function(X)
        if self._n_classes == 2:
            # logistic: threshold at 0; hinge: sign at 0
            yhat01 = (logits >= 0.0).astype(int)
            return self.classes_[yhat01]
        idx = np.argmax(logits, axis=1)
        return self.classes_[idx]
