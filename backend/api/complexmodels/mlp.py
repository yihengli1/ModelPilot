from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict, Any

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler


def _activation(name: str) -> nn.Module:
    n = (name or "relu").lower()
    if n == "relu":
        return nn.ReLU()
    if n == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    if n == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


class _MLPNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: List[int],
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()

        layers: List[nn.Module] = []
        prev = input_dim

        act = _activation(activation)
        p = float(dropout or 0.0)

        for h in hidden_layers:
            h = int(h)
            layers.append(nn.Linear(prev, h))
            layers.append(act.__class__())  # new instance
            if p > 0:
                layers.append(nn.Dropout(p))
            prev = h

        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _to_tensor(x: np.ndarray, dtype: torch.dtype, device: str) -> torch.Tensor:
    return torch.as_tensor(x, dtype=dtype, device=device)


@dataclass
class _FitStats:
    epochs_trained: int = 0
    best_val_loss: Optional[float] = None


class MLPBaseTorchNN:
    """
    sklearn-like wrapper:
      - fit(X, y, X_val=None, y_val=None)
      - predict(X)

    Also stores artifacts for serialize_artifact().
    """

    def __init__(
        self,
        hidden_layers: Optional[List[int]] = None,
        activation: str = "relu",
        dropout: float = 0.0,
        optimizer: str = "adam",
        learning_rate: float = 1e-3,
        epochs: int = 300,
        batch_size: int = 64,
        weight_decay: float = 0.0,
        random_state: int = 42,
        patience: int = 15,
        device: Optional[str] = None,
    ):
        self.hidden_layers = hidden_layers if hidden_layers is not None else [
            128, 64]
        self.activation = activation
        self.dropout = float(dropout or 0.0)

        self.optimizer = (optimizer or "adam").lower()
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)

        self.weight_decay = float(weight_decay or 0.0)
        self.random_state = int(random_state)
        self.patience = int(patience)

        self.device = device or _default_device()

        # learned objects
        self.scaler_: Optional[StandardScaler] = None
        self.model_: Optional[nn.Module] = None

        # artifacts
        self.input_dim_: Optional[int] = None
        self.output_dim_: Optional[int] = None
        self.n_params_: Optional[int] = None
        self.fit_stats_ = _FitStats()

        self.is_fit_ = False

    def _build_optimizer(self, params):
        if self.optimizer == "sgd":
            return torch.optim.SGD(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        if self.optimizer == "adam":
            return torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        raise ValueError(f"Unsupported optimizer: {self.optimizer}")

    def _count_params(self) -> int:
        assert self.model_ is not None
        return int(sum(p.numel() for p in self.model_.parameters() if p.requires_grad))

    def _prep_X(self, X: np.ndarray, fit: bool) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if fit or self.scaler_ is None:
            self.scaler_ = StandardScaler()
            return self.scaler_.fit_transform(X).astype(np.float32, copy=False)
        return self.scaler_.transform(X).astype(np.float32, copy=False)

    # --- override these in child classes ---
    def _infer_output_dim_and_loss(self, y: np.ndarray) -> Tuple[int, nn.Module]:
        raise NotImplementedError

    def _predict_from_logits(self, logits: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit(self, X, y, X_val=None, y_val=None):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        Xn = self._prep_X(X, fit=True)
        y = np.asarray(y)

        self.input_dim_ = int(Xn.shape[1])
        out_dim, loss_fn = self._infer_output_dim_and_loss(y)
        self.output_dim_ = int(out_dim)

        self.model_ = _MLPNet(
            input_dim=self.input_dim_,
            output_dim=self.output_dim_,
            hidden_layers=list(self.hidden_layers),
            activation=self.activation,
            dropout=self.dropout,
        ).to(self.device)

        self.n_params_ = self._count_params()
        opt = self._build_optimizer(self.model_.parameters())

        # training loader
        Xt = _to_tensor(Xn, torch.float32, self.device)

        # dtype: classifier uses long, regressor uses float handled in subclass
        yt = self._y_tensor(y)

        ds = TensorDataset(Xt, yt)
        bs = max(1, min(self.batch_size, len(ds)))
        dl = DataLoader(ds, batch_size=bs, shuffle=True)

        # optional val
        has_val = X_val is not None and y_val is not None and len(
            np.asarray(y_val)) > 0
        if has_val:
            Xv = self._prep_X(X_val, fit=False)
            Xvt = _to_tensor(Xv, torch.float32, self.device)
            yvt = self._y_tensor(np.asarray(y_val))
        else:
            Xvt = yvt = None

        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_val = None
        no_improve = 0

        self.model_.train()
        for epoch in range(self.epochs):
            epoch_losses = []

            for xb, yb in dl:
                opt.zero_grad(set_to_none=True)
                logits = self.model_(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
                epoch_losses.append(float(loss.detach().cpu().item()))

            # val loss for early stopping
            val_loss = None
            if has_val:
                self.model_.eval()
                with torch.no_grad():
                    vlogits = self.model_(Xvt)
                    vloss = loss_fn(vlogits, yvt)
                    val_loss = float(vloss.detach().cpu().item())
                self.model_.train()

                if best_val is None or val_loss < best_val - 1e-6:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in self.model_.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.patience:
                        # stop early
                        self.fit_stats_.epochs_trained = epoch + 1
                        self.fit_stats_.best_val_loss = best_val
                        break

            self.fit_stats_.epochs_trained = epoch + 1
            self.fit_stats_.best_val_loss = best_val

        # restore best if we had val
        if best_state is not None:
            self.model_.load_state_dict(
                {k: v.to(self.device) for k, v in best_state.items()})

        self.is_fit_ = True
        return self

    def predict(self, X):
        if not self.is_fit_ or self.model_ is None:
            raise ValueError("Model is not fit yet.")

        Xn = self._prep_X(X, fit=False)
        Xt = _to_tensor(Xn, torch.float32, self.device)

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(Xt).detach().cpu().numpy()

        return self._predict_from_logits(logits)

    def _y_tensor(self, y: np.ndarray) -> torch.Tensor:
        raise NotImplementedError


class MLPClassifierTorchNN(MLPBaseTorchNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss = "cross_entropy"
        self.n_classes_: Optional[int] = None
        self.classes_: Optional[np.ndarray] = None
        self._class_to_index: Optional[dict] = None

    def _py(self, v):
        try:
            return v.item()
        except Exception:
            return v

    def _setup_label_map(self, y: np.ndarray):
        y = np.asarray(y).reshape(-1)
        classes = np.unique(y)
        self.classes_ = classes
        self.n_classes_ = int(len(classes))
        self._class_to_index = {self._py(c): i for i, c in enumerate(classes)}

    def _encode_y(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y).reshape(-1)
        if self._class_to_index is None or self.classes_ is None:
            self._setup_label_map(y)

        try:
            if (
                np.issubdtype(self.classes_.dtype, np.number)
                and np.issubdtype(y.dtype, np.number)
            ):
                idx = np.searchsorted(self.classes_, y)
                if idx.size > 0 and not np.all(self.classes_[idx] == y):
                    raise ValueError("Found unknown label in y.")
                return idx.astype(np.int64, copy=False)
        except Exception:
            pass

        return np.asarray([self._class_to_index[self._py(v)] for v in y], dtype=np.int64)

    def _infer_output_dim_and_loss(self, y: np.ndarray) -> Tuple[int, nn.Module]:
        self._setup_label_map(y)
        if self.n_classes_ is None or self.n_classes_ < 2:
            raise ValueError("Classification requires at least 2 classes.")
        return int(self.n_classes_), nn.CrossEntropyLoss()

    def _y_tensor(self, y: np.ndarray) -> torch.Tensor:
        y_idx = self._encode_y(y)
        return _to_tensor(y_idx, torch.long, self.device)

    def _predict_from_logits(self, logits: np.ndarray) -> np.ndarray:
        pred_idx = np.asarray(np.argmax(logits, axis=1), dtype=np.int64)
        if self.classes_ is None:
            return pred_idx
        return self.classes_[pred_idx]


class MLPRegressorTorchNN(MLPBaseTorchNN):
    def __init__(self, loss: str = "l2", **kwargs):
        super().__init__(**kwargs)
        self.loss = (loss or "l2").lower()

    def _infer_output_dim_and_loss(self, y: np.ndarray) -> Tuple[int, nn.Module]:
        # output dim 1
        if self.loss == "l1":
            loss_fn = nn.L1Loss()
        elif self.loss == "huber":
            loss_fn = nn.SmoothL1Loss()
        else:
            loss_fn = nn.MSELoss()
        return 1, loss_fn

    def _y_tensor(self, y: np.ndarray) -> torch.Tensor:
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        return _to_tensor(y, torch.float32, self.device)

    def _predict_from_logits(self, logits: np.ndarray) -> np.ndarray:
        # logits shape: [N, 1]
        return np.asarray(logits.reshape(-1), dtype=np.float32)
