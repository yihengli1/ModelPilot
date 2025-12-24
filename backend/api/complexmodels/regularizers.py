import torch


def reg_penalty_linear(layer: torch.nn.Linear, reg: str, alpha: float, l1_ratio: float = 0.5) -> torch.Tensor:
    # L2 is through tensor.optim (weight decay)
    if alpha <= 0 or reg in ("none", "", None):
        return torch.tensor(0.0, device=layer.weight.device)

    reg = reg.lower()
    w = layer.weight
    if reg == "l1":
        return alpha * w.abs().sum()

    if reg == "elasticnet":
        r = min(max(l1_ratio, 0.0), 1.0)
        l1 = w.abs().sum()
        l2 = 0.5 * (w ** 2).sum()
        return alpha * (r * l1 + (1 - r) * l2)

    raise ValueError(f"Unknown regularization: {reg}")
