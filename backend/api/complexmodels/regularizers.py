import torch


def reg_penalty_linear(layer: torch.nn.Linear, reg: str, alpha: float) -> torch.Tensor:
    if alpha <= 0 or reg in ("none", "", None):
        return torch.tensor(0.0, device=layer.weight.device)

    reg = reg.lower()
    w = layer.weight
    if reg == "l1":
        return alpha * w.abs().sum()

    if reg == "l2":
        return 0.5 * alpha * (w ** 2).sum()

    raise ValueError(f"Unknown regularization: {reg}")
