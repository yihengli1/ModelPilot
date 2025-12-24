import torch


class Regularizer:
    def penalty(self):
        return torch.tensor(0.0)


class NoRegularization(Regularizer):
    pass


class L2Regularization(Regularizer):
    def __init__(self, alpha: float = 0.0, include_bias: bool = False):
        self.alpha = float(alpha)
        self.include_bias = include_bias

    def penalty(self, params):
        if self.alpha <= 0:
            return torch.tensor(0.0, device=next(iter(params.values())).device)

        w = params["w"]
        p = 0.5 * self.alpha * (w ** 2).sum()
        if self.include_bias and "b" in params and params["b"] is not None:
            p = p + 0.5 * self.alpha * (params["b"] ** 2).sum()
        return p


class L1Regularization(Regularizer):
    def __init__(self, alpha: float = 0.0, include_bias: bool = False):
        self.alpha = float(alpha)
        self.include_bias = include_bias

    def penalty(self, params):
        if self.alpha <= 0:
            return torch.tensor(0.0, device=next(iter(params.values())).device)

        w = params["w"]
        p = self.alpha * w.abs().sum()
        if self.include_bias and "b" in params and params["b"] is not None:
            p = p + self.alpha * params["b"].abs().sum()
        return p
