import torch


class LinearModel(torch.nn.Module):
    """Linear Forecasting Model with Reversible Instance Normalization."""

    def __init__(self, context_size: int, prediction_horizon: int = 1, use_rev_in: bool = True) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_features=context_size, out_features=prediction_horizon, bias=False)
        self.gamma = torch.nn.Parameter(torch.ones(1))
        self.beta = torch.nn.Parameter(torch.zeros(1))
        self.use_rev_in = use_rev_in

    def forward(self, x: torch.Tensor, features: torch.Tensor | None = None, eps: float = 1e-7) -> torch.Tensor:
        if self.use_rev_in:
            std, mean = torch.std_mean(x, dim=-1, keepdim=True, unbiased=False)
            x = (self.gamma + eps) * (x - mean) / (std + eps) + self.beta
        x = self.linear(x)
        if self.use_rev_in:
            x = (x - self.beta) * (std + eps) / (self.gamma + eps) + mean
        return x
