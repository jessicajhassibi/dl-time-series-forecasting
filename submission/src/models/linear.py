"""Linear baseline implementation."""

import torch


class LinearModel(torch.nn.Module):
    """Linear Forecasting Model with Reversible Instance Normalization.
    This implementation does not use features for prediction and only works on the main target column.
    """

    def __init__(self, context_size: int, prediction_horizon: int, use_rev_in: bool = True,
                 eps: float = 1e-7) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_features=context_size, out_features=prediction_horizon, bias=False)
        self.gamma = torch.nn.Parameter(torch.ones(1))
        self.beta = torch.nn.Parameter(torch.zeros(1))
        self.use_rev_in = use_rev_in
        self.eps = eps

    def forward(self, x: torch.Tensor,
                x_features: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.use_rev_in:
            std, mean = torch.std_mean(x, dim=-1, keepdim=True, unbiased=False)
            x = (self.gamma + self.eps) * (x - mean) / (std + self.eps) + self.beta
        x = self.linear(x)
        if self.use_rev_in:
            x = (x - self.beta) * (std + self.eps) / (self.gamma + self.eps) + mean
        return x, None


class LinearModelWithFeatures(torch.nn.Module):
    def __init__(self, context_size: int, prediction_horizon: int, n_features: int,
                 use_rev_in: bool = True, eps: float = 1e-7):
        super().__init__()
        self.column_linear = LinearModel(context_size, prediction_horizon, use_rev_in, eps)
        self.row_linear = torch.nn.Linear(in_features=n_features + 1, out_features=n_features + 1, bias=False)

    def forward(self, x: torch.Tensor, x_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        # combine x and x_features together into one tensor
        columns = torch.cat([x[:, :, None], x_features], dim=-1)
        batch_size, size, n_cols = columns.shape

        # apply linear layer to all columns
        column_output, _ = self.column_linear(columns.permute(0, 2, 1).reshape(-1, size))
        column_output = column_output.reshape(batch_size, n_cols, -1).permute(0, 2, 1)

        # apply another linear layer to all rows to get final prediction
        row_output = self.row_linear(column_output)

        y = row_output[:, :, 0]
        y_features = row_output[:, :, 1:]

        return y, y_features
