from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm


class Chomp1d(nn.Module):
    """Trim right padding so each conv stays causal."""

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Residual block: two weight-normed dilated causal convs, each with ReLU + dropout."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation))
        self.net = nn.Sequential(
            self.conv1, Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
            self.conv2, Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNDeep(nn.Module):
    """Single-shot TCN forecaster with RevIN on the target and per-instance feature normalization."""

    def __init__(self, prediction_horizon: int, num_features: int,
                 hidden: int = 64, levels: int = 7, kernel_size: int = 3,
                 dropout: float = 0.1, use_rev_in: bool = True, eps: float = 1e-7) -> None:
        super().__init__()
        self.use_rev_in = use_rev_in
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

        in_channels = num_features + 1
        layers, ch_in = [], in_channels
        for i in range(levels):
            layers.append(TemporalBlock(ch_in, hidden, kernel_size, dilation=2 ** i, dropout=dropout))
            ch_in = hidden
        self.tcn = nn.Sequential(*layers)
        # pooled readout: last-step + mean-pool + max-pool over time
        self.head = nn.Linear(hidden * 3, prediction_horizon)

    def forward(self, x: torch.Tensor,
                x_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.use_rev_in:
            std, mean = torch.std_mean(x, dim=-1, keepdim=True, unbiased=False)
            x_norm = (self.gamma + self.eps) * (x - mean) / (std + self.eps) + self.beta
        else:
            x_norm = x

        # normalize each feature channel per-instance over time
        f_std, f_mean = torch.std_mean(x_features, dim=1, keepdim=True, unbiased=False)
        x_feat_norm = (x_features - f_mean) / (f_std + self.eps)

        seq = torch.cat([x_norm[:, :, None], x_feat_norm], dim=-1).permute(0, 2, 1)  # [B, C, L]
        h = self.tcn(seq)                                                            # [B, hidden, L]

        summary = torch.cat([h[:, :, -1], h.mean(dim=-1), h.max(dim=-1).values], dim=-1)  # [B, 3*hidden]
        y = self.head(summary)                                                       # [B, horizon]

        if self.use_rev_in:
            y = (y - self.beta) * (std + self.eps) / (self.gamma + self.eps) + mean
        return y, None