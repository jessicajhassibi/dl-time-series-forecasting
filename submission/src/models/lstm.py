import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_features, hidden_size, num_layers, output_steps=24):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

        self.fc = nn.Linear(hidden_size, output_steps)

    def forward(self, x, x_features):
        print("x, x_features")
        print(x.shape)
        print(x[0])
        print("x_features")
        print(x_features.shape)
        print(x_features[0])

        eps = 1e-7

        std, mean = torch.std_mean(x, dim=1, keepdim=True, unbiased=False)
        x_norm = (self.gamma + eps) * (x - mean) / (std + eps) + self.beta

        f_std, f_mean = torch.std_mean(x_features, dim=1, keepdim=True, unbiased=False)
        x_feat_norm = (x_features - f_mean) / (f_std + eps)

        if x.dim() == 2:
            x = x.unsqueeze(-1)

        combined_x = torch.cat((x_norm.unsqueeze(-1), x_feat_norm), dim=-1)

        lstm_out, _ = self.lstm(combined_x)
        last_step_out = lstm_out[:, -1, :]

        y = self.fc(last_step_out)
        predictions = (y - self.beta) * (std + eps) / (
            self.gamma + eps
        ) + mean

        return predictions, None
