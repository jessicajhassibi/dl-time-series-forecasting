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

        self.fc = nn.Linear(hidden_size, output_steps)

    def forward(self, x, x_features):
        # If x is shape (Batch, 168), make it (Batch, 168, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
            
        # If x_features is shape (Batch, 168), make it (Batch, 168, 1)
        # (It shouldn't be 2D if it has 22 columns, but this prevents future crashes)
        if x_features.dim() == 2:
            x_features = x_features.unsqueeze(-1)
            
        # Now both are guaranteed to be 3D. 
        # You can safely concatenate along dimension 2 (the features)
        combined_x = torch.cat((x, x_features), dim=2)
        
        # Standard LSTM processing
        lstm_out, _ = self.lstm(combined_x)
        last_step_out = lstm_out[:, -1, :]
        predictions = self.fc(last_step_out)
        
        return predictions, None
