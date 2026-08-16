import torch
from torch import nn


class Seq2SeqEncoderDecoder(nn.Module):
    def __init__(self, encoder_input_dim, decoder_input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.encoder = nn.LSTM(
            input_size=encoder_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        
        self.decoder = nn.LSTM(
            input_size=decoder_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, history_x, future_x, teacher_forcing_ratio=0.5):
        """
        history_x: [Batch, history_length (168), encoder_input_dim]
        future_x: [Batch, forecast_horizon (24), future_covariate_dim]
        """
        batch_size = history_x.size(0)
        forecast_horizon = future_x.size(1)
        
        _, (hidden, cell) = self.encoder(history_x)
        
        decoder_input_dim = future_x.size(2) + 1
        decoder_inputs = torch.zeros(batch_size, forecast_horizon, decoder_input_dim, device=history_x.device)
        
        decoder_inputs[:, :, 1:] = future_x
        
        outputs = torch.zeros(batch_size, forecast_horizon, 1, device=history_x.device)
        
        last_target = history_x[:, -1, 0:1]
        
        for t in range(forecast_horizon):
            decoder_inputs[:, t, 0:1] = last_target
            
            decoder_out, (hidden, cell) = self.decoder(decoder_inputs[:, t:t+1, :], (hidden, cell))
            
            step_output = self.fc_out(decoder_out)
            outputs[:, t:t+1, :] = step_output
            
            if self.training and torch.rand(1).item() < teacher_forcing_ratio:
                pass 
            else:
                last_target = step_output
                
        return outputs