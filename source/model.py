import torch
import torch.nn as nn


class LSTMBlock(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(LSTMBlock, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.forget_gate = nn.Sequential(
            nn.Linear(input_size + output_size, output_size),
            nn.Sigmoid()
        )

        self.input_gate = nn.Sequential(
            nn.Linear(input_size + output_size, output_size),
            nn.Sigmoid()
        )
        self.cell_input = nn.Sequential(
            nn.Linear(input_size + output_size, output_size),
            nn.Tanh()
        )

        self.output_gate = nn.Sequential(
            nn.Linear(input_size + output_size, output_size),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        short_term_memory: torch.Tensor,
        long_term_memory: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of a single LSTM block.
        
        Args:
            x (torch.Tensor): input tensor
            short_term_memory (torch.Tensor): short term hidden state
            long_term_memory (torch.Tensor): long term hidden state
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: 
                updated hidden state, updated long term memory
        """
        combined = torch.cat([x, short_term_memory], dim=1)

        forget_weights = self.forget_gate(combined)
        long_term_memory = forget_weights * long_term_memory

        input_weights = self.input_gate(combined)
        input_values = input_weights * self.cell_input(combined)

        long_term_memory = long_term_memory + input_values

        output_weights = self.output_gate(combined)
        output_values = output_weights * torch.tanh(long_term_memory)

        return output_values, long_term_memory


class MyLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int):
        assert num_layers >= 1
        super(MyLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        layers = []
        if num_layers == 1:
            layers = [LSTMBlock(input_size, output_size)]
        else:
            layers.append(LSTMBlock(input_size, hidden_size))
            for i in range(num_layers - 2):
                layers.append(LSTMBlock(hidden_size, hidden_size))
            layers.append(LSTMBlock(hidden_size, output_size))
        self.layers = nn.ModuleList(layers)

    
    def forward(self, x: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): in [batch, seq_len, features]
            dim (int): axis of seq_len
        """
        batch_size, seq_len, _ = x.shape
        short_terms = [torch.zeros(batch_size, layer.output_size).to(x.device) for layer in self.layers]
        long_terms = [torch.zeros(batch_size, layer.output_size).to(x.device) for layer in self.layers]
        output_xs = []

        for i in range(seq_len):
            current_x = x[:, i, :]
            for j, layer in enumerate(self.layers):
                current_short_term, current_long_term = layer(current_x, short_terms[j], long_terms[j])
                short_terms[j] = current_short_term
                long_terms[j] = current_long_term
                current_x = current_short_term  # Pass the output to the next layer
            output_xs.append(short_terms[-1])

        return torch.stack(output_xs, dim=dim)

class Model(nn.Module):
    def __init__(self, num_features: int, input_length: int, output_length: int):
        super(Model, self).__init__()
        self.input_length = input_length
        self.output_length = output_length
        self.lstm = MyLSTM(num_features, num_features, 1, num_layers=5)
        self.feedforward = nn.Linear(input_length, output_length)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lstm(x)
        x = x.squeeze()
        return self.feedforward(x)