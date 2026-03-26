"""
model.py
--------
Definições do Dataset e do modelo LSTM em PyTorch para
previsão univariada de séries temporais financeiras.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """
    Dataset PyTorch para séries temporais.

    Recebe um array 1D de valores escalados e cria janelas deslizantes:
        X[i] = data[i : i + seq_len]   → shape (seq_len, 1)
        y[i] = data[i + seq_len]        → scalar

    Parâmetros
    ----------
    data : np.ndarray
        Array 1D de floats (valores já normalizados pelo MinMaxScaler).
    seq_len : int
        Número de passos de tempo usados como entrada (janela de contexto).
    """

    def __init__(self, data: np.ndarray, seq_len: int = 30) -> None:
        if len(data) <= seq_len:
            raise ValueError(
                f"Array de dados ({len(data)}) deve ser maior que seq_len ({seq_len})."
            )

        n_samples = len(data) - seq_len

        # Constrói X e y como arrays numpy antes de converter para tensor
        X = np.array([data[i : i + seq_len] for i in range(n_samples)], dtype=np.float32)
        y = np.array([data[i + seq_len] for i in range(n_samples)], dtype=np.float32)

        # X: (n_samples, seq_len, 1) — dimensão extra para input_size=1
        self.X = torch.from_numpy(X).unsqueeze(-1)   # (n, seq_len, 1)
        self.y = torch.from_numpy(y)                  # (n,)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    """
    Modelo LSTM para previsão univariada de um passo à frente.

    Arquitetura:
        LSTM (empilhado, batch_first=True)
          → Dropout
          → Linear

    Parâmetros
    ----------
    input_size : int
        Número de features de entrada por passo (1 para univariado).
    hidden_size : int
        Número de unidades ocultas em cada camada LSTM.
    num_layers : int
        Número de camadas LSTM empilhadas.
    dropout : float
        Taxa de dropout aplicada entre camadas LSTM e antes da camada Linear.
    output_size : int
        Número de valores previstos (1 para previsão de 1 passo).
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # nn.LSTM com dropout=0 quando num_layers==1 para evitar UserWarning
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,       # entrada: (batch, seq_len, input_size)
            dropout=lstm_dropout,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passagem direta pelo modelo.

        Parâmetros
        ----------
        x : torch.Tensor de shape (batch, seq_len, input_size)

        Retorna
        -------
        torch.Tensor de shape (batch, output_size)
        """
        # out: (batch, seq_len, hidden_size)
        # h_n: (num_layers, batch, hidden_size)
        out, (h_n, _) = self.lstm(x)

        # Usa o hidden state da última camada no último passo de tempo
        last_hidden = h_n[-1]              # (batch, hidden_size)
        last_hidden = self.dropout(last_hidden)
        return self.fc(last_hidden)        # (batch, output_size)
