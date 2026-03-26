"""
train.py
--------
Funções para preparação de dados, treinamento do modelo LSTM,
avaliação no conjunto de teste e previsão multi-step para 2025.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from model import LSTMModel, TimeSeriesDataset


# ─────────────────────────────────────────────────────────────────────────────
# 1. Preparação dos dados
# ─────────────────────────────────────────────────────────────────────────────

def prepare_data(
    series: np.ndarray,
    seq_len: int = 30,
    train_ratio: float = 0.8,
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader, MinMaxScaler, np.ndarray, np.ndarray]:
    """
    Divide a série em treino/teste, normaliza e cria DataLoaders.

    IMPORTANTE: o MinMaxScaler é fitado APENAS nos dados de treino
    para evitar data leakage (vazamento de informação futura).

    Parâmetros
    ----------
    series : np.ndarray
        Array 1D com os preços brutos (sem normalização).
    seq_len : int
        Tamanho da janela temporal (dias de contexto).
    train_ratio : float
        Fração dos dados usada para treino (0.8 = 80%).
    batch_size : int
        Tamanho do mini-batch para os DataLoaders.

    Retorna
    -------
    (train_loader, test_loader, scaler, train_raw, test_raw)
    """
    if len(series) < seq_len * 2 + 2:
        raise ValueError(
            f"Série muito curta ({len(series)}). Mínimo: {seq_len * 2 + 2} pontos."
        )

    series_2d = series.reshape(-1, 1)
    split_idx = int(len(series_2d) * train_ratio)

    train_raw = series_2d[:split_idx]
    test_raw  = series_2d[split_idx:]

    # Scaler fitado somente no treino
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_raw)

    train_scaled = scaler.transform(train_raw).flatten()
    test_scaled  = scaler.transform(test_raw).flatten()

    train_ds = TimeSeriesDataset(train_scaled, seq_len)
    test_ds  = TimeSeriesDataset(test_scaled,  seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, test_loader, scaler, train_raw, test_raw


# ─────────────────────────────────────────────────────────────────────────────
# 2. Treinamento
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    model: LSTMModel,
    train_loader: DataLoader,
    epochs: int = 100,
    lr: float = 0.001,
    device: str = "cpu",
) -> list[float]:
    """
    Treina o modelo LSTM usando MSELoss e otimizador Adam.

    Inclui gradient clipping (max_norm=1.0) para estabilizar o treinamento
    com redes recorrentes, onde gradientes explodem com frequência.

    Parâmetros
    ----------
    model : LSTMModel
        Instância do modelo (não inicializado no dispositivo ainda).
    train_loader : DataLoader
        DataLoader com os dados de treino.
    epochs : int
        Número de épocas de treinamento.
    lr : float
        Taxa de aprendizado do Adam.
    device : str
        'cpu' ou 'cuda'.

    Retorna
    -------
    list[float]
        Histórico de loss médio por época.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loss_history: list[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: list[float] = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)                        # (batch, 1)
            loss  = criterion(preds.squeeze(), y_batch)   # escalar
            loss.backward()

            # Clipping de gradiente — essencial para LSTM
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_losses.append(loss.item())

        mean_loss = float(np.mean(epoch_losses))
        loss_history.append(mean_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Época [{epoch:3d}/{epochs}]  Loss: {mean_loss:.6f}")

    return loss_history


# ─────────────────────────────────────────────────────────────────────────────
# 3. Avaliação
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model: LSTMModel,
    test_loader: DataLoader,
    scaler: MinMaxScaler,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """
    Avalia o modelo no conjunto de teste e calcula métricas.

    Parâmetros
    ----------
    model : LSTMModel
        Modelo treinado.
    test_loader : DataLoader
        DataLoader com os dados de teste (shuffle=False).
    scaler : MinMaxScaler
        Scaler usado no pré-processamento (para inverter a normalização).
    device : str
        'cpu' ou 'cuda'.

    Retorna
    -------
    (preds_real, actuals_real, metrics)
        Previsões e valores reais na escala original + dicionário de métricas.
    """
    model.eval()
    all_preds:   list[float] = []
    all_actuals: list[float] = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).squeeze().cpu().numpy()
            actuals = y_batch.numpy()

            # .squeeze() pode retornar escalar quando batch_size=1
            all_preds.extend(np.atleast_1d(preds).tolist())
            all_actuals.extend(np.atleast_1d(actuals).tolist())

    preds_arr   = np.array(all_preds,   dtype=np.float32)
    actuals_arr = np.array(all_actuals, dtype=np.float32)

    # Inverter a normalização
    preds_real   = scaler.inverse_transform(preds_arr.reshape(-1, 1)).flatten()
    actuals_real = scaler.inverse_transform(actuals_arr.reshape(-1, 1)).flatten()

    # Métricas
    rmse = float(np.sqrt(np.mean((preds_real - actuals_real) ** 2)))
    mae  = float(np.mean(np.abs(preds_real - actuals_real)))

    # MAPE com proteção contra divisão por zero
    nonzero = actuals_real != 0
    mape = float(
        np.mean(np.abs((actuals_real[nonzero] - preds_real[nonzero]) / actuals_real[nonzero])) * 100
    ) if nonzero.any() else float("nan")

    metrics = {"RMSE": rmse, "MAE": mae, "MAPE": mape}
    return preds_real, actuals_real, metrics


# ─────────────────────────────────────────────────────────────────────────────
# 4. Previsão para 2025 (multi-step recursivo)
# ─────────────────────────────────────────────────────────────────────────────

def forecast_2025(
    model: LSTMModel,
    last_sequence: np.ndarray,
    scaler: MinMaxScaler,
    steps: int = 252,
    device: str = "cpu",
) -> np.ndarray:
    """
    Realiza previsão multi-step recursiva a partir do final de 2024.

    A cada passo:
      1. O modelo prevê o próximo valor com base na janela atual.
      2. O valor previsto é adicionado ao final da janela.
      3. O primeiro valor da janela é descartado (sliding window).

    O clipping em [0, 1] é aplicado a cada previsão para manter os valores
    dentro do range válido do MinMaxScaler e evitar drift exponencial.

    Parâmetros
    ----------
    model : LSTMModel
        Modelo treinado.
    last_sequence : np.ndarray
        Array 1D de shape (seq_len,) com os últimos valores escalados de 2024.
    scaler : MinMaxScaler
        Scaler para inverter a normalização dos valores previstos.
    steps : int
        Número de passos futuros a prever (~252 pregões em 2025).
    device : str
        'cpu' ou 'cuda'.

    Retorna
    -------
    np.ndarray
        Array de shape (steps,) com os preços previstos na escala original.
    """
    model.eval()
    current_seq = last_sequence.copy().astype(np.float32)
    future_scaled: list[float] = []

    with torch.no_grad():
        for _ in range(steps):
            # Tensor: (1, seq_len, 1) — float32 obrigatório para LSTM
            x = torch.tensor(current_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
            pred = model(x).item()

            # Clipa para manter dentro do espaço [0, 1] do scaler
            pred = float(np.clip(pred, 0.0, 1.0))
            future_scaled.append(pred)

            # Desliza a janela: remove primeiro, adiciona previsão
            current_seq = np.append(current_seq[1:], pred)

    future_arr = np.array(future_scaled, dtype=np.float32).reshape(-1, 1)
    future_real = scaler.inverse_transform(future_arr).flatten()
    return future_real
