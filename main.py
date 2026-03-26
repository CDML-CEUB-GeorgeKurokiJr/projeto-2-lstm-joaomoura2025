"""
main.py
-------
Orquestrador principal do pipeline de análise e previsão de
séries temporais de commodities com LSTM.

Pipeline:
  1. Download de dados históricos (2019–2024) via yfinance
  2. Análise exploratória: séries, correlação, SMA
  3. Treinamento de LSTM independente para cada ativo
  4. Avaliação no conjunto de teste (RMSE, MAE, MAPE)
  5. Previsão multi-step para 2025

Execução:
  python main.py

Saída:
  - Métricas impressas no terminal
  - 13 gráficos PNG salvos em ./plots/
"""

from __future__ import annotations

import os
import random

import matplotlib
matplotlib.use("Agg")  # backend sem janela — graficos salvos como PNG em plots/
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from analysis import (
    plot_correlation_heatmap,
    plot_normalized_prices,
    plot_rolling_correlation,
    plot_scatter_pairs,
    plot_sma,
    plot_time_series,
    print_correlation_matrix,
)
from data_loader import BR_FACTOR, BR_UNIT, TICKERS, UNIT_LABELS, download_data
from model import LSTMModel
from train import evaluate_model, forecast_2025, prepare_data, train_model

# ─────────────────────────────────────────────────────────────────────────────
# Configurações globais
# ─────────────────────────────────────────────────────────────────────────────

SEED          = 42
START_DATE    = "2019-01-01"
END_DATE      = "2024-12-31"
SEQ_LEN       = 30      # Janela de contexto: 30 pregões (~6 semanas)
EPOCHS        = 100     # Épocas de treinamento
LR            = 0.001   # Taxa de aprendizado (Adam)
BATCH_SIZE    = 32      # Tamanho do mini-batch
TRAIN_RATIO   = 0.8     # 80% treino / 20% teste
HIDDEN_SIZE   = 64      # Unidades ocultas da LSTM
NUM_LAYERS    = 2       # Camadas LSTM empilhadas
DROPOUT       = 0.2     # Taxa de dropout
FORECAST_STEPS = 252    # Pregões previstos em 2025

PLOTS_DIR = "plots"


# ─────────────────────────────────────────────────────────────────────────────
# Reprodutibilidade
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = SEED) -> None:
    """Define seed global para reprodutibilidade total."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# Funções de plotagem específicas do main
# ─────────────────────────────────────────────────────────────────────────────

def _slugify(name: str) -> str:
    return name.lower().replace(" ", "_").replace("/", "_")


def plot_loss_curve(loss_history: list[float], asset_name: str) -> None:
    """Plota e salva a curva de loss do treinamento."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, len(loss_history) + 1), loss_history, color="#E91E63", linewidth=1.5)
    ax.set_title(f"Curva de Loss — {asset_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Época", fontsize=11)
    ax.set_ylabel("MSE Loss", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, f"loss_{_slugify(asset_name)}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Gráfico salvo: {path}")
    plt.close("all")


def plot_predictions(
    actuals: np.ndarray,
    preds: np.ndarray,
    test_index: pd.DatetimeIndex,
    asset_name: str,
) -> None:
    """Plota previsões vs. valores reais no conjunto de teste."""
    # Garante alinhamento de tamanho
    n = min(len(actuals), len(preds), len(test_index))
    actuals    = actuals[:n]
    preds      = preds[:n]
    test_index = test_index[:n]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(test_index, actuals, color="#2196F3", linewidth=1.2, label="Real",    zorder=3)
    ax.plot(test_index, preds,   color="#FF5722", linewidth=1.2, label="Previsto", zorder=4, linestyle="--")
    ax.set_title(f"LSTM — Previsão vs. Real no Teste: {asset_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Data", fontsize=11)
    ax.set_ylabel("Preço (USD)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=30)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, f"predictions_{_slugify(asset_name)}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Gráfico salvo: {path}")
    plt.close("all")


def plot_forecast_2025(
    future_values: np.ndarray,
    asset_name: str,
    last_known_series: pd.Series,
) -> None:
    """
    Plota a projeção futura para 2025 concatenada com o histórico de 2024.

    Exibe o último ano de dados reais + a previsão para 2025 no mesmo gráfico,
    facilitando a visualização da continuidade e tendência.
    """
    # Datas de pregão para 2025 (aproximação com dias úteis)
    future_dates = pd.bdate_range(start="2025-01-02", periods=len(future_values))

    fig, ax = plt.subplots(figsize=(14, 5))

    # Histórico de 2024 (último ano disponível)
    recent = last_known_series.loc["2024-01-01":]
    ax.plot(recent.index, recent.values, color="#2196F3", linewidth=1.4,
            label="Histórico 2024", zorder=3)

    # Linha de transição
    ax.axvline(x=pd.Timestamp("2025-01-01"), color="gray", linestyle=":", linewidth=1.2)
    ax.text(pd.Timestamp("2025-01-01"), ax.get_ylim()[0],
            " 2025 →", fontsize=9, color="gray", va="bottom")

    # Projeção 2025
    ax.plot(future_dates, future_values, color="#FF5722", linewidth=1.5,
            linestyle="--", label="Previsão 2025", zorder=4)

    # Banda de incerteza (±5% — ilustrativa, drift acumulado em multi-step)
    uncertainty = future_values * 0.05
    ax.fill_between(
        future_dates,
        future_values - uncertainty,
        future_values + uncertainty,
        alpha=0.15,
        color="#FF5722",
        label="Intervalo ±5%",
    )

    ax.set_title(f"Previsão LSTM para 2025 — {asset_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Data", fontsize=11)
    ax.set_ylabel("Preço (USD)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=30)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, f"forecast_2025_{_slugify(asset_name)}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Gráfico salvo: {path}")
    plt.close("all")


# ─────────────────────────────────────────────────────────────────────────────
# Análise final de correlação e limitações
# ─────────────────────────────────────────────────────────────────────────────

def print_final_analysis(df: pd.DataFrame, all_metrics: dict[str, dict[str, float]]) -> None:
    """Imprime analise final didatica: unidades, correlacoes em %, tendencias e limitacoes."""
    sep = "=" * 70

    # ── Unidades e precos atuais ───────────────────────────────────────────
    print("\n" + sep)
    print("ANALISE FINAL — COMMODITIES AGRICOLAS (2019-2024)")
    print(sep)

    print("\n[1] COTACOES FINAIS (dez/2024) — UNIDADES BRASILEIRAS")
    print("-" * 70)
    print(f"  {'Ativo':<12} {'Cotacao CBOT':>16}  {'Unidade CBOT':<22}  {'Equiv. BR':>14}  {'Unidade BR'}")
    print(f"  {'-'*12} {'-'*16}  {'-'*22}  {'-'*14}  {'-'*16}")
    for col in df.columns:
        preco_cbot = df[col].iloc[-1]
        fator      = BR_FACTOR.get(col, 1.0)
        preco_br   = preco_cbot * fator
        unidade_cbot = UNIT_LABELS.get(col, "")
        unidade_br   = BR_UNIT.get(col, "")
        print(f"  {col:<12} {preco_cbot:>16.2f}  {unidade_cbot:<22}  {preco_br:>14.2f}  {unidade_br}")

    # ── Correlacao entre ativos ───────────────────────────────────────────
    print("\n[2] CORRELACAO DE PEARSON ENTRE OS ATIVOS")
    print("-" * 70)
    print("  O coeficiente r varia de -1 a +1:")
    print("  r > 0.7  = correlacao FORTE positiva  (sobem e caem juntos)")
    print("  r 0.4-0.7 = correlacao MODERADA       (tendencia em comum)")
    print("  r < 0.4  = correlacao FRACA            (comportamento independente)")
    print()

    corr = df.corr(method="pearson")
    cols = list(df.columns)
    for i, col_a in enumerate(cols):
        for col_b in cols[i + 1:]:
            r   = corr.loc[col_a, col_b]
            r2  = r ** 2
            pct = abs(r) * 100
            if abs(r) >= 0.7:
                nivel = "FORTE"
                interp = f"{pct:.1f}% dos movimentos sao compartilhados"
            elif abs(r) >= 0.4:
                nivel = "MODERADA"
                interp = f"{pct:.1f}% de tendencia em comum"
            else:
                nivel = "FRACA"
                interp = f"pouca relacao direta ({pct:.1f}%)"
            direcao = "positiva" if r > 0 else "negativa"
            print(f"  {col_a} x {col_b}")
            print(f"    r = {r:+.4f}  |  r^2 = {r2:.4f}  |  correlacao {nivel} {direcao}")
            print(f"    Interpretacao: {interp}")
            if abs(r) >= 0.7:
                print(f"    -> Quando {col_a} SOBE, {col_b} tende a SUBIR tambem")
            elif abs(r) >= 0.4:
                print(f"    -> Ha uma tendencia moderada de co-movimento")
            else:
                print(f"    -> Os ativos se movem de forma relativamente independente")
            print()

    # ── Tendencias 2019-2024 ──────────────────────────────────────────────
    print("[3] TENDENCIAS DE PRECO (2019 vs 2024) — VARIACAO ACUMULADA")
    print("-" * 70)
    for col in df.columns:
        preco_ini  = df[col].iloc[:30].mean()
        preco_fim  = df[col].iloc[-30:].mean()
        var_pct    = (preco_fim - preco_ini) / preco_ini * 100
        seta       = "ALTA" if var_pct > 0 else "BAIXA"
        fator      = BR_FACTOR.get(col, 1.0)
        unidade_br = BR_UNIT.get(col, "")
        print(f"  {col}:")
        print(f"    Media inicio 2019 : {preco_ini:>8.2f} ({preco_ini * fator:.2f} {unidade_br})")
        print(f"    Media fim   2024  : {preco_fim:>8.2f} ({preco_fim * fator:.2f} {unidade_br})")
        print(f"    Variacao acumulada: {var_pct:>+.1f}%  [{seta}]")
        print()

    # ── Metricas do modelo ────────────────────────────────────────────────
    print("[4] METRICAS DO MODELO LSTM (conjunto de teste — 20% dos dados)")
    print("-" * 70)
    print(f"  {'Ativo':<12} {'RMSE':>10}  {'MAE':>10}  {'MAPE (%)':>10}")
    print(f"  {'-'*12} {'-'*10}  {'-'*10}  {'-'*10}")
    for col, m in all_metrics.items():
        fator      = BR_FACTOR.get(col, 1.0)
        unidade_br = BR_UNIT.get(col, "")
        print(f"  {col:<12} {m['RMSE']:>10.4f}  {m['MAE']:>10.4f}  {m['MAPE']:>9.2f}%")
        print(f"             RMSE equiv: {m['RMSE'] * fator:.3f} {unidade_br}")
    print()
    print("  RMSE = erro quadratico medio (mesma unidade do preco)")
    print("  MAE  = erro absoluto medio")
    print("  MAPE = erro percentual medio (quanto o modelo erra em %)")

    # ── Limitacoes ───────────────────────────────────────────────────────
    print("\n[5] LIMITACOES DO MODELO")
    print("-" * 70)
    limitacoes = [
        "Modelo UNIVARIADO: prevê cada ativo separadamente (sem cruzar dados).",
        "Previsao RECURSIVA: cada passo usa a previsao anterior — erros acumulam.",
        "Dados AMERICANOS (CME/CBOT): refletem cotacao em dolares, nao BRL.",
        "Sem fatores EXTERNOS: cambio, clima, estoques globais, politica agricola.",
        "Janela fixa 30 dias: pode nao capturar sazonalidade anual do agro.",
    ]
    for item in limitacoes:
        print(f"  - {item}")

    print("\n" + sep)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    set_seed(SEED)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device.upper()}\n")

    # ── 1. Coleta de dados ──────────────────────────────────────────────────
    print("=" * 65)
    print("ETAPA 1: COLETA DE DADOS")
    print("=" * 65)
    df = download_data(TICKERS, start=START_DATE, end=END_DATE)
    print(df.tail(3).to_string())

    # ── 2. Análise exploratória ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("ETAPA 2-4: ANÁLISE EXPLORATÓRIA + SMA")
    print("=" * 65)

    plot_time_series(df)
    print_correlation_matrix(df)

    # Gráficos de correlação explicativos para o professor
    plot_normalized_prices(df)        # todos no mesmo eixo — co-movimento visível
    plot_scatter_pairs(df)            # dispersão com regressão e r de Pearson
    plot_rolling_correlation(df, window=90)  # correlação ao longo do tempo

    plot_correlation_heatmap(df)
    plot_sma(df, windows=[20, 50])

    # ── 3-9. Pipeline LSTM por ativo ────────────────────────────────────────
    all_metrics: dict[str, dict[str, float]] = {}

    for asset_name in df.columns:
        print("\n" + "=" * 65)
        print(f"ETAPAS 5-9: LSTM — {asset_name}")
        print("=" * 65)

        series = df[asset_name].values  # array 1D de preços brutos

        # Preparação dos dados (sem leakage)
        train_loader, test_loader, scaler, _, _ = prepare_data(
            series,
            seq_len=SEQ_LEN,
            train_ratio=TRAIN_RATIO,
            batch_size=BATCH_SIZE,
        )

        # Re-seed antes de cada modelo para reprodutibilidade independente de ordem
        set_seed(SEED)

        # Instanciação do modelo
        model = LSTMModel(
            input_size=1,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            output_size=1,
        )
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parâmetros treináveis: {n_params:,}")

        # Treinamento
        print(f"\n  Treinando por {EPOCHS} épocas...")
        loss_history = train_model(model, train_loader, epochs=EPOCHS, lr=LR, device=device)
        plot_loss_curve(loss_history, asset_name)

        # Avaliação no teste
        preds, actuals, metrics = evaluate_model(model, test_loader, scaler, device=device)
        all_metrics[asset_name] = metrics

        print(f"\n  RMSE : {metrics['RMSE']:.4f}")
        print(f"  MAE  : {metrics['MAE']:.4f}")
        print(f"  MAPE : {metrics['MAPE']:.2f}%")

        # Índice temporal do conjunto de teste para plotagem
        split_idx = int(len(series) * TRAIN_RATIO)
        test_index = df.index[split_idx + SEQ_LEN:]
        plot_predictions(actuals, preds, test_index, asset_name)

        # Previsão para 2025 (multi-step recursivo)
        print(f"\n  Gerando previsão para 2025 ({FORECAST_STEPS} pregões)...")
        full_scaled = scaler.transform(series.reshape(-1, 1)).flatten()
        last_seq    = full_scaled[-SEQ_LEN:]
        future      = forecast_2025(model, last_seq, scaler, steps=FORECAST_STEPS, device=device)

        print(f"  Previsão mínima 2025: {future.min():.2f} | máxima: {future.max():.2f}")
        plot_forecast_2025(future, asset_name, df[asset_name])

    # ── 10. Análise final ───────────────────────────────────────────────────
    print_final_analysis(df, all_metrics)

    print(f"\nConcluído! Todos os gráficos foram salvos em ./{PLOTS_DIR}/")


if __name__ == "__main__":
    main()
