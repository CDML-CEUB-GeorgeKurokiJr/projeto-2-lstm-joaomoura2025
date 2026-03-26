"""
analysis.py
-----------
Análise exploratória de dados (EDA) para séries temporais de commodities.

Funções disponíveis:
  - plot_time_series          : Séries históricas de preços
  - plot_correlation_heatmap  : Heatmap de correlação de Pearson
  - plot_sma                  : Preços com médias móveis simples (SMA)
  - print_correlation_matrix  : Tabela de correlação + estatísticas descritivas
  - plot_normalized_prices    : Todos os ativos no mesmo eixo (base 100)
  - plot_scatter_pairs        : Dispersão entre pares com linha de regressão
  - plot_rolling_correlation  : Correlação móvel ao longo do tempo
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")  # backend sem janela — compativel com qualquer ambiente
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Importa metadados de unidades (usado nos labels dos gráficos)
try:
    from data_loader import BR_FACTOR, BR_UNIT, UNIT_LABELS
except ImportError:
    UNIT_LABELS = {}
    BR_UNIT     = {}
    BR_FACTOR   = {}

PLOTS_DIR = "plots"


def _ensure_plots_dir() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)


def _slugify(name: str) -> str:
    """Converte nome de ativo para formato de nome de arquivo seguro."""
    return name.lower().replace(" ", "_").replace("/", "_")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Séries temporais históricas
# ─────────────────────────────────────────────────────────────────────────────

def plot_time_series(df: pd.DataFrame, save: bool = True) -> None:
    """
    Plota as séries históricas de preços de cada ativo em subplots separados.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com datas como índice e ativos como colunas.
    save : bool
        Se True, salva o gráfico em plots/time_series.png.
    """
    if len(df) < 2:
        raise ValueError("DataFrame precisa ter pelo menos 2 linhas para plotar.")

    n = len(df.columns)
    colors = ["#2196F3", "#4CAF50", "#FF5722"]

    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=True)
    if n == 1:
        axes = [axes]

    fig.suptitle("Preços Históricos de Commodities (2019–2024)", fontsize=16, fontweight="bold", y=1.01)

    for ax, col, color in zip(axes, df.columns, colors):
        ax.plot(df.index, df[col], color=color, linewidth=1.2, label=col)
        unit   = UNIT_LABELS.get(col, "USD")
        br_u   = BR_UNIT.get(col, "")
        br_f   = BR_FACTOR.get(col, None)
        if br_f and br_u:
            ylabel = f"Preco ({unit})\nEquiv.: x{br_f:.4f} = {br_u}"
        else:
            ylabel = f"Preco ({unit})"
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(col, fontsize=12, fontweight="bold")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    axes[-1].set_xlabel("Data", fontsize=11)
    plt.tight_layout()

    if save:
        _ensure_plots_dir()
        path = os.path.join(PLOTS_DIR, "time_series.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Gráfico salvo: {path}")

    plt.close("all")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Heatmap de correlação de Pearson
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame, save: bool = True) -> None:
    """
    Gera e plota o heatmap de correlação de Pearson entre os ativos.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com os preços dos ativos.
    save : bool
        Se True, salva o gráfico em plots/correlation_heatmap.png.
    """
    if len(df) < 2:
        raise ValueError("DataFrame precisa ter pelo menos 2 linhas.")

    corr = df.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"size": 13, "weight": "bold"},
        ax=ax,
    )
    ax.set_title("Correlação de Pearson — Commodities (2019–2024)", fontsize=14, fontweight="bold", pad=15)
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()

    if save:
        _ensure_plots_dir()
        path = os.path.join(PLOTS_DIR, "correlation_heatmap.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Gráfico salvo: {path}")

    plt.close("all")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Médias Móveis Simples (SMA)
# ─────────────────────────────────────────────────────────────────────────────

def plot_sma(
    df: pd.DataFrame,
    windows: list[int] | None = None,
    save: bool = True,
) -> None:
    """
    Plota o preço de cada ativo junto com suas Médias Móveis Simples (SMA).

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com os preços dos ativos.
    windows : list[int], opcional
        Períodos para calcular as SMAs. Padrão: [20, 50].
    save : bool
        Se True, salva cada gráfico em plots/sma_{ativo}.png.
    """
    if windows is None:
        windows = [20, 50]

    if len(df) < 2:
        raise ValueError("DataFrame precisa ter pelo menos 2 linhas.")

    linestyles = ["-", "--", ":", "-."]
    colors_sma = ["#FF9800", "#9C27B0", "#00BCD4", "#795548"]

    for col in df.columns:
        fig, ax = plt.subplots(figsize=(14, 5))

        ax.plot(df.index, df[col], color="#2196F3", linewidth=1.0,
                label=f"{col} (Preço Real)", alpha=0.85, zorder=3)

        for i, window in enumerate(windows):
            sma = df[col].rolling(window=window).mean()
            ax.plot(
                df.index, sma,
                color=colors_sma[i % len(colors_sma)],
                linestyle=linestyles[(i + 1) % len(linestyles)],
                linewidth=1.8,
                label=f"SMA-{window}",
                zorder=4,
            )

        unit = UNIT_LABELS.get(col, "USD")
        br_u = BR_UNIT.get(col, "")
        subtitle = f"  (cotacao em {unit}" + (f" — 1 saca/@ = x{BR_FACTOR.get(col,1):.4f} {br_u})" if br_u else ")")
        ax.set_title(f"{col} — Preco Real vs. Medias Moveis\n{subtitle}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Data", fontsize=11)
        ax.set_ylabel(f"Preco ({unit})", fontsize=11)
        ax.legend(fontsize=10, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.tight_layout()

        if save:
            _ensure_plots_dir()
            path = os.path.join(PLOTS_DIR, f"sma_{_slugify(col)}.png")
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"  Gráfico salvo: {path}")

        plt.show()
        plt.close("all")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Preços normalizados (base 100) — co-movimento visual
# ─────────────────────────────────────────────────────────────────────────────

def plot_normalized_prices(df: pd.DataFrame, save: bool = True) -> None:
    """
    Plota todos os ativos no MESMO gráfico, normalizados a 100 na data inicial.

    Isso elimina diferenças de escala (ex: boi em ~150 USD vs milho em ~4 USD)
    e permite comparar visualmente se os ativos sobem e caem juntos.
    Quando as linhas se movem na mesma direção ao mesmo tempo, há co-movimento.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com preços dos ativos.
    save : bool
        Se True, salva em plots/normalized_prices.png.
    """
    if len(df) < 2:
        raise ValueError("DataFrame precisa ter pelo menos 2 linhas.")

    # Normaliza: divide cada série pelo seu primeiro valor e multiplica por 100
    df_norm = df.div(df.iloc[0]) * 100

    colors = ["#2196F3", "#4CAF50", "#FF5722"]
    fig, ax = plt.subplots(figsize=(14, 6))

    for col, color in zip(df_norm.columns, colors):
        ax.plot(df_norm.index, df_norm[col], color=color, linewidth=1.5, label=col)

    ax.axhline(y=100, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.set_title(
        "Co-movimento dos Ativos — Preços Normalizados (Base 100 em Jan/2019)\n"
        "Linhas se movendo juntas indicam correlação positiva",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Data", fontsize=11)
    ax.set_ylabel("Índice de Preço (Base = 100)", fontsize=11)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Anota variação total de cada ativo ao final
    for col, color in zip(df_norm.columns, colors):
        final_val = df_norm[col].iloc[-1]
        var = final_val - 100
        sinal = "+" if var >= 0 else ""
        ax.annotate(
            f"{col}: {sinal}{var:.0f}%",
            xy=(df_norm.index[-1], final_val),
            xytext=(10, 0), textcoords="offset points",
            fontsize=9, color=color, va="center",
        )

    plt.tight_layout()

    if save:
        _ensure_plots_dir()
        path = os.path.join(PLOTS_DIR, "normalized_prices.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Gráfico salvo: {path}")

    plt.close("all")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Dispersão entre pares (scatter plots com regressão)
# ─────────────────────────────────────────────────────────────────────────────

def plot_scatter_pairs(df: pd.DataFrame, save: bool = True) -> None:
    """
    Gera scatter plots entre todos os pares de ativos com linha de regressão.

    Cada ponto representa um dia. A inclinação da reta indica a direção da
    relação: positiva (ambos sobem juntos) ou negativa (um sobe, outro cai).
    O coeficiente r de Pearson e o p-valor são exibidos no gráfico.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com preços dos ativos.
    save : bool
        Se True, salva em plots/scatter_pairs.png.
    """
    if len(df) < 2:
        raise ValueError("DataFrame precisa ter pelo menos 2 linhas.")

    cols = list(df.columns)
    # Gera todos os pares únicos
    pairs = [(cols[i], cols[j]) for i in range(len(cols)) for j in range(i + 1, len(cols))]
    n_pairs = len(pairs)

    fig, axes = plt.subplots(1, n_pairs, figsize=(6 * n_pairs, 5))
    if n_pairs == 1:
        axes = [axes]

    fig.suptitle(
        "Dispersão entre Pares de Commodities\n"
        "Tendência positiva = quando um sobe, o outro também sobe",
        fontsize=13, fontweight="bold",
    )

    colors_pairs = ["#9C27B0", "#FF9800", "#00BCD4"]

    for ax, (col_a, col_b), color in zip(axes, pairs, colors_pairs):
        x = df[col_a].values
        y = df[col_b].values

        # Remove NaN
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]

        # Regressão linear
        slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 200)
        y_line = slope * x_line + intercept

        ax.scatter(x, y, alpha=0.25, s=8, color=color, label="Observações")
        ax.plot(x_line, y_line, color="black", linewidth=1.8, label=f"Regressão (r={r_value:.3f})")

        # Interpretação da correlação
        if abs(r_value) >= 0.7:
            interpretacao = "Correlação FORTE"
        elif abs(r_value) >= 0.4:
            interpretacao = "Correlação MODERADA"
        else:
            interpretacao = "Correlação FRACA"
        direcao = "positiva" if r_value > 0 else "negativa"

        p_str    = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
        unit_a   = UNIT_LABELS.get(col_a, "USD")
        unit_b   = UNIT_LABELS.get(col_b, "USD")
        ax.set_title(f"{col_a} x {col_b}", fontsize=12, fontweight="bold")
        ax.set_xlabel(f"{col_a} ({unit_a})", fontsize=10)
        ax.set_ylabel(f"{col_b} ({unit_b})", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Caixa de texto com métricas + % explicativa
        r_pct    = abs(r_value) * 100
        r2_pct   = r_value ** 2 * 100
        textstr  = (
            f"r = {r_value:+.3f}  ({r_pct:.1f}%)\n"
            f"r^2 = {r_value**2:.3f}  ({r2_pct:.1f}% variancia explicada)\n"
            f"{p_str}\n"
            f"{interpretacao} {direcao}"
        )
        ax.text(
            0.04, 0.96, textstr,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8),
        )

    plt.tight_layout()

    if save:
        _ensure_plots_dir()
        path = os.path.join(PLOTS_DIR, "scatter_pairs.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Gráfico salvo: {path}")

    plt.close("all")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Correlação móvel ao longo do tempo
# ─────────────────────────────────────────────────────────────────────────────

def plot_rolling_correlation(
    df: pd.DataFrame,
    window: int = 90,
    save: bool = True,
) -> None:
    """
    Plota a correlação de Pearson móvel entre cada par de ativos ao longo do tempo.

    Mostra como a correlação evolui: em alguns períodos os ativos se movem
    muito juntos (correlação próxima de 1), em outros períodos se descolam.
    Isso é útil para identificar épocas de crise ou mudança estrutural.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com preços dos ativos.
    window : int
        Tamanho da janela móvel em pregões. Padrão: 90 (~4 meses).
    save : bool
        Se True, salva em plots/rolling_correlation.png.
    """
    if len(df) < window + 10:
        raise ValueError(f"DataFrame muito curto para janela de {window} dias.")

    cols = list(df.columns)
    pairs = [(cols[i], cols[j]) for i in range(len(cols)) for j in range(i + 1, len(cols))]

    colors_roll = ["#9C27B0", "#FF9800", "#00BCD4"]
    fig, ax = plt.subplots(figsize=(14, 5))

    for (col_a, col_b), color in zip(pairs, colors_roll):
        rolling_corr = df[col_a].rolling(window=window).corr(df[col_b])
        ax.plot(
            df.index, rolling_corr,
            color=color, linewidth=1.5,
            label=f"{col_a} × {col_b}",
        )

    # Bandas de referência
    ax.axhline(y=0.7,  color="green",  linestyle="--", linewidth=0.9, alpha=0.6, label="Corr. forte (0.7)")
    ax.axhline(y=0.4,  color="orange", linestyle="--", linewidth=0.9, alpha=0.6, label="Corr. moderada (0.4)")
    ax.axhline(y=0.0,  color="gray",   linestyle=":",  linewidth=1.0, alpha=0.5)
    ax.axhline(y=-0.4, color="orange", linestyle="--", linewidth=0.9, alpha=0.6)

    ax.set_ylim(-1.05, 1.05)
    ax.set_title(
        f"Correlação de Pearson Móvel entre Pares (janela = {window} pregões)\n"
        "Acima de 0.7 = correlação forte: quando um sobe, o outro tende a subir também",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Data", fontsize=11)
    ax.set_ylabel("Coeficiente de Correlação (r)", fontsize=11)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()

    if save:
        _ensure_plots_dir()
        path = os.path.join(PLOTS_DIR, "rolling_correlation.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Gráfico salvo: {path}")

    plt.close("all")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Impressão de estatísticas no terminal
# ─────────────────────────────────────────────────────────────────────────────

def print_correlation_matrix(df: pd.DataFrame) -> None:
    """
    Imprime a matriz de correlação de Pearson e estatísticas descritivas
    básicas de cada ativo no terminal.
    """
    print("\n" + "=" * 60)
    print("MATRIZ DE CORRELAÇÃO DE PEARSON")
    print("=" * 60)
    print(df.corr(method="pearson").round(4).to_string())

    print("\n" + "=" * 60)
    print("ESTATÍSTICAS DESCRITIVAS")
    print("=" * 60)
    print(df.describe().round(2).to_string())
    print("=" * 60 + "\n")
