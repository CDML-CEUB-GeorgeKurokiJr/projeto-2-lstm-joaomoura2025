# Analise e Previsao de Commodities Agricolas com LSTM

Projeto academico de analise de series temporais financeiras utilizando
dados historicos de commodities do mercado americano (CME/CBOT) como
proxy para o agronegocio brasileiro.

---

## Professor — Como executar o projeto

O ambiente de execucao esta configurado via **GitHub Codespaces**.
Para rodar o projeto sem instalar nada:

1. Clique no botao verde **`<> Code`** (canto superior direito desta pagina)
2. Clique na aba **`Codespaces`**
3. Clique em **`Create codespace on master`**
4. Aguarde o ambiente abrir (VS Code no navegador)
5. No terminal, rode os comandos abaixo:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python main.py
```

Os graficos serao salvos na pasta `plots/` — clique em qualquer `.png` para visualizar.

---

## Ativos Analisados

| Ativo         | Ticker Yahoo Finance | Cotacao CBOT         | Equivalencia BR                     |
|---------------|----------------------|----------------------|--------------------------------------|
| **Boi Gordo** | `LE=F`               | USD cents por libra  | 1 arroba (@) = 15 kg = 33,07 lb      |
| **Soja**      | `ZS=F`               | USD cents por bushel | 1 saca (60 kg) ≈ 2,20 bushels        |
| **Milho**     | `ZC=F`               | USD cents por bushel | 1 saca (60 kg) ≈ 2,36 bushels        |

> **Como converter:** Divida o valor CBOT por 100 e multiplique pelo fator acima.
> Exemplo: Soja a 988 cents/bu → 9,88 USD/bu × 2,2046 = **21,78 USD/saca**

**Periodo:** 01/01/2019 a 31/12/2024 | **Fonte:** Yahoo Finance via `yfinance`

---

## Correlacao entre os Ativos

A correlacao de Pearson (r) mede o quanto dois ativos se movem juntos:

| Par                  | r      | Interpretacao                                      |
|----------------------|--------|----------------------------------------------------|
| **Soja x Milho**     | ~0.93  | **Forte positiva** — quando a soja sobe, o milho tende a subir tambem |
| Boi Gordo x Soja     | ~0.32  | Fraca — comportamentos relativamente independentes |
| Boi Gordo x Milho    | ~0.20  | Fraca — comportamentos relativamente independentes |

> **Por que Soja e Milho tem alta correlacao?**
> Ambas sao graos negociados globalmente, influenciados pelos mesmos fatores:
> clima nos EUA e Brasil, demanda da China, estoques globais e preco do dolar.
> O boi gordo e mais afetado pelo mercado interno brasileiro.

Os graficos `scatter_pairs.png` e `rolling_correlation.png` mostram essa
relacao visualmente, com linha de tendencia e o coeficiente r em %.

---

## Pipeline do Projeto

```
1. Coleta de Dados        → data_loader.py   (yfinance, 2019-2024)
2. Pre-processamento      → data_loader.py   (alinhamento, ffill, normalização)
3. Analise Exploratoria   → analysis.py      (series, correlacao, SMA)
4. Dataset para LSTM      → model.py         (TimeSeriesDataset, janela 30 dias)
5. Modelo LSTM            → model.py         (2 camadas, hidden=64, dropout=0.2)
6. Treinamento            → train.py         (Adam, MSE, 100 epocas, grad clipping)
7. Avaliacao              → train.py         (RMSE, MAE, MAPE no conjunto de teste)
8. Previsao 2025          → train.py         (forecast recursivo, 252 pregoess)
9. Orquestracao           → main.py          (pipeline completo)
```

---

## Estrutura de Arquivos

```
.
├── data_loader.py     # Download e pre-processamento via yfinance
├── model.py           # TimeSeriesDataset + LSTMModel (PyTorch)
├── train.py           # Treinamento, avaliacao e previsao 2025
├── analysis.py        # Graficos: series, correlacao, SMA, scatter, rolling corr
├── main.py            # Orquestrador principal — rode este arquivo
├── requirements.txt   # Dependencias do projeto
└── plots/             # Graficos gerados automaticamente (criado ao rodar)
    ├── time_series.png
    ├── normalized_prices.png      ← todos os ativos no mesmo eixo (base 100)
    ├── scatter_pairs.png          ← dispersao com r de Pearson em %
    ├── rolling_correlation.png    ← correlacao movel ao longo do tempo
    ├── correlation_heatmap.png
    ├── sma_boi_gordo.png
    ├── sma_soja.png
    ├── sma_milho.png
    ├── loss_boi_gordo.png
    ├── loss_soja.png
    ├── loss_milho.png
    ├── predictions_boi_gordo.png
    ├── predictions_soja.png
    ├── predictions_milho.png
    ├── forecast_2025_boi_gordo.png
    ├── forecast_2025_soja.png
    └── forecast_2025_milho.png
```

---

## Como Executar

### 1. Instalar dependencias

```bash
# PyTorch (CPU-only — funciona em qualquer maquina)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Demais pacotes
pip install -r requirements.txt
```

### 2. Rodar o projeto

```bash
python main.py
```

Os graficos serao salvos automaticamente na pasta `plots/`.
O terminal exibe as metricas, correlacoes em % e tendencias de preco.

---

## Arquitetura do Modelo LSTM

```
Entrada: sequencia de 30 dias de precos normalizados (MinMaxScaler)
    ↓
LSTM — 2 camadas empilhadas, 64 unidades ocultas
    ↓
Dropout (20%) — regularizacao para evitar overfitting
    ↓
Camada Linear — saida: proximo valor normalizado
    ↓
Saida: preco previsto (invertendo a normalizacao)
```

**Hiperparametros:**

| Parametro     | Valor | Justificativa                               |
|---------------|-------|---------------------------------------------|
| Janela (seq)  | 30    | ~6 semanas de pregoess — captura tendencias |
| Hidden size   | 64    | Capacidade sem overfitting nos dados         |
| Num layers    | 2     | LSTM empilhado detecta padroes hierarquicos |
| Dropout       | 0.2   | Regularizacao entre camadas                 |
| Epocas        | 100   | Convergencia tipica antes desse limite      |
| Otimizador    | Adam  | Padrao para redes neurais recorrentes       |
| Loss          | MSE   | Penaliza erros grandes — adequado para preco|

---

## Graficos Gerados

### Correlacao e Co-movimento

| Grafico                    | O que mostra                                                    |
|----------------------------|-----------------------------------------------------------------|
| `normalized_prices.png`    | Os 3 ativos no mesmo eixo (base 100). Linhas juntas = correlacao |
| `scatter_pairs.png`        | Dispersao com reta de regressao, r de Pearson e r² em %          |
| `rolling_correlation.png`  | Correlacao movel 90 dias — mostra quando os ativos se descolam   |
| `correlation_heatmap.png`  | Heatmap de correlacao de Pearson entre todos os pares            |

### Medias Moveis (SMA)

Os graficos `sma_*.png` mostram o preco real junto com:
- **SMA-20** (media dos ultimos 20 pregoess, ~4 semanas): tendencia de curto prazo
- **SMA-50** (media dos ultimos 50 pregoess, ~2,5 meses): tendencia de medio prazo

Quando o preco cruza a SMA de baixo para cima = sinal de alta (Golden Cross).
Quando cruza de cima para baixo = sinal de queda (Death Cross).

### Previsao LSTM

- `predictions_*.png`: previsao vs. real no conjunto de teste (2023-2024)
- `forecast_2025_*.png`: projecao para os ~252 pregoess de 2025
  - A banda cinza indica incerteza de ±5% (drift acumulado da previsao recursiva)

---

## Limitacoes

1. **Modelo univariado**: cada ativo e previsto isoladamente, sem considerar
   a correlacao entre Soja e Milho no proprio modelo.
2. **Previsao recursiva**: cada passo usa a previsao anterior, acumulando erro
   ao longo dos 252 passos de 2025.
3. **Dados americanos**: cotacoes CME/CBOT em dolares — diferem dos precos
   praticados no mercado brasileiro (B3/Cepea) por fatores de cambio e basis.
4. **Sem variaveis externas**: o modelo nao considera cambio (BRL/USD),
   clima, estoques USDA, safra brasileira ou demanda da China.

---

## Tecnologias

- **Python 3.12+**
- **yfinance** — coleta de dados do Yahoo Finance
- **pandas / numpy** — manipulacao de series temporais
- **PyTorch** — modelo LSTM e treinamento
- **scikit-learn** — normalizacao MinMaxScaler
- **matplotlib / seaborn** — visualizacoes
- **scipy** — regressao linear nos scatter plots

---

*Projeto desenvolvido para disciplina de Analise de Dados / Machine Learning*
