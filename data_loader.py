"""
data_loader.py
--------------
Download e pré-processamento de dados históricos de commodities
via Yahoo Finance (yfinance).

Ativos:
  - Boi Gordo  → LE=F  (CME Live Cattle Futures)
  - Soja       → ZS=F  (CBOT Soybean Futures)
  - Milho      → ZC=F  (CBOT Corn Futures)
"""

import time
import warnings

import pandas as pd
import yfinance as yf

# Mapeamento nome legível → ticker Yahoo Finance
TICKERS: dict[str, str] = {
    "Boi Gordo": "LE=F",
    "Soja":      "ZS=F",
    "Milho":     "ZC=F",
}

# Unidades de cotação na bolsa americana (CBOT/CME)
# LE=F: USD cents por libra-peso (lb)
# ZS=F: USD cents por bushel de soja
# ZC=F: USD cents por bushel de milho
UNIT_LABELS: dict[str, str] = {
    "Boi Gordo": "USD cents/lb",
    "Soja":      "USD cents/bushel",
    "Milho":     "USD cents/bushel",
}

# Fatores de conversão para unidades brasileiras
# Boi Gordo: 1 arroba (@) = 15 kg = 33,07 lb → preço/@  = (cents/lb / 100) * 33.07
# Soja     : 1 saca 60 kg ≈ 2,2046 bushels   → preço/saca = (cents/bu / 100) * 2.2046
# Milho    : 1 saca 60 kg ≈ 2,3622 bushels   → preço/saca = (cents/bu / 100) * 2.3622
BR_UNIT: dict[str, str] = {
    "Boi Gordo": "USD/@",
    "Soja":      "USD/saca (60 kg)",
    "Milho":     "USD/saca (60 kg)",
}
BR_FACTOR: dict[str, float] = {
    "Boi Gordo": 33.0693 / 100,   # cents/lb  -> USD/@
    "Soja":      2.2046  / 100,   # cents/bu  -> USD/saca
    "Milho":     2.3622  / 100,   # cents/bu  -> USD/saca
}


def _extract_close(raw: pd.DataFrame, ticker: str) -> pd.Series:
    """
    Extrai a coluna de preço de fechamento (ajustado) para um ticker,
    lidando com as diferentes estruturas de MultiIndex que o yfinance pode retornar.
    """
    if not isinstance(raw.columns, pd.MultiIndex):
        # Caso de coluna simples (apenas 1 ticker baixado separadamente)
        for col in ("Close", "Adj Close"):
            if col in raw.columns:
                return raw[col].rename(ticker)
        raise ValueError(f"Colunas 'Close'/'Adj Close' não encontradas para {ticker}.")

    # MultiIndex com níveis (Price, Ticker) — padrão yfinance >= 0.2.x com auto_adjust=True
    if raw.columns.names[0] in (None, "Price", "price"):
        for price_col in ("Close", "Adj Close"):
            if price_col in raw.columns.get_level_values(0):
                return raw[price_col][ticker].rename(ticker)

    # MultiIndex com níveis (Ticker, Price)
    if ticker in raw.columns.get_level_values(0):
        sub = raw[ticker]
        for price_col in ("Close", "Adj Close"):
            if price_col in sub.columns:
                return sub[price_col].rename(ticker)

    raise ValueError(
        f"Não foi possível extrair preço de fechamento para {ticker}. "
        f"Colunas disponíveis: {list(raw.columns)}"
    )


def _batch_download(
    ticker_list: list[str],
    start: str,
    end: str,
    retries: int = 3,
) -> pd.DataFrame:
    """
    Faz batch download via yf.download com threads=False.
    threads=False evita acesso concorrente ao cache SQLite do yfinance,
    prevenindo o erro 'database is locked'.
    """
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = yf.download(
                    ticker_list,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                    threads=False,   # serial — evita lock SQLite
                )
            if not raw.empty:
                return raw
            raise ValueError("yfinance retornou DataFrame vazio.")
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(3)
    raise RuntimeError(
        f"Falha no download apos {retries} tentativas: {last_exc}"
    ) from last_exc


def download_data(
    tickers: dict[str, str] | None = None,
    start: str = "2019-01-01",
    end: str = "2024-12-31",
) -> pd.DataFrame:
    """
    Baixa dados historicos de commodities do Yahoo Finance e retorna
    um DataFrame com precos de fechamento ajustados, alinhados por data.

    Usa batch download serial (threads=False) para evitar o erro
    'database is locked' do cache SQLite interno do yfinance.

    Parametros
    ----------
    tickers : dict[str, str], opcional
        Dicionario {nome_legivel: ticker_yahoo}. Usa TICKERS global se None.
    start : str
        Data inicial no formato 'YYYY-MM-DD'.
    end : str
        Data final no formato 'YYYY-MM-DD'.

    Retorna
    -------
    pd.DataFrame
        Indice: DatetimeIndex (datas comuns a todos os ativos).
        Colunas: nomes legiveis dos ativos (ex: "Boi Gordo", "Soja", "Milho").
    """
    if tickers is None:
        tickers = TICKERS

    ticker_list = list(tickers.values())
    name_map    = {v: k for k, v in tickers.items()}
    print(f"Baixando dados de {start} a {end} para: {ticker_list}")

    raw = _batch_download(ticker_list, start, end)

    # Extrai serie de fechamento para cada ticker
    series_list = []
    for ticker in ticker_list:
        try:
            series = _extract_close(raw, ticker).ffill()
            name   = name_map[ticker]
            series = series.rename(name)
            if series.index.tz is not None:
                series.index = series.index.tz_localize(None)
            n_ok = series.notna().sum()
            print(f"  {name} ({ticker}): {n_ok} registros")
            series_list.append(series)
        except Exception as exc:
            warnings.warn(f"  AVISO: Ignorando {ticker}: {exc}", UserWarning, stacklevel=2)

    if not series_list:
        raise ValueError("Nenhum ativo pode ser carregado. Verifique os tickers.")

    # Alinha datas (inner join → apenas datas comuns a todos os ativos)
    df = pd.concat(series_list, axis=1, join="inner")
    df = df.dropna()
    df = df.sort_index()

    # Renomeia colunas para nomes legíveis
    df.rename(columns=name_map, inplace=True)

    print(
        f"Dados carregados: {df.shape[0]} pregões × {df.shape[1]} ativos "
        f"({df.index[0].date()} a {df.index[-1].date()})"
    )
    return df
