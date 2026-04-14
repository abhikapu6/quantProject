"""
data.py — Data fetching and cleaning utilities for the GLD/GDX pairs analysis.
Can be imported by notebook.ipynb or run standalone to refresh the CSVs.
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf


TICKERS    = ['GLD', 'GDX']
START_DATE = '2010-01-01'
END_DATE   = '2024-12-31'


def fetch_adjusted_close(
    tickers: list[str] = TICKERS,
    start: str = START_DATE,
    end: str = END_DATE,
) -> pd.DataFrame:
    """
    Download daily adjusted close prices from Yahoo Finance.

    Returns:
        DataFrame with columns ['GLD_adj', 'GDX_adj'] indexed by date.
    """
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    adj = raw['Close'][tickers].copy()
    adj.columns = [f"{t}_adj" for t in tickers]
    return adj


def clean_prices(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Forward-fill isolated NaN gaps, then drop any remaining NaNs.

    Returns:
        (cleaned DataFrame, report dict documenting transformations)
    """
    rows_before  = len(df)
    nans_before  = int(df.isnull().sum().sum())

    df_filled    = df.ffill()
    nans_ffilled = nans_before - int(df_filled.isnull().sum().sum())

    df_clean     = df_filled.dropna()
    rows_dropped = rows_before - len(df_clean)

    report = {
        'rows_raw'    : rows_before,
        'nans_raw'    : nans_before,
        'nans_ffilled': nans_ffilled,
        'rows_dropped': rows_dropped,
        'rows_final'  : len(df_clean),
        'nans_final'  : int(df_clean.isnull().sum().sum()),
    }
    return df_clean, report


def add_log_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add log price and log return columns for GLD and GDX.

    Expects columns 'GLD_adj' and 'GDX_adj'.
    Adds: 'log_GLD', 'log_GDX', 'ret_GLD', 'ret_GDX'.
    """
    df = df.copy()
    df['log_GLD'] = np.log(df['GLD_adj'])
    df['log_GDX'] = np.log(df['GDX_adj'])
    df['ret_GLD'] = df['log_GLD'].diff()
    df['ret_GDX'] = df['log_GDX'].diff()
    return df


def load_data(
    use_cache: bool = True,
    cache_dir: str = 'data',
) -> pd.DataFrame:
    """
    Full pipeline: fetch → clean → log-transform.
    If use_cache=True and clean CSV exists, load from disk instead of re-downloading.

    Returns:
        DataFrame with columns: GLD_adj, GDX_adj, log_GLD, log_GDX, ret_GLD, ret_GDX
    """
    clean_path = os.path.join(cache_dir, 'gld_gdx_clean.csv')

    if use_cache and os.path.exists(clean_path):
        print(f"Loading from cache: {clean_path}")
        df = pd.read_csv(clean_path, index_col=0, parse_dates=True)
    else:
        os.makedirs(cache_dir, exist_ok=True)
        raw = fetch_adjusted_close()
        raw.to_csv(os.path.join(cache_dir, 'gld_gdx_raw.csv'))
        df, report = clean_prices(raw)
        df.to_csv(clean_path)
        print("Cleaning report:", report)

    df = add_log_columns(df)
    return df


if __name__ == '__main__':
    df = load_data(use_cache=False)
    print(df.shape)
    print(df.tail())
