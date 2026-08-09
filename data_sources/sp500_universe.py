"""Current S&P 500 constituent list, scraped from Wikipedia.

Finnhub's /index/constituents is 403 on the free tier, so we use
Wikipedia's maintained "List of S&P 500 companies" table instead. It is
kept in sync with actual index changes closely enough for this project's
purposes (universe definition, not point-in-time backtesting of index
membership).
"""
import io
import datetime as dt
import requests
import pandas as pd

_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch_constituents():
    """Returns a DataFrame with columns: ticker, security, gics_sector,
    gics_sub_industry, date_added, snapshot_date. Tickers are normalized to
    the Yahoo/Finnhub convention (dot -> dash, e.g. BRK.B -> BRK-B)."""
    resp = requests.get(_WIKI_URL, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    df = tables[0]
    df = df.rename(columns={
        "Symbol": "ticker",
        "Security": "security",
        "GICS Sector": "gics_sector",
        "GICS Sub-Industry": "gics_sub_industry",
        "Date added": "date_added",
    })[["ticker", "security", "gics_sector", "gics_sub_industry", "date_added"]]
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    df["snapshot_date"] = dt.date.today().isoformat()
    return df
