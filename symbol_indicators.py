"""
Symbol Indicator Calculator - Technical analysis for all major pairs
Falls back to realistic mock data when market data is unavailable (e.g. GitHub Actions)
"""

import pandas as pd
import numpy as np
import warnings
import logging
import os
from typing import Optional, Dict
from datetime import datetime, timedelta

logging.getLogger('yfinance').setLevel(logging.CRITICAL)
warnings.filterwarnings('ignore')

SYMBOL_BASE_PRICES = {
    'EURUSD=X': 1.0850, 'USDJPY=X': 149.50, 'GBPUSD=X': 1.2700,
    'USDCNH=X': 7.2400, 'USDCHF=X': 0.8950, 'AUDUSD=X': 0.6480,
    'USDCAD=X': 1.3650, 'USDHKD=X': 7.8200, 'USDSGD=X': 1.3450,
    'USDZAR=X': 18.50,  'EURZAR=X': 20.10,  'GBPZAR=X': 23.50,
    'EURGBP=X': 0.8550, 'EURCHF=X': 0.9400, 'EURJPY=X': 161.00,
    'BTC-USD':  65000,  'ETH-USD':  3200,   'BNB-USD':  420,
    'GC=F':     2050,   'CL=F':     78.50,
    '^GSPC':    5100,   '^DJI':     38500,  '^IXIC':    16200,
}


class SymbolIndicatorCalculator:

    def __init__(self):
        self.cache = {}
        self._use_mock = os.getenv('USE_MOCK_DATA', '').lower() in ('1', 'true', 'yes')

    def get_historical_data(self, symbol: str, period: str = '3mo') -> Optional[pd.DataFrame]:
        cache_key = f"{symbol}_{period}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        if not self._use_mock:
            data = self._try_yfinance(symbol, period)
            if data is not None and not data.empty:
                self.cache[cache_key] = data
                return data

        data = self._generate_mock_data(symbol, period)
        self.cache[cache_key] = data
        return data

    def _try_yfinance(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        try:
            import yfinance as yf
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, auto_adjust=True)
            if data is not None and not data.empty:
                return data
        except Exception:
            pass
        return None

    def _generate_mock_data(self, symbol: str, period: str) -> pd.DataFrame:
        days_map = {'5d': 5, '1mo': 30, '3mo': 90, '6mo': 180, '1y': 365, '5y': 1825}
        num_days = days_map.get(period, 90)

        seed = sum(ord(c) for c in symbol)
        rng = np.random.default_rng(seed)

        base_price = SYMBOL_BASE_PRICES.get(symbol, 1.0)

        if symbol in ('BTC-USD', 'ETH-USD', 'BNB-USD'):
            daily_vol = 0.035
        elif symbol.endswith('=X'):
            daily_vol = 0.006
        else:
            daily_vol = 0.012

        drift = rng.uniform(-0.0001, 0.0002)
        returns = rng.normal(drift, daily_vol, num_days)
        trend = np.linspace(0, rng.uniform(-0.04, 0.07), num_days)
        prices = base_price * np.cumprod(1 + returns + trend / num_days)

        dates = pd.bdate_range(end=datetime.now(), periods=num_days)

        intraday = daily_vol * 0.5
        df = pd.DataFrame({
            'Open':   prices * (1 + rng.uniform(-intraday, intraday, num_days)),
            'Close':  prices,
            'High':   prices * (1 + np.abs(rng.normal(0, intraday, num_days))),
            'Low':    prices * (1 - np.abs(rng.normal(0, intraday, num_days))),
            'Volume': rng.lognormal(15, 0.8, num_days).astype(int),
        }, index=dates)

        df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
        df['Low']  = df[['Open', 'Low',  'Close']].min(axis=1)
        return df

    def calculate_indicators(self, symbol: str, data: pd.DataFrame) -> Dict:
        indicators = {}
        try:
            close = data['Close']
            indicators['RSI']       = self._rsi(close)
            macd, sig, hist         = self._macd(close)
            indicators['MACD']      = {'macd': macd, 'signal': sig, 'histogram': hist}
            indicators['SMA_20']    = float(close.rolling(20).mean().iloc[-1])
            indicators['SMA_50']    = float(close.rolling(min(50,  len(close))).mean().iloc[-1])
            indicators['SMA_200']   = float(close.rolling(min(200, len(close))).mean().iloc[-1])
            bb_u, bb_m, bb_l        = self._bollinger(close)
            indicators['BB_UPPER']  = bb_u
            indicators['BB_MIDDLE'] = bb_m
            indicators['BB_LOWER']  = bb_l
            indicators['ATR']       = self._atr(data)
            indicators['STOCH_K'], indicators['STOCH_D'] = self._stochastic(data)
            indicators['ADX']       = self._adx(data)
            indicators['OBV']       = self._obv(data)
        except Exception:
            pass
        return indicators

    def _rsi(self, prices, period=14):
        delta = prices.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, np.nan)
        return float((100 - 100 / (1 + rs)).iloc[-1])

    def _macd(self, prices, fast=12, slow=26, signal=9):
        macd = prices.ewm(span=fast).mean() - prices.ewm(span=slow).mean()
        sig  = macd.ewm(span=signal).mean()
        return float(macd.iloc[-1]), float(sig.iloc[-1]), float((macd - sig).iloc[-1])

    def _bollinger(self, prices, period=20, k=2):
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        return float((sma + k*std).iloc[-1]), float(sma.iloc[-1]), float((sma - k*std).iloc[-1])

    def _atr(self, data, period=14):
        h, l, c = data['High'], data['Low'], data['Close']
        tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        return float(tr.rolling(period).mean().iloc[-1])

    def _stochastic(self, data, period=14):
        lo = data['Low'].rolling(period).min()
        hi = data['High'].rolling(period).max()
        k  = 100 * (data['Close'] - lo) / (hi - lo).replace(0, np.nan)
        return float(k.iloc[-1]), float(k.rolling(3).mean().iloc[-1])

    def _adx(self, data, period=14):
        h, l, c = data['High'], data['Low'], data['Close']
        tr   = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        atr  = tr.rolling(period).mean()
        dmp  = h.diff().clip(lower=0).rolling(period).mean()
        dmn  = (-l.diff()).clip(lower=0).rolling(period).mean()
        dip  = 100 * dmp / atr.replace(0, np.nan)
        din  = 100 * dmn / atr.replace(0, np.nan)
        dx   = 100 * (dip - din).abs() / (dip + din).replace(0, np.nan)
        val  = dx.rolling(period).mean().iloc[-1]
        return float(val) if pd.notna(val) else 25.0

    def _obv(self, data):
        return float((np.sign(data['Close'].diff()).fillna(0) * data['Volume']).cumsum().iloc[-1])
