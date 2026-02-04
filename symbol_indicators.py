import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import requests
import os


class SymbolIndicatorCalculator:
    def __init__(self, api_keys=None):
        self.api_keys = api_keys or self._load_api_keys()
        self.cache = {}
    
    def _load_api_keys(self):
        return {
            'alpha_vantage': os.environ.get('ALPHA_VANTAGE_API_KEY', ''),
            'fred': os.environ.get('FRED_API_KEY', '')
        }
    
    def fetch_data(self, symbol, start_date, end_date=None):
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        symbol_upper = symbol.upper()
        
        if symbol_upper in ['GOLD', 'XAU', 'XAUUSD']:
            return self._fetch_gold_data(start_date, end_date)
        elif symbol_upper in ['US30', 'DJI', 'DJIA']:
            return self._fetch_us30_data(start_date, end_date)
        else:
            return self._fetch_yfinance_data(symbol, start_date, end_date)
    
    def _fetch_yfinance_data(self, symbol, start_date, end_date):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                print(f"No data for {symbol}")
                return None
            
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            return df[['open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            print(f"yfinance error for {symbol}: {e}")
            return None
    
    def _fetch_gold_data(self, start_date, end_date):
        df = None
        
        if self.api_keys.get('fred'):
            df = self._fetch_fred_gold(start_date, end_date)
        
        if df is None:
            df = self._fetch_yfinance_data('GC=F', start_date, end_date)
        
        if df is None:
            df = self._fetch_yfinance_data('GLD', start_date, end_date)
            if df is not None:
                df[['open', 'high', 'low', 'close']] *= 10
        
        return df
    
    def _fetch_fred_gold(self, start_date, end_date):
        if not self.api_keys.get('fred'):
            return None
        
        url = 'https://api.stlouisfed.org/fred/series/observations'
        params = {
            'series_id': 'GOLDAMGBD228NLBM',
            'api_key': self.api_keys['fred'],
            'file_type': 'json',
            'observation_start': start_date,
            'observation_end': end_date
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                observations = data.get('observations', [])
                
                df = pd.DataFrame(observations)
                df['date'] = pd.to_datetime(df['date'])
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                df = df.dropna(subset=['value'])
                df = df.set_index('date')
                df = df.rename(columns={'value': 'close'})
                
                df['open'] = df['close']
                df['high'] = df['close'] * 1.002
                df['low'] = df['close'] * 0.998
                df['volume'] = 1000000
                
                return df[['open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            print(f"FRED error: {e}")
        
        return None
    
    def _fetch_us30_data(self, start_date, end_date):
        symbols = ['^DJI', 'DIA']
        
        for symbol in symbols:
            df = self._fetch_yfinance_data(symbol, start_date, end_date)
            if df is not None and not df.empty:
                if symbol == 'DIA':
                    df[['open', 'high', 'low', 'close']] *= 100 / 3
                return df
        
        return None
    
    def calculate_indicators(self, df):
        df = df.copy()
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
        
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(window=14).mean()
        
        df['+dm'] = np.where(
            (df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
            np.maximum(df['high'] - df['high'].shift(1), 0),
            0
        )
        df['-dm'] = np.where(
            (df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
            np.maximum(df['low'].shift(1) - df['low'], 0),
            0
        )
        
        df['+di'] = 100 * (df['+dm'].rolling(window=14).mean() / df['atr'])
        df['-di'] = 100 * (df['-dm'].rolling(window=14).mean() / df['atr'])
        df['dx'] = 100 * abs(df['+di'] - df['-di']) / (df['+di'] + df['-di'])
        df['adx'] = df['dx'].rolling(window=14).mean()
        
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = tp.rolling(window=20).mean()
        mad = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
        df['cci'] = (tp - sma_tp) / (0.015 * mad)
        
        high_14 = df['high'].rolling(window=14).max()
        low_14 = df['low'].rolling(window=14).min()
        df['willr'] = -100 * ((high_14 - df['close']) / (high_14 - low_14))
        
        return df
    
    def get_indicators_for_date(self, symbol, date, lookback_days=365):
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        date = date.tz_localize(None) if date.tz is not None else date
        
        start_date = (date - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        end_date = (date + timedelta(days=1)).strftime('%Y-%m-%d')
        
        cache_key = f"{symbol}_{start_date}_{end_date}"
        
        if cache_key in self.cache:
            df = self.cache[cache_key]
        else:
            df = self.fetch_data(symbol, start_date, end_date)
            if df is None or df.empty:
                return None
            df = self.calculate_indicators(df)
            self.cache[cache_key] = df
        
        df_sorted = df.sort_index()
        if df_sorted.index.tz is not None:
            df_sorted.index = df_sorted.index.tz_localize(None)
        
        if date not in df_sorted.index:
            idx = df_sorted.index.searchsorted(date)
            if idx >= len(df_sorted):
                idx = len(df_sorted) - 1
            row = df_sorted.iloc[idx]
        else:
            row = df_sorted.loc[date]
        
        return self._format_indicators(row)
    
    def _format_indicators(self, row):
        indicators = {}
        price = row['close']
        
        rsi_val = row['rsi']
        if not pd.isna(rsi_val):
            indicators['RSI'] = {
                'value': round(rsi_val, 2),
                'signal': 'BUY' if rsi_val < 30 else ('SELL' if rsi_val > 70 else 'NEUTRAL'),
                'description': f'RSI at {round(rsi_val, 2)}'
            }
        
        macd_val = row['macd']
        signal_val = row['macd_signal']
        if not pd.isna(macd_val) and not pd.isna(signal_val):
            indicators['MACD'] = {
                'value': round(macd_val, 4),
                'signal': 'BUY' if macd_val > signal_val else 'SELL',
                'description': f'MACD {round(macd_val, 4)} vs Signal {round(signal_val, 4)}'
            }
        
        k_val = row['stoch_k']
        if not pd.isna(k_val):
            indicators['Stochastic'] = {
                'value': round(k_val, 2),
                'signal': 'BUY' if k_val < 20 else ('SELL' if k_val > 80 else 'NEUTRAL'),
                'description': f'Stochastic at {round(k_val, 2)}%'
            }
        
        bb_upper = row['bb_upper']
        bb_lower = row['bb_lower']
        if not pd.isna(bb_upper) and not pd.isna(bb_lower):
            indicators['Bollinger'] = {
                'value': f'{round(bb_lower, 2)}-{round(bb_upper, 2)}',
                'signal': 'SELL' if price > bb_upper else ('BUY' if price < bb_lower else 'NEUTRAL'),
                'description': f'Price at {round(price, 2)}'
            }
        
        sma_20 = row['sma_20']
        sma_50 = row['sma_50']
        if not pd.isna(sma_20) and not pd.isna(sma_50):
            indicators['MA_Cross'] = {
                'value': f'{round(sma_20, 2)}/{round(sma_50, 2)}',
                'signal': 'BUY' if (price > sma_20 and sma_20 > sma_50) else ('SELL' if (price < sma_20 and sma_20 < sma_50) else 'NEUTRAL'),
                'description': f'Price {round(price, 2)} vs SMA20 {round(sma_20, 2)}'
            }
        
        adx_val = row['adx']
        if not pd.isna(adx_val):
            indicators['ADX'] = {
                'value': round(adx_val, 2),
                'signal': 'STRONG TREND' if adx_val > 25 else 'WEAK TREND',
                'description': f'Trend strength {round(adx_val, 2)}'
            }
        
        cci_val = row['cci']
        if not pd.isna(cci_val):
            indicators['CCI'] = {
                'value': round(cci_val, 2),
                'signal': 'BUY' if cci_val < -100 else ('SELL' if cci_val > 100 else 'NEUTRAL'),
                'description': f'CCI at {round(cci_val, 2)}'
            }
        
        willr_val = row['willr']
        if not pd.isna(willr_val):
            indicators['Williams_R'] = {
                'value': round(willr_val, 2),
                'signal': 'BUY' if willr_val < -80 else ('SELL' if willr_val > -20 else 'NEUTRAL'),
                'description': f'Williams %R at {round(willr_val, 2)}'
            }
        
        buy_signals = sum(1 for ind in indicators.values() if ind.get('signal') == 'BUY')
        sell_signals = sum(1 for ind in indicators.values() if ind.get('signal') == 'SELL')
        
        overall = 'BUY' if buy_signals > sell_signals else ('SELL' if sell_signals > buy_signals else 'NEUTRAL')
        
        return {
            'indicators': indicators,
            'overall_signal': overall,
            'buy_count': buy_signals,
            'sell_count': sell_signals,
            'price': round(price, 2)
        }


if __name__ == "__main__":
    calc = SymbolIndicatorCalculator()
    
    symbols = ['GOLD', 'US30', 'AAPL', 'EURUSD=X']
    test_date = '2024-12-01'
    
    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"Symbol: {symbol} | Date: {test_date}")
        print('='*60)
        
        result = calc.get_indicators_for_date(symbol, test_date)
        
        if result:
            print(f"\nPrice: ${result['price']}")
            print(f"Overall Signal: {result['overall_signal']}")
            print(f"BUY signals: {result['buy_count']} | SELL signals: {result['sell_count']}")
            
            print("\nIndicators:")
            for name, data in result['indicators'].items():
                print(f"  {name:15} {data['signal']:12} - {data['description']}")
        else:
            print(f"No data available for {symbol}") 
