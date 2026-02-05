import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta


class VolumeAnalyzer:
    def __init__(self):
        self.cache = {}
    
    def fetch_data(self, symbol, start_date, end_date=None, interval='1d'):
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        cache_key = f"{symbol}_{start_date}_{end_date}_{interval}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                return None
            
            df = df.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low',
                'Close': 'close', 'Volume': 'volume'
            })
            
            self.cache[cache_key] = df
            return df
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None
    
    def calculate_vwap(self, df):
        df = df.copy()
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (df['typical_price'] * df['volume']).cumsum() / df['volume'].cumsum()
        return df
    
    def calculate_obv(self, df):
        df = df.copy()
        df['obv'] = 0
        df.loc[df['close'] > df['close'].shift(1), 'obv'] = df['volume']
        df.loc[df['close'] < df['close'].shift(1), 'obv'] = -df['volume']
        df['obv'] = df['obv'].cumsum()
        return df
    
    def calculate_mfi(self, df, period=14):
        df = df.copy()
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['raw_money_flow'] = df['typical_price'] * df['volume']
        
        df['positive_flow'] = 0
        df['negative_flow'] = 0
        
        df.loc[df['typical_price'] > df['typical_price'].shift(1), 'positive_flow'] = df['raw_money_flow']
        df.loc[df['typical_price'] < df['typical_price'].shift(1), 'negative_flow'] = df['raw_money_flow']
        
        positive_mf = df['positive_flow'].rolling(window=period).sum()
        negative_mf = df['negative_flow'].rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + (positive_mf / negative_mf)))
        df['mfi'] = mfi
        
        return df
    
    def calculate_accumulation_distribution(self, df):
        df = df.copy()
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        clv = clv.fillna(0)
        df['ad'] = (clv * df['volume']).cumsum()
        return df
    
    def calculate_volume_profile(self, df, num_bins=20):
        price_range = df['high'].max() - df['low'].min()
        bin_size = price_range / num_bins
        
        bins = np.arange(df['low'].min(), df['high'].max() + bin_size, bin_size)
        
        volume_profile = []
        for i in range(len(bins) - 1):
            low_bound = bins[i]
            high_bound = bins[i + 1]
            
            mask = (df['low'] <= high_bound) & (df['high'] >= low_bound)
            volume_in_bin = df.loc[mask, 'volume'].sum()
            
            volume_profile.append({
                'price_low': low_bound,
                'price_high': high_bound,
                'price_mid': (low_bound + high_bound) / 2,
                'volume': volume_in_bin
            })
        
        return pd.DataFrame(volume_profile)
    
    def calculate_all_indicators(self, df):
        df = self.calculate_vwap(df)
        df = self.calculate_obv(df)
        df = self.calculate_mfi(df)
        df = self.calculate_accumulation_distribution(df)
        
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        return df
    
    def get_volume_analysis(self, symbol, date, lookback_days=60):
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        start_date = (date - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        end_date = (date + timedelta(days=1)).strftime('%Y-%m-%d')
        
        df = self.fetch_data(symbol, start_date, end_date)
        if df is None or df.empty:
            return None
        
        df = self.calculate_all_indicators(df)
        
        df_sorted = df.sort_index()
        if df_sorted.index.tz is not None:
            df_sorted.index = df_sorted.index.tz_localize(None)
        
        date = date.tz_localize(None) if date.tz is not None else date
        
        if date not in df_sorted.index:
            idx = df_sorted.index.searchsorted(date)
            if idx >= len(df_sorted):
                idx = len(df_sorted) - 1
            row = df_sorted.iloc[idx]
        else:
            row = df_sorted.loc[date]
        
        volume_profile = self.calculate_volume_profile(df_sorted.tail(20))
        poc = volume_profile.loc[volume_profile['volume'].idxmax(), 'price_mid']
        
        analysis = {
            'symbol': symbol,
            'date': row.name.strftime('%Y-%m-%d'),
            'price': round(row['close'], 2),
            'volume': int(row['volume']),
            'indicators': {
                'VWAP': {
                    'value': round(row['vwap'], 2),
                    'signal': 'BULLISH' if row['close'] > row['vwap'] else 'BEARISH',
                    'description': f"Price {'above' if row['close'] > row['vwap'] else 'below'} VWAP"
                },
                'OBV': {
                    'value': int(row['obv']),
                    'trend': 'UP' if row['obv'] > df_sorted['obv'].iloc[-5] else 'DOWN',
                    'description': 'Volume accumulation trending ' + ('up' if row['obv'] > df_sorted['obv'].iloc[-5] else 'down')
                },
                'MFI': {
                    'value': round(row['mfi'], 2),
                    'signal': 'OVERSOLD' if row['mfi'] < 20 else ('OVERBOUGHT' if row['mfi'] > 80 else 'NEUTRAL'),
                    'description': f"Money Flow Index at {round(row['mfi'], 2)}"
                },
                'A/D': {
                    'value': int(row['ad']),
                    'trend': 'ACCUMULATION' if row['ad'] > df_sorted['ad'].iloc[-5] else 'DISTRIBUTION',
                    'description': 'Accumulation/Distribution line trending ' + ('up' if row['ad'] > df_sorted['ad'].iloc[-5] else 'down')
                },
                'Volume_Ratio': {
                    'value': round(row['volume_ratio'], 2),
                    'signal': 'HIGH' if row['volume_ratio'] > 1.5 else ('LOW' if row['volume_ratio'] < 0.5 else 'NORMAL'),
                    'description': f"{round(row['volume_ratio'], 2)}x average volume"
                }
            },
            'volume_profile': {
                'poc': round(poc, 2),
                'signal': 'BULLISH' if row['close'] > poc else 'BEARISH'
            }
        }
        
        bullish_signals = sum(1 for ind in analysis['indicators'].values() 
                             if ind.get('signal') in ['BULLISH', 'OVERSOLD', 'HIGH'] or ind.get('trend') in ['UP', 'ACCUMULATION'])
        bearish_signals = sum(1 for ind in analysis['indicators'].values() 
                             if ind.get('signal') in ['BEARISH', 'OVERBOUGHT', 'LOW'] or ind.get('trend') in ['DOWN', 'DISTRIBUTION'])
        
        analysis['overall_signal'] = 'BULLISH' if bullish_signals > bearish_signals else ('BEARISH' if bearish_signals > bullish_signals else 'NEUTRAL')
        
        return analysis


if __name__ == "__main__":
    analyzer = VolumeAnalyzer()
    
    symbols = ['AAPL', 'GOLD', 'EURUSD=X']
    test_date = '2024-12-01'
    
    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"Volume Analysis: {symbol} | {test_date}")
        print('='*60)
        
        analysis = analyzer.get_volume_analysis(symbol, test_date)
        
        if analysis:
            print(f"\nPrice: ${analysis['price']:,}")
            print(f"Volume: {analysis['volume']:,}")
            print(f"Overall Signal: {analysis['overall_signal']}")
            
            print("\nVolume Indicators:")
            for name, data in analysis['indicators'].items():
                signal = data.get('signal') or data.get('trend', 'N/A')
                print(f"  {name:15} {signal:15} - {data['description']}")
            
            print(f"\nPoint of Control: ${analysis['volume_profile']['poc']:,}")
        else:
            print(f"No data available for {symbol}")
