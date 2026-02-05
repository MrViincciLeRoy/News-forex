import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from scipy.signal import argrelextrema


class MarketStructureAnalyzer:
    def __init__(self):
        self.cache = {}
    
    def fetch_data(self, symbol, start_date, end_date):
        cache_key = f"{symbol}_{start_date}_{end_date}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                return None
            
            df = df.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'
            })
            
            self.cache[cache_key] = df
            return df
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None
    
    def find_support_resistance(self, df, order=5):
        df = df.copy()
        
        local_max = argrelextrema(df['high'].values, np.greater, order=order)[0]
        local_min = argrelextrema(df['low'].values, np.less, order=order)[0]
        
        resistance_levels = df.iloc[local_max]['high'].values
        support_levels = df.iloc[local_min]['low'].values
        
        def cluster_levels(levels, threshold=0.02):
            if len(levels) == 0:
                return []
            
            clustered = []
            sorted_levels = sorted(levels)
            current_cluster = [sorted_levels[0]]
            
            for level in sorted_levels[1:]:
                if (level - current_cluster[-1]) / current_cluster[-1] < threshold:
                    current_cluster.append(level)
                else:
                    clustered.append(np.mean(current_cluster))
                    current_cluster = [level]
            
            clustered.append(np.mean(current_cluster))
            return clustered
        
        resistance_clustered = cluster_levels(resistance_levels)
        support_clustered = cluster_levels(support_levels)
        
        current_price = df['close'].iloc[-1]
        
        nearby_resistance = [r for r in resistance_clustered if r > current_price]
        nearby_support = [s for s in support_clustered if s < current_price]
        
        return {
            'resistance': sorted(nearby_resistance)[:3],
            'support': sorted(nearby_support, reverse=True)[:3],
            'current_price': current_price
        }
    
    def detect_trend(self, df, short_window=20, long_window=50):
        df = df.copy()
        
        df['sma_short'] = df['close'].rolling(window=short_window).mean()
        df['sma_long'] = df['close'].rolling(window=long_window).mean()
        
        current_price = df['close'].iloc[-1]
        sma_short = df['sma_short'].iloc[-1]
        sma_long = df['sma_long'].iloc[-1]
        
        if current_price > sma_short > sma_long:
            trend = 'STRONG_UPTREND'
        elif current_price > sma_short and sma_short < sma_long:
            trend = 'WEAK_UPTREND'
        elif current_price < sma_short < sma_long:
            trend = 'STRONG_DOWNTREND'
        elif current_price < sma_short and sma_short > sma_long:
            trend = 'WEAK_DOWNTREND'
        else:
            trend = 'SIDEWAYS'
        
        slope_short = (df['sma_short'].iloc[-1] - df['sma_short'].iloc[-10]) / 10
        slope_long = (df['sma_long'].iloc[-1] - df['sma_long'].iloc[-20]) / 20
        
        return {
            'trend': trend,
            'current_price': current_price,
            'sma_short': round(sma_short, 2),
            'sma_long': round(sma_long, 2),
            'short_slope': round(slope_short, 4),
            'long_slope': round(slope_long, 4)
        }
    
    def detect_market_regime(self, df, lookback=20):
        df = df.copy()
        
        df['returns'] = df['close'].pct_change()
        volatility = df['returns'].rolling(window=lookback).std() * np.sqrt(252)
        
        df['high_roll'] = df['high'].rolling(window=lookback).max()
        df['low_roll'] = df['low'].rolling(window=lookback).min()
        df['range'] = (df['high_roll'] - df['low_roll']) / df['close']
        
        current_vol = volatility.iloc[-1]
        avg_vol = volatility.mean()
        current_range = df['range'].iloc[-1]
        
        if current_vol > avg_vol * 1.5:
            regime = 'HIGH_VOLATILITY'
        elif current_vol < avg_vol * 0.5:
            regime = 'LOW_VOLATILITY'
        else:
            regime = 'NORMAL_VOLATILITY'
        
        if current_range < 0.05:
            market_type = 'RANGING'
        elif current_range > 0.15:
            market_type = 'TRENDING'
        else:
            market_type = 'MIXED'
        
        return {
            'regime': regime,
            'market_type': market_type,
            'volatility': round(current_vol * 100, 2),
            'avg_volatility': round(avg_vol * 100, 2),
            'range_pct': round(current_range * 100, 2)
        }
    
    def find_higher_highs_lows(self, df, lookback=20):
        df = df.copy()
        
        recent = df.tail(lookback)
        
        highs = recent['high'].values
        lows = recent['low'].values
        
        higher_highs = sum([1 for i in range(1, len(highs)) if highs[i] > highs[i-1]])
        higher_lows = sum([1 for i in range(1, len(lows)) if lows[i] > lows[i-1]])
        
        lower_highs = sum([1 for i in range(1, len(highs)) if highs[i] < highs[i-1]])
        lower_lows = sum([1 for i in range(1, len(lows)) if lows[i] < lows[i-1]])
        
        if higher_highs > lower_highs and higher_lows > lower_lows:
            structure = 'BULLISH'
        elif lower_highs > higher_highs and lower_lows > higher_lows:
            structure = 'BEARISH'
        else:
            structure = 'NEUTRAL'
        
        return {
            'structure': structure,
            'higher_highs': higher_highs,
            'lower_highs': lower_highs,
            'higher_lows': higher_lows,
            'lower_lows': lower_lows
        }
    
    def calculate_pivot_points(self, df):
        last_row = df.iloc[-1]
        
        pivot = (last_row['high'] + last_row['low'] + last_row['close']) / 3
        
        r1 = 2 * pivot - last_row['low']
        r2 = pivot + (last_row['high'] - last_row['low'])
        r3 = r1 + (last_row['high'] - last_row['low'])
        
        s1 = 2 * pivot - last_row['high']
        s2 = pivot - (last_row['high'] - last_row['low'])
        s3 = s1 - (last_row['high'] - last_row['low'])
        
        return {
            'pivot': round(pivot, 2),
            'resistance': {
                'r1': round(r1, 2),
                'r2': round(r2, 2),
                'r3': round(r3, 2)
            },
            'support': {
                's1': round(s1, 2),
                's2': round(s2, 2),
                's3': round(s3, 2)
            }
        }
    
    def get_market_structure(self, symbol, date=None, lookback_days=90):
        if date is None:
            date = datetime.now()
        elif isinstance(date, str):
            date = pd.to_datetime(date)
        
        start_date = (date - timedelta(days=lookback_days + 60)).strftime('%Y-%m-%d')
        end_date = date.strftime('%Y-%m-%d')
        
        df = self.fetch_data(symbol, start_date, end_date)
        
        if df is None or df.empty:
            return None
        
        sr_levels = self.find_support_resistance(df)
        trend = self.detect_trend(df)
        regime = self.detect_market_regime(df)
        structure = self.find_higher_highs_lows(df)
        pivots = self.calculate_pivot_points(df)
        
        return {
            'symbol': symbol,
            'date': end_date,
            'current_price': round(df['close'].iloc[-1], 2),
            'support_resistance': sr_levels,
            'trend_analysis': trend,
            'market_regime': regime,
            'price_structure': structure,
            'pivot_points': pivots,
            'overall_bias': self._determine_overall_bias(trend, structure, sr_levels)
        }
    
    def _determine_overall_bias(self, trend, structure, sr_levels):
        bullish_signals = 0
        bearish_signals = 0
        
        if 'UPTREND' in trend['trend']:
            bullish_signals += 2 if 'STRONG' in trend['trend'] else 1
        elif 'DOWNTREND' in trend['trend']:
            bearish_signals += 2 if 'STRONG' in trend['trend'] else 1
        
        if structure['structure'] == 'BULLISH':
            bullish_signals += 1
        elif structure['structure'] == 'BEARISH':
            bearish_signals += 1
        
        if bullish_signals > bearish_signals:
            return 'BULLISH'
        elif bearish_signals > bullish_signals:
            return 'BEARISH'
        else:
            return 'NEUTRAL'


if __name__ == "__main__":
    analyzer = MarketStructureAnalyzer()
    
    symbols = ['AAPL', 'GC=F', '^GSPC']
    test_date = '2024-12-01'
    
    for symbol in symbols:
        print(f"\n{'='*80}")
        print(f"Market Structure Analysis: {symbol} | {test_date}")
        print('='*80)
        
        analysis = analyzer.get_market_structure(symbol, test_date)
        
        if analysis:
            print(f"\nCurrent Price: ${analysis['current_price']:,}")
            print(f"Overall Bias: {analysis['overall_bias']}")
            
            print(f"\nTrend: {analysis['trend_analysis']['trend']}")
            print(f"Market Regime: {analysis['market_regime']['regime']} - {analysis['market_regime']['market_type']}")
            print(f"Price Structure: {analysis['price_structure']['structure']}")
            
            if analysis['support_resistance']['resistance']:
                print(f"\nResistance Levels: {', '.join([f'${r:.2f}' for r in analysis['support_resistance']['resistance']])}")
            
            if analysis['support_resistance']['support']:
                print(f"Support Levels: {', '.join([f'${s:.2f}' for s in analysis['support_resistance']['support']])}")
            
            print(f"\nPivot Points:")
            print(f"  Pivot: ${analysis['pivot_points']['pivot']}")
            print(f"  R1: ${analysis['pivot_points']['resistance']['r1']} | S1: ${analysis['pivot_points']['support']['s1']}")
