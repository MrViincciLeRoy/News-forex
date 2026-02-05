import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta


class CorrelationAnalyzer:
    def __init__(self):
        self.cache = {}
        
        self.asset_groups = {
            'currencies': ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'DX-Y.NYB'],
            'commodities': ['GC=F', 'SI=F', 'CL=F', 'NG=F'],
            'indices': ['^GSPC', '^IXIC', '^DJI', '^VIX'],
            'bonds': ['TLT', 'IEF', 'SHY'],
            'crypto': ['BTC-USD', 'ETH-USD']
        }
    
    def fetch_data(self, symbol, start_date, end_date):
        cache_key = f"{symbol}_{start_date}_{end_date}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                return None
            
            self.cache[cache_key] = df['Close']
            return df['Close']
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None
    
    def calculate_correlation_matrix(self, symbols, start_date, end_date):
        data = {}
        
        for symbol in symbols:
            series = self.fetch_data(symbol, start_date, end_date)
            if series is not None and not series.empty:
                data[symbol] = series
        
        if not data:
            return None
        
        df = pd.DataFrame(data)
        df = df.dropna()
        
        correlation_matrix = df.corr()
        
        return correlation_matrix
    
    def calculate_rolling_correlation(self, symbol1, symbol2, start_date, end_date, window=30):
        series1 = self.fetch_data(symbol1, start_date, end_date)
        series2 = self.fetch_data(symbol2, start_date, end_date)
        
        if series1 is None or series2 is None:
            return None
        
        df = pd.DataFrame({symbol1: series1, symbol2: series2})
        df = df.dropna()
        
        rolling_corr = df[symbol1].rolling(window=window).corr(df[symbol2])
        
        return rolling_corr
    
    def find_leading_indicators(self, target_symbol, candidate_symbols, start_date, end_date, max_lag=10):
        target_series = self.fetch_data(target_symbol, start_date, end_date)
        if target_series is None:
            return None
        
        target_returns = target_series.pct_change().dropna()
        
        leading_indicators = []
        
        for symbol in candidate_symbols:
            if symbol == target_symbol:
                continue
            
            candidate_series = self.fetch_data(symbol, start_date, end_date)
            if candidate_series is None:
                continue
            
            candidate_returns = candidate_series.pct_change().dropna()
            
            best_correlation = 0
            best_lag = 0
            
            for lag in range(1, max_lag + 1):
                if len(candidate_returns) < lag:
                    continue
                
                aligned = pd.DataFrame({
                    'target': target_returns,
                    'candidate': candidate_returns.shift(lag)
                }).dropna()
                
                if len(aligned) > 0:
                    correlation = aligned.corr().iloc[0, 1]
                    
                    if abs(correlation) > abs(best_correlation):
                        best_correlation = correlation
                        best_lag = lag
            
            if abs(best_correlation) > 0.3:
                leading_indicators.append({
                    'symbol': symbol,
                    'correlation': round(best_correlation, 3),
                    'lag_days': best_lag,
                    'relationship': 'POSITIVE' if best_correlation > 0 else 'NEGATIVE'
                })
        
        leading_indicators.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return leading_indicators
    
    def analyze_intermarket_relationships(self, date, lookback_days=90):
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        start_date = (date - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        end_date = date.strftime('%Y-%m-%d')
        
        key_relationships = {
            'Dollar vs Gold': ('DX-Y.NYB', 'GC=F'),
            'Stocks vs Bonds': ('^GSPC', 'TLT'),
            'Stocks vs VIX': ('^GSPC', '^VIX'),
            'EUR vs Dollar': ('EURUSD=X', 'DX-Y.NYB'),
            'Gold vs Real Yields': ('GC=F', 'TLT'),
            'Oil vs Dollar': ('CL=F', 'DX-Y.NYB')
        }
        
        relationships = {}
        
        for name, (symbol1, symbol2) in key_relationships.items():
            series1 = self.fetch_data(symbol1, start_date, end_date)
            series2 = self.fetch_data(symbol2, start_date, end_date)
            
            if series1 is not None and series2 is not None:
                df = pd.DataFrame({symbol1: series1, symbol2: series2}).dropna()
                
                if len(df) > 0:
                    correlation = df.corr().iloc[0, 1]
                    
                    current_corr = df.tail(30).corr().iloc[0, 1] if len(df) >= 30 else correlation
                    
                    relationships[name] = {
                        'symbols': [symbol1, symbol2],
                        'correlation_90d': round(correlation, 3),
                        'correlation_30d': round(current_corr, 3),
                        'relationship': 'INVERSE' if correlation < -0.3 else ('POSITIVE' if correlation > 0.3 else 'NEUTRAL'),
                        'strength': 'STRONG' if abs(correlation) > 0.7 else ('MODERATE' if abs(correlation) > 0.4 else 'WEAK')
                    }
        
        return {
            'date': end_date,
            'relationships': relationships
        }
    
    def get_correlation_analysis(self, symbol, date, lookback_days=90):
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        start_date = (date - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        end_date = date.strftime('%Y-%m-%d')
        
        all_symbols = []
        for group in self.asset_groups.values():
            all_symbols.extend(group)
        
        all_symbols = list(set(all_symbols))
        if symbol not in all_symbols:
            all_symbols.append(symbol)
        
        corr_matrix = self.calculate_correlation_matrix(all_symbols, start_date, end_date)
        
        if corr_matrix is None or symbol not in corr_matrix.columns:
            return None
        
        symbol_correlations = corr_matrix[symbol].drop(symbol).sort_values(ascending=False)
        
        top_positive = symbol_correlations.head(5).to_dict()
        top_negative = symbol_correlations.tail(5).to_dict()
        
        leading_indicators = self.find_leading_indicators(
            symbol, 
            all_symbols, 
            start_date, 
            end_date
        )
        
        intermarket = self.analyze_intermarket_relationships(date, lookback_days)
        
        return {
            'symbol': symbol,
            'date': end_date,
            'top_positive_correlations': {k: round(v, 3) for k, v in top_positive.items()},
            'top_negative_correlations': {k: round(v, 3) for k, v in top_negative.items()},
            'leading_indicators': leading_indicators[:5] if leading_indicators else [],
            'intermarket_relationships': intermarket['relationships']
        }


if __name__ == "__main__":
    analyzer = CorrelationAnalyzer()
    
    symbols = ['EURUSD=X', 'GC=F', '^GSPC']
    test_date = '2024-12-01'
    
    for symbol in symbols:
        print(f"\n{'='*80}")
        print(f"Correlation Analysis: {symbol} | {test_date}")
        print('='*80)
        
        analysis = analyzer.get_correlation_analysis(symbol, test_date, lookback_days=90)
        
        if analysis:
            print(f"\nTop Positive Correlations:")
            for sym, corr in list(analysis['top_positive_correlations'].items())[:3]:
                print(f"  {sym:20} {corr:6.3f}")
            
            print(f"\nTop Negative Correlations:")
            for sym, corr in list(analysis['top_negative_correlations'].items())[:3]:
                print(f"  {sym:20} {corr:6.3f}")
            
            if analysis['leading_indicators']:
                print(f"\nLeading Indicators:")
                for ind in analysis['leading_indicators'][:3]:
                    print(f"  {ind['symbol']:20} Lag: {ind['lag_days']}d  Corr: {ind['correlation']:6.3f}  {ind['relationship']}")
            
            print(f"\nKey Intermarket Relationships:")
            for name, data in list(analysis['intermarket_relationships'].items())[:3]:
                print(f"  {name:25} {data['relationship']:10} ({data['strength']}) - {data['correlation_90d']:.3f}")
