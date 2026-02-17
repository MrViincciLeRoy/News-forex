"""
Symbol Indicator Calculator - Technical analysis for symbols
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict
from datetime import datetime, timedelta

class SymbolIndicatorCalculator:
    
    def __init__(self):
        self.cache = {}
    
    def get_historical_data(self, symbol: str, period: str = '3mo') -> Optional[pd.DataFrame]:
        """Get historical price data"""
        
        # Try to use yfinance
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            return data
        except:
            pass
        
        # Fallback to mock data
        return self._generate_mock_data(symbol, period)
    
    def _generate_mock_data(self, symbol: str, period: str) -> pd.DataFrame:
        """Generate mock price data for testing"""
        
        days = {
            '5d': 5,
            '1mo': 30,
            '3mo': 90,
            '6mo': 180,
            '1y': 365,
            '5y': 1825
        }
        
        num_days = days.get(period, 90)
        
        # Generate realistic-looking price data
        np.random.seed(hash(symbol) % 1000)
        base_price = np.random.uniform(50, 200)
        
        dates = pd.date_range(end=datetime.now(), periods=num_days, freq='D')
        
        # Generate prices with trend and noise
        trend = np.linspace(0, np.random.uniform(-0.2, 0.2), num_days)
        noise = np.random.normal(0, 0.02, num_days)
        returns = trend + noise
        
        prices = base_price * (1 + returns).cumprod()
        
        # Create OHLCV data
        data = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.01, num_days)),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.015, num_days))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.015, num_days))),
            'Close': prices,
            'Volume': np.random.lognormal(15, 1, num_days)
        }, index=dates)
        
        # Ensure High is highest and Low is lowest
        data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
        data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)
        
        return data
    
    def calculate_indicators(self, symbol: str, data: pd.DataFrame) -> Dict:
        """Calculate all technical indicators"""
        
        indicators = {}
        
        try:
            # RSI
            indicators['RSI'] = self._calculate_rsi(data['Close'])
            
            # MACD
            macd, signal, hist = self._calculate_macd(data['Close'])
            indicators['MACD'] = {'macd': macd, 'signal': signal, 'histogram': hist}
            
            # Moving Averages
            indicators['SMA_20'] = float(data['Close'].rolling(20).mean().iloc[-1])
            indicators['SMA_50'] = float(data['Close'].rolling(50).mean().iloc[-1])
            indicators['SMA_200'] = float(data['Close'].rolling(200).mean().iloc[-1]) if len(data) > 200 else None
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(data['Close'])
            indicators['BB_UPPER'] = bb_upper
            indicators['BB_MIDDLE'] = bb_middle
            indicators['BB_LOWER'] = bb_lower
            
            # ATR
            indicators['ATR'] = self._calculate_atr(data)
            
            # Stochastic
            indicators['STOCH_K'], indicators['STOCH_D'] = self._calculate_stochastic(data)
            
            # ADX
            indicators['ADX'] = self._calculate_adx(data)
            
            # OBV
            indicators['OBV'] = self._calculate_obv(data)
            
        except Exception as e:
            print(f"Error calculating indicators for {symbol}: {str(e)}")
        
        return indicators
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1])
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return float(macd_line.iloc[-1]), float(signal_line.iloc[-1]), float(histogram.iloc[-1])
    
    def _calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        return float(upper.iloc[-1]), float(sma.iloc[-1]), float(lower.iloc[-1])
    
    def _calculate_atr(self, data, period=14):
        """Calculate ATR"""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        atr = true_range.rolling(period).mean()
        return float(atr.iloc[-1])
    
    def _calculate_stochastic(self, data, period=14):
        """Calculate Stochastic Oscillator"""
        low_min = data['Low'].rolling(period).min()
        high_max = data['High'].rolling(period).max()
        
        k = 100 * (data['Close'] - low_min) / (high_max - low_min)
        d = k.rolling(3).mean()
        
        return float(k.iloc[-1]), float(d.iloc[-1])
    
    def _calculate_adx(self, data, period=14):
        """Calculate ADX"""
        high_diff = data['High'].diff()
        low_diff = -data['Low'].diff()
        
        pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        atr = self._calculate_atr(data, period)
        
        pos_di = 100 * (pos_dm.rolling(period).mean() / atr)
        neg_di = 100 * (neg_dm.rolling(period).mean() / atr)
        
        dx = 100 * np.abs(pos_di - neg_di) / (pos_di + neg_di)
        adx = dx.rolling(period).mean()
        
        return float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 25.0
    
    def _calculate_obv(self, data):
        """Calculate OBV"""
        obv = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
        return float(obv.iloc[-1])
