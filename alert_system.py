import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json


class AlertSystem:
    def __init__(self):
        self.alerts = []
        self.cache = {}
    
    def fetch_data(self, symbol, lookback_days=30):
        cache_key = f"{symbol}_{lookback_days}"
        
        if cache_key in self.cache:
            cache_time, data = self.cache[cache_key]
            if (datetime.now() - cache_time).seconds < 300:
                return data
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date.strftime('%Y-%m-%d'))
            
            if df.empty:
                return None
            
            df = df.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low', 
                'Close': 'close', 'Volume': 'volume'
            })
            
            self.cache[cache_key] = (datetime.now(), df)
            return df
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None
    
    def check_price_level(self, symbol, target_price, condition='above'):
        df = self.fetch_data(symbol, lookback_days=5)
        if df is None or df.empty:
            return None
        
        current_price = df['close'].iloc[-1]
        
        triggered = False
        if condition == 'above' and current_price > target_price:
            triggered = True
        elif condition == 'below' and current_price < target_price:
            triggered = True
        
        if triggered:
            return {
                'type': 'price_level',
                'symbol': symbol,
                'current_price': round(current_price, 2),
                'target_price': target_price,
                'condition': condition,
                'message': f"{symbol} is {condition} ${target_price} (current: ${current_price:.2f})",
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        
        return None
    
    def check_ma_crossover(self, symbol, short_period=20, long_period=50):
        df = self.fetch_data(symbol, lookback_days=max(short_period, long_period) + 10)
        if df is None or df.empty:
            return None
        
        df['sma_short'] = df['close'].rolling(window=short_period).mean()
        df['sma_long'] = df['close'].rolling(window=long_period).mean()
        
        current_short = df['sma_short'].iloc[-1]
        current_long = df['sma_long'].iloc[-1]
        prev_short = df['sma_short'].iloc[-2]
        prev_long = df['sma_long'].iloc[-2]
        
        crossover_type = None
        if prev_short <= prev_long and current_short > current_long:
            crossover_type = 'BULLISH'
        elif prev_short >= prev_long and current_short < current_long:
            crossover_type = 'BEARISH'
        
        if crossover_type:
            return {
                'type': 'ma_crossover',
                'symbol': symbol,
                'crossover': crossover_type,
                'short_ma': round(current_short, 2),
                'long_ma': round(current_long, 2),
                'message': f"{symbol} {crossover_type} MA crossover: SMA{short_period} crossed {'above' if crossover_type == 'BULLISH' else 'below'} SMA{long_period}",
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        
        return None
    
    def check_rsi_extreme(self, symbol, oversold=30, overbought=70, period=14):
        df = self.fetch_data(symbol, lookback_days=period + 10)
        if df is None or df.empty:
            return None
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = rsi.iloc[-1]
        
        alert_type = None
        if current_rsi < oversold:
            alert_type = 'OVERSOLD'
        elif current_rsi > overbought:
            alert_type = 'OVERBOUGHT'
        
        if alert_type:
            return {
                'type': 'rsi_extreme',
                'symbol': symbol,
                'rsi': round(current_rsi, 2),
                'condition': alert_type,
                'message': f"{symbol} RSI is {alert_type}: {current_rsi:.2f}",
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        
        return None
    
    def check_volume_spike(self, symbol, multiplier=2.0, lookback=20):
        df = self.fetch_data(symbol, lookback_days=lookback + 5)
        if df is None or df.empty:
            return None
        
        avg_volume = df['volume'].iloc[:-1].mean()
        current_volume = df['volume'].iloc[-1]
        
        if current_volume > avg_volume * multiplier:
            return {
                'type': 'volume_spike',
                'symbol': symbol,
                'current_volume': int(current_volume),
                'avg_volume': int(avg_volume),
                'multiplier': round(current_volume / avg_volume, 2),
                'message': f"{symbol} volume spike: {current_volume:,} ({current_volume/avg_volume:.2f}x average)",
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        
        return None
    
    def check_volatility_breakout(self, symbol, period=20, threshold=1.5):
        df = self.fetch_data(symbol, lookback_days=period + 10)
        if df is None or df.empty:
            return None
        
        df['returns'] = df['close'].pct_change()
        volatility = df['returns'].rolling(window=period).std()
        
        current_vol = volatility.iloc[-1]
        avg_vol = volatility.mean()
        
        if current_vol > avg_vol * threshold:
            return {
                'type': 'volatility_breakout',
                'symbol': symbol,
                'current_volatility': round(current_vol * 100, 2),
                'avg_volatility': round(avg_vol * 100, 2),
                'message': f"{symbol} volatility breakout: {current_vol*100:.2f}% ({current_vol/avg_vol:.2f}x average)",
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        
        return None
    
    def check_support_resistance_break(self, symbol, level, break_type='resistance'):
        df = self.fetch_data(symbol, lookback_days=5)
        if df is None or df.empty:
            return None
        
        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2]
        
        broken = False
        if break_type == 'resistance' and prev_price <= level and current_price > level:
            broken = True
        elif break_type == 'support' and prev_price >= level and current_price < level:
            broken = True
        
        if broken:
            return {
                'type': 'level_break',
                'symbol': symbol,
                'level': level,
                'break_type': break_type,
                'current_price': round(current_price, 2),
                'message': f"{symbol} broke {break_type} at ${level} (current: ${current_price:.2f})",
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        
        return None
    
    def scan_multiple_symbols(self, symbols, alert_types=['ma_crossover', 'rsi_extreme', 'volume_spike']):
        alerts = []
        
        for symbol in symbols:
            for alert_type in alert_types:
                alert = None
                
                if alert_type == 'ma_crossover':
                    alert = self.check_ma_crossover(symbol)
                elif alert_type == 'rsi_extreme':
                    alert = self.check_rsi_extreme(symbol)
                elif alert_type == 'volume_spike':
                    alert = self.check_volume_spike(symbol)
                elif alert_type == 'volatility_breakout':
                    alert = self.check_volatility_breakout(symbol)
                
                if alert:
                    alerts.append(alert)
        
        return alerts
    
    def save_alerts(self, alerts, filename='alerts.json'):
        with open(filename, 'w') as f:
            json.dump(alerts, f, indent=2)
    
    def get_watchlist_status(self, watchlist):
        status = []
        
        for item in watchlist:
            symbol = item['symbol']
            df = self.fetch_data(symbol, lookback_days=30)
            
            if df is None or df.empty:
                continue
            
            current_price = df['close'].iloc[-1]
            change_pct = ((current_price - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100
            
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            
            trend = 'BULLISH' if df['sma_20'].iloc[-1] > df['sma_50'].iloc[-1] else 'BEARISH'
            
            status.append({
                'symbol': symbol,
                'price': round(current_price, 2),
                'change_pct': round(change_pct, 2),
                'rsi': round(current_rsi, 2),
                'trend': trend,
                'target_price': item.get('target_price'),
                'stop_loss': item.get('stop_loss')
            })
        
        return status


if __name__ == "__main__":
    alert_system = AlertSystem()
    
    watchlist = [
        {'symbol': 'AAPL', 'target_price': 200, 'stop_loss': 150},
        {'symbol': 'MSFT', 'target_price': 400, 'stop_loss': 350},
        {'symbol': 'GOOGL', 'target_price': 150, 'stop_loss': 130}
    ]
    
    print("="*80)
    print("ALERT SYSTEM - Watchlist Status")
    print("="*80)
    
    status = alert_system.get_watchlist_status(watchlist)
    
    for item in status:
        print(f"\n{item['symbol']}:")
        print(f"  Price: ${item['price']} ({item['change_pct']:+.2f}%)")
        print(f"  RSI: {item['rsi']:.2f}")
        print(f"  Trend: {item['trend']}")
        if item['target_price']:
            print(f"  Target: ${item['target_price']}")
    
    print("\n" + "="*80)
    print("SCANNING FOR ALERTS")
    print("="*80)
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    alerts = alert_system.scan_multiple_symbols(
        symbols, 
        alert_types=['ma_crossover', 'rsi_extreme', 'volume_spike']
    )
    
    if alerts:
        for alert in alerts:
            print(f"\n⚠️  {alert['message']}")
    else:
        print("\nNo alerts triggered")
