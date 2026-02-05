import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta


class RiskManager:
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
            
            self.cache[cache_key] = df
            return df
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None
    
    def calculate_atr(self, df, period=14):
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def calculate_position_size(self, account_size, risk_percent, entry_price, stop_loss_price):
        risk_amount = account_size * (risk_percent / 100)
        price_risk = abs(entry_price - stop_loss_price)
        
        if price_risk == 0:
            return None
        
        position_size = risk_amount / price_risk
        position_value = position_size * entry_price
        
        return {
            'units': round(position_size, 2),
            'position_value': round(position_value, 2),
            'risk_amount': round(risk_amount, 2),
            'risk_per_unit': round(price_risk, 2)
        }
    
    def calculate_volatility_based_position(self, symbol, account_size, risk_percent, lookback_days=30):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days + 30)
        
        df = self.fetch_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        if df is None or df.empty:
            return None
        
        atr = self.calculate_atr(df)
        current_atr = atr.iloc[-1]
        current_price = df['Close'].iloc[-1]
        
        risk_amount = account_size * (risk_percent / 100)
        
        position_size = risk_amount / (current_atr * 2)
        position_value = position_size * current_price
        
        return {
            'symbol': symbol,
            'current_price': round(current_price, 2),
            'atr': round(current_atr, 2),
            'recommended_units': round(position_size, 2),
            'position_value': round(position_value, 2),
            'stop_loss_2atr': round(current_price - (current_atr * 2), 2),
            'take_profit_3atr': round(current_price + (current_atr * 3), 2),
            'risk_amount': round(risk_amount, 2)
        }
    
    def calculate_stop_loss_levels(self, entry_price, atr, method='atr'):
        levels = {}
        
        if method == 'atr':
            levels = {
                'tight': round(entry_price - (atr * 1), 2),
                'moderate': round(entry_price - (atr * 2), 2),
                'wide': round(entry_price - (atr * 3), 2)
            }
        elif method == 'percentage':
            levels = {
                'tight': round(entry_price * 0.98, 2),
                'moderate': round(entry_price * 0.95, 2),
                'wide': round(entry_price * 0.90, 2)
            }
        
        return levels
    
    def calculate_take_profit_levels(self, entry_price, stop_loss, risk_reward_ratios=[1.5, 2, 3]):
        risk = abs(entry_price - stop_loss)
        
        levels = {}
        for rr in risk_reward_ratios:
            levels[f'RR_{rr}'] = round(entry_price + (risk * rr), 2)
        
        return levels
    
    def calculate_risk_reward_ratio(self, entry_price, stop_loss, take_profit):
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        
        if risk == 0:
            return None
        
        return round(reward / risk, 2)
    
    def calculate_max_drawdown(self, symbol, lookback_days=365):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        df = self.fetch_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        if df is None or df.empty:
            return None
        
        cumulative_returns = (1 + df['Close'].pct_change()).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        
        recovery_date = None
        if max_dd_date in drawdown.index:
            post_dd = drawdown[drawdown.index > max_dd_date]
            recovery_idx = post_dd[post_dd >= 0].first_valid_index()
            if recovery_idx:
                recovery_date = recovery_idx
        
        return {
            'max_drawdown': round(max_dd * 100, 2),
            'drawdown_date': max_dd_date.strftime('%Y-%m-%d'),
            'recovery_date': recovery_date.strftime('%Y-%m-%d') if recovery_date else 'Not recovered',
            'current_drawdown': round(drawdown.iloc[-1] * 100, 2)
        }
    
    def calculate_kelly_criterion(self, win_rate, avg_win, avg_loss):
        if avg_loss == 0:
            return None
        
        win_loss_ratio = avg_win / avg_loss
        kelly_pct = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        kelly_fraction = max(0, kelly_pct) * 0.25
        
        return {
            'kelly_percentage': round(kelly_pct * 100, 2),
            'recommended_fraction': round(kelly_fraction * 100, 2),
            'win_loss_ratio': round(win_loss_ratio, 2)
        }
    
    def get_risk_analysis(self, symbol, account_size=10000, risk_percent=2, lookback_days=90):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days + 30)
        
        df = self.fetch_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        if df is None or df.empty:
            return None
        
        atr = self.calculate_atr(df)
        current_atr = atr.iloc[-1]
        current_price = df['Close'].iloc[-1]
        
        volatility_position = self.calculate_volatility_based_position(
            symbol, account_size, risk_percent, lookback_days
        )
        
        stop_loss_levels = self.calculate_stop_loss_levels(current_price, current_atr, 'atr')
        
        take_profit_levels = self.calculate_take_profit_levels(
            current_price, 
            stop_loss_levels['moderate'],
            [1.5, 2, 3]
        )
        
        drawdown = self.calculate_max_drawdown(symbol, lookback_days)
        
        returns = df['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100
        
        return {
            'symbol': symbol,
            'current_price': round(current_price, 2),
            'account_size': account_size,
            'risk_per_trade': risk_percent,
            'atr': round(current_atr, 2),
            'volatility_annual': round(volatility, 2),
            'position_sizing': volatility_position,
            'stop_loss_levels': stop_loss_levels,
            'take_profit_levels': take_profit_levels,
            'drawdown_analysis': drawdown,
            'risk_metrics': {
                'max_position_value': round(account_size * 0.20, 2),
                'max_portfolio_risk': '6% (3 positions at 2% each)',
                'leverage_warning': 'Not recommended' if volatility > 30 else 'Low risk'
            }
        }


if __name__ == "__main__":
    rm = RiskManager()
    
    symbols = ['AAPL', 'GC=F', 'EURUSD=X']
    account_size = 10000
    risk_percent = 2
    
    for symbol in symbols:
        print(f"\n{'='*80}")
        print(f"Risk Management Analysis: {symbol}")
        print('='*80)
        
        analysis = rm.get_risk_analysis(symbol, account_size, risk_percent)
        
        if analysis:
            print(f"\nAccount: ${account_size:,} | Risk per trade: {risk_percent}%")
            print(f"Current Price: ${analysis['current_price']:,}")
            print(f"ATR: ${analysis['atr']:.2f}")
            print(f"Annual Volatility: {analysis['volatility_annual']:.1f}%")
            
            print(f"\nPosition Sizing:")
            print(f"  Recommended Units: {analysis['position_sizing']['recommended_units']}")
            print(f"  Position Value: ${analysis['position_sizing']['position_value']:,.2f}")
            
            print(f"\nStop Loss Levels (ATR-based):")
            for level, price in analysis['stop_loss_levels'].items():
                print(f"  {level.capitalize()}: ${price}")
            
            print(f"\nTake Profit Targets:")
            for level, price in analysis['take_profit_levels'].items():
                print(f"  {level}: ${price}")
            
            print(f"\nDrawdown Analysis:")
            print(f"  Max Drawdown: {analysis['drawdown_analysis']['max_drawdown']}%")
            print(f"  Current Drawdown: {analysis['drawdown_analysis']['current_drawdown']}%")
