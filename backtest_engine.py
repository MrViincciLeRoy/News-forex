import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta


class BacktestEngine:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.results = None
    
    def fetch_data(self, symbol, start_date, end_date):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                return None
            
            df = df.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low',
                'Close': 'close', 'Volume': 'volume'
            })
            
            return df
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None
    
    def calculate_indicators(self, df):
        df = df.copy()
        
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['atr'] = ranges.max(axis=1).rolling(14).mean()
        
        return df
    
    def generate_signals(self, df, strategy='sma_cross'):
        df = df.copy()
        df['signal'] = 0
        
        if strategy == 'sma_cross':
            df.loc[df['sma_20'] > df['sma_50'], 'signal'] = 1
            df.loc[df['sma_20'] < df['sma_50'], 'signal'] = -1
        
        elif strategy == 'macd':
            df.loc[df['macd'] > df['macd_signal'], 'signal'] = 1
            df.loc[df['macd'] < df['macd_signal'], 'signal'] = -1
        
        elif strategy == 'rsi':
            df.loc[df['rsi'] < 30, 'signal'] = 1
            df.loc[df['rsi'] > 70, 'signal'] = -1
        
        elif strategy == 'trend_following':
            df.loc[(df['close'] > df['sma_200']) & (df['sma_20'] > df['sma_50']), 'signal'] = 1
            df.loc[(df['close'] < df['sma_200']) & (df['sma_20'] < df['sma_50']), 'signal'] = -1
        
        df['position'] = df['signal'].diff()
        
        return df
    
    def run_backtest(self, symbol, start_date, end_date, strategy='sma_cross', 
                    risk_per_trade=0.02, use_stops=True):
        
        df = self.fetch_data(symbol, start_date, end_date)
        
        if df is None or df.empty:
            return None
        
        df = self.calculate_indicators(df)
        df = self.generate_signals(df, strategy)
        
        capital = self.initial_capital
        position = 0
        entry_price = 0
        trades = []
        equity_curve = []
        
        for i in range(len(df)):
            row = df.iloc[i]
            date = df.index[i]
            price = row['close']
            
            if use_stops and position != 0:
                stop_loss = entry_price - (row['atr'] * 2) if position > 0 else entry_price + (row['atr'] * 2)
                
                if (position > 0 and price <= stop_loss) or (position < 0 and price >= stop_loss):
                    pnl = position * (price - entry_price)
                    capital += pnl
                    
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': price,
                        'position': position,
                        'pnl': pnl,
                        'pnl_pct': (pnl / (abs(position) * entry_price)) * 100,
                        'exit_reason': 'stop_loss'
                    })
                    
                    position = 0
            
            if row['position'] == 2:
                if position == 0:
                    risk_amount = capital * risk_per_trade
                    position_size = risk_amount / (row['atr'] * 2)
                    position = position_size
                    entry_price = price
                    entry_date = date
                
                elif position < 0:
                    pnl = position * (price - entry_price)
                    capital += pnl
                    
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': price,
                        'position': position,
                        'pnl': pnl,
                        'pnl_pct': (pnl / (abs(position) * entry_price)) * 100,
                        'exit_reason': 'signal_reversal'
                    })
                    
                    risk_amount = capital * risk_per_trade
                    position_size = risk_amount / (row['atr'] * 2)
                    position = position_size
                    entry_price = price
                    entry_date = date
            
            elif row['position'] == -2:
                if position == 0:
                    risk_amount = capital * risk_per_trade
                    position_size = risk_amount / (row['atr'] * 2)
                    position = -position_size
                    entry_price = price
                    entry_date = date
                
                elif position > 0:
                    pnl = position * (price - entry_price)
                    capital += pnl
                    
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': price,
                        'position': position,
                        'pnl': pnl,
                        'pnl_pct': (pnl / (abs(position) * entry_price)) * 100,
                        'exit_reason': 'signal_reversal'
                    })
                    
                    risk_amount = capital * risk_per_trade
                    position_size = risk_amount / (row['atr'] * 2)
                    position = -position_size
                    entry_price = price
                    entry_date = date
            
            current_value = capital
            if position != 0:
                current_value += position * (price - entry_price)
            
            equity_curve.append({
                'date': date,
                'equity': current_value,
                'position': position
            })
        
        if position != 0:
            final_price = df.iloc[-1]['close']
            pnl = position * (final_price - entry_price)
            capital += pnl
            
            trades.append({
                'entry_date': entry_date,
                'exit_date': df.index[-1],
                'entry_price': entry_price,
                'exit_price': final_price,
                'position': position,
                'pnl': pnl,
                'pnl_pct': (pnl / (abs(position) * entry_price)) * 100,
                'exit_reason': 'end_of_period'
            })
        
        self.results = {
            'trades': trades,
            'equity_curve': equity_curve,
            'final_capital': capital
        }
        
        return self.calculate_performance_metrics()
    
    def calculate_performance_metrics(self):
        if not self.results or not self.results['trades']:
            return None
        
        trades = pd.DataFrame(self.results['trades'])
        equity = pd.DataFrame(self.results['equity_curve'])
        
        total_trades = len(trades)
        winning_trades = len(trades[trades['pnl'] > 0])
        losing_trades = len(trades[trades['pnl'] < 0])
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        total_profit = trades[trades['pnl'] > 0]['pnl'].sum()
        total_loss = abs(trades[trades['pnl'] < 0]['pnl'].sum())
        
        profit_factor = (total_profit / total_loss) if total_loss > 0 else 0
        
        avg_win = trades[trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = abs(trades[trades['pnl'] < 0]['pnl'].mean()) if losing_trades > 0 else 0
        
        total_return = ((self.results['final_capital'] - self.initial_capital) / self.initial_capital) * 100
        
        equity['returns'] = equity['equity'].pct_change()
        sharpe_ratio = (equity['returns'].mean() / equity['returns'].std()) * np.sqrt(252) if equity['returns'].std() > 0 else 0
        
        equity['cummax'] = equity['equity'].cummax()
        equity['drawdown'] = (equity['equity'] - equity['cummax']) / equity['cummax']
        max_drawdown = equity['drawdown'].min() * 100
        
        downside_returns = equity['returns'][equity['returns'] < 0]
        downside_std = downside_returns.std()
        sortino_ratio = (equity['returns'].mean() / downside_std) * np.sqrt(252) if downside_std > 0 else 0
        
        return {
            'summary': {
                'initial_capital': self.initial_capital,
                'final_capital': round(self.results['final_capital'], 2),
                'total_return': round(total_return, 2),
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': round(win_rate, 2)
            },
            'performance': {
                'profit_factor': round(profit_factor, 2),
                'avg_win': round(avg_win, 2),
                'avg_loss': round(avg_loss, 2),
                'avg_win_loss_ratio': round(avg_win / avg_loss, 2) if avg_loss > 0 else 0,
                'sharpe_ratio': round(sharpe_ratio, 2),
                'sortino_ratio': round(sortino_ratio, 2),
                'max_drawdown': round(max_drawdown, 2)
            },
            'trades': trades.to_dict('records')[:10],
            'equity_curve': equity[['date', 'equity']].to_dict('records')
        }


if __name__ == "__main__":
    engine = BacktestEngine(initial_capital=10000)
    
    strategies = ['sma_cross', 'macd', 'trend_following']
    symbol = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2024-12-01'
    
    for strategy in strategies:
        print(f"\n{'='*80}")
        print(f"Backtesting {strategy.upper()} on {symbol}")
        print('='*80)
        
        results = engine.run_backtest(symbol, start_date, end_date, strategy, risk_per_trade=0.02)
        
        if results:
            print(f"\nInitial Capital: ${results['summary']['initial_capital']:,}")
            print(f"Final Capital: ${results['summary']['final_capital']:,}")
            print(f"Total Return: {results['summary']['total_return']}%")
            print(f"Total Trades: {results['summary']['total_trades']}")
            print(f"Win Rate: {results['summary']['win_rate']}%")
            
            print(f"\nPerformance Metrics:")
            print(f"  Profit Factor: {results['performance']['profit_factor']}")
            print(f"  Sharpe Ratio: {results['performance']['sharpe_ratio']}")
            print(f"  Max Drawdown: {results['performance']['max_drawdown']}%")
            print(f"  Avg Win/Loss: {results['performance']['avg_win_loss_ratio']}")
