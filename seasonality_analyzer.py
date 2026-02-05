import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from collections import defaultdict


class SeasonalityAnalyzer:
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
            
            df = df.rename(columns={'Close': 'close', 'Open': 'open', 'High': 'high', 'Low': 'low'})
            self.cache[cache_key] = df
            return df
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None
    
    def analyze_monthly_patterns(self, df):
        df = df.copy()
        df['month'] = df.index.month
        df['returns'] = df['close'].pct_change() * 100
        
        monthly_stats = df.groupby('month')['returns'].agg(['mean', 'std', 'count']).round(3)
        monthly_stats.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        best_month = monthly_stats['mean'].idxmax()
        worst_month = monthly_stats['mean'].idxmin()
        
        return {
            'monthly_stats': monthly_stats.to_dict('index'),
            'best_month': best_month,
            'worst_month': worst_month,
            'best_return': round(monthly_stats['mean'].max(), 2),
            'worst_return': round(monthly_stats['mean'].min(), 2)
        }
    
    def analyze_day_of_week_patterns(self, df):
        df = df.copy()
        df['day_of_week'] = df.index.dayofweek
        df['returns'] = df['close'].pct_change() * 100
        
        dow_stats = df.groupby('day_of_week')['returns'].agg(['mean', 'std', 'count']).round(3)
        dow_stats.index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        
        best_day = dow_stats['mean'].idxmax()
        worst_day = dow_stats['mean'].idxmin()
        
        return {
            'dow_stats': dow_stats.to_dict('index'),
            'best_day': best_day,
            'worst_day': worst_day,
            'best_return': round(dow_stats['mean'].max(), 2),
            'worst_return': round(dow_stats['mean'].min(), 2)
        }
    
    def analyze_quarterly_patterns(self, df):
        df = df.copy()
        df['quarter'] = df.index.quarter
        df['returns'] = df['close'].pct_change() * 100
        
        quarterly_stats = df.groupby('quarter')['returns'].agg(['mean', 'std', 'count']).round(3)
        quarterly_stats.index = ['Q1', 'Q2', 'Q3', 'Q4']
        
        return quarterly_stats.to_dict('index')
    
    def analyze_hour_of_day_patterns(self, df):
        if df.index.tz is None:
            return None
        
        df = df.copy()
        df['hour'] = df.index.hour
        df['returns'] = df['close'].pct_change() * 100
        
        hourly_stats = df.groupby('hour')['returns'].agg(['mean', 'std', 'count']).round(3)
        
        return hourly_stats.to_dict('index')
    
    def detect_holiday_effects(self, df):
        df = df.copy()
        df['returns'] = df['close'].pct_change() * 100
        
        us_holidays = [
            (1, 1), (7, 4), (12, 25),  # New Year, Independence, Christmas
            (1, 15), (2, 15), (5, 25), (9, 1), (11, 25)  # Approx MLK, Presidents, Memorial, Labor, Thanksgiving
        ]
        
        pre_holiday_returns = []
        post_holiday_returns = []
        
        for idx in range(1, len(df) - 1):
            date = df.index[idx]
            next_date = df.index[idx + 1] if idx + 1 < len(df) else None
            
            if next_date:
                for month, day in us_holidays:
                    if abs(next_date.month - month) == 0 and abs(next_date.day - day) <= 3:
                        pre_holiday_returns.append(df.iloc[idx]['returns'])
                    if abs(date.month - month) == 0 and abs(date.day - day) <= 3:
                        post_holiday_returns.append(df.iloc[idx]['returns'])
        
        return {
            'pre_holiday_avg': round(np.mean(pre_holiday_returns), 3) if pre_holiday_returns else 0,
            'post_holiday_avg': round(np.mean(post_holiday_returns), 3) if post_holiday_returns else 0,
            'samples': len(pre_holiday_returns)
        }
    
    def analyze_turn_of_month_effect(self, df):
        df = df.copy()
        df['day'] = df.index.day
        df['returns'] = df['close'].pct_change() * 100
        
        early_month = df[df['day'] <= 5]['returns'].mean()
        mid_month = df[(df['day'] > 5) & (df['day'] <= 25)]['returns'].mean()
        late_month = df[df['day'] > 25]['returns'].mean()
        
        return {
            'early_month': round(early_month, 3),
            'mid_month': round(mid_month, 3),
            'late_month': round(late_month, 3),
            'turn_of_month_premium': round(early_month - mid_month, 3)
        }
    
    def get_seasonality_analysis(self, symbol, years_back=5):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years_back * 365)
        
        df = self.fetch_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        if df is None or df.empty:
            return None
        
        monthly = self.analyze_monthly_patterns(df)
        dow = self.analyze_day_of_week_patterns(df)
        quarterly = self.analyze_quarterly_patterns(df)
        holiday = self.detect_holiday_effects(df)
        turn_month = self.analyze_turn_of_month_effect(df)
        
        current_month = datetime.now().strftime('%b')
        current_day = datetime.now().strftime('%A')
        
        current_month_bias = monthly['monthly_stats'].get(current_month, {}).get('mean', 0)
        current_day_bias = dow['dow_stats'].get(current_day, {}).get('mean', 0)
        
        return {
            'symbol': symbol,
            'analysis_period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            'monthly_patterns': monthly,
            'day_of_week_patterns': dow,
            'quarterly_patterns': quarterly,
            'holiday_effects': holiday,
            'turn_of_month': turn_month,
            'current_biases': {
                'month': current_month,
                'month_avg_return': round(current_month_bias, 2),
                'day': current_day,
                'day_avg_return': round(current_day_bias, 2)
            }
        }


if __name__ == "__main__":
    analyzer = SeasonalityAnalyzer()
    
    symbols = ['AAPL', 'GC=F', '^GSPC']
    
    for symbol in symbols:
        print(f"\n{'='*80}")
        print(f"Seasonality Analysis: {symbol}")
        print('='*80)
        
        analysis = analyzer.get_seasonality_analysis(symbol, years_back=5)
        
        if analysis:
            print(f"\nBest Performing Month: {analysis['monthly_patterns']['best_month']} "
                  f"({analysis['monthly_patterns']['best_return']}%)")
            print(f"Worst Performing Month: {analysis['monthly_patterns']['worst_month']} "
                  f"({analysis['monthly_patterns']['worst_return']}%)")
            
            print(f"\nBest Day of Week: {analysis['day_of_week_patterns']['best_day']} "
                  f"({analysis['day_of_week_patterns']['best_return']}%)")
            print(f"Worst Day of Week: {analysis['day_of_week_patterns']['worst_day']} "
                  f"({analysis['day_of_week_patterns']['worst_return']}%)")
            
            print(f"\nCurrent Period Biases:")
            print(f"  {analysis['current_biases']['month']}: {analysis['current_biases']['month_avg_return']}% avg")
            print(f"  {analysis['current_biases']['day']}: {analysis['current_biases']['day_avg_return']}% avg")
            
            print(f"\nTurn of Month Effect: {analysis['turn_of_month']['turn_of_month_premium']}% premium")
            print(f"Holiday Effects: Pre={analysis['holiday_effects']['pre_holiday_avg']}% "
                  f"Post={analysis['holiday_effects']['post_holiday_avg']}%")
