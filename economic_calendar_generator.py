import pandas as pd
import asyncio
import aiohttp
import numpy as np
import os
import json
from datetime import datetime, timedelta

class GoldIndicatorCalculator:
    def __init__(self):
        self.alpha_keys = self._load_alpha_keys()
        self.fred_key = os.environ.get('FRED_API_KEY', '')
        self.current_key_index = 0
        self.key_lock = asyncio.Lock()
        self.gold_cache = None
    
    def _load_alpha_keys(self):
        keys = []
        i = 1
        while True:
            key = os.environ.get(f'ALPHA_VANTAGE_API_KEY_{i}')
            if key:
                keys.append(key)
                i += 1
            else:
                break
        
        if not keys:
            single_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
            if single_key:
                keys.append(single_key)
        
        if not keys:
            print("⚠️  No Alpha Vantage API keys found in environment variables")
            print("   Set ALPHA_VANTAGE_API_KEY or ALPHA_VANTAGE_API_KEY_1, _2, etc.")
        
        return keys
    
    async def get_next_alpha_key(self):
        async with self.key_lock:
            key = self.alpha_keys[self.current_key_index]
            self.current_key_index = (self.current_key_index + 1) % len(self.alpha_keys)
            return key
    
    async def fetch_fred_gold_data(self, start_date='2001-01-01', end_date=None):
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        url = 'https://api.stlouisfed.org/fred/series/observations'
        params = {
            'series_id': 'GOLDAMGBD228NLBM',
            'api_key': self.fred_key,
            'file_type': 'json',
            'observation_start': start_date,
            'observation_end': end_date
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
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
    
    async def fetch_alpha_vantage_gold(self, start_date='2001-01-01'):
        api_key = await self.get_next_alpha_key()
        url = 'https://www.alphavantage.co/query'
        params = {
            'function': 'FX_DAILY',
            'from_symbol': 'XAU',
            'to_symbol': 'USD',
            'outputsize': 'full',
            'apikey': api_key
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'Time Series FX (Daily)' in data:
                            df = pd.DataFrame.from_dict(
                                data['Time Series FX (Daily)'], 
                                orient='index'
                            )
                            df.index = pd.to_datetime(df.index)
                            df = df.rename(columns={
                                '1. open': 'open',
                                '2. high': 'high',
                                '3. low': 'low',
                                '4. close': 'close'
                            })
                            df = df.astype(float)
                            df['volume'] = 1000000
                            df = df.sort_index()
                            df = df[df.index >= start_date]
                            return df
        except Exception as e:
            print(f"Alpha Vantage error: {e}")
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
    
    def get_signals_for_date(self, df, date):
        try:
            if isinstance(date, str):
                date = pd.to_datetime(date)
            
            df_sorted = df.sort_index()
            if date not in df_sorted.index:
                idx = df_sorted.index.searchsorted(date)
                if idx >= len(df_sorted):
                    idx = len(df_sorted) - 1
                row = df_sorted.iloc[idx]
            else:
                row = df_sorted.loc[date]
            
            return self._format_indicators(row)
        except Exception as e:
            print(f"Error getting signals: {e}")
            return None
    
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
    
    async def load_and_cache_gold_data(self, start_year, end_year):
        start_date = f'{start_year}-01-01'
        end_date = f'{end_year}-12-31'
        
        print("Fetching gold data from FRED...")
        df = await self.fetch_fred_gold_data(start_date, end_date)
        
        if df is None and self.alpha_keys:
            print("FRED failed, trying Alpha Vantage...")
            df = await self.fetch_alpha_vantage_gold(start_date)
        
        if df is not None:
            print(f"Got {len(df)} days of gold data")
            print("Calculating indicators...")
            self.gold_cache = self.calculate_indicators(df)
            return True
        return False


async def get_gdelt_news_async(session, event_name, event_date, num_articles=2):
    keywords_map = {
        'Non-Farm Payrolls': 'nonfarm payrolls jobs report economy',
        'Consumer Price Index': 'CPI inflation consumer prices',
        'Producer Price Index': 'PPI producer prices inflation',
        'Retail Sales': 'retail sales consumer spending economy',
        'ISM Manufacturing': 'ISM manufacturing PMI economy',
        'ISM Services': 'ISM services PMI economy',
        'FOMC': 'Federal Reserve FOMC interest rate',
        'Jobless Claims': 'unemployment jobless claims'
    }
    
    search_terms = 'economic data US'
    for key, value in keywords_map.items():
        if key.lower() in event_name.lower():
            search_terms = value
            break
    
    date_obj = datetime.strptime(event_date, '%Y-%m-%d')
    start_datetime = date_obj.strftime('%Y%m%d') + '000000'
    end_datetime = date_obj.strftime('%Y%m%d') + '235959'
    
    url = 'https://api.gdeltproject.org/api/v2/doc/doc'
    params = {
        'query': search_terms,
        'mode': 'artlist',
        'maxrecords': num_articles * 3,
        'format': 'json',
        'startdatetime': start_datetime,
        'enddatetime': end_datetime,
        'sourcelang': 'english',
        'theme': 'TAX_FNCACT_ECONOMY'
    }
    
    try:
        async with session.get(url, params=params, timeout=15) as response:
            if response.status == 200:
                data = await response.json()
                articles = []
                if 'articles' in data:
                    for article in data['articles'][:num_articles]:
                        articles.append({
                            'title': article.get('title', 'No title'),
                            'url': article.get('url', ''),
                            'source': article.get('domain', 'Unknown')
                        })
                return articles
    except:
        pass
    return []


async def process_event(session, event_date, event_name, gold_calc, fetch_news):
    event = {
        'date': event_date,
        'time': '08:30',
        'event': event_name,
        'impact': 'High',
        'frequency': 'Monthly',
        'indicators': None,
        'news': []
    }
    
    if gold_calc and gold_calc.gold_cache is not None:
        event['indicators'] = gold_calc.get_signals_for_date(gold_calc.gold_cache, event_date)
    
    if fetch_news:
        event['news'] = await get_gdelt_news_async(session, event_name, event_date, 2)
    
    return event


def get_nth_weekday(year, month, n, weekday):
    first_day = datetime(year, month, 1)
    first_weekday = first_day.weekday()
    offset = (weekday - first_weekday) % 7
    target_date = first_day + timedelta(days=offset + (n - 1) * 7)
    if target_date.month == month:
        return target_date
    return None


async def create_calendar_with_gold_indicators(start_year=2024, end_year=2025, fetch_news=True):
    print("=" * 80)
    print(f"ECONOMIC CALENDAR + GOLD TECHNICAL INDICATORS + NEWS ({start_year}-{end_year})")
    print("=" * 80 + "\n")
    
    gold_calc = GoldIndicatorCalculator()
    success = await gold_calc.load_and_cache_gold_data(start_year, end_year)
    
    if not success:
        print("⚠️  Could not load gold data, continuing without indicators")
        gold_calc = None
    
    all_events_tasks = []
    
    async with aiohttp.ClientSession() as session:
        for year in range(start_year, end_year + 1):
            print(f"Preparing {year}...", flush=True)
            
            for month in range(1, 13):
                first_friday = get_nth_weekday(year, month, 1, 4)
                if first_friday:
                    event_date = first_friday.strftime('%Y-%m-%d')
                    all_events_tasks.append(
                        process_event(session, event_date, 'Non-Farm Payrolls (NFP)', 
                                    gold_calc, fetch_news)
                    )
                
                cpi_date = datetime(year, month, 13)
                while cpi_date.weekday() >= 5:
                    cpi_date += timedelta(days=1)
                event_date = cpi_date.strftime('%Y-%m-%d')
                all_events_tasks.append(
                    process_event(session, event_date, 'Consumer Price Index (CPI)', 
                                gold_calc, fetch_news)
                )
        
        print(f"\nProcessing {len(all_events_tasks)} events...")
        events = await asyncio.gather(*all_events_tasks)
    
    events = sorted(events, key=lambda x: x['date'])
    
    events_with_indicators = sum(1 for e in events if e.get('indicators'))
    events_with_news = sum(1 for e in events if e.get('news'))
    
    json_filename = f'gold_calendar_{start_year}_{end_year}.json'
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(events, f, indent=2, ensure_ascii=False)
    
    csv_data = []
    for event in events:
        row = {
            'date': event['date'],
            'time': event['time'],
            'event': event['event'],
            'impact': event['impact']
        }
        
        if event.get('indicators'):
            row['gold_price'] = event['indicators'].get('price', '')
            row['overall_signal'] = event['indicators'].get('overall_signal', '')
            row['buy_signals'] = event['indicators'].get('buy_count', 0)
            row['sell_signals'] = event['indicators'].get('sell_count', 0)
            
            for name, data in event['indicators'].get('indicators', {}).items():
                row[f'{name}_signal'] = data['signal']
                row[f'{name}_value'] = data['value']
        
        row['news_count'] = len(event.get('news', []))
        for i, article in enumerate(event.get('news', [])[:2], 1):
            row[f'news_title_{i}'] = article.get('title', '')
            row[f'news_url_{i}'] = article.get('url', '')
        
        csv_data.append(row)
    
    df = pd.DataFrame(csv_data)
    csv_filename = f'gold_calendar_{start_year}_{end_year}.csv'
    df.to_csv(csv_filename, index=False)
    
    print("\n" + "=" * 80)
    print(f"✓ Total events: {len(events)}")
    print(f"✓ Events with gold indicators: {events_with_indicators}")
    print(f"✓ Events with news: {events_with_news}")
    print(f"✓ JSON: {json_filename}")
    print(f"✓ CSV: {csv_filename}")
    print("=" * 80)
    
    print("\nSAMPLE EVENT:")
    print("=" * 80)
    for event in events:
        if event.get('indicators') and event.get('news'):
            print(f"\n📅 {event['date']} - {event['event']}")
            print(f"🥇 Gold Price: ${event['indicators']['price']}")
            print(f"📊 Overall Signal: {event['indicators']['overall_signal']}")
            print(f"   BUY signals: {event['indicators']['buy_count']}")
            print(f"   SELL signals: {event['indicators']['sell_count']}")
            
            print("\n📈 Gold Indicators:")
            for name, data in event['indicators']['indicators'].items():
                print(f"   {name}: {data['signal']} ({data['description']})")
            
            print("\n📰 News:")
            for i, article in enumerate(event['news'], 1):
                print(f"   {i}. {article['title']}")
                print(f"      {article['url'][:70]}...")
            break
    
    print("\n" + "=" * 80)
    
    return events


if __name__ == "__main__":
    asyncio.run(create_calendar_with_gold_indicators(2024, 2025, True))
