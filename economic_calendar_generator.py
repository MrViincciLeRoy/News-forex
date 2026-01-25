import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import json
import yfinance as yf
import numpy as np

def calculate_indicators(ticker='SPY', date=None, period_before=100):
    """
    Calculate 10+ technical indicators for a given date
    Returns buy/sell/neutral signals
    """
    try:
        end_date = datetime.strptime(date, '%Y-%m-%d')
        start_date = end_date - timedelta(days=period_before)
        
        df = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), 
                        end=(end_date + timedelta(days=1)).strftime('%Y-%m-%d'), 
                        progress=False)
        
        if len(df) < 50:
            return None
        
        # Get scalar values at the end
        price = float(df['Close'].iloc[-1])
        current_volume = float(df['Volume'].iloc[-1])
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_val = float(rsi.iloc[-1])
        rsi_signal = 'BUY' if rsi_val < 30 else ('SELL' if rsi_val > 70 else 'NEUTRAL')
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_val = float(macd.iloc[-1])
        signal_val = float(signal.iloc[-1])
        macd_signal = 'BUY' if macd_val > signal_val else 'SELL'
        
        # Moving Averages
        sma_20 = float(df['Close'].rolling(window=20).mean().iloc[-1])
        sma_50 = float(df['Close'].rolling(window=50).mean().iloc[-1])
        ma_signal = 'BUY' if (price > sma_20 and sma_20 > sma_50) else ('SELL' if (price < sma_20 and sma_20 < sma_50) else 'NEUTRAL')
        
        # Bollinger Bands
        bb_middle = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)
        bb_upper_val = float(bb_upper.iloc[-1])
        bb_lower_val = float(bb_lower.iloc[-1])
        bb_signal = 'SELL' if price > bb_upper_val else ('BUY' if price < bb_lower_val else 'NEUTRAL')
        
        # Stochastic
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        k_percent = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        k_val = float(k_percent.iloc[-1])
        stoch_signal = 'BUY' if k_val < 20 else ('SELL' if k_val > 80 else 'NEUTRAL')
        
        # ADX (trend strength) - simplified ATR calculation
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        
        tr = pd.DataFrame({
            'hl': high_low,
            'hc': high_close,
            'lc': low_close
        }).max(axis=1)
        
        atr = tr.rolling(14).mean()
        adx_val = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0
        adx_signal = 'STRONG TREND' if adx_val > 25 else 'WEAK TREND'
        
        # Volume
        avg_volume = float(df['Volume'].rolling(window=20).mean().iloc[-1])
        volume_signal = 'HIGH' if current_volume > avg_volume * 1.5 else ('LOW' if current_volume < avg_volume * 0.5 else 'NORMAL')
        
        # OBV (On Balance Volume)
        obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        obv_ma = obv.rolling(window=20).mean()
        obv_val = float(obv.iloc[-1])
        obv_ma_val = float(obv_ma.iloc[-1])
        obv_signal = 'BUY' if obv_val > obv_ma_val else 'SELL'
        
        # ATR (volatility)
        atr_percent = (adx_val / price) * 100 if price > 0 else 0
        atr_signal = 'HIGH VOLATILITY' if atr_percent > 2 else 'LOW VOLATILITY'
        
        # Williams %R
        high_14_val = float(high_14.iloc[-1])
        low_14_val = float(low_14.iloc[-1])
        
        if high_14_val != low_14_val:
            williams_r = -100 * ((high_14_val - price) / (high_14_val - low_14_val))
        else:
            williams_r = -50
            
        williams_signal = 'BUY' if williams_r < -80 else ('SELL' if williams_r > -20 else 'NEUTRAL')
        
        # CCI (Commodity Channel Index)
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = tp.rolling(window=20).mean()
        mad = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
        cci = (tp - sma_tp) / (0.015 * mad)
        cci_val = float(cci.iloc[-1])
        cci_signal = 'BUY' if cci_val < -100 else ('SELL' if cci_val > 100 else 'NEUTRAL')
        
        # ROC (Rate of Change)
        if len(df) >= 10:
            roc = ((price - float(df['Close'].iloc[-10])) / float(df['Close'].iloc[-10])) * 100
        else:
            roc = 0
        roc_signal = 'BUY' if roc > 5 else ('SELL' if roc < -5 else 'NEUTRAL')
        
        indicators = {
            'RSI': {'value': round(rsi_val, 2), 'signal': rsi_signal, 'description': f'RSI at {round(rsi_val, 2)}'},
            'MACD': {'value': round(macd_val, 4), 'signal': macd_signal, 'description': f'MACD {round(macd_val, 4)} vs Signal {round(signal_val, 4)}'},
            'MA_Cross': {'value': f'{round(sma_20, 2)}/{round(sma_50, 2)}', 'signal': ma_signal, 'description': f'Price {round(price, 2)} vs SMA20 {round(sma_20, 2)}'},
            'Bollinger': {'value': f'{round(bb_lower_val, 2)}-{round(bb_upper_val, 2)}', 'signal': bb_signal, 'description': f'Price at {round(price, 2)}'},
            'Stochastic': {'value': round(k_val, 2), 'signal': stoch_signal, 'description': f'Stochastic at {round(k_val, 2)}%'},
            'ADX': {'value': round(adx_val, 2), 'signal': adx_signal, 'description': f'Trend strength {round(adx_val, 2)}'},
            'Volume': {'value': int(current_volume), 'signal': volume_signal, 'description': f'Vol {int(current_volume):,} vs Avg {int(avg_volume):,}'},
            'OBV': {'value': int(obv_val), 'signal': obv_signal, 'description': 'Volume trend indicator'},
            'ATR': {'value': round(atr_percent, 2), 'signal': atr_signal, 'description': f'Volatility {round(atr_percent, 2)}%'},
            'Williams_R': {'value': round(williams_r, 2), 'signal': williams_signal, 'description': f'Williams %R at {round(williams_r, 2)}'},
            'CCI': {'value': round(cci_val, 2), 'signal': cci_signal, 'description': f'CCI at {round(cci_val, 2)}'},
            'ROC': {'value': round(roc, 2), 'signal': roc_signal, 'description': f'Rate of change {round(roc, 2)}%'}
        }
        
        buy_signals = sum(1 for ind in indicators.values() if ind['signal'] == 'BUY')
        sell_signals = sum(1 for ind in indicators.values() if ind['signal'] == 'SELL')
        total_signals = buy_signals + sell_signals
        
        if total_signals > 0:
            overall = 'BUY' if buy_signals > sell_signals else ('SELL' if sell_signals > buy_signals else 'NEUTRAL')
        else:
            overall = 'NEUTRAL'
        
        return {
            'indicators': indicators,
            'overall_signal': overall,
            'buy_count': buy_signals,
            'sell_count': sell_signals,
            'price': round(price, 2)
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def get_gdelt_news(event_name, event_date, num_articles=2):
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
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
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


def create_calendar_with_indicators_and_news(start_year=2001, end_year=2026, 
                                             fetch_news=True, fetch_indicators=True,
                                             ticker='SPY'):
    print("=" * 80)
    print(f"ECONOMIC CALENDAR + TECHNICAL INDICATORS + NEWS ({start_year}-{end_year})")
    print(f"Ticker: {ticker}")
    print("=" * 80 + "\n")
    
    events = []
    total_events = 0
    events_with_indicators = 0
    events_with_news = 0
    
    for year in range(start_year, end_year + 1):
        print(f"Processing {year}...", flush=True)
        
        for month in range(1, 13):
            # NFP
            first_friday = get_nth_weekday(year, month, 1, 4)
            if first_friday:
                event_date = first_friday.strftime('%Y-%m-%d')
                
                indicators = None
                if fetch_indicators:
                    print(f"  Calculating indicators for {event_date}...", end=" ")
                    indicators = calculate_indicators(ticker, event_date)
                    if indicators:
                        events_with_indicators += 1
                        print("✓")
                    else:
                        print("✗")
                    time.sleep(0.5)
                
                news = []
                if fetch_news:
                    news = get_gdelt_news('Non-Farm Payrolls', event_date, 2)
                    if news:
                        events_with_news += 1
                    time.sleep(0.5)
                
                events.append({
                    'date': event_date,
                    'time': '08:30',
                    'event': 'Non-Farm Payrolls (NFP)',
                    'impact': 'High',
                    'frequency': 'Monthly',
                    'indicators': indicators,
                    'news': news
                })
                total_events += 1
            
            # CPI
            cpi_date = datetime(year, month, 13)
            while cpi_date.weekday() >= 5:
                cpi_date += timedelta(days=1)
            event_date = cpi_date.strftime('%Y-%m-%d')
            
            indicators = None
            if fetch_indicators:
                print(f"  Calculating indicators for {event_date}...", end=" ")
                indicators = calculate_indicators(ticker, event_date)
                if indicators:
                    events_with_indicators += 1
                    print("✓")
                else:
                    print("✗")
                time.sleep(0.5)
            
            news = []
            if fetch_news:
                news = get_gdelt_news('Consumer Price Index', event_date, 2)
                if news:
                    events_with_news += 1
                time.sleep(0.5)
            
            events.append({
                'date': event_date,
                'time': '08:30',
                'event': 'Consumer Price Index (CPI)',
                'impact': 'High',
                'frequency': 'Monthly',
                'indicators': indicators,
                'news': news
            })
            total_events += 1
    
    # Sort by date
    events = sorted(events, key=lambda x: x['date'])
    
    # Save JSON
    json_filename = f'economic_calendar_full_{start_year}_{end_year}.json'
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(events, f, indent=2, ensure_ascii=False)
    
    # Save CSV
    csv_data = []
    for event in events:
        row = {
            'date': event['date'],
            'time': event['time'],
            'event': event['event'],
            'impact': event['impact']
        }
        
        if event.get('indicators'):
            row['price'] = event['indicators'].get('price', '')
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
    csv_filename = f'economic_calendar_full_{start_year}_{end_year}.csv'
    df.to_csv(csv_filename, index=False)
    
    print("\n" + "=" * 80)
    print(f"✓ Total events: {total_events}")
    print(f"✓ Events with indicators: {events_with_indicators}")
    print(f"✓ Events with news: {events_with_news}")
    print(f"✓ JSON: {json_filename}")
    print(f"✓ CSV: {csv_filename}")
    print("=" * 80)
    
    # Show sample
    print("\nSAMPLE EVENT:")
    print("=" * 80)
    for event in events:
        if event.get('indicators') and event.get('news'):
            print(f"\n📅 {event['date']} - {event['event']}")
            print(f"💰 Price: ${event['indicators']['price']}")
            print(f"📊 Overall Signal: {event['indicators']['overall_signal']}")
            print(f"   BUY signals: {event['indicators']['buy_count']}")
            print(f"   SELL signals: {event['indicators']['sell_count']}")
            
            print("\n📈 Indicators:")
            for name, data in event['indicators']['indicators'].items():
                print(f"   {name}: {data['signal']} ({data['description']})")
            
            print("\n📰 News:")
            for i, article in enumerate(event['news'], 1):
                print(f"   {i}. {article['title']}")
                print(f"      {article['url'][:70]}...")
            break
    
    print("\n" + "=" * 80)
    
    return events


def get_nth_weekday(year, month, n, weekday):
    first_day = datetime(year, month, 1)
    first_weekday = first_day.weekday()
    offset = (weekday - first_weekday) % 7
    target_date = first_day + timedelta(days=offset + (n - 1) * 7)
    if target_date.month == month:
        return target_date
    return None


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("ECONOMIC CALENDAR + TECHNICAL INDICATORS + NEWS")
    print("=" * 80)
    print("\nThis creates a calendar with:")
    print("• Economic events (NFP, CPI, etc.)")
    print("• 12 technical indicators (RSI, MACD, Bollinger, Stochastic, etc.)")
    print("• News articles from GDELT")
    print("\n⚠ NOTE: Full run (2001-2026) takes 30-60 minutes")
    print("=" * 80)
    
    choice = "y" #input("\n1. Quick test (2024-2025)\n2. Full run (2001-2026)\nChoice: ").strip()
    
    if choice == '1':
        create_calendar_with_indicators_and_news(2024, 2025, True, True, 'SPY')
    else:
        create_calendar_with_indicators_and_news(2001, 2026, True, True, 'SPY')
