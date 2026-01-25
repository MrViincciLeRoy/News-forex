import pandas as pd
import requests
from datetime import datetime, timedelta
import json
import os
import asyncio
import aiohttp

def get_api_keys():
    """
    Get API keys from environment variables
    Supports multiple Alpha Vantage keys for rotation
    """
    # Get all Alpha Vantage API keys from environment
    alpha_keys = []
    i = 1
    while True:
        key = os.environ.get(f'ALPHA_VANTAGE_API_KEY_{i}')
        if key:
            alpha_keys.append(key)
            i += 1
        else:
            break
    
    # Fallback to single key if no numbered keys found
    if not alpha_keys:
        single_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
        if single_key:
            alpha_keys.append(single_key)
    
    return alpha_keys

# Global variables for API key rotation
ALPHA_KEYS = get_api_keys()
CURRENT_KEY_INDEX = 0
KEY_LOCK = asyncio.Lock()

async def get_next_api_key():
    """Rotate through available API keys (thread-safe)"""
    global CURRENT_KEY_INDEX
    if not ALPHA_KEYS:
        raise ValueError("No Alpha Vantage API keys found in environment variables")
    
    async with KEY_LOCK:
        key = ALPHA_KEYS[CURRENT_KEY_INDEX]
        CURRENT_KEY_INDEX = (CURRENT_KEY_INDEX + 1) % len(ALPHA_KEYS)
        return key

async def fetch_indicator(session, base_url, params, indicator_name):
    """Fetch a single indicator from API"""
    try:
        async with session.get(base_url, params=params, timeout=10) as response:
            if response.status == 200:
                return indicator_name, await response.json()
    except Exception as e:
        print(f"Error fetching {indicator_name}: {str(e)}")
    return indicator_name, None

async def get_technical_indicators_api(ticker='SPY', date=None):
    """
    Fetch technical indicators from Alpha Vantage API asynchronously
    """
    try:
        indicators = {}
        base_url = 'https://www.alphavantage.co/query'
        
        async with aiohttp.ClientSession() as session:
            # Prepare all requests
            tasks = []
            
            # RSI
            api_key = await get_next_api_key()
            rsi_params = {
                'function': 'RSI',
                'symbol': ticker,
                'interval': 'daily',
                'time_period': 14,
                'series_type': 'close',
                'apikey': api_key
            }
            tasks.append(fetch_indicator(session, base_url, rsi_params, 'RSI'))
            
            # MACD
            api_key = await get_next_api_key()
            macd_params = {
                'function': 'MACD',
                'symbol': ticker,
                'interval': 'daily',
                'series_type': 'close',
                'apikey': api_key
            }
            tasks.append(fetch_indicator(session, base_url, macd_params, 'MACD'))
            
            # Stochastic
            api_key = await get_next_api_key()
            stoch_params = {
                'function': 'STOCH',
                'symbol': ticker,
                'interval': 'daily',
                'apikey': api_key
            }
            tasks.append(fetch_indicator(session, base_url, stoch_params, 'STOCH'))
            
            # ADX
            api_key = await get_next_api_key()
            adx_params = {
                'function': 'ADX',
                'symbol': ticker,
                'interval': 'daily',
                'time_period': 14,
                'apikey': api_key
            }
            tasks.append(fetch_indicator(session, base_url, adx_params, 'ADX'))
            
            # CCI
            api_key = await get_next_api_key()
            cci_params = {
                'function': 'CCI',
                'symbol': ticker,
                'interval': 'daily',
                'time_period': 20,
                'apikey': api_key
            }
            tasks.append(fetch_indicator(session, base_url, cci_params, 'CCI'))
            
            # Bollinger Bands
            api_key = await get_next_api_key()
            bbands_params = {
                'function': 'BBANDS',
                'symbol': ticker,
                'interval': 'daily',
                'time_period': 20,
                'series_type': 'close',
                'apikey': api_key
            }
            tasks.append(fetch_indicator(session, base_url, bbands_params, 'BBANDS'))
            
            # SMA 20
            api_key = await get_next_api_key()
            sma20_params = {
                'function': 'SMA',
                'symbol': ticker,
                'interval': 'daily',
                'time_period': 20,
                'series_type': 'close',
                'apikey': api_key
            }
            tasks.append(fetch_indicator(session, base_url, sma20_params, 'SMA20'))
            
            # SMA 50
            api_key = await get_next_api_key()
            sma50_params = {
                'function': 'SMA',
                'symbol': ticker,
                'interval': 'daily',
                'time_period': 50,
                'series_type': 'close',
                'apikey': api_key
            }
            tasks.append(fetch_indicator(session, base_url, sma50_params, 'SMA50'))
            
            # Williams %R
            api_key = await get_next_api_key()
            willr_params = {
                'function': 'WILLR',
                'symbol': ticker,
                'interval': 'daily',
                'time_period': 14,
                'apikey': api_key
            }
            tasks.append(fetch_indicator(session, base_url, willr_params, 'WILLR'))
            
            # Price Quote
            api_key = await get_next_api_key()
            quote_params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': ticker,
                'apikey': api_key
            }
            tasks.append(fetch_indicator(session, base_url, quote_params, 'QUOTE'))
            
            # Execute all requests concurrently
            results = await asyncio.gather(*tasks)
            
            # Process results
            price = 0
            sma_20 = 0
            sma_50 = 0
            
            for indicator_name, data in results:
                if data is None:
                    continue
                
                if indicator_name == 'RSI' and 'Technical Analysis: RSI' in data:
                    latest_date = list(data['Technical Analysis: RSI'].keys())[0]
                    rsi_val = float(data['Technical Analysis: RSI'][latest_date]['RSI'])
                    rsi_signal = 'BUY' if rsi_val < 30 else ('SELL' if rsi_val > 70 else 'NEUTRAL')
                    indicators['RSI'] = {
                        'value': round(rsi_val, 2),
                        'signal': rsi_signal,
                        'description': f'RSI at {round(rsi_val, 2)}'
                    }
                
                elif indicator_name == 'MACD' and 'Technical Analysis: MACD' in data:
                    latest_date = list(data['Technical Analysis: MACD'].keys())[0]
                    macd_val = float(data['Technical Analysis: MACD'][latest_date]['MACD'])
                    signal_val = float(data['Technical Analysis: MACD'][latest_date]['MACD_Signal'])
                    macd_signal = 'BUY' if macd_val > signal_val else 'SELL'
                    indicators['MACD'] = {
                        'value': round(macd_val, 4),
                        'signal': macd_signal,
                        'description': f'MACD {round(macd_val, 4)} vs Signal {round(signal_val, 4)}'
                    }
                
                elif indicator_name == 'STOCH' and 'Technical Analysis: STOCH' in data:
                    latest_date = list(data['Technical Analysis: STOCH'].keys())[0]
                    k_val = float(data['Technical Analysis: STOCH'][latest_date]['SlowK'])
                    stoch_signal = 'BUY' if k_val < 20 else ('SELL' if k_val > 80 else 'NEUTRAL')
                    indicators['Stochastic'] = {
                        'value': round(k_val, 2),
                        'signal': stoch_signal,
                        'description': f'Stochastic at {round(k_val, 2)}%'
                    }
                
                elif indicator_name == 'ADX' and 'Technical Analysis: ADX' in data:
                    latest_date = list(data['Technical Analysis: ADX'].keys())[0]
                    adx_val = float(data['Technical Analysis: ADX'][latest_date]['ADX'])
                    adx_signal = 'STRONG TREND' if adx_val > 25 else 'WEAK TREND'
                    indicators['ADX'] = {
                        'value': round(adx_val, 2),
                        'signal': adx_signal,
                        'description': f'Trend strength {round(adx_val, 2)}'
                    }
                
                elif indicator_name == 'CCI' and 'Technical Analysis: CCI' in data:
                    latest_date = list(data['Technical Analysis: CCI'].keys())[0]
                    cci_val = float(data['Technical Analysis: CCI'][latest_date]['CCI'])
                    cci_signal = 'BUY' if cci_val < -100 else ('SELL' if cci_val > 100 else 'NEUTRAL')
                    indicators['CCI'] = {
                        'value': round(cci_val, 2),
                        'signal': cci_signal,
                        'description': f'CCI at {round(cci_val, 2)}'
                    }
                
                elif indicator_name == 'BBANDS' and 'Technical Analysis: BBANDS' in data:
                    latest_date = list(data['Technical Analysis: BBANDS'].keys())[0]
                    upper = float(data['Technical Analysis: BBANDS'][latest_date]['Real Upper Band'])
                    lower = float(data['Technical Analysis: BBANDS'][latest_date]['Real Lower Band'])
                    indicators['bbands_upper'] = upper
                    indicators['bbands_lower'] = lower
                
                elif indicator_name == 'SMA20' and 'Technical Analysis: SMA' in data:
                    latest_date = list(data['Technical Analysis: SMA'].keys())[0]
                    sma_20 = float(data['Technical Analysis: SMA'][latest_date]['SMA'])
                
                elif indicator_name == 'SMA50' and 'Technical Analysis: SMA' in data:
                    latest_date = list(data['Technical Analysis: SMA'].keys())[0]
                    sma_50 = float(data['Technical Analysis: SMA'][latest_date]['SMA'])
                
                elif indicator_name == 'WILLR' and 'Technical Analysis: WILLR' in data:
                    latest_date = list(data['Technical Analysis: WILLR'].keys())[0]
                    willr_val = float(data['Technical Analysis: WILLR'][latest_date]['WILLR'])
                    williams_signal = 'BUY' if willr_val < -80 else ('SELL' if willr_val > -20 else 'NEUTRAL')
                    indicators['Williams_R'] = {
                        'value': round(willr_val, 2),
                        'signal': williams_signal,
                        'description': f'Williams %R at {round(willr_val, 2)}'
                    }
                
                elif indicator_name == 'QUOTE' and 'Global Quote' in data:
                    if '05. price' in data['Global Quote']:
                        price = float(data['Global Quote']['05. price'])
            
            # Process Bollinger Bands with price
            if 'bbands_upper' in indicators and 'bbands_lower' in indicators:
                upper = indicators.pop('bbands_upper')
                lower = indicators.pop('bbands_lower')
                bb_signal = 'SELL' if price > upper else ('BUY' if price < lower else 'NEUTRAL')
                indicators['Bollinger'] = {
                    'value': f'{round(lower, 2)}-{round(upper, 2)}',
                    'signal': bb_signal,
                    'description': f'Price at {round(price, 2)}'
                }
            
            # Process MA Cross
            if sma_20 and sma_50:
                ma_signal = 'BUY' if (price > sma_20 and sma_20 > sma_50) else ('SELL' if (price < sma_20 and sma_20 < sma_50) else 'NEUTRAL')
                indicators['MA_Cross'] = {
                    'value': f'{round(sma_20, 2)}/{round(sma_50, 2)}',
                    'signal': ma_signal,
                    'description': f'Price {round(price, 2)} vs SMA20 {round(sma_20, 2)}'
                }
            
            # Calculate overall signal
            buy_signals = sum(1 for ind in indicators.values() if isinstance(ind, dict) and ind.get('signal') == 'BUY')
            sell_signals = sum(1 for ind in indicators.values() if isinstance(ind, dict) and ind.get('signal') == 'SELL')
            total_signals = buy_signals + sell_signals
            
            if total_signals > 0:
                overall = 'BUY' if buy_signals > sell_signals else ('SELL' if sell_signals > buy_signals else 'NEUTRAL')
            else:
                overall = 'NEUTRAL'
            
            return {
                'indicators': {k: v for k, v in indicators.items() if isinstance(v, dict)},
                'overall_signal': overall,
                'buy_count': buy_signals,
                'sell_count': sell_signals,
                'price': round(price, 2)
            }
        
    except Exception as e:
        print(f"API Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

async def get_gdelt_news_async(session, event_name, event_date, num_articles=2):
    """Fetch news articles from GDELT API asynchronously"""
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

async def process_event(session, event_date, event_name, ticker, fetch_indicators, fetch_news):
    """Process a single event asynchronously"""
    event = {
        'date': event_date,
        'time': '08:30',
        'event': event_name,
        'impact': 'High',
        'frequency': 'Monthly',
        'indicators': None,
        'news': []
    }
    
    tasks = []
    
    if fetch_indicators:
        tasks.append(get_technical_indicators_api(ticker, event_date))
    else:
        tasks.append(asyncio.sleep(0))
    
    if fetch_news:
        tasks.append(get_gdelt_news_async(session, event_name, event_date, 2))
    else:
        tasks.append(asyncio.sleep(0))
    
    results = await asyncio.gather(*tasks)
    
    if fetch_indicators and results[0]:
        event['indicators'] = results[0]
    
    if fetch_news and results[1]:
        event['news'] = results[1]
    
    return event

async def create_calendar_with_indicators_and_news(start_year=2024, end_year=2025, 
                                                   fetch_news=True, fetch_indicators=True,
                                                   ticker='SPY'):
    """
    Create comprehensive economic calendar with indicators and news (async version)
    Uses Alpha Vantage API with automatic key rotation
    """
    print("=" * 80)
    print(f"ECONOMIC CALENDAR + TECHNICAL INDICATORS + NEWS ({start_year}-{end_year})")
    print(f"Ticker: {ticker}")
    print(f"API Keys Available: {len(ALPHA_KEYS)}")
    print("=" * 80 + "\n")
    
    all_events_tasks = []
    
    async with aiohttp.ClientSession() as session:
        for year in range(start_year, end_year + 1):
            print(f"Preparing {year}...", flush=True)
            
            for month in range(1, 13):
                # NFP
                first_friday = get_nth_weekday(year, month, 1, 4)
                if first_friday:
                    event_date = first_friday.strftime('%Y-%m-%d')
                    all_events_tasks.append(
                        process_event(session, event_date, 'Non-Farm Payrolls (NFP)', 
                                    ticker, fetch_indicators, fetch_news)
                    )
                
                # CPI
                cpi_date = datetime(year, month, 13)
                while cpi_date.weekday() >= 5:
                    cpi_date += timedelta(days=1)
                event_date = cpi_date.strftime('%Y-%m-%d')
                all_events_tasks.append(
                    process_event(session, event_date, 'Consumer Price Index (CPI)', 
                                ticker, fetch_indicators, fetch_news)
                )
        
        print(f"\nProcessing {len(all_events_tasks)} events asynchronously...")
        events = await asyncio.gather(*all_events_tasks)
    
    # Sort by date
    events = sorted(events, key=lambda x: x['date'])
    
    # Count successes
    events_with_indicators = sum(1 for e in events if e.get('indicators'))
    events_with_news = sum(1 for e in events if e.get('news'))
    
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
    print(f"✓ Total events: {len(events)}")
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
    """Get the nth occurrence of a weekday in a month"""
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
    print("\nThis fetches data from APIs (Async):")
    print("• Technical indicators from Alpha Vantage")
    print("• News articles from GDELT")
    print(f"\nAPI Keys detected: {len(ALPHA_KEYS)}")
    print("=" * 80 + "\n")
    
    # Run with default settings (2024-2025)
    asyncio.run(create_calendar_with_indicators_and_news(2024, 2025, True, True, 'SPY'))
