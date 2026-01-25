import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import json

def get_technical_indicators_api(ticker='SPY', date=None):
    """
    Fetch technical indicators from Alpha Vantage API
    You'll need a free API key from https://www.alphavantage.co/support/#api-key
    """
    API_KEY = 'OW8A4KQB2V0DQJV3'  # Replace with your API key
    
    try:
        indicators = {}
        base_url = 'https://www.alphavantage.co/query'
        
        # RSI
        rsi_params = {
            'function': 'RSI',
            'symbol': ticker,
            'interval': 'daily',
            'time_period': 14,
            'series_type': 'close',
            'apikey': API_KEY
        }
        rsi_response = requests.get(base_url, params=rsi_params, timeout=10)
        if rsi_response.status_code == 200:
            rsi_data = rsi_response.json()
            if 'Technical Analysis: RSI' in rsi_data:
                latest_date = list(rsi_data['Technical Analysis: RSI'].keys())[0]
                rsi_val = float(rsi_data['Technical Analysis: RSI'][latest_date]['RSI'])
                rsi_signal = 'BUY' if rsi_val < 30 else ('SELL' if rsi_val > 70 else 'NEUTRAL')
                indicators['RSI'] = {
                    'value': round(rsi_val, 2),
                    'signal': rsi_signal,
                    'description': f'RSI at {round(rsi_val, 2)}'
                }
        time.sleep(12)  # Alpha Vantage free tier: 5 calls/minute
        
        # MACD
        macd_params = {
            'function': 'MACD',
            'symbol': ticker,
            'interval': 'daily',
            'series_type': 'close',
            'apikey': API_KEY
        }
        macd_response = requests.get(base_url, params=macd_params, timeout=10)
        if macd_response.status_code == 200:
            macd_data = macd_response.json()
            if 'Technical Analysis: MACD' in macd_data:
                latest_date = list(macd_data['Technical Analysis: MACD'].keys())[0]
                macd_val = float(macd_data['Technical Analysis: MACD'][latest_date]['MACD'])
                signal_val = float(macd_data['Technical Analysis: MACD'][latest_date]['MACD_Signal'])
                macd_signal = 'BUY' if macd_val > signal_val else 'SELL'
                indicators['MACD'] = {
                    'value': round(macd_val, 4),
                    'signal': macd_signal,
                    'description': f'MACD {round(macd_val, 4)} vs Signal {round(signal_val, 4)}'
                }
        time.sleep(12)
        
        # Stochastic
        stoch_params = {
            'function': 'STOCH',
            'symbol': ticker,
            'interval': 'daily',
            'apikey': API_KEY
        }
        stoch_response = requests.get(base_url, params=stoch_params, timeout=10)
        if stoch_response.status_code == 200:
            stoch_data = stoch_response.json()
            if 'Technical Analysis: STOCH' in stoch_data:
                latest_date = list(stoch_data['Technical Analysis: STOCH'].keys())[0]
                k_val = float(stoch_data['Technical Analysis: STOCH'][latest_date]['SlowK'])
                stoch_signal = 'BUY' if k_val < 20 else ('SELL' if k_val > 80 else 'NEUTRAL')
                indicators['Stochastic'] = {
                    'value': round(k_val, 2),
                    'signal': stoch_signal,
                    'description': f'Stochastic at {round(k_val, 2)}%'
                }
        time.sleep(12)
        
        # ADX
        adx_params = {
            'function': 'ADX',
            'symbol': ticker,
            'interval': 'daily',
            'time_period': 14,
            'apikey': API_KEY
        }
        adx_response = requests.get(base_url, params=adx_params, timeout=10)
        if adx_response.status_code == 200:
            adx_data = adx_response.json()
            if 'Technical Analysis: ADX' in adx_data:
                latest_date = list(adx_data['Technical Analysis: ADX'].keys())[0]
                adx_val = float(adx_data['Technical Analysis: ADX'][latest_date]['ADX'])
                adx_signal = 'STRONG TREND' if adx_val > 25 else 'WEAK TREND'
                indicators['ADX'] = {
                    'value': round(adx_val, 2),
                    'signal': adx_signal,
                    'description': f'Trend strength {round(adx_val, 2)}'
                }
        time.sleep(12)
        
        # CCI
        cci_params = {
            'function': 'CCI',
            'symbol': ticker,
            'interval': 'daily',
            'time_period': 20,
            'apikey': API_KEY
        }
        cci_response = requests.get(base_url, params=cci_params, timeout=10)
        if cci_response.status_code == 200:
            cci_data = cci_response.json()
            if 'Technical Analysis: CCI' in cci_data:
                latest_date = list(cci_data['Technical Analysis: CCI'].keys())[0]
                cci_val = float(cci_data['Technical Analysis: CCI'][latest_date]['CCI'])
                cci_signal = 'BUY' if cci_val < -100 else ('SELL' if cci_val > 100 else 'NEUTRAL')
                indicators['CCI'] = {
                    'value': round(cci_val, 2),
                    'signal': cci_signal,
                    'description': f'CCI at {round(cci_val, 2)}'
                }
        time.sleep(12)
        
        # Bollinger Bands
        bbands_params = {
            'function': 'BBANDS',
            'symbol': ticker,
            'interval': 'daily',
            'time_period': 20,
            'series_type': 'close',
            'apikey': API_KEY
        }
        bbands_response = requests.get(base_url, params=bbands_params, timeout=10)
        if bbands_response.status_code == 200:
            bbands_data = bbands_response.json()
            if 'Technical Analysis: BBANDS' in bbands_data:
                latest_date = list(bbands_data['Technical Analysis: BBANDS'].keys())[0]
                upper = float(bbands_data['Technical Analysis: BBANDS'][latest_date]['Real Upper Band'])
                lower = float(bbands_data['Technical Analysis: BBANDS'][latest_date]['Real Lower Band'])
                middle = float(bbands_data['Technical Analysis: BBANDS'][latest_date]['Real Middle Band'])
                
                # Get current price
                quote_params = {
                    'function': 'GLOBAL_QUOTE',
                    'symbol': ticker,
                    'apikey': API_KEY
                }
                quote_response = requests.get(base_url, params=quote_params, timeout=10)
                price = 0
                if quote_response.status_code == 200:
                    quote_data = quote_response.json()
                    if 'Global Quote' in quote_data and '05. price' in quote_data['Global Quote']:
                        price = float(quote_data['Global Quote']['05. price'])
                
                bb_signal = 'SELL' if price > upper else ('BUY' if price < lower else 'NEUTRAL')
                indicators['Bollinger'] = {
                    'value': f'{round(lower, 2)}-{round(upper, 2)}',
                    'signal': bb_signal,
                    'description': f'Price at {round(price, 2)}'
                }
                indicators['price'] = round(price, 2)
        time.sleep(12)
        
        # SMA
        sma20_params = {
            'function': 'SMA',
            'symbol': ticker,
            'interval': 'daily',
            'time_period': 20,
            'series_type': 'close',
            'apikey': API_KEY
        }
        sma20_response = requests.get(base_url, params=sma20_params, timeout=10)
        sma_20 = 0
        if sma20_response.status_code == 200:
            sma20_data = sma20_response.json()
            if 'Technical Analysis: SMA' in sma20_data:
                latest_date = list(sma20_data['Technical Analysis: SMA'].keys())[0]
                sma_20 = float(sma20_data['Technical Analysis: SMA'][latest_date]['SMA'])
        time.sleep(12)
        
        sma50_params = {
            'function': 'SMA',
            'symbol': ticker,
            'interval': 'daily',
            'time_period': 50,
            'series_type': 'close',
            'apikey': API_KEY
        }
        sma50_response = requests.get(base_url, params=sma50_params, timeout=10)
        sma_50 = 0
        if sma50_response.status_code == 200:
            sma50_data = sma50_response.json()
            if 'Technical Analysis: SMA' in sma50_data:
                latest_date = list(sma50_data['Technical Analysis: SMA'].keys())[0]
                sma_50 = float(sma50_data['Technical Analysis: SMA'][latest_date]['SMA'])
        
        price = indicators.get('price', 0)
        ma_signal = 'BUY' if (price > sma_20 and sma_20 > sma_50) else ('SELL' if (price < sma_20 and sma_20 < sma_50) else 'NEUTRAL')
        indicators['MA_Cross'] = {
            'value': f'{round(sma_20, 2)}/{round(sma_50, 2)}',
            'signal': ma_signal,
            'description': f'Price {round(price, 2)} vs SMA20 {round(sma_20, 2)}'
        }
        time.sleep(12)
        
        # Williams %R
        willr_params = {
            'function': 'WILLR',
            'symbol': ticker,
            'interval': 'daily',
            'time_period': 14,
            'apikey': API_KEY
        }
        willr_response = requests.get(base_url, params=willr_params, timeout=10)
        if willr_response.status_code == 200:
            willr_data = willr_response.json()
            if 'Technical Analysis: WILLR' in willr_data:
                latest_date = list(willr_data['Technical Analysis: WILLR'].keys())[0]
                willr_val = float(willr_data['Technical Analysis: WILLR'][latest_date]['WILLR'])
                williams_signal = 'BUY' if willr_val < -80 else ('SELL' if willr_val > -20 else 'NEUTRAL')
                indicators['Williams_R'] = {
                    'value': round(willr_val, 2),
                    'signal': williams_signal,
                    'description': f'Williams %R at {round(willr_val, 2)}'
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
            'indicators': {k: v for k, v in indicators.items() if k != 'price'},
            'overall_signal': overall,
            'buy_count': buy_signals,
            'sell_count': sell_signals,
            'price': indicators.get('price', 0)
        }
        
    except Exception as e:
        print(f"API Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def get_twelve_data_indicators(ticker='SPY', date=None):
    """
    Alternative: Fetch technical indicators from Twelve Data API
    Free tier: 800 requests/day - Get API key from https://twelvedata.com/
    """
    API_KEY = 'YOUR_TWELVE_DATA_API_KEY'  # Replace with your API key
    
    try:
        base_url = 'https://api.twelvedata.com'
        
        # Get multiple indicators in one call
        indicators_list = 'rsi,macd,stoch,adx,cci,bbands,sma,willr'
        
        params = {
            'symbol': ticker,
            'interval': '1day',
            'apikey': API_KEY,
            'outputsize': 1
        }
        
        all_indicators = {}
        
        # RSI
        rsi_response = requests.get(f'{base_url}/rsi', params={**params, 'time_period': 14}, timeout=10)
        if rsi_response.status_code == 200:
            rsi_data = rsi_response.json()
            if 'values' in rsi_data and len(rsi_data['values']) > 0:
                rsi_val = float(rsi_data['values'][0]['rsi'])
                rsi_signal = 'BUY' if rsi_val < 30 else ('SELL' if rsi_val > 70 else 'NEUTRAL')
                all_indicators['RSI'] = {
                    'value': round(rsi_val, 2),
                    'signal': rsi_signal,
                    'description': f'RSI at {round(rsi_val, 2)}'
                }
        time.sleep(1)
        
        # MACD
        macd_response = requests.get(f'{base_url}/macd', params=params, timeout=10)
        if macd_response.status_code == 200:
            macd_data = macd_response.json()
            if 'values' in macd_data and len(macd_data['values']) > 0:
                macd_val = float(macd_data['values'][0]['macd'])
                signal_val = float(macd_data['values'][0]['macd_signal'])
                macd_signal = 'BUY' if macd_val > signal_val else 'SELL'
                all_indicators['MACD'] = {
                    'value': round(macd_val, 4),
                    'signal': macd_signal,
                    'description': f'MACD {round(macd_val, 4)} vs Signal {round(signal_val, 4)}'
                }
        time.sleep(1)
        
        # Get current price
        quote_response = requests.get(f'{base_url}/price', params={'symbol': ticker, 'apikey': API_KEY}, timeout=10)
        price = 0
        if quote_response.status_code == 200:
            price_data = quote_response.json()
            if 'price' in price_data:
                price = float(price_data['price'])
        
        # Stochastic
        stoch_response = requests.get(f'{base_url}/stoch', params=params, timeout=10)
        if stoch_response.status_code == 200:
            stoch_data = stoch_response.json()
            if 'values' in stoch_data and len(stoch_data['values']) > 0:
                k_val = float(stoch_data['values'][0]['slow_k'])
                stoch_signal = 'BUY' if k_val < 20 else ('SELL' if k_val > 80 else 'NEUTRAL')
                all_indicators['Stochastic'] = {
                    'value': round(k_val, 2),
                    'signal': stoch_signal,
                    'description': f'Stochastic at {round(k_val, 2)}%'
                }
        time.sleep(1)
        
        # Continue with other indicators...
        # ADX, CCI, Bollinger Bands, SMA, Williams %R, etc.
        
        buy_signals = sum(1 for ind in all_indicators.values() if ind.get('signal') == 'BUY')
        sell_signals = sum(1 for ind in all_indicators.values() if ind.get('signal') == 'SELL')
        total_signals = buy_signals + sell_signals
        
        overall = 'BUY' if buy_signals > sell_signals else ('SELL' if sell_signals > buy_signals else 'NEUTRAL')
        
        return {
            'indicators': all_indicators,
            'overall_signal': overall,
            'buy_count': buy_signals,
            'sell_count': sell_signals,
            'price': round(price, 2)
        }
        
    except Exception as e:
        print(f"API Error: {str(e)}")
        return None


def get_gdelt_news(event_name, event_date, num_articles=2):
    """Fetch news articles from GDELT API"""
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
                                             ticker='SPY', api_choice='twelve'):
    """
    Create comprehensive economic calendar with indicators and news
    
    api_choice: 'alpha' for Alpha Vantage or 'twelve' for Twelve Data
    """
    print("=" * 80)
    print(f"ECONOMIC CALENDAR + TECHNICAL INDICATORS + NEWS ({start_year}-{end_year})")
    print(f"Ticker: {ticker}")
    print(f"API: {api_choice.upper()}")
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
                    print(f"  Fetching indicators for {event_date}...", end=" ")
                    if api_choice == 'alpha':
                        indicators = get_technical_indicators_api(ticker, event_date)
                    else:
                        indicators = get_twelve_data_indicators(ticker, event_date)
                    
                    if indicators:
                        events_with_indicators += 1
                        print("✓")
                    else:
                        print("✗")
                
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
                print(f"  Fetching indicators for {event_date}...", end=" ")
                if api_choice == 'alpha':
                    indicators = get_technical_indicators_api(ticker, event_date)
                else:
                    indicators = get_twelve_data_indicators(ticker, event_date)
                
                if indicators:
                    events_with_indicators += 1
                    print("✓")
                else:
                    print("✗")
            
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
    print("\nThis fetches data from APIs:")
    print("• Technical indicators from Alpha Vantage or Twelve Data")
    print("• News articles from GDELT")
    print("\n⚠ SETUP REQUIRED:")
    print("1. Get FREE API key from:")
    print("   - Alpha Vantage: https://www.alphavantage.co/support/#api-key")
    print("   - OR Twelve Data: https://twelvedata.com/")
    print("2. Replace 'YOUR_API_KEY' in the code with your actual key")
    print("\n⚠ NOTE: API rate limits apply - Full run may take several hours")
    print("=" * 80)
    
    choice = 1 # input("\n1. Quick test (2024-2025)\n2. Full run (2001-2026)\nChoice: ").strip()
    api_choice ="alpha" #input("API Choice (alpha/twelve): ").strip().lower()
    
    if api_choice not in ['alpha', 'twelve']:
        api_choice = 'twelve'
    
    if choice == '1':
        create_calendar_with_indicators_and_news(2024, 2025, True, True, 'SPY', api_choice)
    else:
        create_calendar_with_indicators_and_news(2001, 2026, True, True, 'SPY', api_choice)
