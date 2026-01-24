import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import json
from urllib.parse import quote

def get_gdelt_news(event_name, event_date, num_articles=2):
    """
    Fetch news articles using GDELT (100% FREE, historical data back to 1979)
    GDELT monitors news from around the world in real-time
    """
    
    # Map event types to search terms
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
    
    # Get search keywords
    search_terms = 'economic data US'
    for key, value in keywords_map.items():
        if key.lower() in event_name.lower():
            search_terms = value
            break
    
    # Parse date
    date_obj = datetime.strptime(event_date, '%Y-%m-%d')
    
    # GDELT uses YYYYMMDDHHMMSS format
    start_datetime = date_obj.strftime('%Y%m%d') + '000000'
    end_datetime = date_obj.strftime('%Y%m%d') + '235959'
    
    # GDELT DOC 2.0 API endpoint
    url = 'https://api.gdeltproject.org/api/v2/doc/doc'
    
    params = {
        'query': search_terms,
        'mode': 'artlist',
        'maxrecords': num_articles * 3,  # Get extra in case some are bad
        'format': 'json',
        'startdatetime': start_datetime,
        'enddatetime': end_datetime,
        'sourcelang': 'english',
        'theme': 'TAX_FNCACT_ECONOMY'  # Economy theme filter
    }
    
    try:
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code == 200:
            try:
                data = response.json()
                articles = []
                
                if 'articles' in data:
                    for article in data['articles'][:num_articles]:
                        articles.append({
                            'title': article.get('title', 'No title'),
                            'url': article.get('url', ''),
                            'source': article.get('domain', 'Unknown'),
                            'seendate': article.get('seendate', ''),
                            'language': article.get('language', 'en')
                        })
                
                return articles
            except json.JSONDecodeError:
                return []
        else:
            return []
            
    except Exception as e:
        print(f"  ⚠ Error fetching news for {event_date}: {str(e)[:50]}")
        return []


def create_event_calendar_with_news(start_year=2001, end_year=2026, fetch_news=True):
    """
    Creates calendar based on known release schedules with GDELT news
    """
    
    events = []
    
    print("=" * 70)
    print(f"CREATING ECONOMIC CALENDAR: {start_year} - {end_year}")
    if fetch_news:
        print("✓ Using GDELT for news (FREE, historical data available)")
    else:
        print("⚠ Skipping news fetching")
    print("=" * 70 + "\n")
    
    total_events = 0
    events_with_news = 0
    
    for year in range(start_year, end_year + 1):
        print(f"Processing {year}...", end=" ", flush=True)
        year_events = []
        
        for month in range(1, 13):
            
            # NFP - First Friday of every month at 8:30 AM EST
            first_friday = get_nth_weekday(year, month, 1, 4)
            if first_friday:
                event_date = first_friday.strftime('%Y-%m-%d')
                news = []
                
                if fetch_news:
                    news = get_gdelt_news('Non-Farm Payrolls', event_date, 2)
                    if news:
                        events_with_news += 1
                    time.sleep(0.5)  # Be nice to GDELT
                
                year_events.append({
                    'date': event_date,
                    'time': '08:30',
                    'event': 'Non-Farm Payrolls (NFP)',
                    'impact': 'High',
                    'frequency': 'Monthly',
                    'news': news
                })
                total_events += 1
            
            # CPI - Around 13th-15th of each month at 8:30 AM EST
            cpi_date = datetime(year, month, 13)
            while cpi_date.weekday() >= 5:
                cpi_date += timedelta(days=1)
            
            event_date = cpi_date.strftime('%Y-%m-%d')
            news = []
            
            if fetch_news:
                news = get_gdelt_news('Consumer Price Index', event_date, 2)
                if news:
                    events_with_news += 1
                time.sleep(0.5)
            
            year_events.append({
                'date': event_date,
                'time': '08:30',
                'event': 'Consumer Price Index (CPI)',
                'impact': 'High',
                'frequency': 'Monthly',
                'news': news
            })
            total_events += 1
            
            # PPI - Around 14th-16th
            ppi_date = datetime(year, month, 14)
            while ppi_date.weekday() >= 5:
                ppi_date += timedelta(days=1)
            
            event_date = ppi_date.strftime('%Y-%m-%d')
            news = []
            
            if fetch_news:
                news = get_gdelt_news('Producer Price Index', event_date, 2)
                if news:
                    events_with_news += 1
                time.sleep(0.5)
            
            year_events.append({
                'date': event_date,
                'time': '08:30',
                'event': 'Producer Price Index (PPI)',
                'impact': 'High',
                'frequency': 'Monthly',
                'news': news
            })
            total_events += 1
            
            # Retail Sales - Mid-month around 15th-17th at 8:30 AM
            retail_date = datetime(year, month, 16)
            while retail_date.weekday() >= 5:
                retail_date += timedelta(days=1)
            
            event_date = retail_date.strftime('%Y-%m-%d')
            news = []
            
            if fetch_news:
                news = get_gdelt_news('Retail Sales', event_date, 2)
                if news:
                    events_with_news += 1
                time.sleep(0.5)
            
            year_events.append({
                'date': event_date,
                'time': '08:30',
                'event': 'Retail Sales',
                'impact': 'High',
                'frequency': 'Monthly',
                'news': news
            })
            total_events += 1
            
            # ISM Manufacturing PMI - First business day at 10:00 AM
            ism_date = datetime(year, month, 1)
            while ism_date.weekday() >= 5:
                ism_date += timedelta(days=1)
            
            event_date = ism_date.strftime('%Y-%m-%d')
            news = []
            
            if fetch_news:
                news = get_gdelt_news('ISM Manufacturing', event_date, 2)
                if news:
                    events_with_news += 1
                time.sleep(0.5)
            
            year_events.append({
                'date': event_date,
                'time': '10:00',
                'event': 'ISM Manufacturing PMI',
                'impact': 'High',
                'frequency': 'Monthly',
                'news': news
            })
            total_events += 1
            
            # ISM Services PMI - Third business day at 10:00 AM
            ism_services = datetime(year, month, 3)
            while ism_services.weekday() >= 5:
                ism_services += timedelta(days=1)
            
            event_date = ism_services.strftime('%Y-%m-%d')
            news = []
            
            if fetch_news:
                news = get_gdelt_news('ISM Services', event_date, 2)
                if news:
                    events_with_news += 1
                time.sleep(0.5)
            
            year_events.append({
                'date': event_date,
                'time': '10:00',
                'event': 'ISM Services PMI',
                'impact': 'High',
                'frequency': 'Monthly',
                'news': news
            })
            total_events += 1
        
        events.extend(year_events)
        print(f"✓ ({len(year_events)} events)")
    
    # Add FOMC meetings
    print("\nAdding FOMC meetings...", end=" ", flush=True)
    fomc_dates = get_fomc_dates(start_year, end_year)
    for event_date in fomc_dates:
        news = []
        if fetch_news:
            news = get_gdelt_news('FOMC', event_date, 2)
            if news:
                events_with_news += 1
            time.sleep(0.5)
        
        events.append({
            'date': event_date,
            'time': '14:00',
            'event': 'FOMC Rate Decision',
            'impact': 'High',
            'frequency': 'Every 6 weeks',
            'news': news
        })
        total_events += 1
    print("✓")
    
    # Sort by date
    events = sorted(events, key=lambda x: x['date'])
    
    # Save as JSON
    json_filename = f'economic_calendar_{start_year}_{end_year}_with_news.json'
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(events, f, indent=2, ensure_ascii=False)
    
    # Save as CSV
    csv_data = []
    for event in events:
        news_titles = []
        news_urls = []
        for article in event.get('news', []):
            news_titles.append(article.get('title', ''))
            news_urls.append(article.get('url', ''))
        
        csv_row = {
            'date': event['date'],
            'time': event['time'],
            'event': event['event'],
            'impact': event['impact'],
            'frequency': event['frequency'],
            'news_count': len(event.get('news', [])),
            'news_title_1': news_titles[0] if len(news_titles) > 0 else '',
            'news_url_1': news_urls[0] if len(news_urls) > 0 else '',
            'news_title_2': news_titles[1] if len(news_titles) > 1 else '',
            'news_url_2': news_urls[1] if len(news_urls) > 1 else '',
        }
        csv_data.append(csv_row)
    
    df = pd.DataFrame(csv_data)
    csv_filename = f'economic_calendar_{start_year}_{end_year}_with_news.csv'
    df.to_csv(csv_filename, index=False)
    
    print("\n" + "=" * 70)
    print(f"✓ Created calendar with {total_events} events")
    print(f"✓ JSON saved to: {json_filename}")
    print(f"✓ CSV saved to: {csv_filename}")
    
    if fetch_news:
        total_news = sum(len(e.get('news', [])) for e in events)
        print(f"✓ Events with news: {events_with_news}/{total_events}")
        print(f"✓ Total news articles fetched: {total_news}")
    
    print("\nEvent breakdown:")
    event_types = {}
    for event in events:
        event_name = event['event']
        event_types[event_name] = event_types.get(event_name, 0) + 1
    
    for event_name, count in sorted(event_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {event_name}: {count}")
    
    print("\n" + "=" * 70)
    print("SAMPLE EVENTS WITH NEWS:")
    print("=" * 70)
    
    # Show sample events with news
    events_shown = 0
    for event in events:
        if event.get('news') and events_shown < 3:
            print(f"\n📅 {event['date']} {event['time']} - {event['event']}")
            for i, article in enumerate(event['news'], 1):
                print(f"  📰 News {i}: {article['title']}")
                print(f"     🔗 {article['url'][:80]}...")
            events_shown += 1
    
    if events_shown == 0:
        print("\n⚠ No news articles found in sample")
        print("Note: GDELT coverage may be sparse for older dates")
    
    print("\n" + "=" * 70)
    
    return events


def get_nth_weekday(year, month, n, weekday):
    """Get nth weekday of month (e.g., 1st Friday)"""
    first_day = datetime(year, month, 1)
    first_weekday = first_day.weekday()
    
    offset = (weekday - first_weekday) % 7
    target_date = first_day + timedelta(days=offset + (n - 1) * 7)
    
    if target_date.month == month:
        return target_date
    return None


def get_fomc_dates(start_year, end_year):
    """FOMC meets 8 times per year"""
    dates = []
    months = [1, 3, 5, 6, 7, 9, 11, 12]
    
    for year in range(start_year, end_year + 1):
        for month in months:
            if year >= 2001 and year <= 2026:
                last_day = datetime(year, month, 28)
                while last_day.weekday() != 2:  # Wednesday
                    last_day -= timedelta(days=1)
                dates.append(last_day.strftime('%Y-%m-%d'))
    
    return dates


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("US ECONOMIC CALENDAR WITH NEWS (2001-2026)")
    print("Using GDELT - 100% FREE, Historical Data Available")
    print("=" * 70)
    print("\nThis will:")
    print("• Create calendar with major US economic events")
    print("• Fetch 2 news articles per event from GDELT")
    print("• Save as JSON and CSV")
    print("\n⚠ NOTE: This will take 15-30 minutes due to API rate limiting")
    print("=" * 70)
    
    choice = input("\nFetch news articles? (y/n): ").strip().lower()
    
    if choice == 'y':
        print("\n🚀 Starting... This will take a while, please be patient!\n")
        create_event_calendar_with_news(2001, 2026, fetch_news=True)
    else:
        print("\n🚀 Creating calendar without news...\n")
        create_event_calendar_with_news(2001, 2026, fetch_news=False)
