import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import json

# ============================================================================
# METHOD 1: TradingEconomics (Free tier - limited but good)
# ============================================================================
def get_tradingeconomics_calendar(api_key=None):
    """
    TradingEconomics has a free tier (limited to 500 requests/month)
    Get API key: https://tradingeconomics.com/api/register
    """
    if not api_key:
        print("Get free API key from: https://tradingeconomics.com/api/register")
        return None
    
    base_url = "https://api.tradingeconomics.com/calendar"
    
    # Get events from 2010 to now
    params = {
        'c': api_key,
        'country': 'united states',
        'importance': '3',  # High importance only
        'f': 'json'
    }
    
    try:
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            df.to_csv('economic_calendar_tradingeconomics.csv', index=False)
            print(f"✓ Downloaded {len(df)} events")
            return df
        else:
            print(f"Error: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")
    
    return None


# ============================================================================
# METHOD 2: Alpha Vantage (FREE - Good for economic indicators)
# ============================================================================
def get_alphavantage_data(api_key='demo'):
    """
    Alpha Vantage is FREE and has economic data
    Get API key: https://www.alphavantage.co/support/#api-key
    """
    
    indicators = {
        'NONFARM_PAYROLL': 'Non-Farm Payrolls',
        'CPI': 'Consumer Price Index',
        'RETAIL_SALES': 'Retail Sales',
        'UNEMPLOYMENT': 'Unemployment Rate',
        'FEDERAL_FUNDS_RATE': 'Fed Funds Rate'
    }
    
    all_data = []
    
    for indicator, name in indicators.items():
        url = f'https://www.alphavantage.co/query'
        params = {
            'function': indicator,
            'apikey': api_key,
            'datatype': 'json'
        }
        
        try:
            print(f"Fetching {name}...", end=" ")
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data:
                    for item in data['data']:
                        all_data.append({
                            'date': item['date'],
                            'indicator': name,
                            'value': item['value']
                        })
                    print(f"✓ {len(data['data'])} records")
                else:
                    print("✗ No data")
            
            time.sleep(12)  # Free tier: 5 calls per minute
            
        except Exception as e:
            print(f"✗ Error: {e}")
    
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv('economic_data_alphavantage.csv', index=False)
        print(f"\n✓ Saved {len(df)} records to economic_data_alphavantage.csv")
        return df
    
    return None


# ============================================================================
# METHOD 3: Manual Calendar Creation (Most Reliable)
# ============================================================================
def create_event_calendar_from_rules(start_year=2001, end_year=2026):
    """
    Creates calendar based on known release schedules
    This is the MOST RELIABLE method since events follow patterns
    """
    
    events = []
    
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            
            # NFP - First Friday of every month at 8:30 AM EST
            first_friday = get_nth_weekday(year, month, 1, 4)  # 4 = Friday
            if first_friday:
                events.append({
                    'date': first_friday.strftime('%Y-%m-%d'),
                    'time': '08:30',
                    'event': 'Non-Farm Payrolls (NFP)',
                    'impact': 'High',
                    'frequency': 'Monthly'
                })
            
            # CPI - Around 13th-15th of each month at 8:30 AM EST
            cpi_date = datetime(year, month, 13)
            # Adjust if weekend
            while cpi_date.weekday() >= 5:  # Sat=5, Sun=6
                cpi_date += timedelta(days=1)
            
            events.append({
                'date': cpi_date.strftime('%Y-%m-%d'),
                'time': '08:30',
                'event': 'Consumer Price Index (CPI)',
                'impact': 'High',
                'frequency': 'Monthly'
            })
            
            # PPI - Around 14th-16th
            ppi_date = datetime(year, month, 14)
            while ppi_date.weekday() >= 5:
                ppi_date += timedelta(days=1)
            
            events.append({
                'date': ppi_date.strftime('%Y-%m-%d'),
                'time': '08:30',
                'event': 'Producer Price Index (PPI)',
                'impact': 'High',
                'frequency': 'Monthly'
            })
            
            # Retail Sales - Mid-month around 15th-17th at 8:30 AM
            retail_date = datetime(year, month, 16)
            while retail_date.weekday() >= 5:
                retail_date += timedelta(days=1)
            
            events.append({
                'date': retail_date.strftime('%Y-%m-%d'),
                'time': '08:30',
                'event': 'Retail Sales',
                'impact': 'High',
                'frequency': 'Monthly'
            })
            
            # ISM Manufacturing PMI - First business day of month at 10:00 AM
            ism_date = datetime(year, month, 1)
            while ism_date.weekday() >= 5:
                ism_date += timedelta(days=1)
            
            events.append({
                'date': ism_date.strftime('%Y-%m-%d'),
                'time': '10:00',
                'event': 'ISM Manufacturing PMI',
                'impact': 'High',
                'frequency': 'Monthly'
            })
            
            # ISM Services PMI - Third business day at 10:00 AM
            ism_services = datetime(year, month, 3)
            while ism_services.weekday() >= 5:
                ism_services += timedelta(days=1)
            
            events.append({
                'date': ism_services.strftime('%Y-%m-%d'),
                'time': '10:00',
                'event': 'ISM Services PMI',
                'impact': 'High',
                'frequency': 'Monthly'
            })
            
            # Unemployment Claims - Every Thursday at 8:30 AM
            for day in range(1, 32):
                try:
                    date = datetime(year, month, day)
                    if date.weekday() == 3:  # Thursday
                        events.append({
                            'date': date.strftime('%Y-%m-%d'),
                            'time': '08:30',
                            'event': 'Initial Jobless Claims',
                            'impact': 'Medium',
                            'frequency': 'Weekly'
                        })
                except ValueError:
                    break
    
    # Add FOMC meetings (8 per year, known schedule)
    fomc_dates = get_fomc_dates(start_year, end_year)
    for date in fomc_dates:
        events.append({
            'date': date,
            'time': '14:00',
            'event': 'FOMC Rate Decision',
            'impact': 'High',
            'frequency': 'Every 6 weeks'
        })
    
    df = pd.DataFrame(events)
    df = df.sort_values('date')
    df.to_csv('us_economic_calendar_2001_2026.csv', index=False)
    
    print("=" * 70)
    print(f"✓ Created calendar with {len(df)} events from {start_year} to {end_year}")
    print(f"✓ Saved to: us_economic_calendar_2001_2026.csv")
    print("\nEvent breakdown:")
    print(df['event'].value_counts())
    print("\nFirst 10 events:")
    print(df.head(10))
    
    return df


def get_nth_weekday(year, month, n, weekday):
    """Get nth weekday of month (e.g., 1st Friday)"""
    first_day = datetime(year, month, 1)
    first_weekday = first_day.weekday()
    
    # Calculate offset to desired weekday
    offset = (weekday - first_weekday) % 7
    target_date = first_day + timedelta(days=offset + (n - 1) * 7)
    
    # Check if still in same month
    if target_date.month == month:
        return target_date
    return None


def get_fomc_dates(start_year, end_year):
    """
    FOMC meets 8 times per year, roughly every 6 weeks
    Approximate schedule (usually Tue/Wed in these months)
    """
    dates = []
    months = [1, 3, 5, 6, 7, 9, 11, 12]  # Typical FOMC meeting months
    
    for year in range(start_year, end_year + 1):
        for month in months:
            if year >= 2001 and year <= 2026:
                # Usually last Tue/Wed of these months
                last_day = datetime(year, month, 28)
                # Find last Wednesday
                while last_day.weekday() != 2:  # 2 = Wednesday
                    last_day -= timedelta(days=1)
                dates.append(last_day.strftime('%Y-%m-%d'))
    
    return dates


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("US ECONOMIC CALENDAR DOWNLOADER")
    print("=" * 70)
    print("\nChoose method:")
    print("1. Alpha Vantage API (FREE - requires API key)")
    print("2. TradingEconomics API (FREE tier - requires API key)")
    print("3. Rule-based Calendar (NO API needed - RECOMMENDED)")
    print("\n" + "=" * 70)
    
    choice = input("\nEnter choice (1, 2, or 3): ").strip()
    
    if choice == '1':
        print("\nGet FREE API key: https://www.alphavantage.co/support/#api-key")
        api_key = input("Enter your API key (or press Enter for demo): ").strip()
        if not api_key:
            api_key = 'demo'
        get_alphavantage_data(api_key)
        
    elif choice == '2':
        print("\nGet FREE API key: https://tradingeconomics.com/api/register")
        api_key = input("Enter your API key: ").strip()
        get_tradingeconomics_calendar(api_key)
        
    else:  # Option 3 - Recommended
        print("\n✓ Creating calendar based on known release schedules...")
        print("This creates events for 2001-2026 with NO API needed\n")
        create_event_calendar_from_rules(2001, 2026)
