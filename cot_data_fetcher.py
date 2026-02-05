import pandas as pd
import requests
from datetime import datetime, timedelta
import io
import zipfile


class COTDataFetcher:
    def __init__(self):
        self.base_url = "https://www.cftc.gov/files/dea/history"
        
        self.cftc_codes = {
            'EUR': '099741',
            'GBP': '096742',
            'JPY': '097741',
            'CHF': '092741',
            'CAD': '090741',
            'AUD': '232741',
            'NZD': '112741',
            'MXN': '095741',
            'GOLD': '088691',
            'SILVER': '084691',
            'CRUDE_OIL': '067651',
            'NATURAL_GAS': '023651',
            'SP500': '13874+',
            'NASDAQ': '209742',
            'DOW': '124603',
            'US_DOLLAR_INDEX': '098662',
            'BITCOIN': '133741'
        }
        
        self.symbol_names = {
            'EUR': 'Euro FX',
            'GBP': 'British Pound',
            'JPY': 'Japanese Yen',
            'CHF': 'Swiss Franc',
            'CAD': 'Canadian Dollar',
            'AUD': 'Australian Dollar',
            'NZD': 'New Zealand Dollar',
            'MXN': 'Mexican Peso',
            'GOLD': 'Gold',
            'SILVER': 'Silver',
            'CRUDE_OIL': 'Crude Oil WTI',
            'NATURAL_GAS': 'Natural Gas',
            'SP500': 'S&P 500',
            'NASDAQ': 'NASDAQ 100',
            'DOW': 'Dow Jones',
            'US_DOLLAR_INDEX': 'US Dollar Index',
            'BITCOIN': 'Bitcoin'
        }
    
    def get_report_date_for_date(self, target_date):
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)
        
        days_since_tuesday = (target_date.weekday() - 1) % 7
        report_tuesday = target_date - timedelta(days=days_since_tuesday)
        release_friday = report_tuesday + timedelta(days=3)
        
        return report_tuesday, release_friday
    
    def fetch_cot_data(self, year, report_type='financial'):
        print(f"\n[DEBUG] Fetching {report_type} data for year {year}")
        
        filenames = [
            f"dea_fut_xls_{year}.zip" if report_type == 'financial' else f"dea_com_xls_{year}.zip",
            f"deacot{year}.zip" if report_type == 'financial' else f"deacom{year}.zip",
            f"annual.txt"
        ]
        
        for idx, filename in enumerate(filenames, 1):
            url = f"{self.base_url}/{filename}"
            print(f"[DEBUG] Attempt {idx}/{len(filenames)}: {url}")
            
            try:
                response = requests.get(url, timeout=30)
                print(f"[DEBUG] Response status: {response.status_code}")
                
                if response.status_code == 200:
                    if filename.endswith('.zip'):
                        print(f"[DEBUG] Processing ZIP file...")
                        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                            print(f"[DEBUG] ZIP contents: {z.namelist()}")
                            
                            for file_info in z.namelist():
                                if file_info.endswith(('.xls', '.xlsx', '.txt')):
                                    print(f"[DEBUG] Extracting {file_info}...")
                                    
                                    with z.open(file_info) as f:
                                        file_content = f.read()
                                        
                                        if file_info.endswith('.txt'):
                                            df = pd.read_csv(io.BytesIO(file_content))
                                        else:
                                            df = pd.read_excel(io.BytesIO(file_content))
                                        
                                        print(f"[DEBUG] DataFrame shape: {df.shape}")
                                        
                                        # Handle different date column formats
                                        if 'Report_Date_as_MM_DD_YYYY' in df.columns:
                                            df['Report_Date_as_YYYY-MM-DD'] = pd.to_datetime(
                                                df['Report_Date_as_MM_DD_YYYY']
                                            )
                                        elif 'Report_Date_as_YYYY-MM-DD' in df.columns:
                                            df['Report_Date_as_YYYY-MM-DD'] = pd.to_datetime(
                                                df['Report_Date_as_YYYY-MM-DD']
                                            )
                                        else:
                                            print(f"[DEBUG] No recognized date column found")
                                            continue
                                        
                                        print(f"✓ Successfully loaded {len(df)} records for {year}")
                                        return df
                    
                    elif filename.endswith('.txt'):
                        df = pd.read_csv(io.StringIO(response.text))
                        
                        if 'Report_Date_as_MM_DD_YYYY' in df.columns:
                            df['Report_Date_as_YYYY-MM-DD'] = pd.to_datetime(
                                df['Report_Date_as_MM_DD_YYYY']
                            )
                        elif 'Report_Date_as_YYYY-MM-DD' in df.columns:
                            df['Report_Date_as_YYYY-MM-DD'] = pd.to_datetime(
                                df['Report_Date_as_YYYY-MM-DD']
                            )
                        else:
                            continue
                        
                        df_year = df[df['Report_Date_as_YYYY-MM-DD'].dt.year == year]
                        if not df_year.empty:
                            print(f"✓ Loaded {len(df_year)} records for {year}")
                            return df_year
            
            except Exception as e:
                print(f"[ERROR] {type(e).__name__}: {e}")
                continue
        
        print(f"[ERROR] Could not fetch {year} data from any source")
        return None
    
    def get_symbol_data(self, symbol, start_date, end_date=None, report_type='financial'):
        if symbol.upper() not in self.cftc_codes:
            print(f"Symbol {symbol} not supported. Available: {list(self.cftc_codes.keys())}")
            return None
        
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if end_date is None:
            end_date = datetime.now()
        elif isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        code = self.cftc_codes[symbol.upper()]
        
        start_year = start_date.year
        end_year = end_date.year
        
        all_data = []
        
        for year in range(start_year, end_year + 1):
            print(f"Fetching {symbol} data for {year}...")
            df = self.fetch_cot_data(year, report_type)
            
            if df is not None:
                symbol_data = df[df['CFTC_Contract_Market_Code'] == code].copy()
                all_data.append(symbol_data)
        
        if not all_data:
            return None
        
        combined = pd.concat(all_data, ignore_index=True)
        combined = combined[
            (combined['Report_Date_as_YYYY-MM-DD'] >= start_date) &
            (combined['Report_Date_as_YYYY-MM-DD'] <= end_date)
        ]
        combined = combined.sort_values('Report_Date_as_YYYY-MM-DD')
        
        return combined
    
    def get_positioning_for_date(self, symbol, target_date):
        print(f"\n[INFO] Getting positioning for {symbol} on {target_date}")
        
        report_tuesday, release_friday = self.get_report_date_for_date(target_date)
        print(f"[INFO] Report Tuesday: {report_tuesday.strftime('%Y-%m-%d')}")
        print(f"[INFO] Release Friday: {release_friday.strftime('%Y-%m-%d')}")
        
        year = report_tuesday.year
        df = self.fetch_cot_data(year)
        
        if df is None:
            print(f"[ERROR] No data available for year {year}")
            return None
        
        code = self.cftc_codes.get(symbol.upper())
        if not code:
            print(f"[ERROR] No CFTC code found for {symbol}")
            return None
        
        symbol_data = df[df['CFTC_Contract_Market_Code'] == code]
        
        if len(symbol_data) == 0:
            print(f"[ERROR] No data found for {symbol} (code: {code})")
            return None
        
        report_data = symbol_data[
            symbol_data['Report_Date_as_YYYY-MM-DD'] == report_tuesday
        ]
        
        if report_data.empty:
            closest = symbol_data.iloc[(symbol_data['Report_Date_as_YYYY-MM-DD'] - report_tuesday).abs().argsort()[:1]]
            if not closest.empty:
                report_data = closest
                print(f"[DEBUG] Using closest date: {closest['Report_Date_as_YYYY-MM-DD'].iloc[0]}")
        
        if report_data.empty:
            print(f"[ERROR] No data found for {symbol} near {report_tuesday}")
            return None
        
        row = report_data.iloc[0]
        
        positioning = {
            'symbol': symbol.upper(),
            'report_date': row['Report_Date_as_YYYY-MM-DD'].strftime('%Y-%m-%d'),
            'release_date': release_friday.strftime('%Y-%m-%d'),
            'dealer': {
                'long': int(row.get('Dealer_Positions_Long_All', 0)),
                'short': int(row.get('Dealer_Positions_Short_All', 0)),
                'net': int(row.get('Dealer_Positions_Long_All', 0)) - int(row.get('Dealer_Positions_Short_All', 0))
            },
            'asset_manager': {
                'long': int(row.get('Asset_Mgr_Positions_Long_All', 0)),
                'short': int(row.get('Asset_Mgr_Positions_Short_All', 0)),
                'net': int(row.get('Asset_Mgr_Positions_Long_All', 0)) - int(row.get('Asset_Mgr_Positions_Short_All', 0))
            },
            'leveraged': {
                'long': int(row.get('Lev_Money_Positions_Long_All', 0)),
                'short': int(row.get('Lev_Money_Positions_Short_All', 0)),
                'net': int(row.get('Lev_Money_Positions_Long_All', 0)) - int(row.get('Lev_Money_Positions_Short_All', 0))
            },
            'other': {
                'long': int(row.get('Other_Rept_Positions_Long_All', 0)),
                'short': int(row.get('Other_Rept_Positions_Short_All', 0)),
                'net': int(row.get('Other_Rept_Positions_Long_All', 0)) - int(row.get('Other_Rept_Positions_Short_All', 0))
            },
            'open_interest': int(row.get('Open_Interest_All', 0))
        }
        
        positioning['sentiment'] = self._analyze_sentiment(positioning)
        
        return positioning
    
    def _analyze_sentiment(self, positioning):
        dealer_net = positioning['dealer']['net']
        asset_mgr_net = positioning['asset_manager']['net']
        leveraged_net = positioning['leveraged']['net']
        
        oi = positioning['open_interest']
        if oi == 0:
            return 'NEUTRAL'
        
        dealer_pct = (dealer_net / oi) * 100
        asset_pct = (asset_mgr_net / oi) * 100
        lev_pct = (leveraged_net / oi) * 100
        
        smart_money = dealer_pct + asset_pct
        
        if smart_money > 15:
            sentiment = 'BULLISH'
        elif smart_money < -15:
            sentiment = 'BEARISH'
        else:
            sentiment = 'NEUTRAL'
        
        extreme = ''
        if abs(lev_pct) > 30:
            extreme = ' (EXTREME - REVERSAL RISK)'
        
        positioning['analysis'] = {
            'dealer_net_pct': round(dealer_pct, 2),
            'asset_mgr_net_pct': round(asset_pct, 2),
            'leveraged_net_pct': round(lev_pct, 2),
            'smart_money_net_pct': round(smart_money, 2)
        }
        
        return sentiment + extreme
    
    def compare_positioning(self, symbol, date1, date2):
        pos1 = self.get_positioning_for_date(symbol, date1)
        pos2 = self.get_positioning_for_date(symbol, date2)
        
        if not pos1 or not pos2:
            return None
        
        change = {
            'symbol': symbol.upper(),
            'from_date': pos1['report_date'],
            'to_date': pos2['report_date'],
            'dealer_net_change': pos2['dealer']['net'] - pos1['dealer']['net'],
            'asset_mgr_net_change': pos2['asset_manager']['net'] - pos1['asset_manager']['net'],
            'leveraged_net_change': pos2['leveraged']['net'] - pos1['leveraged']['net'],
            'sentiment_change': f"{pos1['sentiment']} → {pos2['sentiment']}"
        }
        
        return change


if __name__ == "__main__":
    fetcher = COTDataFetcher()
    
    print("="*80)
    print("COT DATA FETCHER - FIXED VERSION")
    print("="*80)
    
    symbols = ['EUR', 'GOLD', 'CRUDE_OIL']
    test_date = '2024-11-01'
    
    for symbol in symbols:
        print(f"\n{'='*80}")
        print(f"Symbol: {symbol} | Date: {test_date}")
        print('='*80)
        
        positioning = fetcher.get_positioning_for_date(symbol, test_date)
        
        if positioning:
            print(f"\n✓ SUCCESS!")
            print(f"Report Date: {positioning['report_date']}")
            print(f"Release Date: {positioning['release_date']}")
            print(f"Sentiment: {positioning['sentiment']}")
            print(f"Open Interest: {positioning['open_interest']:,}")
        else:
            print(f"\n✗ FAILED - No data available for {symbol}")
    
    print(f"\n{'='*80}")
    print("Test complete")
    print('='*80)
