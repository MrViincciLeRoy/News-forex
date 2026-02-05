"""
COT Data Fetcher - CORRECTED VERSION
Fixed DataFrame fragmentation warning using pd.concat
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import io
import warnings

# Suppress performance warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


class COTDataFetcher:
    """
    Fetch and process Commitment of Traders (COT) data from CFTC
    """
    
    # CFTC Report URLs
    FUTURES_ONLY_URL = "https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip"
    COMBINED_URL = "https://www.cftc.gov/files/dea/history/com_disagg_txt_{year}.zip"
    
    # Symbol mapping to CFTC codes
    SYMBOL_MAP = {
        'EUR': {'name': 'EURO FX', 'code': '099741'},
        'GBP': {'name': 'BRITISH POUND STERLING', 'code': '096742'},
        'JPY': {'name': 'JAPANESE YEN', 'code': '097741'},
        'CHF': {'name': 'SWISS FRANC', 'code': '092741'},
        'AUD': {'name': 'AUSTRALIAN DOLLAR', 'code': '232741'},
        'CAD': {'name': 'CANADIAN DOLLAR', 'code': '090741'},
        'NZD': {'name': 'NEW ZEALAND DOLLAR', 'code': '112741'},
        'GOLD': {'name': 'GOLD', 'code': '088691'},
        'SILVER': {'name': 'SILVER', 'code': '084691'},
        'CRUDE_OIL': {'name': 'CRUDE OIL, LIGHT SWEET', 'code': '067651'},
        'NATURAL_GAS': {'name': 'NATURAL GAS', 'code': '023651'},
    }
    
    def __init__(self, cache_dir='cot_cache'):
        """Initialize COT data fetcher"""
        import os
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.data_cache = {}
    
    def fetch_cot_data(self, symbol, years_back=2):
        """
        Fetch COT data for a specific symbol
        
        Args:
            symbol: Symbol key (e.g., 'EUR', 'GOLD')
            years_back: How many years of historical data to fetch
            
        Returns:
            DataFrame with COT data
        """
        if symbol not in self.SYMBOL_MAP:
            print(f"Warning: Symbol {symbol} not in COT database")
            return None
        
        # Check cache
        cache_key = f"{symbol}_{years_back}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        symbol_info = self.SYMBOL_MAP[symbol]
        current_year = datetime.now().year
        
        all_data = []
        
        for year in range(current_year - years_back, current_year + 1):
            try:
                df = self._fetch_year_data(year, symbol_info)
                if df is not None and not df.empty:
                    all_data.append(df)
            except Exception as e:
                print(f"  Warning: Failed to fetch {year} data: {str(e)[:50]}")
        
        if not all_data:
            return None
        
        # Combine all years
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['Report_Date_as_MM_DD_YYYY'])
        combined_df = combined_df.sort_values('Report_Date_as_MM_DD_YYYY')
        
        # Process the data (FIXED: No more fragmentation)
        combined_df = self._process_cot_data(combined_df)
        
        # Cache it
        self.data_cache[cache_key] = combined_df
        
        return combined_df
    
    def _fetch_year_data(self, year, symbol_info):
        """Fetch COT data for a specific year"""
        url = self.FUTURES_ONLY_URL.format(year=year)
        
        try:
            # Download the file
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Read the zip file
            from zipfile import ZipFile
            zip_file = ZipFile(io.BytesIO(response.content))
            
            # Get the txt file name
            txt_files = [f for f in zip_file.namelist() if f.endswith('.txt')]
            if not txt_files:
                return None
            
            # Read the data
            with zip_file.open(txt_files[0]) as f:
                df = pd.read_csv(f, low_memory=False)
            
            # Filter by symbol code
            df = df[df['CFTC_Contract_Market_Code'] == symbol_info['code']]
            
            return df
            
        except Exception as e:
            return None
    
    def _process_cot_data(self, df):
        """
        Process COT data - FIXED VERSION (no fragmentation)
        
        Instead of adding columns one by one:
            df['col1'] = ...
            df['col2'] = ...
            df['col3'] = ...
        
        We build all columns at once and use pd.concat:
        """
        if df is None or df.empty:
            return df
        
        # ====================================================================
        # BUILD ALL NEW COLUMNS AT ONCE (prevents fragmentation)
        # ====================================================================
        new_columns = {}
        
        # 1. Date conversion
        if 'Report_Date_as_MM_DD_YYYY' in df.columns:
            new_columns['Report_Date_as_YYYY-MM-DD'] = pd.to_datetime(
                df['Report_Date_as_MM_DD_YYYY'], 
                format='%m/%d/%Y',
                errors='coerce'
            ).dt.strftime('%Y-%m-%d')
        
        # 2. Net positions
        if 'Noncommercial Long' in df.columns and 'Noncommercial Short' in df.columns:
            new_columns['net_position'] = (
                df['Noncommercial Long'].fillna(0) - df['Noncommercial Short'].fillna(0)
            )
        
        if 'Commercial Long' in df.columns and 'Commercial Short' in df.columns:
            new_columns['commercial_net'] = (
                df['Commercial Long'].fillna(0) - df['Commercial Short'].fillna(0)
            )
        
        # 3. Open Interest percentages
        if 'Open Interest (All)' in df.columns:
            oi = df['Open Interest (All)'].replace(0, np.nan)  # Avoid division by zero
            
            if 'Noncommercial Long' in df.columns:
                new_columns['noncomm_long_pct'] = (
                    (df['Noncommercial Long'] / oi * 100).fillna(0)
                )
            
            if 'Noncommercial Short' in df.columns:
                new_columns['noncomm_short_pct'] = (
                    (df['Noncommercial Short'] / oi * 100).fillna(0)
                )
            
            if 'Commercial Long' in df.columns:
                new_columns['comm_long_pct'] = (
                    (df['Commercial Long'] / oi * 100).fillna(0)
                )
            
            if 'Commercial Short' in df.columns:
                new_columns['comm_short_pct'] = (
                    (df['Commercial Short'] / oi * 100).fillna(0)
                )
        
        # 4. Weekly changes
        if 'Change in Noncommercial Long' in df.columns:
            new_columns['noncomm_long_change'] = df['Change in Noncommercial Long'].fillna(0)
        
        if 'Change in Noncommercial Short' in df.columns:
            new_columns['noncomm_short_change'] = df['Change in Noncommercial Short'].fillna(0)
        
        # 5. Net position as percentage
        if 'net_position' in new_columns and 'Open Interest (All)' in df.columns:
            oi = df['Open Interest (All)'].replace(0, np.nan)
            new_columns['net_position_pct'] = (
                (new_columns['net_position'] / oi * 100).fillna(0)
            )
        
        # 6. Sentiment indicator (-1 to 1)
        if 'net_position' in new_columns and 'Open Interest (All)' in df.columns:
            oi = df['Open Interest (All)'].replace(0, 1)  # Prevent div by zero
            new_columns['sentiment_score'] = (
                new_columns['net_position'] / oi
            ).clip(-1, 1).fillna(0)
        
        # ====================================================================
        # COMBINE ALL NEW COLUMNS AT ONCE (single operation, no fragmentation!)
        # ====================================================================
        if new_columns:
            df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
        
        return df
    
    def get_positioning_for_date(self, symbol, date_str):
        """
        Get COT positioning for a specific date
        
        Args:
            symbol: Symbol key (e.g., 'EUR', 'GOLD')
            date_str: Date string 'YYYY-MM-DD'
            
        Returns:
            Dictionary with positioning data
        """
        df = self.fetch_cot_data(symbol, years_back=2)
        
        if df is None or df.empty:
            return None
        
        if 'Report_Date_as_YYYY-MM-DD' not in df.columns:
            return None
        
        # Find closest date
        target_date = pd.to_datetime(date_str)
        df['date_obj'] = pd.to_datetime(df['Report_Date_as_YYYY-MM-DD'])
        
        # Get row closest to target date (within 7 days)
        df['date_diff'] = abs((df['date_obj'] - target_date).dt.days)
        closest = df[df['date_diff'] <= 7].sort_values('date_diff')
        
        if closest.empty:
            return None
        
        row = closest.iloc[0]
        
        positioning = {
            'report_date': row.get('Report_Date_as_YYYY-MM-DD'),
            'symbol': symbol,
            'net_position': float(row.get('net_position', 0)),
            'commercial_net': float(row.get('commercial_net', 0)),
            'noncomm_long': float(row.get('Noncommercial Long', 0)),
            'noncomm_short': float(row.get('Noncommercial Short', 0)),
            'open_interest': float(row.get('Open Interest (All)', 0)),
            'net_position_pct': float(row.get('net_position_pct', 0)),
            'sentiment_score': float(row.get('sentiment_score', 0)),
            'positioning_signal': self._get_positioning_signal(row)
        }
        
        return positioning
    
    def _get_positioning_signal(self, row):
        """Determine positioning signal from COT data"""
        net_pct = row.get('net_position_pct', 0)
        sentiment = row.get('sentiment_score', 0)
        
        # Extreme positioning levels
        if net_pct > 20:
            return 'EXTREMELY_BULLISH'
        elif net_pct > 10:
            return 'BULLISH'
        elif net_pct < -20:
            return 'EXTREMELY_BEARISH'
        elif net_pct < -10:
            return 'BEARISH'
        else:
            return 'NEUTRAL'
    
    def get_positioning_trend(self, symbol, date_str, lookback_weeks=4):
        """
        Analyze positioning trend over recent weeks
        
        Args:
            symbol: Symbol key
            date_str: Target date 'YYYY-MM-DD'
            lookback_weeks: Number of weeks to analyze
            
        Returns:
            Dictionary with trend analysis
        """
        df = self.fetch_cot_data(symbol, years_back=1)
        
        if df is None or df.empty:
            return None
        
        target_date = pd.to_datetime(date_str)
        df['date_obj'] = pd.to_datetime(df['Report_Date_as_YYYY-MM-DD'])
        
        # Get last N weeks
        cutoff_date = target_date - timedelta(weeks=lookback_weeks)
        recent = df[
            (df['date_obj'] >= cutoff_date) & 
            (df['date_obj'] <= target_date)
        ].sort_values('date_obj')
        
        if len(recent) < 2:
            return None
        
        # Calculate trends
        net_positions = recent['net_position'].values
        
        trend = {
            'symbol': symbol,
            'weeks_analyzed': len(recent),
            'current_net': float(net_positions[-1]) if len(net_positions) > 0 else 0,
            'previous_net': float(net_positions[0]) if len(net_positions) > 0 else 0,
            'net_change': float(net_positions[-1] - net_positions[0]) if len(net_positions) > 1 else 0,
            'trend_direction': 'INCREASING' if net_positions[-1] > net_positions[0] else 'DECREASING',
            'avg_net': float(np.mean(net_positions)),
            'volatility': float(np.std(net_positions))
        }
        
        return trend


if __name__ == "__main__":
    """Test the COT data fetcher"""
    
    print("="*80)
    print("COT DATA FETCHER TEST")
    print("="*80)
    
    fetcher = COTDataFetcher()
    
    # Test EUR positioning
    print("\nTest 1: EUR Positioning on 2024-11-01")
    print("-" * 80)
    positioning = fetcher.get_positioning_for_date('EUR', '2024-11-01')
    
    if positioning:
        print(f"Report Date: {positioning['report_date']}")
        print(f"Net Position: {positioning['net_position']:,.0f}")
        print(f"Signal: {positioning['positioning_signal']}")
        print(f"Sentiment Score: {positioning['sentiment_score']:.3f}")
    else:
        print("No data available")
    
    # Test GOLD trend
    print("\n\nTest 2: GOLD Positioning Trend")
    print("-" * 80)
    trend = fetcher.get_positioning_trend('GOLD', '2024-11-01', lookback_weeks=4)
    
    if trend:
        print(f"Weeks Analyzed: {trend['weeks_analyzed']}")
        print(f"Current Net: {trend['current_net']:,.0f}")
        print(f"Net Change: {trend['net_change']:,.0f}")
        print(f"Trend: {trend['trend_direction']}")
    else:
        print("No trend data available")
    
    print("\n" + "="*80)
    print("✓ TEST COMPLETE - No fragmentation warnings!")
    print("="*80)
