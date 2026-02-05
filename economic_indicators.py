import pandas as pd
import requests
import os
from datetime import datetime, timedelta
import yfinance as yf


class EconomicIndicatorIntegration:
    def __init__(self):
        self.fred_key = os.environ.get('FRED_API_KEY', '')
        self.cache = {}
        
        self.indicators = {
            'GDP': 'GDP',
            'CPI': 'CPIAUCSL',
            'Core_CPI': 'CPILFESL',
            'Unemployment': 'UNRATE',
            'Fed_Funds_Rate': 'FEDFUNDS',
            'PCE': 'PCE',
            'Retail_Sales': 'RSXFS',
            'Industrial_Production': 'INDPRO',
            'Consumer_Sentiment': 'UMCSENT',
            'Housing_Starts': 'HOUST',
            'Payrolls': 'PAYEMS',
            'ISM_Manufacturing': 'MANEMP',
            'Real_Yields_10Y': 'DFII10',
            'Treasury_10Y': 'DGS10',
            'Treasury_2Y': 'DGS2',
            'VIX': '^VIX'
        }
    
    def fetch_fred_data(self, series_id, start_date=None, end_date=None):
        if not self.fred_key:
            print("FRED_API_KEY not set")
            return None
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
        
        cache_key = f"{series_id}_{start_date}_{end_date}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        url = 'https://api.stlouisfed.org/fred/series/observations'
        params = {
            'series_id': series_id,
            'api_key': self.fred_key,
            'file_type': 'json',
            'observation_start': start_date,
            'observation_end': end_date
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                observations = data.get('observations', [])
                
                df = pd.DataFrame(observations)
                df['date'] = pd.to_datetime(df['date'])
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                df = df.dropna(subset=['value'])
                df = df.set_index('date')
                
                self.cache[cache_key] = df
                return df
            else:
                print(f"FRED API error: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching FRED data: {e}")
            return None
    
    def fetch_market_data(self, symbol, start_date, end_date):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                return None
            
            return df['Close']
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None
    
    def calculate_interest_rate_differential(self, date=None):
        if date is None:
            date = datetime.now()
        
        start_date = (date - timedelta(days=365)).strftime('%Y-%m-%d')
        end_date = date.strftime('%Y-%m-%d')
        
        fed_funds = self.fetch_fred_data('FEDFUNDS', start_date, end_date)
        treasury_10y = self.fetch_fred_data('DGS10', start_date, end_date)
        treasury_2y = self.fetch_fred_data('DGS2', start_date, end_date)
        
        if fed_funds is None or treasury_10y is None or treasury_2y is None:
            return None
        
        if len(fed_funds) == 0 or len(treasury_10y) == 0 or len(treasury_2y) == 0:
            return None
        
        current_fed = fed_funds['value'].iloc[-1]
        current_10y = treasury_10y['value'].iloc[-1]
        current_2y = treasury_2y['value'].iloc[-1]
        
        yield_curve = current_10y - current_2y
        
        return {
            'fed_funds_rate': round(current_fed, 2),
            'treasury_10y': round(current_10y, 2),
            'treasury_2y': round(current_2y, 2),
            'yield_curve_spread': round(yield_curve, 2),
            'yield_curve_status': 'INVERTED' if yield_curve < 0 else 'NORMAL',
            'recession_risk': 'HIGH' if yield_curve < -0.5 else ('MEDIUM' if yield_curve < 0 else 'LOW')
        }
    
    def analyze_inflation_trend(self, lookback_months=12):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_months * 30 + 60)
        
        cpi = self.fetch_fred_data('CPIAUCSL', start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        core_cpi = self.fetch_fred_data('CPILFESL', start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        if cpi is None or core_cpi is None:
            return None
        
        if len(cpi) < 14 or len(core_cpi) < 14:
            print(f"Insufficient data: CPI has {len(cpi)} rows, Core CPI has {len(core_cpi)} rows")
            return None
        
        # Calculate year-over-year change (use available data if less than 13 months)
        yoy_index = min(13, len(cpi) - 1)
        cpi_yoy = ((cpi['value'].iloc[-1] - cpi['value'].iloc[-yoy_index]) / cpi['value'].iloc[-yoy_index]) * 100
        core_cpi_yoy = ((core_cpi['value'].iloc[-1] - core_cpi['value'].iloc[-yoy_index]) / core_cpi['value'].iloc[-yoy_index]) * 100
        
        # Calculate month-over-month change
        if len(cpi) >= 2:
            cpi_mom = ((cpi['value'].iloc[-1] - cpi['value'].iloc[-2]) / cpi['value'].iloc[-2]) * 100
        else:
            cpi_mom = 0
        
        # Calculate trend using recent data (use available data if less than 6 months)
        trend_window = min(6, len(cpi))
        trend_recent = cpi['value'].iloc[-trend_window:].pct_change().mean() * 100
        
        return {
            'cpi_yoy': round(cpi_yoy, 2),
            'core_cpi_yoy': round(core_cpi_yoy, 2),
            'cpi_mom': round(cpi_mom, 2),
            'trend': 'RISING' if trend_recent > 0 else 'FALLING',
            'inflation_status': 'HIGH' if cpi_yoy > 3 else ('MODERATE' if cpi_yoy > 2 else 'LOW')
        }
    
    def get_employment_indicators(self):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        unemployment = self.fetch_fred_data('UNRATE', start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        payrolls = self.fetch_fred_data('PAYEMS', start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        if unemployment is None or payrolls is None:
            return None
        
        if len(unemployment) < 2 or len(payrolls) < 2:
            return None
        
        current_unemp = unemployment['value'].iloc[-1]
        prev_unemp = unemployment['value'].iloc[-2]
        
        current_payrolls = payrolls['value'].iloc[-1]
        prev_payrolls = payrolls['value'].iloc[-2]
        payroll_change = (current_payrolls - prev_payrolls) * 1000
        
        return {
            'unemployment_rate': round(current_unemp, 1),
            'unemployment_change': round(current_unemp - prev_unemp, 1),
            'payroll_change_monthly': int(payroll_change),
            'employment_trend': 'IMPROVING' if current_unemp < prev_unemp else 'WEAKENING'
        }
    
    def correlate_indicator_with_asset(self, indicator_series_id, asset_symbol, lookback_days=365):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        indicator_data = self.fetch_fred_data(indicator_series_id, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        if indicator_series_id == '^VIX':
            asset_data = self.fetch_market_data(asset_symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        else:
            asset_data = self.fetch_market_data(asset_symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        if indicator_data is None or asset_data is None:
            return None
        
        if len(indicator_data) == 0 or len(asset_data) == 0:
            return None
        
        # Ensure both indexes are timezone-naive for compatibility
        if hasattr(asset_data.index, 'tz') and asset_data.index.tz is not None:
            asset_data.index = asset_data.index.tz_localize(None)
        
        merged = pd.DataFrame({
            'indicator': indicator_data['value'],
            'asset': asset_data
        }).dropna()
        
        if len(merged) < 2:
            return None
        
        correlation = merged['indicator'].corr(merged['asset'])
        
        return {
            'indicator': indicator_series_id,
            'asset': asset_symbol,
            'correlation': round(correlation, 3),
            'relationship': 'POSITIVE' if correlation > 0.3 else ('NEGATIVE' if correlation < -0.3 else 'NEUTRAL'),
            'strength': 'STRONG' if abs(correlation) > 0.7 else ('MODERATE' if abs(correlation) > 0.4 else 'WEAK')
        }
    
    def get_economic_snapshot(self):
        interest_rates = self.calculate_interest_rate_differential()
        inflation = self.analyze_inflation_trend()
        employment = self.get_employment_indicators()
        
        economic_health = []
        
        if inflation and inflation['inflation_status'] == 'HIGH':
            economic_health.append('High Inflation')
        if interest_rates and interest_rates['recession_risk'] == 'HIGH':
            economic_health.append('Yield Curve Inverted')
        if employment and employment['employment_trend'] == 'WEAKENING':
            economic_health.append('Weakening Employment')
        
        overall_status = 'WEAK' if len(economic_health) >= 2 else ('MODERATE' if len(economic_health) == 1 else 'STRONG')
        
        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'interest_rates': interest_rates,
            'inflation': inflation,
            'employment': employment,
            'economic_health_indicators': economic_health,
            'overall_economic_status': overall_status
        }


if __name__ == "__main__":
    econ = EconomicIndicatorIntegration()
    
    print("="*80)
    print("ECONOMIC INDICATORS SNAPSHOT")
    print("="*80)
    
    snapshot = econ.get_economic_snapshot()
    
    if snapshot['interest_rates']:
        print("\nInterest Rates:")
        print(f"  Fed Funds: {snapshot['interest_rates']['fed_funds_rate']}%")
        print(f"  10Y Treasury: {snapshot['interest_rates']['treasury_10y']}%")
        print(f"  Yield Curve: {snapshot['interest_rates']['yield_curve_spread']}% ({snapshot['interest_rates']['yield_curve_status']})")
        print(f"  Recession Risk: {snapshot['interest_rates']['recession_risk']}")
    else:
        print("\nInterest Rates: Data unavailable")
    
    if snapshot['inflation']:
        print("\nInflation:")
        print(f"  CPI (YoY): {snapshot['inflation']['cpi_yoy']}%")
        print(f"  Core CPI (YoY): {snapshot['inflation']['core_cpi_yoy']}%")
        print(f"  Trend: {snapshot['inflation']['trend']}")
        print(f"  Status: {snapshot['inflation']['inflation_status']}")
    else:
        print("\nInflation: Data unavailable")
    
    if snapshot['employment']:
        print("\nEmployment:")
        print(f"  Unemployment: {snapshot['employment']['unemployment_rate']}%")
        print(f"  Payroll Change: {snapshot['employment']['payroll_change_monthly']:,}")
        print(f"  Trend: {snapshot['employment']['employment_trend']}")
    else:
        print("\nEmployment: Data unavailable")
    
    print(f"\nOverall Economic Status: {snapshot['overall_economic_status']}")
    
    print("\n" + "="*80)
    print("ASSET CORRELATIONS")
    print("="*80)
    
    correlations = [
        econ.correlate_indicator_with_asset('DGS10', 'GC=F', 365),
        econ.correlate_indicator_with_asset('CPIAUCSL', 'GC=F', 365),
        econ.correlate_indicator_with_asset('UNRATE', '^GSPC', 365)
    ]
    
    for corr in correlations:
        if corr:
            print(f"\n{corr['indicator']} vs {corr['asset']}: {corr['correlation']:.3f} ({corr['relationship']}, {corr['strength']})")
        else:
            print(f"\nCorrelation data unavailable")
