import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from symbol_indicators import SymbolIndicatorCalculator


class NewsImpactAnalyzer:
    def __init__(self):
        self.calc = SymbolIndicatorCalculator()
        
        self.event_impact_map = {
            'Non-Farm Payrolls': {
                'high_impact': ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'DX-Y.NYB', 'GOLD', 'US30', '^GSPC', '^IXIC'],
                'medium_impact': ['XLE', 'XLF', 'TLT', 'GLD', 'SLV'],
                'low_impact': ['BTC-USD', 'ETH-USD']
            },
            'CPI': {
                'high_impact': ['GOLD', 'EURUSD=X', 'GBPUSD=X', 'TLT', 'DX-Y.NYB', 'BTC-USD'],
                'medium_impact': ['US30', '^GSPC', '^IXIC', 'XLF', 'XLE'],
                'low_impact': ['JPM', 'BAC', 'GS']
            },
            'Consumer Price Index': {
                'high_impact': ['GOLD', 'EURUSD=X', 'GBPUSD=X', 'TLT', 'DX-Y.NYB', 'BTC-USD'],
                'medium_impact': ['US30', '^GSPC', '^IXIC', 'XLF', 'XLE'],
                'low_impact': ['JPM', 'BAC', 'GS']
            },
            'FOMC': {
                'high_impact': ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'GOLD', 'US30', '^GSPC', '^IXIC', 'TLT', 'DX-Y.NYB'],
                'medium_impact': ['XLF', 'XLE', 'XLU', 'BTC-USD', 'ETH-USD'],
                'low_impact': ['GLD', 'SLV']
            },
            'Federal Reserve': {
                'high_impact': ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'GOLD', 'US30', '^GSPC', '^IXIC', 'TLT', 'DX-Y.NYB'],
                'medium_impact': ['XLF', 'XLE', 'XLU', 'BTC-USD', 'ETH-USD'],
                'low_impact': ['GLD', 'SLV']
            },
            'PPI': {
                'high_impact': ['GOLD', 'EURUSD=X', 'TLT', 'DX-Y.NYB'],
                'medium_impact': ['US30', '^GSPC', 'XLE', 'XLB'],
                'low_impact': ['BTC-USD', 'GBPUSD=X']
            },
            'Producer Price Index': {
                'high_impact': ['GOLD', 'EURUSD=X', 'TLT', 'DX-Y.NYB'],
                'medium_impact': ['US30', '^GSPC', 'XLE', 'XLB'],
                'low_impact': ['BTC-USD', 'GBPUSD=X']
            },
            'Retail Sales': {
                'high_impact': ['US30', '^GSPC', '^IXIC', 'XLY', 'EURUSD=X'],
                'medium_impact': ['GOLD', 'TLT', 'XRT'],
                'low_impact': ['BTC-USD', 'DX-Y.NYB']
            },
            'ISM Manufacturing': {
                'high_impact': ['US30', '^GSPC', 'XLI', 'EURUSD=X'],
                'medium_impact': ['GOLD', 'XLE', 'XLB'],
                'low_impact': ['TLT', 'BTC-USD']
            },
            'ISM Services': {
                'high_impact': ['US30', '^GSPC', '^IXIC', 'EURUSD=X'],
                'medium_impact': ['XLF', 'XLK', 'GOLD'],
                'low_impact': ['TLT', 'BTC-USD']
            },
            'Jobless Claims': {
                'high_impact': ['EURUSD=X', 'USDJPY=X', 'DX-Y.NYB'],
                'medium_impact': ['GOLD', 'US30', '^GSPC', 'TLT'],
                'low_impact': ['BTC-USD', 'XLF']
            },
            'GDP': {
                'high_impact': ['EURUSD=X', 'GBPUSD=X', 'US30', '^GSPC', '^IXIC'],
                'medium_impact': ['GOLD', 'TLT', 'DX-Y.NYB', 'BTC-USD'],
                'low_impact': ['XLE', 'XLF']
            },
            'Unemployment': {
                'high_impact': ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'DX-Y.NYB', 'GOLD'],
                'medium_impact': ['US30', '^GSPC', 'TLT'],
                'low_impact': ['BTC-USD', 'XLF']
            }
        }
        
        self.keyword_symbols = {
            'inflation': ['GOLD', 'TLT', 'EURUSD=X', 'BTC-USD', 'DX-Y.NYB'],
            'interest rate': ['EURUSD=X', 'GBPUSD=X', 'GOLD', 'TLT', 'US30', '^GSPC'],
            'jobs': ['EURUSD=X', 'DX-Y.NYB', 'GOLD', 'US30'],
            'employment': ['EURUSD=X', 'DX-Y.NYB', 'GOLD', 'US30'],
            'recession': ['GOLD', 'TLT', 'US30', '^GSPC', '^IXIC', 'DX-Y.NYB'],
            'trade': ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'XLI', 'US30'],
            'oil': ['XLE', 'USO', 'CL=F', 'EURUSD=X'],
            'energy': ['XLE', 'USO', 'CL=F'],
            'tech': ['^IXIC', 'XLK', 'AAPL', 'MSFT', 'GOOGL'],
            'bank': ['XLF', 'JPM', 'BAC', 'GS', 'WFC'],
            'crypto': ['BTC-USD', 'ETH-USD'],
            'dollar': ['DX-Y.NYB', 'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'GOLD'],
            'stock market': ['US30', '^GSPC', '^IXIC'],
            'bond': ['TLT', 'IEF', 'SHY'],
            'gold': ['GOLD', 'GLD', 'GC=F'],
            'silver': ['SLV', 'SI=F']
        }
    
    def get_affected_symbols(self, event_name, news_title='', impact_level='all'):
        affected = set()
        
        for event_key, impacts in self.event_impact_map.items():
            if event_key.lower() in event_name.lower():
                if impact_level == 'all':
                    for level in ['high_impact', 'medium_impact', 'low_impact']:
                        affected.update(impacts.get(level, []))
                else:
                    affected.update(impacts.get(f'{impact_level}_impact', []))
                break
        
        news_text = f"{event_name} {news_title}".lower()
        for keyword, symbols in self.keyword_symbols.items():
            if keyword in news_text:
                affected.update(symbols)
        
        return list(affected)
    
    def analyze_event_impact(self, event_date, event_name, news_articles=None, 
                            comparison_days=5, impact_level='all'):
        if isinstance(event_date, str):
            event_date = pd.to_datetime(event_date)
        
        news_titles = []
        if news_articles:
            news_titles = [article.get('title', '') for article in news_articles]
        
        affected_symbols = self.get_affected_symbols(
            event_name, 
            ' '.join(news_titles), 
            impact_level
        )
        
        if not affected_symbols:
            return None
        
        results = {
            'event_date': event_date.strftime('%Y-%m-%d'),
            'event_name': event_name,
            'symbols_analyzed': len(affected_symbols),
            'symbols': {}
        }
        
        before_date = event_date - timedelta(days=comparison_days)
        after_date = event_date + timedelta(days=comparison_days)
        
        for symbol in affected_symbols:
            try:
                before_indicators = self.calc.get_indicators_for_date(symbol, before_date)
                event_indicators = self.calc.get_indicators_for_date(symbol, event_date)
                after_indicators = self.calc.get_indicators_for_date(symbol, after_date)
                
                if all([before_indicators, event_indicators, after_indicators]):
                    price_change_event = ((event_indicators['price'] - before_indicators['price']) 
                                         / before_indicators['price'] * 100)
                    price_change_after = ((after_indicators['price'] - event_indicators['price']) 
                                         / event_indicators['price'] * 100)
                    
                    signal_change = (
                        before_indicators['overall_signal'] != event_indicators['overall_signal'] or
                        event_indicators['overall_signal'] != after_indicators['overall_signal']
                    )
                    
                    volatility = abs(price_change_event) + abs(price_change_after)
                    
                    results['symbols'][symbol] = {
                        'before': {
                            'price': before_indicators['price'],
                            'signal': before_indicators['overall_signal']
                        },
                        'event_day': {
                            'price': event_indicators['price'],
                            'signal': event_indicators['overall_signal']
                        },
                        'after': {
                            'price': after_indicators['price'],
                            'signal': after_indicators['overall_signal']
                        },
                        'price_change_pct': {
                            'before_to_event': round(price_change_event, 2),
                            'event_to_after': round(price_change_after, 2),
                            'total': round(price_change_event + price_change_after, 2)
                        },
                        'signal_changed': signal_change,
                        'volatility_score': round(volatility, 2),
                        'impact_assessment': self._assess_impact(volatility, signal_change)
                    }
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
                continue
        
        results['high_impact_symbols'] = [
            sym for sym, data in results['symbols'].items() 
            if data['impact_assessment'] == 'HIGH'
        ]
        results['medium_impact_symbols'] = [
            sym for sym, data in results['symbols'].items() 
            if data['impact_assessment'] == 'MEDIUM'
        ]
        results['low_impact_symbols'] = [
            sym for sym, data in results['symbols'].items() 
            if data['impact_assessment'] == 'LOW'
        ]
        
        return results
    
    def _assess_impact(self, volatility, signal_changed):
        if volatility > 5 or (volatility > 2 and signal_changed):
            return 'HIGH'
        elif volatility > 2 or signal_changed:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def batch_analyze_events(self, events, comparison_days=5, impact_level='all'):
        results = []
        
        for i, event in enumerate(events):
            print(f"Analyzing event {i+1}/{len(events)}: {event['date']} - {event['event']}")
            
            analysis = self.analyze_event_impact(
                event['date'],
                event['event'],
                event.get('news', []),
                comparison_days,
                impact_level
            )
            
            if analysis:
                results.append(analysis)
        
        return results


if __name__ == "__main__":
    import json
    
    analyzer = NewsImpactAnalyzer()
    
    print("="*80)
    print("NEWS IMPACT ANALYZER - Testing")
    print("="*80)
    
    test_events = [
        {
            'date': '2024-11-01',
            'event': 'Non-Farm Payrolls',
            'news': [
                {'title': 'US Jobs Report Shows Strong Growth'},
                {'title': 'Employment Numbers Beat Expectations'}
            ]
        },
        {
            'date': '2024-10-10',
            'event': 'Consumer Price Index (CPI)',
            'news': [
                {'title': 'Inflation Rises More Than Expected'},
                {'title': 'CPI Data Impacts Fed Rate Decision'}
            ]
        }
    ]
    
    for event in test_events:
        print(f"\n{'='*80}")
        print(f"Event: {event['event']} on {event['date']}")
        print('='*80)
        
        affected = analyzer.get_affected_symbols(event['event'])
        print(f"\nAffected symbols ({len(affected)}): {', '.join(affected[:10])}")
        
        analysis = analyzer.analyze_event_impact(
            event['date'],
            event['event'],
            event.get('news'),
            comparison_days=3
        )
        
        if analysis and analysis['symbols']:
            print(f"\nAnalyzed {len(analysis['symbols'])} symbols:")
            print(f"  High impact: {len(analysis['high_impact_symbols'])}")
            print(f"  Medium impact: {len(analysis['medium_impact_symbols'])}")
            print(f"  Low impact: {len(analysis['low_impact_symbols'])}")
            
            print("\nTop 5 Most Impacted:")
            sorted_symbols = sorted(
                analysis['symbols'].items(),
                key=lambda x: x[1]['volatility_score'],
                reverse=True
            )[:5]
            
            for symbol, data in sorted_symbols:
                print(f"\n  {symbol}:")
                print(f"    Price change: {data['price_change_pct']['total']}%")
                print(f"    Signal changed: {data['signal_changed']}")
                print(f"    Impact: {data['impact_assessment']}")
    
    print(f"\n{'='*80}")
    print("Test complete")
    print('='*80)
