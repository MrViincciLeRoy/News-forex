from cot_data_fetcher import COTDataFetcher
from news_impact_analyzer import NewsImpactAnalyzer
from symbol_indicators import SymbolIndicatorCalculator
import json
from datetime import datetime, timedelta


class COTNewsIntegration:
    def __init__(self):
        self.cot = COTDataFetcher()
        self.news_analyzer = NewsImpactAnalyzer()
        self.indicators = SymbolIndicatorCalculator()
        
        self.symbol_map = {
            'EURUSD=X': 'EUR',
            'GBPUSD=X': 'GBP',
            'USDJPY=X': 'JPY',
            'AUDUSD=X': 'AUD',
            'NZDUSD=X': 'NZD',
            'USDCAD=X': 'CAD',
            'USDCHF=X': 'CHF',
            'GOLD': 'GOLD',
            'GC=F': 'GOLD',
            'XAUUSD': 'GOLD',
            'SI=F': 'SILVER',
            'XAGUSD': 'SILVER',
            'CL=F': 'CRUDE_OIL',
            'NG=F': 'NATURAL_GAS',
            '^GSPC': 'SP500',
            '^IXIC': 'NASDAQ',
            'US30': 'DOW',
            '^DJI': 'DOW'
        }
    
    def get_cot_symbol(self, market_symbol):
        return self.symbol_map.get(market_symbol, market_symbol)
    
    def analyze_event_with_cot(self, event_date, event_name, news_articles=None):
        if isinstance(event_date, str):
            event_date = datetime.strptime(event_date, '%Y-%m-%d')
        
        affected_symbols = self.news_analyzer.get_affected_symbols(
            event_name,
            ' '.join([n.get('title', '') for n in (news_articles or [])])
        )
        
        report_tuesday, release_friday = self.cot.get_report_date_for_date(event_date)
        
        pre_event_date = event_date - timedelta(days=14)
        pre_report_tuesday, _ = self.cot.get_report_date_for_date(pre_event_date)
        
        analysis = {
            'event_date': event_date.strftime('%Y-%m-%d'),
            'event_name': event_name,
            'cot_report_date': report_tuesday.strftime('%Y-%m-%d'),
            'cot_release_date': release_friday.strftime('%Y-%m-%d'),
            'symbols_analyzed': [],
            'high_risk_setups': [],
            'confluence_trades': []
        }
        
        for market_symbol in affected_symbols[:15]:
            cot_symbol = self.get_cot_symbol(market_symbol)
            
            if cot_symbol not in self.cot.cftc_codes:
                continue
            
            try:
                current_cot = self.cot.get_positioning_for_date(cot_symbol, event_date)
                previous_cot = self.cot.get_positioning_for_date(cot_symbol, pre_event_date)
                
                if not current_cot:
                    continue
                
                indicators = self.indicators.get_indicators_for_date(market_symbol, event_date)
                
                if not indicators:
                    continue
                
                symbol_analysis = {
                    'market_symbol': market_symbol,
                    'cot_symbol': cot_symbol,
                    'price': indicators['price'],
                    'technical_signal': indicators['overall_signal'],
                    'cot_sentiment': current_cot['sentiment'],
                    'cot_positioning': {
                        'smart_money_net': current_cot['dealer']['net'] + current_cot['asset_manager']['net'],
                        'hedge_funds_net': current_cot['leveraged']['net'],
                        'open_interest': current_cot['open_interest']
                    }
                }
                
                if previous_cot:
                    smart_money_change = (
                        (current_cot['dealer']['net'] + current_cot['asset_manager']['net']) -
                        (previous_cot['dealer']['net'] + previous_cot['asset_manager']['net'])
                    )
                    
                    symbol_analysis['positioning_change_2w'] = {
                        'smart_money': smart_money_change,
                        'hedge_funds': current_cot['leveraged']['net'] - previous_cot['leveraged']['net']
                    }
                
                risk_level = self._assess_risk(current_cot, indicators)
                symbol_analysis['risk_level'] = risk_level
                
                if self._is_confluence_trade(current_cot, indicators):
                    symbol_analysis['confluence'] = True
                    analysis['confluence_trades'].append({
                        'symbol': market_symbol,
                        'setup': self._describe_setup(current_cot, indicators)
                    })
                
                if risk_level == 'HIGH':
                    analysis['high_risk_setups'].append({
                        'symbol': market_symbol,
                        'reason': self._describe_risk(current_cot, indicators)
                    })
                
                analysis['symbols_analyzed'].append(symbol_analysis)
                
            except Exception as e:
                print(f"Error analyzing {market_symbol}: {e}")
                continue
        
        return analysis
    
    def _assess_risk(self, cot_data, indicators):
        if 'EXTREME' in cot_data['sentiment']:
            return 'HIGH'
        
        smart_money_net = cot_data['dealer']['net'] + cot_data['asset_manager']['net']
        hedge_net = cot_data['leveraged']['net']
        
        if smart_money_net * hedge_net < 0 and abs(hedge_net) > abs(smart_money_net):
            return 'HIGH'
        
        cot_bullish = smart_money_net > 0
        tech_bullish = indicators['overall_signal'] == 'BUY'
        
        if cot_bullish != tech_bullish:
            return 'MEDIUM'
        
        return 'LOW'
    
    def _is_confluence_trade(self, cot_data, indicators):
        smart_money_net = cot_data['dealer']['net'] + cot_data['asset_manager']['net']
        
        cot_bullish = smart_money_net > 0
        tech_bullish = indicators['overall_signal'] == 'BUY'
        
        return cot_bullish == tech_bullish and 'EXTREME' not in cot_data['sentiment']
    
    def _describe_setup(self, cot_data, indicators):
        smart_money_net = cot_data['dealer']['net'] + cot_data['asset_manager']['net']
        direction = "BULLISH" if smart_money_net > 0 else "BEARISH"
        
        return f"{direction} - Smart money and technicals aligned"
    
    def _describe_risk(self, cot_data, indicators):
        if 'EXTREME' in cot_data['sentiment']:
            return f"Extreme positioning detected - {cot_data['sentiment']}"
        
        smart_money_net = cot_data['dealer']['net'] + cot_data['asset_manager']['net']
        hedge_net = cot_data['leveraged']['net']
        
        if smart_money_net * hedge_net < 0:
            return "Smart money vs Hedge funds - Potential reversal"
        
        cot_bullish = smart_money_net > 0
        tech_bullish = indicators['overall_signal'] == 'BUY'
        
        if cot_bullish != tech_bullish:
            return "COT sentiment conflicts with technical analysis"
        
        return "Elevated risk"


if __name__ == "__main__":
    integration = COTNewsIntegration()
    
    print("="*80)
    print("COT + NEWS + TECHNICAL ANALYSIS INTEGRATION")
    print("="*80)
    
    test_events = [
        {
            'date': '2024-11-01',
            'event': 'Non-Farm Payrolls',
            'news': [
                {'title': 'US Jobs Report Beats Expectations'},
                {'title': 'Strong Employment Growth Continues'}
            ]
        },
        {
            'date': '2024-10-10',
            'event': 'Consumer Price Index',
            'news': [
                {'title': 'Inflation Rises Above Forecast'},
                {'title': 'CPI Data Surprises Markets'}
            ]
        }
    ]
    
    results = []
    
    for event in test_events:
        print(f"\n{'='*80}")
        print(f"Analyzing: {event['event']} on {event['date']}")
        print('='*80)
        
        analysis = integration.analyze_event_with_cot(
            event['date'],
            event['event'],
            event.get('news')
        )
        
        print(f"\nCOT Report Date: {analysis['cot_report_date']}")
        print(f"Symbols Analyzed: {len(analysis['symbols_analyzed'])}")
        print(f"High Risk Setups: {len(analysis['high_risk_setups'])}")
        print(f"Confluence Trades: {len(analysis['confluence_trades'])}")
        
        if analysis['confluence_trades']:
            print("\n✓ CONFLUENCE TRADES (COT + Technicals Aligned):")
            for trade in analysis['confluence_trades'][:5]:
                print(f"  {trade['symbol']}: {trade['setup']}")
        
        if analysis['high_risk_setups']:
            print("\n⚠ HIGH RISK SETUPS (Extreme Positioning):")
            for setup in analysis['high_risk_setups'][:5]:
                print(f"  {setup['symbol']}: {setup['reason']}")
        
        results.append(analysis)
    
    output_file = 'cot_news_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✓ Full analysis saved to {output_file}")
    print('='*80)
