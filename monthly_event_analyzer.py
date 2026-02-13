"""
Monthly Event Analyzer - Pre & Post Event Reports
Generates comprehensive reports for all major events in a month
Tracks institutional flows, major currency pairs, and crypto impacts
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


class MonthlyEventAnalyzer:
    """
    Analyzes all major economic events in a month
    Generates both PRE-event and POST-event reports
    """
    
    def __init__(self, output_dir: str = 'monthly_reports'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Major currency pairs (from your requirements)
        self.major_pairs = {
            'EURUSD=X': {'share': 21.2, 'name': 'EUR/USD'},
            'USDJPY=X': {'share': 14.3, 'name': 'USD/JPY'},
            'CNY=X': {'share': 8.1, 'name': 'USD/CNY'},
            'GBPUSD=X': {'share': 10.2, 'name': 'GBP/USD'}
        }
        
        # Top 10 institutional players
        self.major_institutions = [
            {'name': 'JPMorgan Chase', 'share': 10.78, 'country': 'US'},
            {'name': 'UBS', 'share': 8.13, 'country': 'Switzerland'},
            {'name': 'XTX Markets', 'share': 7.58, 'country': 'UK'},
            {'name': 'Deutsche Bank', 'share': 7.38, 'country': 'Germany'},
            {'name': 'Citi', 'share': 5.50, 'country': 'US'},
            {'name': 'HSBC', 'share': 5.33, 'country': 'UK'},
            {'name': 'Jump Trading', 'share': 5.23, 'country': 'US'},
            {'name': 'Goldman Sachs', 'share': 4.62, 'country': 'US'},
            {'name': 'State Street', 'share': 4.61, 'country': 'US'},
            {'name': 'Bank of America', 'share': 4.50, 'country': 'US'}
        ]
        
        # Crypto symbols to track
        self.crypto_symbols = [
            'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD',
            'SOL-USD', 'DOGE-USD', 'MATIC-USD', 'DOT-USD', 'AVAX-USD'
        ]
        
        # Major event calendar
        self.event_calendar = self._load_event_calendar()
    
    def _load_event_calendar(self) -> Dict:
        """Load economic event calendar"""
        return {
            'NFP': {
                'name': 'Non-Farm Payrolls',
                'frequency': 'monthly',
                'day': 'first_friday',
                'time': '08:30 ET',
                'impact': 'HIGH',
                'affected_markets': ['USD', 'GOLD', 'EQUITIES', 'CRYPTO']
            },
            'CPI': {
                'name': 'Consumer Price Index',
                'frequency': 'monthly',
                'day': 'mid_month',
                'time': '08:30 ET',
                'impact': 'HIGH',
                'affected_markets': ['USD', 'GOLD', 'BONDS', 'CRYPTO']
            },
            'FOMC': {
                'name': 'Federal Reserve Meeting',
                'frequency': 'every_6_weeks',
                'day': 'wednesday',
                'time': '14:00 ET',
                'impact': 'EXTREME',
                'affected_markets': ['ALL']
            },
            'PPI': {
                'name': 'Producer Price Index',
                'frequency': 'monthly',
                'day': 'mid_month',
                'time': '08:30 ET',
                'impact': 'MEDIUM',
                'affected_markets': ['USD', 'COMMODITIES']
            },
            'RETAIL_SALES': {
                'name': 'Retail Sales',
                'frequency': 'monthly',
                'day': 'mid_month',
                'time': '08:30 ET',
                'impact': 'MEDIUM',
                'affected_markets': ['USD', 'EQUITIES']
            },
            'GDP': {
                'name': 'GDP Report',
                'frequency': 'quarterly',
                'day': 'end_month',
                'time': '08:30 ET',
                'impact': 'HIGH',
                'affected_markets': ['USD', 'EQUITIES', 'BONDS']
            },
            'JOBLESS_CLAIMS': {
                'name': 'Initial Jobless Claims',
                'frequency': 'weekly',
                'day': 'thursday',
                'time': '08:30 ET',
                'impact': 'LOW',
                'affected_markets': ['USD']
            },
            'ISM_MANUFACTURING': {
                'name': 'ISM Manufacturing PMI',
                'frequency': 'monthly',
                'day': 'first_business_day',
                'time': '10:00 ET',
                'impact': 'MEDIUM',
                'affected_markets': ['USD', 'EQUITIES']
            },
            'ISM_SERVICES': {
                'name': 'ISM Services PMI',
                'frequency': 'monthly',
                'day': 'third_business_day',
                'time': '10:00 ET',
                'impact': 'MEDIUM',
                'affected_markets': ['USD', 'EQUITIES']
            }
        }
    
    def get_events_for_month(self, year: int, month: int) -> List[Dict]:
        """Get all major events for a specific month"""
        events = []
        
        # Example: February 2026 events
        if year == 2026 and month == 2:
            events = [
                {'date': '2026-02-03', 'event': 'ISM_MANUFACTURING'},
                {'date': '2026-02-05', 'event': 'ISM_SERVICES'},
                {'date': '2026-02-06', 'event': 'NFP'},
                {'date': '2026-02-12', 'event': 'CPI'},
                {'date': '2026-02-13', 'event': 'PPI'},
                {'date': '2026-02-14', 'event': 'RETAIL_SALES'},
                {'date': '2026-02-26', 'event': 'GDP'}
            ]
        
        # Expand with event details
        expanded_events = []
        for event in events:
            event_key = event['event']
            if event_key in self.event_calendar:
                expanded_events.append({
                    **event,
                    **self.event_calendar[event_key],
                    'event_key': event_key
                })
        
        return expanded_events
    
    def analyze_month(self, year: int, month: int) -> Dict:
        """
        Analyze all events in a month
        Generate pre and post reports for each
        """
        print(f"\n{'='*80}")
        print(f"MONTHLY EVENT ANALYSIS: {year}-{month:02d}")
        print(f"{'='*80}\n")
        
        events = self.get_events_for_month(year, month)
        
        results = {
            'year': year,
            'month': month,
            'total_events': len(events),
            'events': [],
            'monthly_summary': {}
        }
        
        for event in events:
            print(f"\n📅 {event['name']} - {event['date']}")
            print(f"   Impact: {event['impact']}")
            print(f"   Markets: {', '.join(event['affected_markets'])}")
            
            # Generate PRE-event report
            pre_report = self._generate_pre_event_report(event)
            
            # Generate POST-event report
            post_report = self._generate_post_event_report(event)
            
            event_result = {
                'event_key': event['event_key'],
                'event_name': event['name'],
                'date': event['date'],
                'impact_level': event['impact'],
                'pre_event_report': pre_report,
                'post_event_report': post_report
            }
            
            results['events'].append(event_result)
            
            # Save individual reports
            self._save_event_reports(event, pre_report, post_report)
        
        # Generate monthly summary
        results['monthly_summary'] = self._generate_monthly_summary(results['events'])
        
        # Save consolidated monthly report
        self._save_monthly_report(year, month, results)
        
        return results
    
    def _generate_pre_event_report(self, event: Dict) -> Dict:
        """
        Generate PRE-event report
        Focus: Positioning, expectations, setup
        """
        from comprehensive_pipeline import ComprehensiveAnalysisPipeline
        
        event_date = event['date']
        lookback_date = (datetime.strptime(event_date, '%Y-%m-%d') - timedelta(days=7)).strftime('%Y-%m-%d')
        
        print(f"   🔍 PRE-event analysis (lookback to {lookback_date})")
        
        # All symbols to analyze
        symbols = list(self.major_pairs.keys()) + ['DX-Y.NYB', 'GC=F'] + self.crypto_symbols
        
        pipeline = ComprehensiveAnalysisPipeline(
            output_dir=str(self.output_dir / f"{event['event_key']}_{event_date}_PRE"),
            enable_viz=True,
            enable_hf=True,
            max_articles=30
        )
        
        # Run analysis for PRE period
        pre_results = pipeline.analyze(
            date=lookback_date,
            event_name=f"PRE-{event['name']}",
            symbols=symbols
        )
        
        # Enhanced institutional analysis
        institutional_analysis = self._analyze_institutional_positioning(
            event, lookback_date, is_pre=True
        )
        
        # Currency flow analysis
        currency_flows = self._analyze_currency_flows(symbols[:4], lookback_date)
        
        # Crypto correlation analysis
        crypto_analysis = self._analyze_crypto_correlations(
            self.crypto_symbols, lookback_date
        )
        
        return {
            'analysis_date': lookback_date,
            'event_date': event_date,
            'type': 'PRE-EVENT',
            'pipeline_results': pre_results,
            'institutional_positioning': institutional_analysis,
            'currency_flows': currency_flows,
            'crypto_analysis': crypto_analysis,
            'market_expectations': self._get_market_expectations(event)
        }
    
    def _generate_post_event_report(self, event: Dict) -> Dict:
        """
        Generate POST-event report
        Focus: Actual impact, moves, winners/losers
        """
        from comprehensive_pipeline import ComprehensiveAnalysisPipeline
        
        event_date = event['date']
        followup_date = (datetime.strptime(event_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
        
        print(f"   📊 POST-event analysis (impact through {followup_date})")
        
        symbols = list(self.major_pairs.keys()) + ['DX-Y.NYB', 'GC=F'] + self.crypto_symbols
        
        pipeline = ComprehensiveAnalysisPipeline(
            output_dir=str(self.output_dir / f"{event['event_key']}_{event_date}_POST"),
            enable_viz=True,
            enable_hf=True,
            max_articles=40
        )
        
        post_results = pipeline.analyze(
            date=event_date,
            event_name=f"POST-{event['name']}",
            symbols=symbols
        )
        
        # Impact analysis
        impact_analysis = self._analyze_event_impact(event, event_date, symbols)
        
        # Institutional moves
        institutional_moves = self._analyze_institutional_positioning(
            event, event_date, is_pre=False
        )
        
        # Currency reactions
        currency_reactions = self._analyze_currency_reactions(
            symbols[:4], event_date
        )
        
        # Crypto impact
        crypto_impact = self._analyze_crypto_impact(
            self.crypto_symbols, event_date
        )
        
        return {
            'analysis_date': event_date,
            'type': 'POST-EVENT',
            'pipeline_results': post_results,
            'event_impact': impact_analysis,
            'institutional_moves': institutional_moves,
            'currency_reactions': currency_reactions,
            'crypto_impact': crypto_impact,
            'winners_losers': self._identify_winners_losers(impact_analysis)
        }
    
    def _analyze_institutional_positioning(self, event: Dict, date: str, is_pre: bool) -> Dict:
        """Analyze major institutional player positioning"""
        from cot_data_fetcher import COTDataFetcher
        
        cot = COTDataFetcher()
        
        # Analyze major currencies
        currencies = ['EUR', 'JPY', 'GBP', 'GOLD']
        
        positioning = {}
        
        for currency in currencies:
            try:
                pos = cot.get_positioning_for_date(currency, date)
                if pos:
                    positioning[currency] = {
                        'net_positioning': pos['sentiment'],
                        'dealer_net': pos['dealer']['net'],
                        'asset_mgr_net': pos['asset_manager']['net'],
                        'leveraged_net': pos['leveraged']['net'],
                        'smart_money_net': pos['dealer']['net'] + pos['asset_manager']['net'],
                        'report_date': pos['report_date']
                    }
            except:
                pass
        
        # Add institutional share context
        positioning['institutional_context'] = {
            'major_players': self.major_institutions,
            'total_top10_share': sum(inst['share'] for inst in self.major_institutions),
            'analysis_type': 'PRE-EVENT' if is_pre else 'POST-EVENT'
        }
        
        return positioning
    
    def _analyze_currency_flows(self, pairs: List[str], date: str) -> Dict:
        """Analyze major currency pair flows"""
        from symbol_indicators import SymbolIndicatorCalculator
        
        calc = SymbolIndicatorCalculator()
        
        flows = {}
        
        for pair in pairs:
            indicators = calc.get_indicators_for_date(pair, date)
            if indicators:
                pair_name = self.major_pairs.get(pair, {}).get('name', pair)
                market_share = self.major_pairs.get(pair, {}).get('share', 0)
                
                flows[pair] = {
                    'pair_name': pair_name,
                    'market_share': market_share,
                    'price': indicators['price'],
                    'signal': indicators['overall_signal'],
                    'buy_signals': indicators['buy_count'],
                    'sell_signals': indicators['sell_count'],
                    'key_indicators': {
                        name: data for name, data in indicators['indicators'].items()
                        if name in ['RSI', 'MACD', 'MA_Cross']
                    }
                }
        
        return flows
    
    def _analyze_crypto_correlations(self, crypto_symbols: List[str], date: str) -> Dict:
        """Analyze crypto market correlations with traditional markets"""
        from symbol_indicators import SymbolIndicatorCalculator
        
        calc = SymbolIndicatorCalculator()
        
        crypto_data = {}
        
        for symbol in crypto_symbols[:5]:  # Top 5 cryptos
            indicators = calc.get_indicators_for_date(symbol, date)
            if indicators:
                crypto_data[symbol] = {
                    'price': indicators['price'],
                    'signal': indicators['overall_signal'],
                    'trend_strength': indicators.get('indicators', {}).get('ADX', {}).get('value', 0)
                }
        
        return {
            'crypto_positions': crypto_data,
            'btc_dominance': self._calculate_btc_dominance(crypto_data),
            'risk_sentiment': self._assess_crypto_risk_sentiment(crypto_data)
        }
    
    def _analyze_currency_reactions(self, pairs: List[str], event_date: str) -> Dict:
        """Analyze how currencies reacted to event"""
        from symbol_indicators import SymbolIndicatorCalculator
        
        calc = SymbolIndicatorCalculator()
        
        reactions = {}
        
        # Compare day before vs day after
        day_before = (datetime.strptime(event_date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
        day_after = (datetime.strptime(event_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
        
        for pair in pairs:
            before = calc.get_indicators_for_date(pair, day_before)
            after = calc.get_indicators_for_date(pair, day_after)
            
            if before and after:
                change_pct = ((after['price'] - before['price']) / before['price']) * 100
                
                reactions[pair] = {
                    'price_before': before['price'],
                    'price_after': after['price'],
                    'change_pct': round(change_pct, 2),
                    'signal_before': before['overall_signal'],
                    'signal_after': after['overall_signal'],
                    'signal_flip': before['overall_signal'] != after['overall_signal']
                }
        
        return reactions
    
    def _analyze_crypto_impact(self, crypto_symbols: List[str], event_date: str) -> Dict:
        """Analyze crypto market impact from event"""
        from symbol_indicators import SymbolIndicatorCalculator
        
        calc = SymbolIndicatorCalculator()
        
        day_before = (datetime.strptime(event_date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
        day_after = (datetime.strptime(event_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
        
        impacts = {}
        
        for symbol in crypto_symbols[:5]:
            before = calc.get_indicators_for_date(symbol, day_before)
            after = calc.get_indicators_for_date(symbol, day_after)
            
            if before and after:
                change_pct = ((after['price'] - before['price']) / before['price']) * 100
                
                impacts[symbol] = {
                    'change_pct': round(change_pct, 2),
                    'direction': 'UP' if change_pct > 0 else 'DOWN',
                    'magnitude': 'HIGH' if abs(change_pct) > 5 else ('MEDIUM' if abs(change_pct) > 2 else 'LOW')
                }
        
        return {
            'crypto_reactions': impacts,
            'overall_crypto_sentiment': self._assess_overall_crypto_sentiment(impacts),
            'correlation_with_tradfi': self._assess_crypto_tradfi_correlation(impacts)
        }
    
    def _analyze_event_impact(self, event: Dict, date: str, symbols: List[str]) -> Dict:
        """Analyze overall event impact"""
        return {
            'event': event['name'],
            'date': date,
            'impact_level': event['impact'],
            'affected_markets': event['affected_markets'],
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _get_market_expectations(self, event: Dict) -> Dict:
        """Get market expectations for event"""
        return {
            'consensus': 'MODERATE',
            'volatility_expected': event['impact'],
            'key_levels_to_watch': []
        }
    
    def _identify_winners_losers(self, impact: Dict) -> Dict:
        """Identify biggest winners and losers"""
        return {
            'winners': [],
            'losers': [],
            'neutral': []
        }
    
    def _calculate_btc_dominance(self, crypto_data: Dict) -> float:
        """Calculate Bitcoin dominance"""
        return 50.0  # Placeholder
    
    def _assess_crypto_risk_sentiment(self, crypto_data: Dict) -> str:
        """Assess overall crypto risk sentiment"""
        return "NEUTRAL"
    
    def _assess_overall_crypto_sentiment(self, impacts: Dict) -> str:
        """Assess overall crypto market sentiment"""
        positive = sum(1 for v in impacts.values() if v['direction'] == 'UP')
        total = len(impacts)
        return 'BULLISH' if positive > total/2 else 'BEARISH'
    
    def _assess_crypto_tradfi_correlation(self, impacts: Dict) -> str:
        """Assess crypto correlation with traditional finance"""
        return "MODERATE"
    
    def _generate_monthly_summary(self, events: List[Dict]) -> Dict:
        """Generate summary for entire month"""
        return {
            'total_events': len(events),
            'high_impact_events': sum(1 for e in events if e['impact_level'] == 'HIGH'),
            'summary_generated': datetime.now().isoformat()
        }
    
    def _save_event_reports(self, event: Dict, pre_report: Dict, post_report: Dict):
        """Save individual event reports"""
        event_dir = self.output_dir / f"{event['event_key']}_{event['date']}"
        event_dir.mkdir(exist_ok=True)
        
        # Save PRE report
        with open(event_dir / 'pre_event_report.json', 'w') as f:
            json.dump(pre_report, f, indent=2, default=str)
        
        # Save POST report
        with open(event_dir / 'post_event_report.json', 'w') as f:
            json.dump(post_report, f, indent=2, default=str)
        
        print(f"   ✓ Saved to {event_dir}")
    
    def _save_monthly_report(self, year: int, month: int, results: Dict):
        """Save consolidated monthly report"""
        filename = self.output_dir / f"monthly_report_{year}_{month:02d}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✓ Monthly report saved: {filename}")


if __name__ == "__main__":
    analyzer = MonthlyEventAnalyzer()
    
    # Analyze February 2026
    results = analyzer.analyze_month(2026, 2)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Total events analyzed: {results['total_events']}")
    print(f"Reports generated: {results['total_events'] * 2} (PRE + POST)")
