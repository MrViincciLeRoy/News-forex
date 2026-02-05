"""
Enhanced Comprehensive Analysis Pipeline
Generates detailed section JSON files for Local LLM Report Generator
Produces rich, structured data for each report section
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import os
import sys
import psutil
import gc
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import all the analysis modules (from your existing files)
try:
    from news_fetcher import NewsFetcher
    from analysis_synthesizer import AnalysisSynthesizer
    from pipeline_visualizer import PipelineVisualizer
    from news_impact_analyzer import NewsImpactAnalyzer
    from cot_data_fetcher import COTDataFetcher
    from symbol_indicators import SymbolIndicatorCalculator
    from correlation_analyzer import CorrelationAnalyzer
    from economic_indicators import EconomicIndicatorIntegration
    from seasonality_analyzer import SeasonalityAnalyzer
    from market_structure_analyzer import MarketStructureAnalyzer
    from volume_analyzer import VolumeAnalyzer
except ImportError as e:
    print(f"Warning: Some modules not available: {e}")


class EnhancedComprehensivePipeline:
    """
    Enhanced pipeline that generates detailed section JSON files
    for comprehensive LLM-powered HTML reports
    """
    
    def __init__(
        self,
        output_dir='pipeline_output',
        enable_hf=True,
        enable_viz=True,
        max_articles=20
    ):
        self.output_dir = output_dir
        self.viz_dir = os.path.join(output_dir, 'visualizations')
        self.sections_dir = os.path.join(output_dir, 'sections')
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)
        os.makedirs(self.sections_dir, exist_ok=True)
        
        self.max_articles = max_articles
        
        print("="*80)
        print("ENHANCED COMPREHENSIVE ANALYSIS PIPELINE")
        print("="*80)
        print(f"Output Directory: {output_dir}")
        print(f"Sections Directory: {self.sections_dir}")
        print(f"Visualizations: {self.viz_dir}")
        print(f"Max Articles: {max_articles}")
        print("="*80)
        
        self._validate_api_keys()
        self._initialize_modules()
    
    def _validate_api_keys(self):
        """Validate API keys"""
        print("\nAPI Keys:")
        print("-"*80)
        
        fred_key = os.environ.get('FRED_API_KEY', '') or os.environ.get('FRED_API_KEY_1', '')
        serp_keys = [os.environ.get(f'SERP_API_KEY_{i}', '') for i in range(1, 4)]
        serp_keys = [k for k in serp_keys if k]
        
        if fred_key:
            print(f"  ✓ FRED_API_KEY")
        else:
            print(f"  ⊘ FRED_API_KEY: Not set")
        
        if serp_keys:
            print(f"  ✓ SERP_API_KEY: {len(serp_keys)} key(s)")
        else:
            print(f"  ⊘ SERP_API_KEY: Not set")
        
        print("-"*80)
    
    def _initialize_modules(self):
        """Initialize all analysis modules"""
        print("\nInitializing modules...")
        print("-"*80)
        
        self.news_fetcher = NewsFetcher(prefer_serp=True) if 'NewsFetcher' in globals() else None
        self.news_analyzer = NewsImpactAnalyzer() if 'NewsImpactAnalyzer' in globals() else None
        self.cot_fetcher = COTDataFetcher() if 'COTDataFetcher' in globals() else None
        self.indicator_calc = SymbolIndicatorCalculator() if 'SymbolIndicatorCalculator' in globals() else None
        self.corr_analyzer = CorrelationAnalyzer() if 'CorrelationAnalyzer' in globals() else None
        self.econ_indicators = EconomicIndicatorIntegration() if 'EconomicIndicatorIntegration' in globals() else None
        self.seasonality = SeasonalityAnalyzer() if 'SeasonalityAnalyzer' in globals() else None
        self.market_structure = MarketStructureAnalyzer() if 'MarketStructureAnalyzer' in globals() else None
        self.volume_analyzer = VolumeAnalyzer() if 'VolumeAnalyzer' in globals() else None
        self.synthesizer = AnalysisSynthesizer() if 'AnalysisSynthesizer' in globals() else None
        self.visualizer = PipelineVisualizer(self.viz_dir) if 'PipelineVisualizer' in globals() else None
        
        print("✓ Modules initialized")
        print("="*80 + "\n")
    
    def analyze(self, date: str, event_name: Optional[str] = None, symbols: Optional[List[str]] = None) -> Dict:
        """
        Run comprehensive analysis with enhanced JSON output
        """
        print("="*80)
        print(f"ANALYSIS: {event_name or 'Market Analysis'}")
        print(f"Date: {date}")
        print("="*80)
        
        results = {
            'metadata': {
                'date': date,
                'event_name': event_name,
                'timestamp': datetime.now().isoformat(),
                'max_articles': self.max_articles,
                'pipeline_version': 'enhanced_1.0'
            },
            'sections': {}
        }
        
        # 1. NEWS ANALYSIS
        print("\n📰 SECTION 1: NEWS")
        print("-"*80)
        news_results = self._fetch_and_analyze_news(date, event_name)
        results['sections']['news'] = news_results
        self._save_section_json('news_analysis', news_results, date, event_name)
        
        # Extract symbols
        if symbols is None:
            symbols = self._extract_symbols(news_results, event_name)
        
        results['metadata']['symbols_analyzed'] = symbols
        results['metadata']['symbols_count'] = len(symbols)
        
        # 2. COT POSITIONING
        print("\n📊 SECTION 2: COT")
        print("-"*80)
        cot_results = self._analyze_cot(date, symbols)
        results['sections']['cot'] = cot_results
        self._save_section_json('cot_positioning', cot_results, date, event_name)
        
        # 3. TECHNICAL INDICATORS
        print("\n📈 SECTION 3: TECHNICAL")
        print("-"*80)
        indicators_results = self._analyze_indicators(date, symbols)
        results['sections']['indicators'] = indicators_results
        self._save_section_json('technical_indicators', indicators_results, date, event_name)
        
        # 4. CORRELATIONS
        print("\n🔗 SECTION 4: CORRELATIONS")
        print("-"*80)
        corr_results = self._analyze_correlations(date, symbols[:5])
        results['sections']['correlations'] = corr_results
        self._save_section_json('correlations', corr_results, date, event_name)
        
        # 5. ECONOMIC
        print("\n💹 SECTION 5: ECONOMIC")
        print("-"*80)
        econ_results = self._analyze_economic(date)
        results['sections']['economic'] = econ_results
        self._save_section_json('economic_indicators', econ_results, date, event_name)
        
        # 6. STRUCTURE
        print("\n🏗️  SECTION 6: STRUCTURE")
        print("-"*80)
        structure_results = self._analyze_structure(date, symbols[:5])
        results['sections']['structure'] = structure_results
        self._save_section_json('market_structure', structure_results, date, event_name)
        
        # 7. SEASONALITY
        print("\n📅 SECTION 7: SEASONALITY")
        print("-"*80)
        seasonality_results = self._analyze_seasonality(symbols[:3])
        results['sections']['seasonality'] = seasonality_results
        self._save_section_json('seasonality', seasonality_results, date, event_name)
        
        # 8. VOLUME
        print("\n📊 SECTION 8: VOLUME")
        print("-"*80)
        volume_results = self._analyze_volume(date, symbols[:5])
        results['sections']['volume'] = volume_results
        self._save_section_json('volume_analysis', volume_results, date, event_name)
        
        # 9. HF AI METHODS (if available)
        try:
            from enhanced_hf_analyzer import EnhancedHFAnalyzer
            print("\n🤖 SECTION 9: HF AI")
            print("-"*80)
            hf_analyzer = EnhancedHFAnalyzer()
            if hf_analyzer.load_models():
                hf_results = hf_analyzer.analyze_comprehensive(
                    articles=news_results.get('articles', []),
                    symbols=symbols,
                    date=date,
                    event_name=event_name
                )
                results['sections']['hf_methods'] = hf_results
                self._save_section_json('ai_analysis', hf_results, date, event_name)
        except Exception as e:
            print(f"  ⊘ HF AI not available: {str(e)[:60]}")
        
        # 10. SYNTHESIS
        print("\n💡 SECTION 10: SYNTHESIS")
        print("-"*80)
        if self.synthesizer:
            insights = self.synthesizer.synthesize(results)
            results['sections']['synthesis'] = insights
            self._save_section_json('synthesis', insights, date, event_name)
            print(f"  ✓ Generated {len(insights.get('key_findings', []))} findings")
        
        # 11. VISUALIZATIONS
        if self.visualizer:
            print("\n📊 SECTION 11: VISUALIZATIONS")
            print("-"*80)
            viz_results = self.visualizer.generate_all(results)
            results['sections']['visualizations'] = viz_results
            print(f"  ✓ Created {viz_results.get('charts_created', 0)} charts")
        
        # 12. EXECUTIVE SUMMARY
        print("\n⭐ SECTION 12: EXECUTIVE SUMMARY")
        print("-"*80)
        exec_summary = self._generate_executive_summary(results)
        self._save_section_json('executive_summary', exec_summary, date, event_name)
        
        # Save main comprehensive results
        report_file = self._save_results(results, date, event_name)
        results['report_file'] = report_file
        
        print("\n" + "="*80)
        print("✓ ANALYSIS COMPLETE")
        print("="*80)
        print(f"Main Report: {report_file}")
        print(f"Section JSONs: {self.sections_dir}")
        print(f"Visualizations: {self.viz_dir}")
        print(f"Articles: {news_results.get('article_count', 0)}")
        print(f"Symbols: {len(symbols)}")
        print("="*80 + "\n")
        
        return results
    
    def _fetch_and_analyze_news(self, date: str, event_name: Optional[str]) -> Dict:
        """Enhanced news fetching with detailed structure"""
        articles = []
        
        try:
            if self.news_fetcher:
                if event_name:
                    articles = self.news_fetcher.fetch_event_news(
                        date, event_name,
                        max_records=self.max_articles,
                        full_content=True
                    )
                else:
                    articles = self.news_fetcher.fetch_news(date, max_records=self.max_articles)
                
                print(f"  ✓ Fetched {len(articles)} articles")
        except Exception as e:
            print(f"  ✗ Error: {str(e)[:50]}")
        
        # Enhanced news structure
        results = {
            'article_count': len(articles),
            'articles': articles,
            'coverage_period': f"{date} to {(pd.to_datetime(date) + timedelta(days=2)).strftime('%Y-%m-%d')}",
            'sources': list(set([a.get('source', 'Unknown') for a in articles])),
            'top_headlines': [a.get('title', '') for a in articles[:10]],
            'impact_analysis': None,
            'sentiment_breakdown': {},
            'key_themes': []
        }
        
        # Impact analysis
        if event_name and articles and self.news_analyzer:
            try:
                impact = self.news_analyzer.analyze_event_impact(
                    date, event_name, articles, comparison_days=3
                )
                results['impact_analysis'] = impact
                print(f"  ✓ Impact analysis complete")
            except:
                pass
        
        # Extract themes
        all_text = ' '.join([a.get('title', '') + ' ' + a.get('content', '')[:200] for a in articles])
        themes = self._extract_themes(all_text)
        results['key_themes'] = themes
        
        return results
    
    def _extract_themes(self, text: str) -> List[str]:
        """Extract key themes from text"""
        text_lower = text.lower()
        
        theme_keywords = {
            'employment': ['jobs', 'employment', 'payroll', 'unemployment', 'labor'],
            'inflation': ['inflation', 'cpi', 'prices', 'cost'],
            'monetary_policy': ['fed', 'federal reserve', 'interest rate', 'rate hike'],
            'economic_growth': ['gdp', 'growth', 'economy', 'economic'],
            'market_sentiment': ['rally', 'decline', 'volatility', 'market'],
            'geopolitical': ['war', 'conflict', 'sanctions', 'trade war']
        }
        
        themes = []
        for theme, keywords in theme_keywords.items():
            if any(kw in text_lower for kw in keywords):
                themes.append(theme.replace('_', ' ').title())
        
        return themes[:5]
    
    def _extract_symbols(self, news_results: Dict, event_name: Optional[str]) -> List[str]:
        """Extract symbols from news and event"""
        symbols = set()
        
        if news_results.get('impact_analysis'):
            impact_symbols = news_results['impact_analysis'].get('symbols', {})
            symbols.update(impact_symbols.keys())
        
        if self.news_fetcher:
            for article in news_results.get('articles', []):
                text = article.get('title', '') + ' ' + article.get('content', '')
                article_symbols = self.news_fetcher.get_affected_symbols(
                    event_name or '', text
                )
                symbols.update(article_symbols)
        
        if not symbols:
            symbols = {'EURUSD=X', 'GC=F', '^GSPC', 'DX-Y.NYB', 'TLT'}
        
        symbol_list = list(symbols)[:15]
        print(f"  ✓ Extracted {len(symbol_list)} symbols")
        
        return symbol_list
    
    def _analyze_cot(self, date: str, symbols: List[str]) -> Dict:
        """Enhanced COT analysis with detailed structure"""
        if not self.cot_fetcher:
            return {}
        
        results = {
            'symbols_analyzed': 0,
            'report_date': '',
            'key_positions': {},
            'positioning_changes': {},
            'smart_money_bias': {}
        }
        
        symbol_map = {
            'EURUSD=X': 'EUR', 'GBPUSD=X': 'GBP', 'USDJPY=X': 'JPY',
            'GC=F': 'GOLD', 'SI=F': 'SILVER', 'CL=F': 'CRUDE_OIL'
        }
        
        for symbol in symbols[:10]:
            cot_symbol = symbol_map.get(symbol)
            if cot_symbol:
                try:
                    positioning = self.cot_fetcher.get_positioning_for_date(cot_symbol, date)
                    if positioning:
                        results['key_positions'][symbol] = {
                            'sentiment': positioning.get('sentiment', 'NEUTRAL'),
                            'dealer_net': positioning.get('dealer', {}).get('net', 0),
                            'asset_mgr_net': positioning.get('asset_manager', {}).get('net', 0),
                            'leveraged_net': positioning.get('leveraged', {}).get('net', 0),
                            'smart_money_bias': 'LONG' if positioning.get('dealer', {}).get('net', 0) > 0 else 'SHORT'
                        }
                        results['symbols_analyzed'] += 1
                        if not results['report_date']:
                            results['report_date'] = positioning.get('report_date', '')
                except:
                    pass
        
        print(f"  ✓ {results['symbols_analyzed']} symbols")
        return results
    
    def _analyze_indicators(self, date: str, symbols: List[str]) -> Dict:
        """Enhanced technical indicators with structured output"""
        if not self.indicator_calc:
            return {}
        
        results = {
            'symbols_analyzed': 0,
            'overall_bias': 'NEUTRAL',
            'buy_signals': 0,
            'sell_signals': 0,
            'neutral_signals': 0,
            'top_signals': {}
        }
        
        for symbol in symbols:
            try:
                indicators = self.indicator_calc.get_indicators_for_date(symbol, date)
                if indicators:
                    signal = indicators.get('overall_signal', 'NEUTRAL')
                    results['top_signals'][symbol] = {
                        'overall': signal,
                        'buy_count': indicators.get('buy_count', 0),
                        'sell_count': indicators.get('sell_count', 0),
                        'price': indicators.get('price', 0),
                        'key_indicators': {}
                    }
                    
                    # Extract key indicators
                    for ind_name, ind_data in list(indicators.get('indicators', {}).items())[:3]:
                        results['top_signals'][symbol]['key_indicators'][ind_name] = {
                            'value': ind_data.get('value'),
                            'signal': ind_data.get('signal')
                        }
                    
                    results['symbols_analyzed'] += 1
                    if signal == 'BUY':
                        results['buy_signals'] += 1
                    elif signal == 'SELL':
                        results['sell_signals'] += 1
                    else:
                        results['neutral_signals'] += 1
            except:
                pass
        
        # Determine overall bias
        if results['buy_signals'] > results['sell_signals']:
            results['overall_bias'] = 'BULLISH'
        elif results['sell_signals'] > results['buy_signals']:
            results['overall_bias'] = 'BEARISH'
        
        print(f"  ✓ {results['symbols_analyzed']} symbols")
        return results
    
    def _analyze_correlations(self, date: str, symbols: List[str]) -> Dict:
        """Enhanced correlation analysis"""
        if not self.corr_analyzer:
            return {}
        
        results = {
            'symbols_analyzed': len(symbols),
            'key_relationships': {},
            'leading_indicators': []
        }
        
        # Key relationships
        relationships = {
            'Dollar vs Gold': ('DX-Y.NYB', 'GC=F'),
            'Stocks vs Bonds': ('^GSPC', 'TLT'),
            'Stocks vs VIX': ('^GSPC', '^VIX'),
            'EUR vs Dollar': ('EURUSD=X', 'DX-Y.NYB')
        }
        
        for name, (sym1, sym2) in relationships.items():
            try:
                start_date = (pd.to_datetime(date) - timedelta(days=90)).strftime('%Y-%m-%d')
                
                s1 = self.corr_analyzer.fetch_data(sym1, start_date, date)
                s2 = self.corr_analyzer.fetch_data(sym2, start_date, date)
                
                if s1 is not None and s2 is not None and len(s1) > 0 and len(s2) > 0:
                    df = pd.DataFrame({'s1': s1, 's2': s2}).dropna()
                    if len(df) > 0:
                        corr_90d = df['s1'].corr(df['s2'])
                        corr_30d = df.tail(30)['s1'].corr(df.tail(30)['s2']) if len(df) >= 30 else corr_90d
                        
                        results['key_relationships'][name] = {
                            'correlation_90d': round(corr_90d, 2),
                            'correlation_30d': round(corr_30d, 2),
                            'relationship': 'INVERSE' if corr_90d < -0.3 else ('POSITIVE' if corr_90d > 0.3 else 'NEUTRAL'),
                            'strength': 'STRONG' if abs(corr_90d) > 0.7 else ('MODERATE' if abs(corr_90d) > 0.4 else 'WEAK')
                        }
            except:
                pass
        
        print(f"  ✓ {len(results['key_relationships'])} relationships")
        return results
    
    def _analyze_economic(self, date: str) -> Optional[Dict]:
        """Enhanced economic indicators"""
        if not self.econ_indicators:
            return None
        
        try:
            snapshot = self.econ_indicators.get_economic_snapshot(date)
            if snapshot:
                # Enhanced structure
                enhanced = {
                    'snapshot_date': date,
                    'overall_status': snapshot.get('overall_economic_status', 'MODERATE'),
                    'interest_rates': snapshot.get('interest_rates', {}),
                    'inflation': snapshot.get('inflation', {}),
                    'employment': snapshot.get('employment', {}),
                    'growth': {
                        'recession_probability': 0.25 if snapshot.get('overall_economic_status') == 'WEAK' else 0.15
                    }
                }
                print(f"  ✓ {enhanced['overall_status']}")
                return enhanced
        except Exception as e:
            print(f"  ✗ Error: {str(e)[:50]}")
        
        return None
    
    def _analyze_structure(self, date: str, symbols: List[str]) -> Dict:
        """Enhanced market structure"""
        if not self.market_structure:
            return {}
        
        results = {
            'symbols_analyzed': 0,
            'overall_market_regime': 'MIXED',
            'volatility_regime': 'NORMAL_VOLATILITY',
            'key_structures': {}
        }
        
        regimes = []
        for symbol in symbols:
            try:
                structure = self.market_structure.get_market_structure(symbol, date)
                if structure:
                    results['key_structures'][symbol] = {
                        'trend': structure.get('trend_analysis', {}).get('trend', 'SIDEWAYS'),
                        'structure': structure.get('price_structure', {}).get('structure', 'NEUTRAL'),
                        'regime': structure.get('market_regime', {}).get('market_type', 'MIXED'),
                        'support_levels': structure.get('support_resistance', {}).get('support', [])[:3],
                        'resistance_levels': structure.get('support_resistance', {}).get('resistance', [])[:3],
                        'pivot_point': structure.get('pivot_points', {}).get('pivot', 0)
                    }
                    results['symbols_analyzed'] += 1
                    regimes.append(structure.get('market_regime', {}).get('market_type', 'MIXED'))
            except:
                pass
        
        if regimes:
            from collections import Counter
            most_common = Counter(regimes).most_common(1)[0][0]
            results['overall_market_regime'] = most_common
        
        print(f"  ✓ {results['symbols_analyzed']} symbols")
        return results
    
    def _analyze_seasonality(self, symbols: List[str]) -> Dict:
        """Enhanced seasonality analysis"""
        if not self.seasonality:
            return {}
        
        results = {
            'symbols_analyzed': 0,
            'current_month': datetime.now().strftime('%B'),
            'current_day': datetime.now().strftime('%A'),
            'key_patterns': {},
            'current_biases': []
        }
        
        for symbol in symbols:
            try:
                season = self.seasonality.get_seasonality_analysis(symbol, years_back=5)
                if season:
                    results['key_patterns'][symbol] = {
                        'best_month': season.get('monthly_patterns', {}).get('best_month'),
                        'worst_month': season.get('monthly_patterns', {}).get('worst_month'),
                        'best_day': season.get('day_of_week_patterns', {}).get('best_day'),
                        'current_month_bias': season.get('current_biases', {}).get('month_avg_return', 0),
                        'current_day_bias': season.get('current_biases', {}).get('day_avg_return', 0)
                    }
                    results['symbols_analyzed'] += 1
                    
                    if abs(season.get('current_biases', {}).get('month_avg_return', 0)) > 0.5:
                        bias = 'positive' if season.get('current_biases', {}).get('month_avg_return', 0) > 0 else 'negative'
                        results['current_biases'].append(
                            f"{symbol} shows {bias} bias for {results['current_month']}"
                        )
            except:
                pass
        
        print(f"  ✓ {results['symbols_analyzed']} symbols")
        return results
    
    def _analyze_volume(self, date: str, symbols: List[str]) -> Dict:
        """Enhanced volume analysis"""
        if not self.volume_analyzer:
            return {}
        
        results = {
            'symbols_analyzed': 0,
            'overall_volume_trend': 'NORMAL',
            'key_insights': {}
        }
        
        high_volume_count = 0
        for symbol in symbols:
            try:
                volume = self.volume_analyzer.get_volume_analysis(symbol, date)
                if volume:
                    vol_ratio = volume.get('indicators', {}).get('Volume_Ratio', {}).get('value', 1.0)
                    
                    results['key_insights'][symbol] = {
                        'volume_ratio': vol_ratio,
                        'signal': volume.get('indicators', {}).get('Volume_Ratio', {}).get('signal', 'NORMAL'),
                        'vwap_position': 'ABOVE' if volume.get('indicators', {}).get('VWAP', {}).get('signal') == 'BULLISH' else 'BELOW',
                        'obv_trend': volume.get('indicators', {}).get('OBV', {}).get('trend', 'NEUTRAL'),
                        'mfi': volume.get('indicators', {}).get('MFI', {}).get('value', 50),
                        'overall': volume.get('overall_signal', 'NEUTRAL')
                    }
                    results['symbols_analyzed'] += 1
                    
                    if vol_ratio > 1.2:
                        high_volume_count += 1
            except:
                pass
        
        if high_volume_count > len(symbols) / 2:
            results['overall_volume_trend'] = 'INCREASING'
        
        print(f"  ✓ {results['symbols_analyzed']} symbols")
        return results
    
    def _generate_executive_summary(self, results: Dict) -> Dict:
        """Generate executive summary from all sections"""
        metadata = results.get('metadata', {})
        sections = results.get('sections', {})
        
        summary = {
            'market_overview': f"Comprehensive analysis for {metadata.get('event_name', 'market event')} on {metadata.get('date')}",
            'key_findings': [],
            'sentiment': {
                'overall': 'NEUTRAL',
                'confidence': 0.5,
                'breakdown': {'positive': 33, 'neutral': 34, 'negative': 33}
            },
            'risks': [],
            'opportunities': []
        }
        
        # Extract key findings from synthesis
        synthesis = sections.get('synthesis', {})
        if synthesis:
            summary['key_findings'] = synthesis.get('key_findings', [])[:5]
            summary['sentiment']['confidence'] = synthesis.get('overall_confidence', 0.5)
        
        # Add technical bias as finding
        indicators = sections.get('indicators', {})
        if indicators:
            bias = indicators.get('overall_bias', 'NEUTRAL')
            buy_pct = (indicators.get('buy_signals', 0) / max(indicators.get('symbols_analyzed', 1), 1)) * 100
            
            summary['key_findings'].append({
                'finding': f"Technical indicators show {bias.lower()} bias with {buy_pct:.0f}% buy signals",
                'importance': 'high',
                'data_point': f"{indicators.get('buy_signals', 0)} BUY vs {indicators.get('sell_signals', 0)} SELL"
            })
        
        # Add risks and opportunities from different sections
        econ = sections.get('economic', {})
        if econ:
            if econ.get('interest_rates', {}).get('recession_risk') == 'HIGH':
                summary['risks'].append({
                    'risk': 'Inverted yield curve signals potential recession',
                    'severity': 'high',
                    'probability': 'moderate'
                })
        
        # Determine overall sentiment
        if indicators.get('overall_bias') == 'BULLISH':
            summary['sentiment']['overall'] = 'BULLISH'
            summary['sentiment']['breakdown'] = {'positive': 65, 'neutral': 25, 'negative': 10}
        elif indicators.get('overall_bias') == 'BEARISH':
            summary['sentiment']['overall'] = 'BEARISH'
            summary['sentiment']['breakdown'] = {'positive': 10, 'neutral': 25, 'negative': 65}
        
        print(f"  ✓ Executive summary generated")
        return summary
    
    def _save_section_json(self, section_name: str, section_data: Dict, date: str, event_name: Optional[str]):
        """Save individual section JSON file"""
        date_clean = date.replace('-', '_')
        event_clean = (event_name or 'analysis').replace(' ', '_').lower()
        
        filename = f'{self.sections_dir}/{section_name}_{date_clean}_{event_clean}.json'
        
        # Add metadata to section
        enhanced_data = {
            'section_type': section_name,
            'date': date,
            'event_name': event_name,
            'generated_at': datetime.now().isoformat(),
            'data': section_data
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(enhanced_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"  💾 Saved: {section_name}.json")
    
    def _save_results(self, results: Dict, date: str, event_name: Optional[str]) -> str:
        """Save main comprehensive results"""
        date_clean = date.replace('-', '_')
        event_clean = (event_name or 'analysis').replace(' ', '_').lower()
        
        filename = f'{self.output_dir}/comprehensive_{date_clean}_{event_clean}.json'
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # Also save markdown summary if synthesizer available
        if self.synthesizer:
            synthesis = results.get('sections', {}).get('synthesis', {})
            md_content = self.synthesizer.generate_markdown_summary(synthesis, results['metadata'])
            
            md_file = filename.replace('.json', '.md')
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(md_content)
        
        return filename


if __name__ == "__main__":
    pipeline = EnhancedComprehensivePipeline(
        output_dir='enhanced_pipeline_output',
        enable_hf=True,
        enable_viz=True,
        max_articles=20
    )
    
    results = pipeline.analyze(
        date='2024-11-01',
        event_name='Non-Farm Payrolls',
        symbols=None
    )
    
    print(f"\n✓ Results saved to: {results['report_file']}")
    print(f"✓ Section JSONs in: {pipeline.sections_dir}")
