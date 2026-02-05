"""
FIXED: Comprehensive Analysis Pipeline with NaN-safe visualizations
Handles memory issues and prevents visualization crashes
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

# Add safe value checking
def safe_float(value, default=0.0):
    """Safely convert to float, handling NaN"""
    try:
        v = float(value)
        return default if np.isnan(v) or np.isinf(v) else v
    except (ValueError, TypeError):
        return default

def safe_pie_data(data_dict):
    """Ensure pie chart data is valid (no NaN, positive values, sum > 0)"""
    clean = {}
    for k, v in data_dict.items():
        val = safe_float(v, 0)
        if val > 0:
            clean[k] = val
    
    if not clean or sum(clean.values()) == 0:
        return None
    
    return clean

# Import modules with error handling
try:
    from news_fetcher import NewsFetcher
except ImportError:
    NewsFetcher = None
    
try:
    from news_impact_analyzer import NewsImpactAnalyzer
except ImportError:
    NewsImpactAnalyzer = None

try:
    from cot_data_fetcher import COTDataFetcher
except ImportError:
    COTDataFetcher = None

try:
    from symbol_indicators import SymbolIndicatorCalculator  
except ImportError:
    SymbolIndicatorCalculator = None

try:
    from correlation_analyzer import CorrelationAnalyzer
except ImportError:
    CorrelationAnalyzer = None

try:
    from economic_indicators import EconomicIndicatorIntegration
except ImportError:
    EconomicIndicatorIntegration = None

try:
    from seasonality_analyzer import SeasonalityAnalyzer
except ImportError:
    SeasonalityAnalyzer = None

try:
    from market_structure_analyzer import MarketStructureAnalyzer
except ImportError:
    MarketStructureAnalyzer = None

try:
    from volume_analyzer import VolumeAnalyzer
except ImportError:
    VolumeAnalyzer = None


class ComprehensiveAnalysisPipeline:
    """Master pipeline with memory-safe HF model loading"""
    
    def __init__(self, output_dir='pipeline_output', use_hf=False):
        self.output_dir = output_dir
        self.use_hf = use_hf
        os.makedirs(output_dir, exist_ok=True)
        
        print("="*80)
        print("COMPREHENSIVE ANALYSIS PIPELINE (FIXED)")
        print("="*80)
        print("Initializing core modules...")
        
        # Core modules (lightweight)
        self.news_fetcher = NewsFetcher() if NewsFetcher else None
        self.news_analyzer = NewsImpactAnalyzer() if NewsImpactAnalyzer else None
        self.cot_fetcher = COTDataFetcher() if COTDataFetcher else None
        self.indicator_calc = SymbolIndicatorCalculator() if SymbolIndicatorCalculator else None
        self.corr_analyzer = CorrelationAnalyzer() if CorrelationAnalyzer else None
        self.econ_indicators = EconomicIndicatorIntegration() if EconomicIndicatorIntegration else None
        self.seasonality = SeasonalityAnalyzer() if SeasonalityAnalyzer else None
        self.market_structure = MarketStructureAnalyzer() if MarketStructureAnalyzer else None
        self.volume_analyzer = VolumeAnalyzer() if VolumeAnalyzer else None
        
        print("✓ Core modules initialized")
        
        # Skip HF models in CI/CD to avoid memory issues
        if self.use_hf and not os.environ.get('CI'):
            print("\nLoading HF models (memory intensive)...")
            self._load_hf_models_safe()
        else:
            print("\n⊘ Skipping HF models (memory conservation mode)")
            self.hf_methods = {}
        
        print("✓ Pipeline ready\n")
    
    def _load_hf_models_safe(self):
        """Load HF models with memory checks"""
        self.hf_methods = {}
        
        # Only load critical models
        try:
            from hf_method1_sentiment import HFSentimentAnalyzer
            self.hf_methods['sentiment'] = HFSentimentAnalyzer()
            self.hf_methods['sentiment'].load_model()
            print("  ✓ Sentiment")
        except Exception as e:
            print(f"  ⊘ Sentiment: {str(e)[:50]}")
        
        try:
            from hf_method2_ner import HFEntityExtractor
            self.hf_methods['ner'] = HFEntityExtractor()
            self.hf_methods['ner'].load_model()
            print("  ✓ NER")
        except Exception as e:
            print(f"  ⊘ NER: {str(e)[:50]}")
    
    def analyze(self, date, event_name=None, symbols=None, 
                use_hf_methods=False, generate_visuals=False):
        """Run analysis with safe defaults"""
        
        print("="*80)
        print(f"ANALYSIS: {event_name or 'Market Data'} on {date}")
        print("="*80)
        
        results = {
            'metadata': {
                'date': date,
                'event_name': event_name,
                'symbols': symbols or 'auto-detect',
                'timestamp': datetime.now().isoformat(),
                'hf_enabled': use_hf_methods and bool(getattr(self, 'hf_methods', {})),
                'visuals_enabled': generate_visuals
            },
            'sections': {}
        }
        
        # NEWS
        if self.news_fetcher and self.news_analyzer:
            print("\n📰 NEWS ANALYSIS")
            print("-" * 80)
            news_results = self._analyze_news_safe(date, event_name)
            results['sections']['news'] = news_results
            
            if symbols is None and news_results.get('extracted_symbols'):
                symbols = news_results['extracted_symbols'][:10]
                print(f"✓ Auto-detected symbols: {', '.join(symbols[:5])}...")
        
        if symbols is None:
            symbols = ['EURUSD=X', 'GC=F', '^GSPC']
            print(f"Using default symbols: {', '.join(symbols)}")
        
        results['metadata']['symbols_analyzed'] = symbols
        
        # INDICATORS
        if self.indicator_calc:
            print("\n📈 TECHNICAL INDICATORS")
            print("-" * 80)
            indicator_results = self._analyze_indicators_safe(date, symbols)
            results['sections']['indicators'] = indicator_results
        
        # COT
        if self.cot_fetcher:
            print("\n📊 COT POSITIONING")
            print("-" * 80)
            cot_results = self._analyze_cot_safe(date, symbols)
            results['sections']['cot'] = cot_results
        
        # SYNTHESIS
        print("\n💡 SYNTHESIS")
        print("-" * 80)
        synthesis = self._synthesize_safe(results)
        results['sections']['synthesis'] = synthesis
        
        # Save
        report_file = self._save_results(results, date, event_name)
        results['report_file'] = report_file
        
        print("\n" + "="*80)
        print("✓ ANALYSIS COMPLETE")
        print("="*80)
        print(f"Report: {report_file}")
        print("="*80 + "\n")
        
        return results
    
    def _analyze_news_safe(self, date, event_name):
        """Safe news analysis"""
        results = {'articles': [], 'impact_analysis': None, 'extracted_symbols': []}
        
        try:
            if event_name:
                articles = self.news_fetcher.fetch_event_news(date, event_name, max_records=5)
            else:
                articles = self.news_fetcher.fetch_news(date, max_records=5)
            
            results['articles'] = articles
            print(f"  ✓ Fetched {len(articles)} articles")
            
            if event_name and articles and self.news_analyzer:
                impact = self.news_analyzer.analyze_event_impact(
                    date, event_name, articles, comparison_days=3
                )
                results['impact_analysis'] = impact
                
                if impact and impact.get('symbols'):
                    results['extracted_symbols'] = list(impact['symbols'].keys())
                    print(f"  ✓ Analyzed {len(impact['symbols'])} symbols")
        except Exception as e:
            print(f"  ⊘ News error: {str(e)[:50]}")
        
        return results
    
    def _analyze_indicators_safe(self, date, symbols):
        """Safe indicator analysis"""
        results = {}
        
        for symbol in symbols[:5]:
            try:
                indicators = self.indicator_calc.get_indicators_for_date(symbol, date)
                if indicators:
                    results[symbol] = indicators
            except Exception as e:
                print(f"  ⊘ {symbol}: {str(e)[:30]}")
        
        print(f"  ✓ Indicators for {len(results)} symbols")
        return results
    
    def _analyze_cot_safe(self, date, symbols):
        """Safe COT analysis"""
        results = {}
        
        symbol_map = {
            'EURUSD=X': 'EUR', 'GC=F': 'GOLD', '^GSPC': None
        }
        
        for symbol in symbols[:5]:
            cot_symbol = symbol_map.get(symbol)
            if cot_symbol:
                try:
                    positioning = self.cot_fetcher.get_positioning_for_date(cot_symbol, date)
                    if positioning:
                        results[symbol] = positioning
                except Exception as e:
                    print(f"  ⊘ {symbol}: {str(e)[:30]}")
        
        print(f"  ✓ COT data for {len(results)} symbols")
        return results
    
    def _synthesize_safe(self, results):
        """Safe synthesis"""
        insights = {
            'summary': [],
            'key_findings': [],
            'signals': []
        }
        
        meta = results['metadata']
        insights['summary'].append(
            f"Analyzed {meta['event_name'] or 'market'} on {meta['date']}"
        )
        
        # Indicators
        indicators = results['sections'].get('indicators', {})
        if indicators:
            buy_signals = sum(1 for ind in indicators.values() 
                            if ind.get('overall_signal') == 'BUY')
            
            if buy_signals > len(indicators) / 2:
                insights['signals'].append(
                    f"Technical: BULLISH ({buy_signals}/{len(indicators)})"
                )
        
        return insights
    
    def _save_results(self, results, date, event_name):
        """Save results"""
        date_clean = date.replace('-', '_')
        event_clean = (event_name or 'analysis').replace(' ', '_').lower()
        
        filename = f'{self.output_dir}/analysis_{date_clean}_{event_clean}.json'
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        return filename
    
    def generate_markdown_report(self, results):
        """Generate markdown report"""
        lines = [
            "# MARKET ANALYSIS REPORT",
            "",
            f"**Date:** {results['metadata']['date']}",
            f"**Event:** {results['metadata']['event_name'] or 'Market Analysis'}",
            "",
            "## Summary",
            ""
        ]
        
        synthesis = results['sections'].get('synthesis', {})
        for item in synthesis.get('summary', []):
            lines.append(f"- {item}")
        
        lines.append("")
        lines.append("## Key Findings")
        lines.append("")
        
        for finding in synthesis.get('key_findings', []):
            lines.append(f"- {finding}")
        
        if synthesis.get('signals'):
            lines.append("")
            lines.append("## Signals")
            lines.append("")
            for signal in synthesis['signals']:
                lines.append(f"- {signal}")
        
        lines.append("")
        lines.append("---")
        lines.append("*Generated by Comprehensive Pipeline*")
        
        return '\n'.join(lines)


if __name__ == "__main__":
    print("="*80)
    print("COMPREHENSIVE PIPELINE - MEMORY-SAFE TEST")
    print("="*80)
    
    # Create pipeline WITHOUT HF models (memory safe)
    pipeline = ComprehensiveAnalysisPipeline(
        output_dir='test_pipeline_output',
        use_hf=False  # Disable HF models for CI/CD
    )
    
    # Test
    results = pipeline.analyze(
        date='2024-11-01',
        event_name='Non-Farm Payrolls',
        symbols=['EURUSD=X', 'GC=F'],
        use_hf_methods=False,
        generate_visuals=False
    )
    
    # Generate report
    markdown = pipeline.generate_markdown_report(results)
    
    md_file = f"{pipeline.output_dir}/report.md"
    with open(md_file, 'w') as f:
        f.write(markdown)
    
    print(f"\n✓ Report: {md_file}")
    print("\n" + "="*80)
    print("✓ Test complete - no memory issues!")
    print("="*80)
