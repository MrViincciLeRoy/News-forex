"""
Comprehensive Analysis Pipeline - OPTIMIZED for 80% Memory Usage
Intelligently loads HF models based on available memory
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys
import psutil
import gc
from pathlib import Path

# Safe math helpers
def safe_float(value, default=0.0):
    """Safely convert to float, handling NaN"""
    try:
        v = float(value)
        return default if np.isnan(v) or np.isinf(v) else v
    except (ValueError, TypeError):
        return default

def safe_pie_data(data_dict):
    """Ensure pie chart data is valid"""
    clean = {}
    for k, v in data_dict.items():
        val = safe_float(v, 0)
        if val > 0:
            clean[k] = val
    
    if not clean or sum(clean.values()) == 0:
        return None
    
    return clean

# Memory utilities
def get_memory_info():
    """Get current memory stats"""
    vm = psutil.virtual_memory()
    process = psutil.Process()
    
    return {
        'total_gb': round(vm.total / (1024**3), 2),
        'available_gb': round(vm.available / (1024**3), 2),
        'used_percent': vm.percent,
        'process_mb': round(process.memory_info().rss / (1024**2), 2)
    }

def can_use_memory(required_mb):
    """Check if we have enough memory"""
    mem = get_memory_info()
    available_mb = mem['available_gb'] * 1024
    target_usage = 0.80  # Use up to 80%
    
    # Calculate how much we can still use
    total_mb = mem['total_gb'] * 1024
    max_usable = total_mb * target_usage
    currently_used = mem['process_mb']
    remaining = max_usable - currently_used
    
    return remaining > required_mb, remaining

# Import core modules
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


class MemoryAwareHFLoader:
    """Intelligently load HF models based on available memory"""
    
    # Estimated memory requirements (MB)
    MODEL_MEMORY = {
        'sentiment': 500,
        'ner': 400,
        'classification': 800,
        'qa': 600,
        'multimodal': 700
    }
    
    # Priority order (most useful first)
    MODEL_PRIORITY = ['sentiment', 'ner', 'classification', 'qa', 'multimodal']
    
    def __init__(self):
        self.loaded_models = {}
        self.memory_budget_used = 0
        
    def load_models(self, max_memory_mb=None):
        """Load as many models as memory allows"""
        if max_memory_mb is None:
            mem = get_memory_info()
            # Use 80% of available memory
            max_memory_mb = mem['available_gb'] * 1024 * 0.80
        
        print(f"\n🧠 Memory Budget: {max_memory_mb:.0f}MB")
        print(f"Loading HF models in priority order...")
        
        for model_name in self.MODEL_PRIORITY:
            required = self.MODEL_MEMORY[model_name]
            
            if self.memory_budget_used + required > max_memory_mb:
                print(f"  ⊘ {model_name}: Budget exceeded ({self.memory_budget_used + required:.0f}/{max_memory_mb:.0f}MB)")
                continue
            
            can_load, remaining = can_use_memory(required)
            
            if not can_load:
                print(f"  ⊘ {model_name}: Insufficient memory ({remaining:.0f}MB remaining)")
                continue
            
            # Try to load the model
            try:
                model = self._load_model(model_name)
                if model:
                    self.loaded_models[model_name] = model
                    self.memory_budget_used += required
                    print(f"  ✓ {model_name} ({required}MB, total: {self.memory_budget_used:.0f}MB)")
            except Exception as e:
                print(f"  ✗ {model_name}: {str(e)[:50]}")
        
        print(f"\n✓ Loaded {len(self.loaded_models)}/{len(self.MODEL_PRIORITY)} models")
        print(f"  Memory used: {self.memory_budget_used:.0f}MB")
        
        return self.loaded_models
    
    def _load_model(self, model_name):
        """Load specific model"""
        if model_name == 'sentiment':
            from hf_method1_sentiment import HFSentimentAnalyzer
            analyzer = HFSentimentAnalyzer()
            analyzer.load_model()
            return analyzer
        
        elif model_name == 'ner':
            from hf_method2_ner import HFEntityExtractor
            extractor = HFEntityExtractor()
            extractor.load_model()
            return extractor
        
        elif model_name == 'classification':
            from hf_method4_classification import HFEventClassifier
            classifier = HFEventClassifier()
            classifier.load_model()
            return classifier
        
        elif model_name == 'qa':
            from hf_method6_qa import HFMarketQA
            qa = HFMarketQA()
            qa.load_model()
            return qa
        
        elif model_name == 'multimodal':
            from hf_method8_multimodal import HFMultiModalAnalyzer
            multimodal = HFMultiModalAnalyzer()
            multimodal.load_models()
            return multimodal
        
        return None


class ComprehensiveAnalysisPipeline:
    """Memory-optimized comprehensive pipeline"""
    
    def __init__(self, output_dir='pipeline_output', enable_hf=True):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Get initial memory state
        mem_start = get_memory_info()
        
        print("="*80)
        print("COMPREHENSIVE ANALYSIS PIPELINE - MEMORY OPTIMIZED")
        print("="*80)
        print(f"System Memory: {mem_start['total_gb']}GB total, {mem_start['available_gb']}GB available")
        print(f"Target Usage: 80% ({mem_start['total_gb'] * 0.80:.1f}GB max)")
        print("="*80)
        
        print("\nInitializing core modules...")
        
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
        
        # Load HF models intelligently
        self.hf_models = {}
        if enable_hf:
            loader = MemoryAwareHFLoader()
            self.hf_models = loader.load_models()
        else:
            print("\n⊘ HF models disabled")
        
        mem_end = get_memory_info()
        print(f"\n📊 Memory After Init: {mem_end['process_mb']}MB process, {mem_end['used_percent']:.1f}% system")
        print("="*80 + "\n")
    
    def analyze(self, date, event_name=None, symbols=None):
        """Run comprehensive analysis"""
        
        print("="*80)
        print(f"ANALYSIS: {event_name or 'Market Analysis'}")
        print(f"Date: {date}")
        print("="*80)
        
        results = {
            'metadata': {
                'date': date,
                'event_name': event_name,
                'timestamp': datetime.now().isoformat(),
                'hf_models_loaded': list(self.hf_models.keys()),
                'symbols': symbols or 'auto-detect'
            },
            'sections': {}
        }
        
        # 1. NEWS
        print("\n📰 SECTION 1: NEWS ANALYSIS")
        print("-" * 80)
        news_results = self._analyze_news(date, event_name)
        results['sections']['news'] = news_results
        
        if symbols is None and news_results.get('extracted_symbols'):
            symbols = news_results['extracted_symbols'][:15]
            print(f"✓ Auto-detected: {', '.join(symbols[:5])}...")
        elif symbols is None:
            symbols = ['EURUSD=X', 'GC=F', '^GSPC', 'DX-Y.NYB', 'TLT']
        
        results['metadata']['symbols_analyzed'] = symbols
        
        # 2. COT
        print("\n📊 SECTION 2: COT POSITIONING")
        print("-" * 80)
        cot_results = self._analyze_cot(date, symbols)
        results['sections']['cot'] = cot_results
        
        # 3. INDICATORS
        print("\n📈 SECTION 3: TECHNICAL INDICATORS")
        print("-" * 80)
        indicator_results = self._analyze_indicators(date, symbols)
        results['sections']['indicators'] = indicator_results
        
        # 4. CORRELATIONS
        if self.corr_analyzer:
            print("\n🔗 SECTION 4: CORRELATIONS")
            print("-" * 80)
            corr_results = self._analyze_correlations(date, symbols[:5])
            results['sections']['correlations'] = corr_results
        
        # 5. ECONOMIC INDICATORS
        if self.econ_indicators:
            print("\n💹 SECTION 5: ECONOMIC INDICATORS")
            print("-" * 80)
            econ_results = self._analyze_economic(date)
            results['sections']['economic'] = econ_results
        
        # 6. MARKET STRUCTURE
        if self.market_structure:
            print("\n🏗️  SECTION 6: MARKET STRUCTURE")
            print("-" * 80)
            structure_results = self._analyze_structure(date, symbols[:5])
            results['sections']['structure'] = structure_results
        
        # 7. SEASONALITY
        if self.seasonality:
            print("\n📅 SECTION 7: SEASONALITY")
            print("-" * 80)
            season_results = self._analyze_seasonality(symbols[:3])
            results['sections']['seasonality'] = season_results
        
        # 8. VOLUME
        if self.volume_analyzer:
            print("\n📊 SECTION 8: VOLUME ANALYSIS")
            print("-" * 80)
            volume_results = self._analyze_volume(date, symbols[:5])
            results['sections']['volume'] = volume_results
        
        # 9-13. HF METHODS
        if self.hf_models:
            print("\n🤖 SECTIONS 9-13: HF AI METHODS")
            print("-" * 80)
            hf_results = self._run_hf_methods(date, event_name, news_results, symbols, indicator_results)
            results['sections']['hf_methods'] = hf_results
        
        # 14. SYNTHESIS
        print("\n💡 SECTION 14: SYNTHESIS & INSIGHTS")
        print("-" * 80)
        synthesis = self._synthesize(results)
        results['sections']['synthesis'] = synthesis
        
        # Save
        report_file = self._save_results(results, date, event_name)
        results['report_file'] = report_file
        
        mem_final = get_memory_info()
        
        print("\n" + "="*80)
        print("✓ COMPREHENSIVE ANALYSIS COMPLETE")
        print("="*80)
        print(f"Report: {report_file}")
        print(f"Memory: {mem_final['process_mb']}MB ({mem_final['used_percent']:.1f}% system)")
        print("="*80 + "\n")
        
        return results
    
    def _analyze_news(self, date, event_name):
        """News analysis"""
        results = {'articles': [], 'impact_analysis': None, 'extracted_symbols': []}
        
        if not self.news_fetcher:
            return results
        
        try:
            if event_name:
                articles = self.news_fetcher.fetch_event_news(date, event_name, max_records=10)
            else:
                articles = self.news_fetcher.fetch_news(date, max_records=10)
            
            results['articles'] = articles
            print(f"  ✓ Fetched {len(articles)} articles")
            
            if event_name and articles and self.news_analyzer:
                impact = self.news_analyzer.analyze_event_impact(
                    date, event_name, articles, comparison_days=3
                )
                results['impact_analysis'] = impact
                
                if impact and impact.get('symbols'):
                    results['extracted_symbols'] = list(impact['symbols'].keys())
                    print(f"  ✓ Impact analysis: {len(impact['symbols'])} symbols")
        except Exception as e:
            print(f"  ⊘ Error: {str(e)[:50]}")
        
        return results
    
    def _analyze_cot(self, date, symbols):
        """COT positioning"""
        results = {}
        
        if not self.cot_fetcher:
            return results
        
        symbol_map = {
            'EURUSD=X': 'EUR', 'GBPUSD=X': 'GBP', 'USDJPY=X': 'JPY',
            'GC=F': 'GOLD', 'GOLD': 'GOLD', 'SI=F': 'SILVER', 'CL=F': 'CRUDE_OIL'
        }
        
        for symbol in symbols[:10]:
            cot_symbol = symbol_map.get(symbol)
            if cot_symbol:
                try:
                    positioning = self.cot_fetcher.get_positioning_for_date(cot_symbol, date)
                    if positioning:
                        results[symbol] = positioning
                except Exception as e:
                    pass
        
        print(f"  ✓ COT data: {len(results)} symbols")
        return results
    
    def _analyze_indicators(self, date, symbols):
        """Technical indicators"""
        results = {}
        
        if not self.indicator_calc:
            return results
        
        for symbol in symbols:
            try:
                indicators = self.indicator_calc.get_indicators_for_date(symbol, date)
                if indicators:
                    results[symbol] = indicators
            except Exception as e:
                pass
        
        print(f"  ✓ Indicators: {len(results)} symbols")
        return results
    
    def _analyze_correlations(self, date, symbols):
        """Correlation analysis"""
        results = {}
        
        for symbol in symbols:
            try:
                corr = self.corr_analyzer.get_correlation_analysis(symbol, date, lookback_days=90)
                if corr:
                    results[symbol] = corr
            except Exception as e:
                pass
        
        print(f"  ✓ Correlations: {len(results)} symbols")
        return results
    
    def _analyze_economic(self, date):
        """Economic indicators"""
        try:
            snapshot = self.econ_indicators.get_economic_snapshot(date)
            print(f"  ✓ Economic: {snapshot.get('overall_economic_status', 'N/A')}")
            return snapshot
        except Exception as e:
            print(f"  ⊘ Error: {str(e)[:50]}")
            return None
    
    def _analyze_structure(self, date, symbols):
        """Market structure"""
        results = {}
        
        for symbol in symbols:
            try:
                structure = self.market_structure.get_market_structure(symbol, date)
                if structure:
                    results[symbol] = structure
            except Exception as e:
                pass
        
        print(f"  ✓ Structure: {len(results)} symbols")
        return results
    
    def _analyze_seasonality(self, symbols):
        """Seasonality patterns"""
        results = {}
        
        for symbol in symbols:
            try:
                season = self.seasonality.get_seasonality_analysis(symbol, years_back=5)
                if season:
                    results[symbol] = season
            except Exception as e:
                pass
        
        print(f"  ✓ Seasonality: {len(results)} symbols")
        return results
    
    def _analyze_volume(self, date, symbols):
        """Volume analysis"""
        results = {}
        
        for symbol in symbols:
            try:
                volume = self.volume_analyzer.get_volume_analysis(symbol, date)
                if volume:
                    results[symbol] = volume
            except Exception as e:
                pass
        
        print(f"  ✓ Volume: {len(results)} symbols")
        return results
    
    def _run_hf_methods(self, date, event_name, news_results, symbols, indicator_results):
        """Run HF AI methods"""
        hf_results = {}
        articles = news_results.get('articles', [])
        
        # Sentiment
        if 'sentiment' in self.hf_models and articles:
            try:
                analyzer = self.hf_models['sentiment']
                sentiment = analyzer.analyze_news_articles(articles)
                aggregated = analyzer.aggregate_sentiment(sentiment)
                hf_results['sentiment'] = {'articles': sentiment, 'aggregated': aggregated}
                print(f"  ✓ Sentiment: {aggregated.get('overall_sentiment', 'N/A').upper()}")
            except Exception as e:
                print(f"  ⊘ Sentiment: {str(e)[:50]}")
        
        # NER
        if 'ner' in self.hf_models and articles:
            try:
                extractor = self.hf_models['ner']
                entities = extractor.analyze_batch(articles)
                aggregated = extractor.aggregate_symbols(entities)
                hf_results['entities'] = {'analyzed': entities, 'aggregated': aggregated}
                print(f"  ✓ NER: {aggregated.get('symbol_count', 0)} symbols")
            except Exception as e:
                print(f"  ⊘ NER: {str(e)[:50]}")
        
        # Classification
        if 'classification' in self.hf_models and event_name:
            try:
                classifier = self.hf_models['classification']
                classification = classifier.classify_event(event_name)
                hf_results['classification'] = classification
                print(f"  ✓ Classification: {classification.get('impact_level', 'N/A')}")
            except Exception as e:
                print(f"  ⊘ Classification: {str(e)[:50]}")
        
        # Q&A
        if 'qa' in self.hf_models and articles:
            try:
                qa = self.hf_models['qa']
                qa.index_news_impact([news_results.get('impact_analysis')])
                questions = [f"What happened on {date}?", "What was the market sentiment?"]
                qa_results = qa.batch_ask(questions)
                hf_results['qa'] = qa_results
                print(f"  ✓ Q&A: {len(qa_results)} questions answered")
            except Exception as e:
                print(f"  ⊘ Q&A: {str(e)[:50]}")
        
        return hf_results
    
    def _synthesize(self, results):
        """Synthesize insights"""
        insights = {
            'summary': [],
            'key_findings': [],
            'signals': [],
            'risks': []
        }
        
        meta = results['metadata']
        insights['summary'].append(
            f"Analyzed {meta['event_name'] or 'market'} on {meta['date']}"
        )
        insights['summary'].append(
            f"Processed {len(meta.get('symbols_analyzed', []))} symbols"
        )
        
        # HF sentiment
        hf = results['sections'].get('hf_methods', {})
        if hf.get('sentiment'):
            sent = hf['sentiment']['aggregated']
            insights['key_findings'].append(
                f"AI Sentiment: {sent.get('overall_sentiment', 'N/A').upper()} ({sent.get('confidence', 0):.1%})"
            )
        
        # Technical signals
        indicators = results['sections'].get('indicators', {})
        if indicators:
            buy_count = sum(1 for ind in indicators.values() if ind.get('overall_signal') == 'BUY')
            sell_count = sum(1 for ind in indicators.values() if ind.get('overall_signal') == 'SELL')
            
            if buy_count > sell_count:
                insights['signals'].append(f"Technical: BULLISH ({buy_count}/{len(indicators)})")
            elif sell_count > buy_count:
                insights['signals'].append(f"Technical: BEARISH ({sell_count}/{len(indicators)})")
        
        # Economic status
        econ = results['sections'].get('economic', {})
        if econ:
            insights['key_findings'].append(
                f"Economic Status: {econ.get('overall_economic_status', 'N/A')}"
            )
        
        return insights
    
    def _save_results(self, results, date, event_name):
        """Save results"""
        date_clean = date.replace('-', '_')
        event_clean = (event_name or 'analysis').replace(' ', '_').lower()
        
        filename = f'{self.output_dir}/comprehensive_{date_clean}_{event_clean}.json'
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        return filename
    
    def generate_markdown_report(self, results):
        """Generate markdown report"""
        lines = [
            "# COMPREHENSIVE MARKET ANALYSIS",
            "",
            f"**Date:** {results['metadata']['date']}",
            f"**Event:** {results['metadata']['event_name'] or 'Market Analysis'}",
            f"**Symbols:** {len(results['metadata'].get('symbols_analyzed', []))}",
            f"**HF Models:** {', '.join(results['metadata'].get('hf_models_loaded', ['None']))}",
            "",
            "---",
            "",
            "## Executive Summary",
            ""
        ]
        
        synthesis = results['sections'].get('synthesis', {})
        for item in synthesis.get('summary', []):
            lines.append(f"- {item}")
        
        lines.extend(["", "## Key Findings", ""])
        for finding in synthesis.get('key_findings', []):
            lines.append(f"- **{finding}**")
        
        if synthesis.get('signals'):
            lines.extend(["", "## Trading Signals", ""])
            for signal in synthesis['signals']:
                lines.append(f"- 📊 {signal}")
        
        if synthesis.get('risks'):
            lines.extend(["", "## Risk Factors", ""])
            for risk in synthesis['risks']:
                lines.append(f"- ⚠️ {risk}")
        
        lines.extend(["", "---", "*Generated by Comprehensive Pipeline*"])
        
        return '\n'.join(lines)


if __name__ == "__main__":
    print("="*80)
    print("COMPREHENSIVE PIPELINE - 80% MEMORY TEST")
    print("="*80)
    
    # Enable HF models with 80% memory usage
    pipeline = ComprehensiveAnalysisPipeline(
        output_dir='test_pipeline_output',
        enable_hf=True  # Enable HF models!
    )
    
    # Run comprehensive analysis
    results = pipeline.analyze(
        date='2024-11-01',
        event_name='Non-Farm Payrolls',
        symbols=None  # Auto-detect
    )
    
    # Generate markdown report
    markdown = pipeline.generate_markdown_report(results)
    
    md_file = f"{pipeline.output_dir}/comprehensive_report.md"
    with open(md_file, 'w') as f:
        f.write(markdown)
    
    print(f"\n✓ Markdown Report: {md_file}")
    print("\n" + "="*80)
    print("✓ COMPREHENSIVE ANALYSIS COMPLETE")
    print("="*80)
