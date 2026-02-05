"""
Comprehensive Analysis Pipeline - Streamlined Orchestrator
Delegates responsibilities to specialized modules
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

import warnings
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from news_fetcher import NewsFetcher
from analysis_synthesizer import AnalysisSynthesizer
from pipeline_visualizer import PipelineVisualizer

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


def get_memory_info():
    """Get current memory usage"""
    vm = psutil.virtual_memory()
    process = psutil.Process()
    
    return {
        'total_gb': round(vm.total / (1024**3), 2),
        'available_gb': round(vm.available / (1024**3), 2),
        'used_percent': vm.percent,
        'process_mb': round(process.memory_info().rss / (1024**2), 2)
    }


class ComprehensivePipeline:
    """
    Streamlined orchestrator for comprehensive market analysis
    Delegates to specialized modules for cleaner architecture
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
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)
        
        self.max_articles = max_articles
        
        mem_start = get_memory_info()
        
        print("="*80)
        print("COMPREHENSIVE ANALYSIS PIPELINE - STREAMLINED")
        print("="*80)
        print(f"System Memory: {mem_start['total_gb']}GB")
        print(f"Max Articles: {max_articles}")
        print(f"Visualizations: {'Enabled' if enable_viz else 'Disabled'}")
        print("="*80)
        
        self._validate_api_keys()
        
        print("\nInitializing modules...")
        print("-"*80)
        
        # Core modules
        self.news_fetcher = NewsFetcher(prefer_serp=True)
        self.news_analyzer = NewsImpactAnalyzer() if NewsImpactAnalyzer else None
        self.cot_fetcher = COTDataFetcher() if COTDataFetcher else None
        self.indicator_calc = SymbolIndicatorCalculator() if SymbolIndicatorCalculator else None
        self.corr_analyzer = CorrelationAnalyzer() if CorrelationAnalyzer else None
        self.econ_indicators = EconomicIndicatorIntegration() if EconomicIndicatorIntegration else None
        self.seasonality = SeasonalityAnalyzer() if SeasonalityAnalyzer else None
        self.market_structure = MarketStructureAnalyzer() if MarketStructureAnalyzer else None
        self.volume_analyzer = VolumeAnalyzer() if VolumeAnalyzer else None
        
        # Specialized modules
        self.synthesizer = AnalysisSynthesizer()
        self.visualizer = PipelineVisualizer(self.viz_dir) if enable_viz else None
        
        print("✓ Core modules initialized")
        
        # HF models
        self.hf_models = {}
        if enable_hf:
            self._load_hf_models()
        
        mem_end = get_memory_info()
        print(f"\n📊 Memory: {mem_end['process_mb']}MB process, {mem_end['used_percent']:.1f}% system")
        print("="*80 + "\n")
    
    def _validate_api_keys(self):
        """Validate API keys"""
        print("\nAPI Keys:")
        print("-"*80)
        
        fred_key = os.environ.get('FRED_API_KEY', '')
        serp_keys = [os.environ.get(f'SERP_API_KEY_{i}', '') for i in range(1, 4)]
        serp_keys = [k for k in serp_keys if k]
        
        if fred_key:
            print(f"  ✓ FRED_API_KEY")
        else:
            print(f"  ⊘ FRED_API_KEY: Not set")
        
        if serp_keys:
            print(f"  ✓ SERP_API_KEY: {len(serp_keys)} key(s)")
        else:
            print(f"  ⊘ SERP_API_KEY: Not set (will use fallback sources)")
        
        print("-"*80)
    
    def _load_hf_models(self):
        """Load HF models with memory awareness"""
        try:
            from hf_method1_sentiment import HFSentimentAnalyzer
            from hf_method2_ner import HFEntityExtractor
            
            print("\nLoading HF models...")
            
            try:
                sentiment = HFSentimentAnalyzer()
                sentiment.load_model()
                self.hf_models['sentiment'] = sentiment
                print("  ✓ Sentiment analyzer")
            except:
                print("  ⊘ Sentiment analyzer failed")
            
            try:
                ner = HFEntityExtractor()
                ner.load_model()
                self.hf_models['ner'] = ner
                print("  ✓ Entity extractor")
            except:
                print("  ⊘ Entity extractor failed")
        
        except ImportError:
            print("\n⊘ HF models not available")
    
    def analyze(self, date: str, event_name: str = None, symbols: List[str] = None) -> Dict:
        """
        Run comprehensive analysis
        
        Args:
            date: Analysis date (YYYY-MM-DD)
            event_name: Economic event name
            symbols: List of symbols (auto-detected if None)
            
        Returns:
            Complete analysis results dictionary
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
                'hf_models_loaded': list(self.hf_models.keys()),
                'max_articles': self.max_articles
            },
            'sections': {}
        }
        
        # 1. NEWS ANALYSIS
        print("\n📰 SECTION 1: NEWS")
        print("-"*80)
        news_results = self._fetch_news(date, event_name)
        results['sections']['news'] = news_results
        
        # Extract symbols
        if symbols is None:
            symbols = self._extract_symbols(news_results, event_name)
        
        results['metadata']['symbols_analyzed'] = symbols
        results['metadata']['symbols_count'] = len(symbols)
        
        # 2. COT POSITIONING
        print("\n📊 SECTION 2: COT")
        print("-"*80)
        results['sections']['cot'] = self._analyze_cot(date, symbols)
        
        # 3. TECHNICAL INDICATORS
        print("\n📈 SECTION 3: TECHNICAL")
        print("-"*80)
        results['sections']['indicators'] = self._analyze_indicators(date, symbols)
        
        # 4. CORRELATIONS
        print("\n🔗 SECTION 4: CORRELATIONS")
        print("-"*80)
        results['sections']['correlations'] = self._analyze_correlations(date, symbols[:5])
        
        # 5. ECONOMIC
        print("\n💹 SECTION 5: ECONOMIC")
        print("-"*80)
        results['sections']['economic'] = self._analyze_economic(date)
        
        # 6. STRUCTURE
        print("\n🏗️  SECTION 6: STRUCTURE")
        print("-"*80)
        results['sections']['structure'] = self._analyze_structure(date, symbols[:5])
        
        # 7. SEASONALITY
        print("\n📅 SECTION 7: SEASONALITY")
        print("-"*80)
        results['sections']['seasonality'] = self._analyze_seasonality(symbols[:3])
        
        # 8. VOLUME
        print("\n📊 SECTION 8: VOLUME")
        print("-"*80)
        results['sections']['volume'] = self._analyze_volume(date, symbols[:5])
        
        # 9. HF AI METHODS
        if self.hf_models:
            print("\n🤖 SECTION 9: HF AI")
            print("-"*80)
            results['sections']['hf_methods'] = self._run_hf_analysis(
                date, event_name, news_results, symbols
            )
        
        # 10. SYNTHESIS
        print("\n💡 SECTION 10: SYNTHESIS")
        print("-"*80)
        insights = self.synthesizer.synthesize(results)
        results['sections']['synthesis'] = insights
        print(f"  ✓ Generated {len(insights['key_findings'])} findings")
        print(f"  ✓ Overall confidence: {insights.get('overall_confidence', 0):.1%}")
        
        # 11. VISUALIZATIONS
        if self.visualizer:
            print("\n📊 SECTION 11: VISUALIZATIONS")
            print("-"*80)
            viz_results = self.visualizer.generate_all(results)
            results['sections']['visualizations'] = viz_results
            print(f"  ✓ Created {viz_results.get('charts_created', 0)} charts")
        
        # Save results
        report_file = self._save_results(results, date, event_name)
        results['report_file'] = report_file
        
        mem_final = get_memory_info()
        
        print("\n" + "="*80)
        print("✓ ANALYSIS COMPLETE")
        print("="*80)
        print(f"Report: {report_file}")
        print(f"Articles: {news_results.get('article_count', 0)}")
        print(f"Symbols: {len(symbols)}")
        print(f"Memory: {mem_final['process_mb']}MB")
        print("="*80 + "\n")
        
        return results
    
    def _fetch_news(self, date: str, event_name: str) -> Dict:
        """Fetch news articles"""
        articles = []
        
        try:
            if event_name:
                articles = self.news_fetcher.fetch_event_news(
                    date, event_name,
                    max_records=self.max_articles,
                    full_content=True
                )
            else:
                articles = self.news_fetcher.fetch_news(
                    date,
                    max_records=self.max_articles
                )
            
            print(f"  ✓ Fetched {len(articles)} articles")
        
        except Exception as e:
            print(f"  ✗ Error: {str(e)[:50]}")
        
        results = {
            'articles': articles,
            'article_count': len(articles),
            'impact_analysis': None
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
        
        return results
    
    def _extract_symbols(self, news_results: Dict, event_name: str) -> List[str]:
        """Extract symbols from news and event"""
        symbols = set()
        
        # From impact analysis
        if news_results.get('impact_analysis'):
            impact_symbols = news_results['impact_analysis'].get('symbols', {})
            symbols.update(impact_symbols.keys())
        
        # From articles
        for article in news_results.get('articles', []):
            text = article.get('title', '') + ' ' + article.get('content', '')
            article_symbols = self.news_fetcher.get_affected_symbols(
                event_name or '', text
            )
            symbols.update(article_symbols)
        
        # Default symbols if none found
        if not symbols:
            symbols = {'EURUSD=X', 'GC=F', '^GSPC', 'DX-Y.NYB', 'TLT'}
        
        symbol_list = list(symbols)[:15]
        print(f"  ✓ Extracted {len(symbol_list)} symbols")
        
        return symbol_list
    
    def _analyze_cot(self, date: str, symbols: List[str]) -> Dict:
        """Analyze COT positioning"""
        if not self.cot_fetcher:
            return {}
        
        results = {}
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
                        results[symbol] = positioning
                except:
                    pass
        
        print(f"  ✓ {len(results)} symbols")
        return results
    
    def _analyze_indicators(self, date: str, symbols: List[str]) -> Dict:
        """Analyze technical indicators"""
        if not self.indicator_calc:
            return {}
        
        results = {}
        for symbol in symbols:
            try:
                indicators = self.indicator_calc.get_indicators_for_date(symbol, date)
                if indicators:
                    results[symbol] = indicators
            except:
                pass
        
        print(f"  ✓ {len(results)} symbols")
        return results
    
    def _analyze_correlations(self, date: str, symbols: List[str]) -> Dict:
        """Analyze correlations"""
        if not self.corr_analyzer:
            return {}
        
        results = {}
        for symbol in symbols:
            try:
                corr = self.corr_analyzer.get_correlation_analysis(symbol, date, lookback_days=90)
                if corr:
                    results[symbol] = corr
            except:
                pass
        
        print(f"  ✓ {len(results)} symbols")
        return results
    
    def _analyze_economic(self, date: str) -> Dict:
        """Analyze economic indicators"""
        if not self.econ_indicators:
            print("  ⊘ Not available")
            return None
        
        try:
            snapshot = self.econ_indicators.get_economic_snapshot(date)
            if snapshot:
                status = snapshot.get('overall_economic_status', 'N/A')
                print(f"  ✓ {status}")
                return snapshot
        except Exception as e:
            print(f"  ✗ Error: {str(e)[:50]}")
        
        return None
    
    def _analyze_structure(self, date: str, symbols: List[str]) -> Dict:
        """Analyze market structure"""
        if not self.market_structure:
            return {}
        
        results = {}
        for symbol in symbols:
            try:
                structure = self.market_structure.get_market_structure(symbol, date)
                if structure:
                    results[symbol] = structure
            except:
                pass
        
        print(f"  ✓ {len(results)} symbols")
        return results
    
    def _analyze_seasonality(self, symbols: List[str]) -> Dict:
        """Analyze seasonality"""
        if not self.seasonality:
            return {}
        
        results = {}
        for symbol in symbols:
            try:
                season = self.seasonality.get_seasonality_analysis(symbol, years_back=5)
                if season:
                    results[symbol] = season
            except:
                pass
        
        print(f"  ✓ {len(results)} symbols")
        return results
    
    def _analyze_volume(self, date: str, symbols: List[str]) -> Dict:
        """Analyze volume"""
        if not self.volume_analyzer:
            return {}
        
        results = {}
        for symbol in symbols:
            try:
                volume = self.volume_analyzer.get_volume_analysis(symbol, date)
                if volume:
                    results[symbol] = volume
            except:
                pass
        
        print(f"  ✓ {len(results)} symbols")
        return results
    
    def _run_hf_analysis(self, date: str, event_name: str, news_results: Dict, symbols: List[str]) -> Dict:
        """Run HF AI analysis"""
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
                print(f"  ⊘ Sentiment: {str(e)[:40]}")
        
        # NER
        if 'ner' in self.hf_models and articles:
            try:
                extractor = self.hf_models['ner']
                entities = extractor.analyze_batch(articles)
                aggregated = extractor.aggregate_symbols(entities)
                hf_results['entities'] = {'analyzed': entities, 'aggregated': aggregated}
                print(f"  ✓ NER: {aggregated.get('symbol_count', 0)} symbols")
            except Exception as e:
                print(f"  ⊘ NER: {str(e)[:40]}")
        
        return hf_results
    
    def _save_results(self, results: Dict, date: str, event_name: str) -> str:
        """Save results to file"""
        date_clean = date.replace('-', '_')
        event_clean = (event_name or 'analysis').replace(' ', '_').lower()
        
        filename = f'{self.output_dir}/comprehensive_{date_clean}_{event_clean}.json'
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # Also save markdown
        md_content = self.synthesizer.generate_markdown_summary(
            results['sections'].get('synthesis', {}),
            results['metadata']
        )
        
        md_file = filename.replace('.json', '.md')
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        return filename


if __name__ == "__main__":
    pipeline = ComprehensivePipeline(
        output_dir='test_pipeline_output',
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
