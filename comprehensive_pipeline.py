"""
Comprehensive Analysis Pipeline - FIXED VERSION
- Better entity extraction
- Improved data parsing
- Visualization support
- Enhanced error handling
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
    """Memory-optimized comprehensive pipeline with better parsing"""
    
    def __init__(self, output_dir='pipeline_output', enable_hf=True, enable_viz=True):
        self.output_dir = output_dir
        self.viz_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)
        
        self.enable_viz = enable_viz
        
        # Get initial memory state
        mem_start = get_memory_info()
        
        print("="*80)
        print("COMPREHENSIVE ANALYSIS PIPELINE - FIXED VERSION")
        print("="*80)
        print(f"System Memory: {mem_start['total_gb']}GB total, {mem_start['available_gb']}GB available")
        print(f"Target Usage: 80% ({mem_start['total_gb'] * 0.80:.1f}GB max)")
        print(f"Visualizations: {'Enabled' if enable_viz else 'Disabled'}")
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
        
        # Extract symbols from multiple sources
        extracted_symbols = set()
        
        # From news impact
        if news_results.get('extracted_symbols'):
            extracted_symbols.update(news_results['extracted_symbols'])
        
        # From news articles (parse titles and content)
        for article in news_results.get('articles', []):
            symbols_from_article = self._extract_symbols_from_text(
                article.get('title', '') + ' ' + article.get('content', '')
            )
            extracted_symbols.update(symbols_from_article)
        
        if symbols is None and extracted_symbols:
            symbols = list(extracted_symbols)[:15]
            print(f"✓ Auto-detected {len(symbols)} symbols: {', '.join(symbols[:5])}...")
        elif symbols is None:
            symbols = ['EURUSD=X', 'GC=F', '^GSPC', 'DX-Y.NYB', 'TLT']
            print(f"✓ Using default symbols: {', '.join(symbols)}")
        
        results['metadata']['symbols_analyzed'] = symbols
        results['metadata']['symbols_count'] = len(symbols)
        
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
        
        # 15. VISUALIZATIONS
        if self.enable_viz:
            print("\n📊 SECTION 15: VISUALIZATIONS")
            print("-" * 80)
            viz_results = self._generate_visualizations(results)
            results['sections']['visualizations'] = viz_results
        
        # Save
        report_file = self._save_results(results, date, event_name)
        results['report_file'] = report_file
        
        mem_final = get_memory_info()
        
        print("\n" + "="*80)
        print("✓ COMPREHENSIVE ANALYSIS COMPLETE")
        print("="*80)
        print(f"Report: {report_file}")
        print(f"Symbols: {len(symbols)} analyzed")
        print(f"Memory: {mem_final['process_mb']}MB ({mem_final['used_percent']:.1f}% system)")
        if self.enable_viz and viz_results:
            print(f"Visualizations: {viz_results.get('charts_created', 0)} charts created")
        print("="*80 + "\n")
        
        return results
    
    def _extract_symbols_from_text(self, text):
        """Extract financial symbols from text"""
        symbols = set()
        
        # Common forex pairs
        forex_patterns = [
            'EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'AUD/USD', 'USD/CAD', 'NZD/USD',
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD'
        ]
        
        # Convert to Yahoo Finance format
        symbol_map = {
            'EUR/USD': 'EURUSD=X', 'EURUSD': 'EURUSD=X',
            'GBP/USD': 'GBPUSD=X', 'GBPUSD': 'GBPUSD=X',
            'USD/JPY': 'USDJPY=X', 'USDJPY': 'USDJPY=X',
            'USD/CHF': 'USDCHF=X', 'USDCHF': 'USDCHF=X',
            'AUD/USD': 'AUDUSD=X', 'AUDUSD': 'AUDUSD=X',
            'USD/CAD': 'USDCAD=X', 'USDCAD': 'USDCAD=X',
            'NZD/USD': 'NZDUSD=X', 'NZDUSD': 'NZDUSD=X',
            'GOLD': 'GC=F',
            'SILVER': 'SI=F',
            'OIL': 'CL=F',
            'S&P 500': '^GSPC',
            'DOW': '^DJI',
            'NASDAQ': '^IXIC',
            'DOLLAR INDEX': 'DX-Y.NYB',
            'DXY': 'DX-Y.NYB'
        }
        
        text_upper = text.upper()
        
        for pattern, symbol in symbol_map.items():
            if pattern.upper() in text_upper:
                symbols.add(symbol)
        
        return list(symbols)
    
    def _analyze_news(self, date, event_name):
        """News analysis with better symbol extraction"""
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
            status = snapshot.get('overall_economic_status', 'N/A')
            print(f"  ✓ Economic: {status}")
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
        """Run HF AI methods with better entity extraction"""
        hf_results = {}
        articles = news_results.get('articles', [])
        
        # Sentiment
        if 'sentiment' in self.hf_models and articles:
            try:
                analyzer = self.hf_models['sentiment']
                sentiment = analyzer.analyze_news_articles(articles)
                aggregated = analyzer.aggregate_sentiment(sentiment)
                hf_results['sentiment'] = {'articles': sentiment, 'aggregated': aggregated}
                
                overall = aggregated.get('overall_sentiment', 'N/A').upper()
                confidence = aggregated.get('confidence', 0)
                print(f"  ✓ Sentiment: {overall} ({confidence:.1%} confidence)")
            except Exception as e:
                print(f"  ⊘ Sentiment: {str(e)[:50]}")
        
        # NER - Enhanced extraction
        if 'ner' in self.hf_models and articles:
            try:
                extractor = self.hf_models['ner']
                entities = extractor.analyze_batch(articles)
                
                # Manual symbol extraction as fallback
                manual_symbols = set()
                for article in articles:
                    text = article.get('title', '') + ' ' + article.get('content', '')
                    manual_symbols.update(self._extract_symbols_from_text(text))
                
                # Combine NER and manual extraction
                aggregated = extractor.aggregate_symbols(entities)
                
                if manual_symbols:
                    if 'symbols' not in aggregated:
                        aggregated['symbols'] = {}
                    for sym in manual_symbols:
                        if sym not in aggregated['symbols']:
                            aggregated['symbols'][sym] = {'count': 1, 'confidence': 0.7}
                
                aggregated['symbol_count'] = len(aggregated.get('symbols', {}))
                
                hf_results['entities'] = {
                    'analyzed': entities,
                    'aggregated': aggregated,
                    'manual_extraction': list(manual_symbols)
                }
                
                symbol_count = aggregated.get('symbol_count', 0)
                print(f"  ✓ NER: {symbol_count} symbols extracted")
                
                if manual_symbols:
                    print(f"     Manual extraction: {', '.join(list(manual_symbols)[:5])}")
                
            except Exception as e:
                print(f"  ⊘ NER: {str(e)[:50]}")
        
        # Classification
        if 'classification' in self.hf_models and event_name:
            try:
                classifier = self.hf_models['classification']
                classification = classifier.classify_event(event_name)
                hf_results['classification'] = classification
                
                impact = classification.get('impact_level', 'N/A')
                category = classification.get('category', 'N/A')
                print(f"  ✓ Classification: {impact} impact, {category} category")
            except Exception as e:
                print(f"  ⊘ Classification: {str(e)[:50]}")
        
        # Q&A
        if 'qa' in self.hf_models and articles:
            try:
                qa = self.hf_models['qa']
                
                # Index news impact
                if news_results.get('impact_analysis'):
                    qa.index_news_impact([news_results['impact_analysis']])
                
                # Ask relevant questions
                questions = [
                    f"What happened on {date}?",
                    f"What was the market sentiment regarding {event_name or 'the event'}?",
                    "Which symbols were most affected?",
                    "What was the overall impact on financial markets?"
                ]
                
                qa_results = qa.batch_ask(questions)
                hf_results['qa'] = qa_results
                
                print(f"  ✓ Q&A: {len(qa_results)} questions answered")
            except Exception as e:
                print(f"  ⊘ Q&A: {str(e)[:50]}")
        
        return hf_results
    
    def _synthesize(self, results):
        """Synthesize insights with better formatting"""
        insights = {
            'summary': [],
            'key_findings': [],
            'signals': [],
            'risks': [],
            'opportunities': []
        }
        
        meta = results['metadata']
        insights['summary'].append(
            f"Analyzed {meta['event_name'] or 'market'} on {meta['date']}"
        )
        insights['summary'].append(
            f"Processed {len(meta.get('symbols_analyzed', []))} symbols using {len(meta.get('hf_models_loaded', []))} AI models"
        )
        
        # HF sentiment
        hf = results['sections'].get('hf_methods', {})
        if hf.get('sentiment'):
            sent = hf['sentiment']['aggregated']
            sentiment = sent.get('overall_sentiment', 'N/A').upper()
            confidence = sent.get('confidence', 0)
            
            insights['key_findings'].append(
                f"AI Sentiment: {sentiment} ({confidence:.1%} confidence)"
            )
            
            # Add risk/opportunity based on sentiment
            if sentiment == 'NEGATIVE':
                insights['risks'].append("Negative market sentiment detected")
            elif sentiment == 'POSITIVE':
                insights['opportunities'].append("Positive market sentiment detected")
        
        # Entity extraction
        if hf.get('entities'):
            symbol_count = hf['entities']['aggregated'].get('symbol_count', 0)
            if symbol_count > 0:
                insights['key_findings'].append(
                    f"AI extracted {symbol_count} financial instruments from news"
                )
        
        # Technical signals
        indicators = results['sections'].get('indicators', {})
        if indicators:
            buy_count = sum(1 for ind in indicators.values() if ind.get('overall_signal') == 'BUY')
            sell_count = sum(1 for ind in indicators.values() if ind.get('overall_signal') == 'SELL')
            neutral_count = len(indicators) - buy_count - sell_count
            
            if buy_count > sell_count:
                insights['signals'].append(f"Technical: BULLISH ({buy_count} BUY, {sell_count} SELL, {neutral_count} NEUTRAL)")
                insights['opportunities'].append(f"{buy_count} symbols showing bullish technical signals")
            elif sell_count > buy_count:
                insights['signals'].append(f"Technical: BEARISH ({sell_count} SELL, {buy_count} BUY, {neutral_count} NEUTRAL)")
                insights['risks'].append(f"{sell_count} symbols showing bearish technical signals")
            else:
                insights['signals'].append(f"Technical: NEUTRAL ({neutral_count} symbols)")
        
        # Economic status
        econ = results['sections'].get('economic', {})
        if econ:
            status = econ.get('overall_economic_status', 'N/A')
            insights['key_findings'].append(f"Economic Status: {status}")
            
            if status == 'RECESSION_WARNING':
                insights['risks'].append("Economic indicators show recession warning signs")
            elif status == 'STRONG_GROWTH':
                insights['opportunities'].append("Economic indicators show strong growth")
        
        # Market structure
        structure = results['sections'].get('structure', {})
        if structure:
            trending_count = sum(1 for s in structure.values() if s.get('trend') == 'TRENDING')
            ranging_count = sum(1 for s in structure.values() if s.get('trend') == 'RANGING')
            
            if trending_count > 0:
                insights['key_findings'].append(f"Market Structure: {trending_count} trending, {ranging_count} ranging")
        
        # Volume analysis
        volume = results['sections'].get('volume', {})
        if volume:
            high_volume_count = sum(1 for v in volume.values() if v.get('volume_status') == 'HIGH')
            if high_volume_count > 0:
                insights['key_findings'].append(f"Volume: {high_volume_count} symbols with elevated activity")
        
        return insights
    
    def _generate_visualizations(self, results):
        """Generate visualizations from analysis results"""
        viz_results = {'charts_created': 0, 'chart_files': []}
        
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            sns.set_style("whitegrid")
            
            # 1. Sentiment Distribution (if available)
            hf = results['sections'].get('hf_methods', {})
            if hf.get('sentiment'):
                sent = hf['sentiment']['aggregated']
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                sentiments = ['Positive', 'Negative', 'Neutral']
                values = [
                    sent.get('positive_count', 0),
                    sent.get('negative_count', 0),
                    sent.get('neutral_count', 0)
                ]
                
                colors = ['#2ecc71', '#e74c3c', '#95a5a6']
                ax.bar(sentiments, values, color=colors)
                ax.set_title(f'News Sentiment Distribution\n{results["metadata"]["event_name"] or "Market Analysis"}')
                ax.set_ylabel('Number of Articles')
                
                chart_file = os.path.join(self.viz_dir, 'sentiment_distribution.png')
                plt.tight_layout()
                plt.savefig(chart_file, dpi=150, bbox_inches='tight')
                plt.close()
                
                viz_results['charts_created'] += 1
                viz_results['chart_files'].append(chart_file)
                print(f"  ✓ Created: sentiment_distribution.png")
            
            # 2. Technical Signals Summary
            indicators = results['sections'].get('indicators', {})
            if indicators:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                symbols = list(indicators.keys())[:10]  # Top 10
                signals = [indicators[s].get('overall_signal', 'NEUTRAL') for s in symbols]
                
                signal_colors = {'BUY': '#2ecc71', 'SELL': '#e74c3c', 'NEUTRAL': '#95a5a6'}
                colors = [signal_colors.get(s, '#95a5a6') for s in signals]
                
                ax.barh(symbols, [1]*len(symbols), color=colors)
                ax.set_xlabel('Signal')
                ax.set_title('Technical Signals by Symbol')
                ax.set_xlim(0, 1)
                ax.set_xticks([])
                
                # Add legend
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='#2ecc71', label='BUY'),
                    Patch(facecolor='#e74c3c', label='SELL'),
                    Patch(facecolor='#95a5a6', label='NEUTRAL')
                ]
                ax.legend(handles=legend_elements, loc='lower right')
                
                chart_file = os.path.join(self.viz_dir, 'technical_signals.png')
                plt.tight_layout()
                plt.savefig(chart_file, dpi=150, bbox_inches='tight')
                plt.close()
                
                viz_results['charts_created'] += 1
                viz_results['chart_files'].append(chart_file)
                print(f"  ✓ Created: technical_signals.png")
            
            # 3. Economic Status
            econ = results['sections'].get('economic', {})
            if econ and econ.get('indicator_statuses'):
                fig, ax = plt.subplots(figsize=(10, 8))
                
                indicators = list(econ['indicator_statuses'].items())
                names = [i[0].replace('_', ' ').title() for i in indicators]
                statuses = [i[1] for i in indicators]
                
                status_colors = {
                    'STRONG_GROWTH': '#2ecc71',
                    'MODERATE': '#f39c12',
                    'WEAK': '#e74c3c',
                    'RECESSION_WARNING': '#c0392b',
                    'NORMAL': '#95a5a6'
                }
                colors = [status_colors.get(s, '#95a5a6') for s in statuses]
                
                ax.barh(names, [1]*len(names), color=colors)
                ax.set_xlabel('Status')
                ax.set_title('Economic Indicators Status')
                ax.set_xlim(0, 1)
                ax.set_xticks([])
                
                chart_file = os.path.join(self.viz_dir, 'economic_status.png')
                plt.tight_layout()
                plt.savefig(chart_file, dpi=150, bbox_inches='tight')
                plt.close()
                
                viz_results['charts_created'] += 1
                viz_results['chart_files'].append(chart_file)
                print(f"  ✓ Created: economic_status.png")
            
            print(f"\n  ✓ Total: {viz_results['charts_created']} visualizations created")
            
        except ImportError:
            print("  ⊘ Matplotlib not available - skipping visualizations")
        except Exception as e:
            print(f"  ⊘ Visualization error: {str(e)[:50]}")
        
        return viz_results
    
    def _save_results(self, results, date, event_name):
        """Save results"""
        date_clean = date.replace('-', '_')
        event_clean = (event_name or 'analysis').replace(' ', '_').lower()
        
        filename = f'{self.output_dir}/comprehensive_{date_clean}_{event_clean}.json'
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        return filename
    
    def generate_markdown_report(self, results):
        """Generate enhanced markdown report"""
        lines = [
            "# COMPREHENSIVE MARKET ANALYSIS",
            "",
            f"**Date:** {results['metadata']['date']}",
            f"**Event:** {results['metadata']['event_name'] or 'Market Analysis'}",
            f"**Symbols Analyzed:** {results['metadata'].get('symbols_count', 0)}",
            f"**AI Models Used:** {', '.join(results['metadata'].get('hf_models_loaded', ['None']))}",
            f"**Generated:** {results['metadata']['timestamp']}",
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
        
        if synthesis.get('opportunities'):
            lines.extend(["", "## Opportunities", ""])
            for opp in synthesis['opportunities']:
                lines.append(f"- 💡 {opp}")
        
        if synthesis.get('risks'):
            lines.extend(["", "## Risk Factors", ""])
            for risk in synthesis['risks']:
                lines.append(f"- ⚠️ {risk}")
        
        # Add visualizations if available
        viz = results['sections'].get('visualizations', {})
        if viz and viz.get('charts_created', 0) > 0:
            lines.extend(["", "## Visualizations", ""])
            lines.append(f"Generated {viz['charts_created']} charts in `visualizations/` directory")
        
        # Add symbols analyzed
        symbols = results['metadata'].get('symbols_analyzed', [])
        if symbols:
            lines.extend(["", "## Symbols Analyzed", ""])
            lines.append(", ".join(symbols))
        
        lines.extend(["", "---", "", "*Generated by Comprehensive Analysis Pipeline (Fixed Version)*"])
        
        return '\n'.join(lines)


if __name__ == "__main__":
    print("="*80)
    print("COMPREHENSIVE PIPELINE - FIXED VERSION TEST")
    print("="*80)
    
    # Enable HF models and visualizations
    pipeline = ComprehensiveAnalysisPipeline(
        output_dir='test_pipeline_output',
        enable_hf=True,
        enable_viz=True
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
