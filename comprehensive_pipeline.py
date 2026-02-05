"""
Comprehensive Analysis Pipeline
Combines ALL data sources for a given date/event with symbols
Includes: News, COT, Indicators, Correlations, HF Methods (1-10), Visualizations
Generates complete report with images
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

# Import all analysis modules
from news_fetcher import NewsFetcher
from news_impact_analyzer import NewsImpactAnalyzer
from cot_data_fetcher import COTDataFetcher
from symbol_indicators import SymbolIndicatorCalculator
from correlation_analyzer import CorrelationAnalyzer
from economic_indicators import EconomicIndicatorIntegration
from seasonality_analyzer import SeasonalityAnalyzer
from market_structure_analyzer import MarketStructureAnalyzer
from volume_analyzer import VolumeAnalyzer

# Import HF methods
from hf_method1_sentiment import HFSentimentAnalyzer
from hf_method2_ner import HFEntityExtractor
from hf_method3_forecasting import HFTimeSeriesForecaster
from hf_method4_classification import HFEventClassifier
from hf_method5_embeddings import HFCorrelationDiscovery
from hf_method6_qa import HFMarketQA
from hf_method7_anomaly import HFAnomalyDetector
from hf_method8_multimodal import HFMultiModalAnalyzer
from hf_method9_zeroshot import HFZeroShotCategorizer
from hf_method10_causal import HFCausalExplainer

# Import visualization
from visualization_wrapper import VisualizationWrapper


class ComprehensiveAnalysisPipeline:
    """
    Master pipeline combining all analysis methods
    Input: Date, Event Name, Symbols (optional)
    Output: Complete report + visualizations
    """
    
    def __init__(self, output_dir='pipeline_output'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print("="*80)
        print("COMPREHENSIVE ANALYSIS PIPELINE")
        print("="*80)
        print("Initializing all modules...")
        
        # Core modules
        self.news_fetcher = NewsFetcher()
        self.news_analyzer = NewsImpactAnalyzer()
        self.cot_fetcher = COTDataFetcher()
        self.indicator_calc = SymbolIndicatorCalculator()
        self.corr_analyzer = CorrelationAnalyzer()
        self.econ_indicators = EconomicIndicatorIntegration()
        self.seasonality = SeasonalityAnalyzer()
        self.market_structure = MarketStructureAnalyzer()
        self.volume_analyzer = VolumeAnalyzer()
        
        # HF modules
        self.hf_sentiment = HFSentimentAnalyzer()
        self.hf_ner = HFEntityExtractor()
        self.hf_forecast = HFTimeSeriesForecaster()
        self.hf_classifier = HFEventClassifier()
        self.hf_correlation = HFCorrelationDiscovery()
        self.hf_qa = HFMarketQA()
        self.hf_anomaly = HFAnomalyDetector()
        self.hf_multimodal = HFMultiModalAnalyzer()
        self.hf_zeroshot = HFZeroShotCategorizer()
        self.hf_causal = HFCausalExplainer()
        
        # Visualization
        self.viz = VisualizationWrapper(output_dir=f'{output_dir}/visualizations')
        
        # Load HF models
        self._load_hf_models()
        
        print("✓ All modules initialized\n")
    
    def _load_hf_models(self):
        """Load all HF models"""
        print("Loading HF models...")
        
        try:
            self.hf_sentiment.load_model()
            print("  ✓ Sentiment")
        except:
            print("  ⊘ Sentiment (fallback mode)")
        
        try:
            self.hf_ner.load_model()
            print("  ✓ NER")
        except:
            print("  ⊘ NER (fallback mode)")
        
        try:
            self.hf_forecast.load_model()
            print("  ✓ Forecasting")
        except:
            print("  ⊘ Forecasting (statistical mode)")
        
        try:
            self.hf_classifier.load_model()
            print("  ✓ Classification")
        except:
            print("  ⊘ Classification (fallback mode)")
        
        try:
            self.hf_correlation.load_model()
            print("  ✓ Correlation Discovery")
        except:
            print("  ⊘ Correlation Discovery (unavailable)")
        
        try:
            self.hf_qa.load_model()
            print("  ✓ Q&A")
        except:
            print("  ⊘ Q&A (fallback mode)")
        
        try:
            self.hf_multimodal.load_models()
            print("  ✓ Multi-Modal")
        except:
            print("  ⊘ Multi-Modal (text only)")
        
        try:
            self.hf_zeroshot.load_model()
            print("  ✓ Zero-Shot")
        except:
            print("  ⊘ Zero-Shot (fallback mode)")
        
        try:
            self.hf_causal.load_model()
            print("  ✓ Causal Explainer")
        except:
            print("  ⊘ Causal Explainer (rule-based)")
        
        print()
    
    def analyze(self, date, event_name=None, symbols=None, 
                use_hf_methods=True, generate_visuals=True):
        """
        Run comprehensive analysis for given date/event
        
        Args:
            date: Date in 'YYYY-MM-DD' format
            event_name: Economic event name (e.g., 'Non-Farm Payrolls')
            symbols: List of symbols to analyze (auto-detected if None)
            use_hf_methods: Run all 10 HF methods (default True)
            generate_visuals: Generate visualizations (default True)
        
        Returns:
            Complete analysis results
        """
        
        print("="*80)
        print(f"COMPREHENSIVE ANALYSIS: {event_name or 'Market Analysis'}")
        print(f"Date: {date}")
        print("="*80 + "\n")
        
        results = {
            'metadata': {
                'date': date,
                'event_name': event_name,
                'symbols': symbols or 'auto-detected',
                'timestamp': datetime.now().isoformat(),
                'hf_methods_used': use_hf_methods,
                'visualizations_generated': generate_visuals
            },
            'sections': {}
        }
        
        # 1. NEWS ANALYSIS
        print("📰 SECTION 1: NEWS ANALYSIS")
        print("-" * 80)
        news_results = self._analyze_news(date, event_name)
        results['sections']['news'] = news_results
        
        # Extract symbols from news if not provided
        if symbols is None and news_results.get('extracted_symbols'):
            symbols = news_results['extracted_symbols'][:15]
            print(f"✓ Auto-detected symbols: {', '.join(symbols[:5])}...")
        elif symbols is None:
            # Default symbols
            symbols = ['EURUSD=X', 'GC=F', '^GSPC', 'DX-Y.NYB', 'TLT']
            print(f"Using default symbols: {', '.join(symbols)}")
        
        results['metadata']['symbols_analyzed'] = symbols
        
        # 2. COT POSITIONING
        print("\n📊 SECTION 2: COT POSITIONING")
        print("-" * 80)
        cot_results = self._analyze_cot(date, symbols)
        results['sections']['cot'] = cot_results
        
        # 3. TECHNICAL INDICATORS
        print("\n📈 SECTION 3: TECHNICAL INDICATORS")
        print("-" * 80)
        indicator_results = self._analyze_indicators(date, symbols)
        results['sections']['indicators'] = indicator_results
        
        # 4. CORRELATION ANALYSIS
        print("\n🔗 SECTION 4: CORRELATION ANALYSIS")
        print("-" * 80)
        correlation_results = self._analyze_correlations(date, symbols)
        results['sections']['correlations'] = correlation_results
        
        # 5. ECONOMIC INDICATORS
        print("\n💹 SECTION 5: ECONOMIC INDICATORS")
        print("-" * 80)
        econ_results = self._analyze_economic_indicators(date)
        results['sections']['economic_indicators'] = econ_results
        
        # 6. MARKET STRUCTURE
        print("\n🏗️  SECTION 6: MARKET STRUCTURE")
        print("-" * 80)
        structure_results = self._analyze_market_structure(date, symbols)
        results['sections']['market_structure'] = structure_results
        
        # 7. SEASONALITY
        print("\n📅 SECTION 7: SEASONALITY")
        print("-" * 80)
        seasonality_results = self._analyze_seasonality(date, symbols)
        results['sections']['seasonality'] = seasonality_results
        
        # 8. VOLUME ANALYSIS
        print("\n📊 SECTION 8: VOLUME ANALYSIS")
        print("-" * 80)
        volume_results = self._analyze_volume(date, symbols)
        results['sections']['volume'] = volume_results
        
        # HF METHODS (9-18)
        if use_hf_methods:
            print("\n🤖 SECTIONS 9-18: HF AI METHODS")
            print("-" * 80)
            hf_results = self._run_all_hf_methods(
                date, event_name, news_results, symbols, indicator_results
            )
            results['sections']['hf_methods'] = hf_results
        
        # 19. SYNTHESIS & INSIGHTS
        print("\n💡 SECTION 19: SYNTHESIS & INSIGHTS")
        print("-" * 80)
        synthesis = self._synthesize_insights(results)
        results['sections']['synthesis'] = synthesis
        
        # VISUALIZATIONS
        if generate_visuals:
            print("\n📊 GENERATING VISUALIZATIONS")
            print("-" * 80)
            viz_files = self._generate_visualizations(results, date, event_name)
            results['visualizations'] = viz_files
        
        # Save results
        report_file = self._save_results(results, date, event_name)
        results['report_file'] = report_file
        
        print("\n" + "="*80)
        print("✓ COMPREHENSIVE ANALYSIS COMPLETE")
        print("="*80)
        print(f"Report: {report_file}")
        if generate_visuals:
            print(f"Visualizations: {len(results['visualizations'])} images")
        print("="*80 + "\n")
        
        return results
    
    def _analyze_news(self, date, event_name):
        """Fetch and analyze news"""
        results = {
            'articles': [],
            'impact_analysis': None,
            'extracted_symbols': []
        }
        
        # Fetch news
        if event_name:
            articles = self.news_fetcher.fetch_event_news(date, event_name, max_records=10)
        else:
            articles = self.news_fetcher.fetch_news(date, max_records=10)
        
        results['articles'] = articles
        print(f"  ✓ Fetched {len(articles)} articles")
        
        # Analyze impact
        if event_name and articles:
            impact = self.news_analyzer.analyze_event_impact(
                date, event_name, articles, comparison_days=3
            )
            results['impact_analysis'] = impact
            
            if impact and impact.get('symbols'):
                results['extracted_symbols'] = list(impact['symbols'].keys())
                print(f"  ✓ Analyzed impact on {len(impact['symbols'])} symbols")
        
        return results
    
    def _analyze_cot(self, date, symbols):
        """Analyze COT positioning"""
        results = {}
        
        # Map symbols to COT symbols
        symbol_map = {
            'EURUSD=X': 'EUR', 'GBPUSD=X': 'GBP', 'USDJPY=X': 'JPY',
            'AUDUSD=X': 'AUD', 'GC=F': 'GOLD', 'GOLD': 'GOLD',
            'SI=F': 'SILVER', 'CL=F': 'CRUDE_OIL'
        }
        
        for symbol in symbols[:10]:
            cot_symbol = symbol_map.get(symbol)
            if cot_symbol:
                try:
                    positioning = self.cot_fetcher.get_positioning_for_date(cot_symbol, date)
                    if positioning:
                        results[symbol] = positioning
                except Exception as e:
                    print(f"  ⊘ {symbol}: {str(e)[:50]}")
        
        print(f"  ✓ COT data for {len(results)} symbols")
        return results
    
    def _analyze_indicators(self, date, symbols):
        """Analyze technical indicators"""
        results = {}
        
        for symbol in symbols:
            try:
                indicators = self.indicator_calc.get_indicators_for_date(symbol, date)
                if indicators:
                    results[symbol] = indicators
            except Exception as e:
                print(f"  ⊘ {symbol}: {str(e)[:50]}")
        
        print(f"  ✓ Indicators for {len(results)} symbols")
        return results
    
    def _analyze_correlations(self, date, symbols):
        """Analyze correlations"""
        results = {}
        
        for symbol in symbols[:5]:  # Top 5 to save time
            try:
                corr = self.corr_analyzer.get_correlation_analysis(symbol, date, lookback_days=90)
                if corr:
                    results[symbol] = corr
            except Exception as e:
                print(f"  ⊘ {symbol}: {str(e)[:50]}")
        
        print(f"  ✓ Correlations for {len(results)} symbols")
        return results
    
    def _analyze_economic_indicators(self, date):
        """Analyze economic indicators"""
        try:
            snapshot = self.econ_indicators.get_economic_snapshot(date)
            print(f"  ✓ Economic snapshot: {snapshot['overall_economic_status']}")
            return snapshot
        except Exception as e:
            print(f"  ⊘ Error: {str(e)[:50]}")
            return None
    
    def _analyze_market_structure(self, date, symbols):
        """Analyze market structure"""
        results = {}
        
        for symbol in symbols[:5]:
            try:
                structure = self.market_structure.get_market_structure(symbol, date)
                if structure:
                    results[symbol] = structure
            except Exception as e:
                print(f"  ⊘ {symbol}: {str(e)[:50]}")
        
        print(f"  ✓ Market structure for {len(results)} symbols")
        return results
    
    def _analyze_seasonality(self, date, symbols):
        """Analyze seasonality"""
        results = {}
        
        for symbol in symbols[:5]:
            try:
                seasonality = self.seasonality.get_seasonality_analysis(symbol, years_back=5)
                if seasonality:
                    results[symbol] = seasonality
            except Exception as e:
                print(f"  ⊘ {symbol}: {str(e)[:50]}")
        
        print(f"  ✓ Seasonality for {len(results)} symbols")
        return results
    
    def _analyze_volume(self, date, symbols):
        """Analyze volume"""
        results = {}
        
        for symbol in symbols[:5]:
            try:
                volume = self.volume_analyzer.get_volume_analysis(symbol, date)
                if volume:
                    results[symbol] = volume
            except Exception as e:
                print(f"  ⊘ {symbol}: {str(e)[:50]}")
        
        print(f"  ✓ Volume analysis for {len(results)} symbols")
        return results
    
    def _run_all_hf_methods(self, date, event_name, news_results, symbols, indicator_results):
        """Run all 10 HF methods"""
        hf_results = {}
        
        articles = news_results.get('articles', [])
        
        # Method 1: Sentiment
        print("  Running HF Method 1: Sentiment Analysis...")
        try:
            if articles:
                sentiment = self.hf_sentiment.analyze_news_articles(articles)
                aggregated = self.hf_sentiment.aggregate_sentiment(sentiment)
                hf_results['sentiment'] = {
                    'articles': sentiment,
                    'aggregated': aggregated
                }
                print(f"    ✓ Sentiment: {aggregated.get('overall_sentiment', 'N/A').upper()}")
        except Exception as e:
            print(f"    ⊘ Error: {str(e)[:50]}")
        
        # Method 2: NER
        print("  Running HF Method 2: Named Entity Recognition...")
        try:
            if articles:
                entities = self.hf_ner.analyze_batch(articles)
                aggregated = self.hf_ner.aggregate_symbols(entities)
                hf_results['entities'] = {
                    'analyzed': entities,
                    'aggregated': aggregated
                }
                print(f"    ✓ Entities: {aggregated.get('symbol_count', 0)} symbols")
        except Exception as e:
            print(f"    ⊘ Error: {str(e)[:50]}")
        
        # Method 3: Forecasting
        print("  Running HF Method 3: Time Series Forecasting...")
        hf_results['forecasts'] = {}
        for symbol in symbols[:3]:
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                prices = ticker.history(period='3mo')['Close']
                
                if len(prices) > 20:
                    forecast = self.hf_forecast.forecast(prices, forecast_horizon=7)
                    hf_results['forecasts'][symbol] = forecast
            except Exception as e:
                pass
        print(f"    ✓ Forecasts: {len(hf_results['forecasts'])} symbols")
        
        # Method 4: Classification
        print("  Running HF Method 4: Event Classification...")
        try:
            if event_name:
                classification = self.hf_classifier.classify_event(event_name)
                hf_results['classification'] = classification
                print(f"    ✓ Classification: {classification.get('impact_level', 'N/A')}")
        except Exception as e:
            print(f"    ⊘ Error: {str(e)[:50]}")
        
        # Method 5: Correlation Discovery
        print("  Running HF Method 5: Correlation Discovery...")
        try:
            if articles:
                events = [{'event': a.get('title', ''), 'date': date} for a in articles[:10]]
                if events:
                    correlations = self.hf_correlation.discover_hidden_correlations(events)
                    hf_results['correlation_discovery'] = correlations
                    print(f"    ✓ Discovered {correlations.get('semantic_clusters', 0)} clusters")
        except Exception as e:
            print(f"    ⊘ Error: {str(e)[:50]}")
        
        # Method 6: Q&A
        print("  Running HF Method 6: Market Q&A...")
        try:
            if articles and indicator_results:
                # Index data
                self.hf_qa.index_news_impact([news_results.get('impact_analysis')])
                
                questions = [
                    f"What happened on {date}?",
                    f"What was the market sentiment?",
                    f"Which symbols were most affected?"
                ]
                
                qa_results = self.hf_qa.batch_ask(questions[:2])
                hf_results['qa'] = qa_results
                print(f"    ✓ Answered {len(qa_results)} questions")
        except Exception as e:
            print(f"    ⊘ Error: {str(e)[:50]}")
        
        # Method 7: Anomaly Detection
        print("  Running HF Method 7: Anomaly Detection...")
        hf_results['anomalies'] = {}
        for symbol in symbols[:3]:
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                df = ticker.history(period='3mo')
                
                if len(df) > 20:
                    df = df.rename(columns={
                        'Close': 'close', 'Open': 'open', 
                        'High': 'high', 'Low': 'low', 'Volume': 'volume'
                    })
                    
                    current = df.iloc[-1]
                    scan = self.hf_anomaly.comprehensive_anomaly_scan(current, df, symbol)
                    
                    if scan.get('anomalies_detected', 0) > 0:
                        hf_results['anomalies'][symbol] = scan
            except Exception as e:
                pass
        print(f"    ✓ Anomaly scans: {len(hf_results['anomalies'])} symbols")
        
        # Method 8: Multi-Modal
        print("  Running HF Method 8: Multi-Modal Analysis...")
        try:
            if articles and symbols:
                import yfinance as yf
                symbol = symbols[0]
                ticker = yf.Ticker(symbol)
                df = ticker.history(period='1mo')
                
                if len(df) > 0:
                    df = df.rename(columns={'Close': 'close', 'High': 'high', 'Low': 'low'})
                    
                    multimodal = self.hf_multimodal.analyze_event_impact(
                        {'event': event_name, 'date': date},
                        articles[:5],
                        df
                    )
                    hf_results['multimodal'] = multimodal
                    print(f"    ✓ Signal: {multimodal['combined_analysis']['combined_signal']}")
        except Exception as e:
            print(f"    ⊘ Error: {str(e)[:50]}")
        
        # Method 9: Zero-Shot
        print("  Running HF Method 9: Zero-Shot Categorization...")
        try:
            if articles:
                events = [{'event': a.get('title', ''), 'date': date} for a in articles[:10]]
                categorized = self.hf_zeroshot.categorize_batch(events[:5])
                hf_results['zeroshot'] = categorized
                print(f"    ✓ Categorized {len(categorized)} items")
        except Exception as e:
            print(f"    ⊘ Error: {str(e)[:50]}")
        
        # Method 10: Causal Explanations
        print("  Running HF Method 10: Causal Explanations...")
        try:
            # Generate explanations for key findings
            explanations = {}
            
            if hf_results.get('sentiment'):
                sentiment = hf_results['sentiment']['aggregated']['overall_sentiment']
                explanations['sentiment'] = f"Market sentiment is {sentiment} based on news analysis"
            
            if hf_results.get('classification'):
                impact = hf_results['classification']['impact_level']
                explanations['impact'] = self.hf_causal.explain_event_impact(
                    event_name or 'Market Event',
                    symbols[:5]
                )
            
            hf_results['causal_explanations'] = explanations
            print(f"    ✓ Generated {len(explanations)} explanations")
        except Exception as e:
            print(f"    ⊘ Error: {str(e)[:50]}")
        
        return hf_results
    
    def _synthesize_insights(self, results):
        """Synthesize key insights from all analyses"""
        insights = {
            'summary': [],
            'key_findings': [],
            'trading_signals': [],
            'risk_factors': [],
            'opportunities': []
        }
        
        # Summary
        metadata = results['metadata']
        insights['summary'].append(
            f"Analysis of {metadata['event_name'] or 'market conditions'} on {metadata['date']}"
        )
        insights['summary'].append(
            f"Analyzed {len(metadata.get('symbols_analyzed', []))} symbols"
        )
        
        # News sentiment
        news = results['sections'].get('news', {})
        if news.get('articles'):
            insights['summary'].append(f"Processed {len(news['articles'])} news articles")
        
        # Economic indicators
        econ = results['sections'].get('economic_indicators', {})
        if econ:
            status = econ.get('overall_economic_status', 'N/A')
            insights['key_findings'].append(f"Economic status: {status}")
        
        # HF sentiment
        hf_methods = results['sections'].get('hf_methods', {})
        if hf_methods.get('sentiment'):
            sentiment = hf_methods['sentiment']['aggregated']['overall_sentiment']
            confidence = hf_methods['sentiment']['aggregated'].get('confidence', 0)
            insights['key_findings'].append(
                f"AI Sentiment: {sentiment.upper()} (confidence: {confidence:.1%})"
            )
        
        # Technical signals
        indicators = results['sections'].get('indicators', {})
        if indicators:
            buy_signals = sum(1 for ind in indicators.values() 
                            if ind.get('overall_signal') == 'BUY')
            sell_signals = sum(1 for ind in indicators.values() 
                             if ind.get('overall_signal') == 'SELL')
            
            if buy_signals > sell_signals:
                insights['trading_signals'].append(
                    f"Technical bias: BULLISH ({buy_signals}/{len(indicators)} symbols)"
                )
            elif sell_signals > buy_signals:
                insights['trading_signals'].append(
                    f"Technical bias: BEARISH ({sell_signals}/{len(indicators)} symbols)"
                )
        
        # COT positioning
        cot = results['sections'].get('cot', {})
        bullish_cot = sum(1 for pos in cot.values() 
                         if 'BULLISH' in pos.get('sentiment', ''))
        if bullish_cot > len(cot) / 2:
            insights['trading_signals'].append(
                f"COT positioning: BULLISH ({bullish_cot}/{len(cot)} assets)"
            )
        
        # Anomalies
        if hf_methods.get('anomalies'):
            anomaly_count = sum(a.get('anomalies_detected', 0) 
                              for a in hf_methods['anomalies'].values())
            if anomaly_count > 0:
                insights['risk_factors'].append(
                    f"Detected {anomaly_count} market anomalies - elevated risk"
                )
        
        # Market structure
        structure = results['sections'].get('market_structure', {})
        uptrends = sum(1 for s in structure.values() 
                      if 'UPTREND' in s.get('trend_analysis', {}).get('trend', ''))
        if uptrends > len(structure) / 2:
            insights['opportunities'].append(
                f"Trending markets: {uptrends}/{len(structure)} in uptrend"
            )
        
        return insights
    
    def _generate_visualizations(self, results, date, event_name):
        """Generate all visualizations"""
        viz_files = []
        
        date_clean = date.replace('-', '_')
        event_clean = (event_name or 'analysis').replace(' ', '_').lower()
        
        # 1. Sentiment visualization
        hf_methods = results['sections'].get('hf_methods', {})
        if hf_methods.get('sentiment'):
            files = self.viz.visualize(
                hf_methods['sentiment'],
                f'{date_clean}_{event_clean}_sentiment',
                'sentiment'
            )
            viz_files.extend(files)
            print(f"  ✓ Sentiment: {len(files)} image(s)")
        
        # 2. Entity extraction
        if hf_methods.get('entities'):
            files = self.viz.visualize(
                hf_methods['entities'],
                f'{date_clean}_{event_clean}_entities',
                'entities'
            )
            viz_files.extend(files)
            print(f"  ✓ Entities: {len(files)} image(s)")
        
        # 3. Forecasts
        if hf_methods.get('forecasts'):
            for symbol, forecast in list(hf_methods['forecasts'].items())[:3]:
                symbol_clean = symbol.replace('=', '_').replace('^', '')
                files = self.viz.visualize(
                    forecast,
                    f'{date_clean}_{symbol_clean}_forecast',
                    'forecast'
                )
                viz_files.extend(files)
            print(f"  ✓ Forecasts: {len(hf_methods['forecasts'])} symbol(s)")
        
        # 4. Indicators
        indicators = results['sections'].get('indicators', {})
        if indicators:
            # Visualize first symbol's indicators
            first_symbol = list(indicators.keys())[0]
            files = self.viz.visualize(
                indicators[first_symbol],
                f'{date_clean}_{first_symbol}_indicators',
                'indicators'
            )
            viz_files.extend(files)
            print(f"  ✓ Indicators: {len(files)} image(s)")
        
        # 5. COT positioning
        cot = results['sections'].get('cot', {})
        if cot:
            first_cot = list(cot.values())[0]
            files = self.viz.visualize(
                first_cot,
                f'{date_clean}_cot_positioning',
                'cot'
            )
            viz_files.extend(files)
            print(f"  ✓ COT: {len(files)} image(s)")
        
        # 6. Correlations
        correlations = results['sections'].get('correlations', {})
        if correlations:
            first_corr = list(correlations.values())[0]
            files = self.viz.visualize(
                first_corr,
                f'{date_clean}_correlations',
                'correlation'
            )
            viz_files.extend(files)
            print(f"  ✓ Correlations: {len(files)} image(s)")
        
        # 7. Anomalies
        if hf_methods.get('anomalies'):
            for symbol, anomaly in list(hf_methods['anomalies'].items())[:2]:
                symbol_clean = symbol.replace('=', '_').replace('^', '')
                files = self.viz.visualize(
                    anomaly,
                    f'{date_clean}_{symbol_clean}_anomaly',
                    'anomaly'
                )
                viz_files.extend(files)
            print(f"  ✓ Anomalies: {len(hf_methods['anomalies'])} symbol(s)")
        
        print(f"  Total visualizations: {len(viz_files)}")
        return viz_files
    
    def _save_results(self, results, date, event_name):
        """Save complete results to JSON"""
        date_clean = date.replace('-', '_')
        event_clean = (event_name or 'analysis').replace(' ', '_').lower()
        
        filename = f'{self.output_dir}/comprehensive_{date_clean}_{event_clean}.json'
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        return filename
    
    def generate_markdown_report(self, results):
        """Generate human-readable markdown report"""
        lines = []
        
        lines.append("# COMPREHENSIVE MARKET ANALYSIS REPORT")
        lines.append("")
        
        # Metadata
        meta = results['metadata']
        lines.append(f"**Date:** {meta['date']}")
        lines.append(f"**Event:** {meta['event_name'] or 'Market Analysis'}")
        lines.append(f"**Symbols Analyzed:** {len(meta.get('symbols_analyzed', []))}")
        lines.append(f"**Generated:** {meta['timestamp']}")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Executive Summary
        synthesis = results['sections'].get('synthesis', {})
        lines.append("## Executive Summary")
        lines.append("")
        for item in synthesis.get('summary', []):
            lines.append(f"- {item}")
        lines.append("")
        
        # Key Findings
        lines.append("## Key Findings")
        lines.append("")
        for finding in synthesis.get('key_findings', []):
            lines.append(f"- **{finding}**")
        lines.append("")
        
        # Trading Signals
        if synthesis.get('trading_signals'):
            lines.append("## Trading Signals")
            lines.append("")
            for signal in synthesis['trading_signals']:
                lines.append(f"- 📊 {signal}")
            lines.append("")
        
        # Risk Factors
        if synthesis.get('risk_factors'):
            lines.append("## Risk Factors")
            lines.append("")
            for risk in synthesis['risk_factors']:
                lines.append(f"- ⚠️ {risk}")
            lines.append("")
        
        # Opportunities
        if synthesis.get('opportunities'):
            lines.append("## Opportunities")
            lines.append("")
            for opp in synthesis['opportunities']:
                lines.append(f"- 💡 {opp}")
            lines.append("")
        
        # Section summaries
        lines.append("---")
        lines.append("")
        lines.append("## Detailed Analysis")
        lines.append("")
        
        # News
        news = results['sections'].get('news', {})
        if news.get('articles'):
            lines.append(f"### 📰 News ({len(news['articles'])} articles)")
            lines.append("")
            for i, article in enumerate(news['articles'][:5], 1):
                lines.append(f"{i}. {article.get('title', 'N/A')}")
            lines.append("")
        
        # COT
        cot = results['sections'].get('cot', {})
        if cot:
            lines.append(f"### 📊 COT Positioning ({len(cot)} assets)")
            lines.append("")
            for symbol, pos in list(cot.items())[:5]:
                lines.append(f"- **{symbol}**: {pos.get('sentiment', 'N/A')}")
            lines.append("")
        
        # Technical Indicators
        indicators = results['sections'].get('indicators', {})
        if indicators:
            lines.append(f"### 📈 Technical Indicators ({len(indicators)} symbols)")
            lines.append("")
            for symbol, ind in list(indicators.items())[:5]:
                signal = ind.get('overall_signal', 'N/A')
                lines.append(f"- **{symbol}**: {signal}")
            lines.append("")
        
        # HF Methods Summary
        hf_methods = results['sections'].get('hf_methods', {})
        if hf_methods:
            lines.append("### 🤖 AI Analysis (HF Methods)")
            lines.append("")
            
            if hf_methods.get('sentiment'):
                sent = hf_methods['sentiment']['aggregated']
                lines.append(f"- **Sentiment**: {sent.get('overall_sentiment', 'N/A').upper()}")
            
            if hf_methods.get('classification'):
                impact = hf_methods['classification'].get('impact_level', 'N/A')
                lines.append(f"- **Event Impact**: {impact}")
            
            if hf_methods.get('forecasts'):
                lines.append(f"- **Forecasts Generated**: {len(hf_methods['forecasts'])} symbols")
            
            if hf_methods.get('anomalies'):
                total_anomalies = sum(a.get('anomalies_detected', 0) 
                                     for a in hf_methods['anomalies'].values())
                if total_anomalies > 0:
                    lines.append(f"- **Anomalies Detected**: {total_anomalies}")
            
            lines.append("")
        
        # Visualizations
        if results.get('visualizations'):
            lines.append(f"### 📊 Visualizations Generated: {len(results['visualizations'])}")
            lines.append("")
        
        # Footer
        lines.append("---")
        lines.append("")
        lines.append("*Report generated by Comprehensive Analysis Pipeline*")
        
        return '\n'.join(lines)


if __name__ == "__main__":
    print("="*80)
    print("COMPREHENSIVE ANALYSIS PIPELINE - TEST")
    print("="*80)
    
    pipeline = ComprehensiveAnalysisPipeline(output_dir='test_pipeline_output')
    
    # Test cases
    test_cases = [
        {
            'date': '2024-11-01',
            'event_name': 'Non-Farm Payrolls',
            'symbols': ['EURUSD=X', 'GC=F', '^GSPC', 'DX-Y.NYB']
        },
        {
            'date': '2024-10-10',
            'event_name': 'Consumer Price Index',
            'symbols': None  # Auto-detect
        }
    ]
    
    for i, test in enumerate(test_cases[:1], 1):  # Run first test only
        print(f"\nTEST CASE {i}/{len(test_cases)}")
        print("="*80)
        
        results = pipeline.analyze(
            date=test['date'],
            event_name=test['event_name'],
            symbols=test['symbols'],
            use_hf_methods=True,
            generate_visuals=True
        )
        
        # Generate markdown report
        markdown = pipeline.generate_markdown_report(results)
        
        md_file = f"{pipeline.output_dir}/report_{test['date'].replace('-', '_')}.md"
        with open(md_file, 'w') as f:
            f.write(markdown)
        
        print(f"\nMarkdown report: {md_file}")
    
    print("\n" + "="*80)
    print("✓ Pipeline test complete")
    print("="*80)
