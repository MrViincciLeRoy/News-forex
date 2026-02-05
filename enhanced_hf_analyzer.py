"""
Enhanced HF AI Analysis Module - PRODUCTION VERSION
Integrates all 10 HF methods with robust error handling and detailed output
"""

import json
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np


class EnhancedHFAnalyzer:
    """Production-ready HF analyzer with all 10 methods"""
    
    def __init__(self):
        self.methods = {}
        self.available_methods = []
        self.results_cache = {}
    
    def load_models(self):
        """Load all available HF models"""
        print("\n" + "="*80)
        print("LOADING HF AI MODELS")
        print("="*80)
        
        loaded_count = 0
        
        # Method 1: Sentiment Analysis
        try:
            from hf_method1_sentiment import HFSentimentAnalyzer
            analyzer = HFSentimentAnalyzer()
            analyzer.load_model()
            self.methods['sentiment'] = analyzer
            self.available_methods.append('sentiment')
            print("  ✓ Method 1: Sentiment Analysis (FinBERT)")
            loaded_count += 1
        except Exception as e:
            print(f"  ⊘ Method 1: Sentiment - {str(e)[:50]}")
        
        # Method 2: Named Entity Recognition
        try:
            from hf_method2_ner import HFEntityExtractor
            extractor = HFEntityExtractor()
            extractor.load_model()
            self.methods['ner'] = extractor
            self.available_methods.append('ner')
            print("  ✓ Method 2: Named Entity Recognition (BERT-NER)")
            loaded_count += 1
        except Exception as e:
            print(f"  ⊘ Method 2: NER - {str(e)[:50]}")
        
        # Method 3: Time Series Forecasting
        try:
            from hf_method3_forecasting import HFTimeSeriesForecaster
            forecaster = HFTimeSeriesForecaster()
            forecaster.load_model()
            self.methods['forecasting'] = forecaster
            self.available_methods.append('forecasting')
            print("  ✓ Method 3: Time Series Forecasting (Chronos)")
            loaded_count += 1
        except Exception as e:
            print(f"  ⊘ Method 3: Forecasting - {str(e)[:50]}")
        
        # Method 4: Event Classification
        try:
            from hf_method4_classification import HFEventClassifier
            classifier = HFEventClassifier()
            classifier.load_model()
            self.methods['classification'] = classifier
            self.available_methods.append('classification')
            print("  ✓ Method 4: Event Classification (BART-MNLI)")
            loaded_count += 1
        except Exception as e:
            print(f"  ⊘ Method 4: Classification - {str(e)[:50]}")
        
        # Method 5: Correlation Discovery
        try:
            from hf_method5_embeddings import HFCorrelationDiscovery
            discovery = HFCorrelationDiscovery()
            discovery.load_model()
            self.methods['correlation'] = discovery
            self.available_methods.append('correlation')
            print("  ✓ Method 5: Correlation Discovery (Embeddings)")
            loaded_count += 1
        except Exception as e:
            print(f"  ⊘ Method 5: Correlation - {str(e)[:50]}")
        
        # Method 6: Market Q&A
        try:
            from hf_method6_qa import HFMarketQA
            qa = HFMarketQA()
            qa.load_model()
            self.methods['qa'] = qa
            self.available_methods.append('qa')
            print("  ✓ Method 6: Market Q&A System (RoBERTa-SQuAD)")
            loaded_count += 1
        except Exception as e:
            print(f"  ⊘ Method 6: Q&A - {str(e)[:50]}")
        
        # Method 7: Anomaly Detection
        try:
            from hf_method7_anomaly import HFAnomalyDetector
            anomaly = HFAnomalyDetector()
            anomaly.load_model()
            self.methods['anomaly'] = anomaly
            self.available_methods.append('anomaly')
            print("  ✓ Method 7: Anomaly Detection (Statistical)")
            loaded_count += 1
        except Exception as e:
            print(f"  ⊘ Method 7: Anomaly - {str(e)[:50]}")
        
        # Method 8: Multi-Modal Analysis
        try:
            from hf_method8_multimodal import HFMultiModalAnalyzer
            multimodal = HFMultiModalAnalyzer()
            multimodal.load_models()
            self.methods['multimodal'] = multimodal
            self.available_methods.append('multimodal')
            print("  ✓ Method 8: Multi-Modal Analysis (Text+Technical)")
            loaded_count += 1
        except Exception as e:
            print(f"  ⊘ Method 8: Multi-Modal - {str(e)[:50]}")
        
        # Method 9: Zero-Shot Categorization
        try:
            from hf_method9_zeroshot import HFZeroShotCategorizer
            zeroshot = HFZeroShotCategorizer()
            zeroshot.load_model()
            
            # Define categories
            zeroshot.define_categories({
                'employment': 'Jobs and labor market',
                'inflation': 'Price levels and inflation',
                'monetary_policy': 'Central bank policy',
                'growth': 'Economic growth',
                'trade': 'International trade',
                'consumer': 'Consumer spending',
                'manufacturing': 'Industrial production',
                'housing': 'Real estate market'
            })
            
            self.methods['zeroshot'] = zeroshot
            self.available_methods.append('zeroshot')
            print("  ✓ Method 9: Zero-Shot Categorization (DeBERTa)")
            loaded_count += 1
        except Exception as e:
            print(f"  ⊘ Method 9: Zero-Shot - {str(e)[:50]}")
        
        # Method 10: Causal Explanations
        try:
            from hf_method10_causal import HFCausalExplainer
            causal = HFCausalExplainer()
            causal.load_model()
            self.methods['causal'] = causal
            self.available_methods.append('causal')
            print("  ✓ Method 10: Causal Explanations (FLAN-T5)")
            loaded_count += 1
        except Exception as e:
            print(f"  ⊘ Method 10: Causal - {str(e)[:50]}")
        
        print("="*80)
        print(f"✓ Loaded {loaded_count}/10 HF AI methods")
        print("="*80 + "\n")
        
        return loaded_count > 0
    
    def analyze_comprehensive(
        self, 
        articles: List[Dict],
        symbols: List[str],
        date: str,
        event_name: Optional[str] = None,
        price_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Dict:
        """Run comprehensive HF analysis with all available methods"""
        
        results = {
            'metadata': {
                'methods_available': self.available_methods,
                'methods_count': len(self.available_methods),
                'articles_count': len(articles),
                'symbols_count': len(symbols),
                'date': date,
                'event_name': event_name,
                'timestamp': datetime.now().isoformat()
            },
            'methods': {},
            'summary': {}
        }
        
        if not articles:
            print("  ⊘ No articles to analyze")
            return results
        
        print(f"\nRunning {len(self.available_methods)} HF AI methods...")
        print("="*80)
        
        # 1. Sentiment Analysis
        if 'sentiment' in self.methods:
            results['methods']['sentiment'] = self._run_sentiment(articles)
        
        # 2. Named Entity Recognition
        if 'ner' in self.methods:
            results['methods']['ner'] = self._run_ner(articles)
        
        # 3. Event Classification
        if 'classification' in self.methods:
            results['methods']['classification'] = self._run_classification(articles, event_name)
        
        # 4. Zero-Shot Categorization
        if 'zeroshot' in self.methods:
            results['methods']['zeroshot'] = self._run_zeroshot(articles, event_name)
        
        # 5. Correlation Discovery
        if 'correlation' in self.methods:
            results['methods']['correlation'] = self._run_correlation(articles, symbols)
        
        # 6. Multi-Modal Analysis
        if 'multimodal' in self.methods and price_data:
            results['methods']['multimodal'] = self._run_multimodal(articles, symbols, price_data)
        
        # 7. Market Q&A
        if 'qa' in self.methods:
            results['methods']['qa'] = self._run_qa(articles, event_name)
        
        # 8. Time Series Forecasting
        if 'forecasting' in self.methods and price_data:
            results['methods']['forecasting'] = self._run_forecasting(symbols[:3], price_data)
        
        # 9. Anomaly Detection
        if 'anomaly' in self.methods and price_data:
            results['methods']['anomaly'] = self._run_anomaly(symbols[:3], price_data)
        
        # 10. Causal Explanations
        if 'causal' in self.methods:
            results['methods']['causal'] = self._run_causal(results['methods'], event_name)
        
        # Generate summary
        results['summary'] = self._generate_summary(results['methods'])
        
        print("\n" + "="*80)
        print(f"✓ HF AI Analysis Complete")
        print(f"  Methods Run: {len([m for m in results['methods'].values() if m.get('success')])}/{len(results['methods'])}")
        print(f"  Insights Generated: {len(results['summary'].get('key_insights', []))}")
        print("="*80 + "\n")
        
        return results
    
    def _run_sentiment(self, articles: List[Dict]) -> Dict:
        """Run sentiment analysis"""
        try:
            analyzer = self.methods['sentiment']
            
            print(f"\n  → Method 1: Sentiment Analysis")
            print(f"     Analyzing {len(articles)} articles...")
            
            sentiment_results = analyzer.analyze_news_articles(articles)
            aggregated = analyzer.aggregate_sentiment(sentiment_results)
            
            overall = aggregated.get('overall_sentiment', 'neutral').upper()
            score = aggregated.get('confidence', 0.0)
            pos = aggregated.get('positive_count', 0)
            neg = aggregated.get('negative_count', 0)
            neu = aggregated.get('neutral_count', 0)
            
            print(f"     ✓ Overall: {overall}")
            print(f"     Confidence: {score:.1%}")
            print(f"     Distribution: Pos={pos} Neu={neu} Neg={neg}")
            
            return {
                'articles': sentiment_results,
                'aggregated': aggregated,
                'success': True,
                'insights': [
                    f"Market sentiment is {overall.lower()} with {score:.0%} confidence",
                    f"Analyzed {len(articles)} articles: {pos} positive, {neg} negative"
                ]
            }
            
        except Exception as e:
            print(f"     ✗ Error: {str(e)[:60]}")
            return {'success': False, 'error': str(e)}
    
    def _run_ner(self, articles: List[Dict]) -> Dict:
        """Run named entity recognition"""
        try:
            extractor = self.methods['ner']
            
            print(f"\n  → Method 2: Named Entity Recognition")
            print(f"     Extracting entities from {len(articles)} articles...")
            
            entities = extractor.analyze_batch(articles)
            aggregated = extractor.aggregate_symbols(entities)
            
            symbol_count = aggregated.get('symbol_count', 0)
            org_count = len(aggregated.get('unique_symbols', []))
            top_symbols = aggregated.get('top_symbols', [])[:5]
            
            print(f"     ✓ Unique Symbols: {symbol_count}")
            print(f"     Organizations: {org_count}")
            if top_symbols:
                print(f"     Top Symbols: {', '.join(top_symbols)}")
            
            return {
                'entities': entities,
                'aggregated': aggregated,
                'success': True,
                'insights': [
                    f"Identified {symbol_count} unique symbols in news coverage",
                    f"Top mentioned: {', '.join(top_symbols[:3])}" if top_symbols else "No symbols extracted"
                ]
            }
            
        except Exception as e:
            print(f"     ✗ Error: {str(e)[:60]}")
            return {'success': False, 'error': str(e)}
    
    def _run_classification(self, articles: List[Dict], event_name: Optional[str]) -> Dict:
        """Run event classification"""
        try:
            classifier = self.methods['classification']
            
            print(f"\n  → Method 4: Event Classification")
            
            # Create events from articles
            events = [{'event': a.get('title', ''), 'date': a.get('date', '')} for a in articles[:10]]
            if event_name:
                events.insert(0, {'event': event_name, 'date': ''})
            
            classifications = classifier.classify_batch(events)
            
            # Count categories
            categories = {}
            for cls in classifications:
                cat = cls.get('impact_level', 'unknown')
                categories[cat] = categories.get(cat, 0) + 1
            
            top_category = max(categories.items(), key=lambda x: x[1])[0] if categories else 'unknown'
            
            print(f"     ✓ Primary Impact: {top_category}")
            print(f"     Categories: {dict(list(categories.items())[:3])}")
            
            return {
                'classifications': classifications,
                'categories': categories,
                'top_category': top_category,
                'success': True,
                'insights': [
                    f"Event classified as {top_category} impact",
                    f"Analyzed {len(classifications)} events"
                ]
            }
            
        except Exception as e:
            print(f"     ✗ Error: {str(e)[:60]}")
            return {'success': False, 'error': str(e)}
    
    def _run_zeroshot(self, articles: List[Dict], event_name: Optional[str]) -> Dict:
        """Run zero-shot categorization"""
        try:
            categorizer = self.methods['zeroshot']
            
            print(f"\n  → Method 9: Zero-Shot Categorization")
            
            # Categorize articles
            events = [{'event': a.get('title', ''), 'date': a.get('date', '')} for a in articles[:10]]
            results = categorizer.categorize_batch(events)
            
            # Find dominant theme
            themes = {}
            for r in results:
                cat = r.get('categorization', {})
                theme = cat.get('primary_category', 'unknown')
                themes[theme] = themes.get(theme, 0) + 1
            
            dominant = max(themes.items(), key=lambda x: x[1])[0] if themes else 'unknown'
            
            print(f"     ✓ Dominant Theme: {dominant}")
            print(f"     Themes: {dict(list(themes.items())[:3])}")
            
            return {
                'categorizations': results,
                'themes': themes,
                'dominant_theme': dominant,
                'success': True,
                'insights': [
                    f"Dominant theme is {dominant}",
                    f"Coverage spans {len(themes)} different categories"
                ]
            }
            
        except Exception as e:
            print(f"     ✗ Error: {str(e)[:60]}")
            return {'success': False, 'error': str(e)}
    
    def _run_correlation(self, articles: List[Dict], symbols: List[str]) -> Dict:
        """Run correlation discovery"""
        try:
            discovery = self.methods['correlation']
            
            print(f"\n  → Method 5: Correlation Discovery")
            
            # Create events from articles
            events = [
                {
                    'event': a.get('title', ''),
                    'description': a.get('content', '')[:200],
                    'date': a.get('date', ''),
                    'category': 'news'
                }
                for a in articles[:10]
            ]
            
            # Discover correlations
            correlations = discovery.discover_hidden_correlations(events)
            
            cluster_count = correlations.get('semantic_clusters', 0)
            
            print(f"     ✓ Semantic Clusters: {cluster_count}")
            
            return {
                'correlations': correlations,
                'cluster_count': cluster_count,
                'success': True,
                'insights': [
                    f"Discovered {cluster_count} semantic event clusters",
                    "Events grouped by thematic similarity"
                ]
            }
            
        except Exception as e:
            print(f"     ✗ Error: {str(e)[:60]}")
            return {'success': False, 'error': str(e)}
    
    def _run_multimodal(self, articles: List[Dict], symbols: List[str], 
                        price_data: Dict[str, pd.DataFrame]) -> Dict:
        """Run multi-modal analysis"""
        try:
            analyzer = self.methods['multimodal']
            
            print(f"\n  → Method 8: Multi-Modal Analysis")
            
            results = {}
            for symbol in symbols[:3]:
                if symbol in price_data and not price_data[symbol].empty:
                    event = {'event': f'{symbol} Analysis', 'date': ''}
                    result = analyzer.analyze_event_impact(event, articles, price_data[symbol])
                    results[symbol] = result
            
            print(f"     ✓ Analyzed {len(results)} symbols")
            
            return {
                'analyses': results,
                'symbols_analyzed': len(results),
                'success': True,
                'insights': [
                    f"Combined text + technical analysis for {len(results)} symbols"
                ]
            }
            
        except Exception as e:
            print(f"     ✗ Error: {str(e)[:60]}")
            return {'success': False, 'error': str(e)}
    
    def _run_qa(self, articles: List[Dict], event_name: Optional[str]) -> Dict:
        """Run market Q&A"""
        try:
            qa_system = self.methods['qa']
            
            print(f"\n  → Method 6: Market Q&A System")
            
            # Index articles
            calendar_data = [
                {
                    'date': a.get('date', ''),
                    'event': event_name or 'Market Event',
                    'indicators': {},
                    'news': [a]
                }
                for a in articles[:5]
            ]
            
            qa_system.index_economic_calendar(calendar_data)
            
            # Ask questions
            questions = [
                "What is the market sentiment?",
                "What are the key risks?",
                "What symbols are most affected?"
            ]
            
            answers = qa_system.batch_ask(questions)
            
            print(f"     ✓ Answered {len(answers)} questions")
            
            return {
                'qa_pairs': answers,
                'questions_answered': len(answers),
                'success': True,
                'insights': [
                    f"Q&A system ready with {len(questions)} market questions"
                ]
            }
            
        except Exception as e:
            print(f"     ✗ Error: {str(e)[:60]}")
            return {'success': False, 'error': str(e)}
    
    def _run_forecasting(self, symbols: List[str], 
                        price_data: Dict[str, pd.DataFrame]) -> Dict:
        """Run time series forecasting"""
        try:
            forecaster = self.methods['forecasting']
            
            print(f"\n  → Method 3: Time Series Forecasting")
            
            forecasts = {}
            for symbol in symbols:
                if symbol in price_data and not price_data[symbol].empty:
                    try:
                        df = price_data[symbol]
                        prices = df['Close'] if 'Close' in df.columns else df['close']
                        forecast = forecaster.forecast(prices, forecast_horizon=7)
                        forecasts[symbol] = forecast
                        
                        method = forecast.get('method', 'unknown')
                        print(f"     {symbol}: {method} forecast (7-day)")
                    except:
                        pass
            
            print(f"     ✓ Forecasted {len(forecasts)} symbols")
            
            return {
                'forecasts': forecasts,
                'symbols_forecasted': len(forecasts),
                'success': True,
                'insights': [
                    f"Generated 7-day forecasts for {len(forecasts)} symbols"
                ]
            }
            
        except Exception as e:
            print(f"     ✗ Error: {str(e)[:60]}")
            return {'success': False, 'error': str(e)}
    
    def _run_anomaly(self, symbols: List[str], 
                     price_data: Dict[str, pd.DataFrame]) -> Dict:
        """Run anomaly detection"""
        try:
            detector = self.methods['anomaly']
            
            print(f"\n  → Method 7: Anomaly Detection")
            
            anomalies_found = {}
            for symbol in symbols:
                if symbol in price_data and len(price_data[symbol]) > 20:
                    try:
                        df = price_data[symbol]
                        df = df.rename(columns={
                            'Open': 'open', 'High': 'high', 'Low': 'low',
                            'Close': 'close', 'Volume': 'volume'
                        })
                        
                        current = df.iloc[-1]
                        scan = detector.comprehensive_anomaly_scan(current, df, symbol)
                        
                        if scan['anomalies_detected'] > 0:
                            anomalies_found[symbol] = scan
                    except:
                        pass
            
            total_anomalies = sum(a['anomalies_detected'] for a in anomalies_found.values())
            print(f"     ✓ Found {total_anomalies} anomalies across {len(anomalies_found)} symbols")
            
            return {
                'scans': anomalies_found,
                'total_anomalies': total_anomalies,
                'success': True,
                'insights': [
                    f"Detected {total_anomalies} market anomalies" if total_anomalies > 0 
                    else "No significant anomalies detected"
                ]
            }
            
        except Exception as e:
            print(f"     ✗ Error: {str(e)[:60]}")
            return {'success': False, 'error': str(e)}
    
    def _run_causal(self, methods_results: Dict, event_name: Optional[str]) -> Dict:
        """Run causal explanations"""
        try:
            explainer = self.methods['causal']
            
            print(f"\n  → Method 10: Causal Explanations")
            
            # Generate comprehensive report
            analysis_data = {}
            
            # Add any available data
            if 'correlation' in methods_results and methods_results['correlation'].get('success'):
                analysis_data['correlations'] = []
            
            if event_name:
                analysis_data['events'] = [
                    {
                        'event_name': event_name,
                        'affected_symbols': []
                    }
                ]
            
            report = explainer.generate_comprehensive_report(analysis_data)
            
            print(f"     ✓ Generated causal explanation report")
            
            return {
                'report': report,
                'success': True,
                'insights': [
                    "Generated natural language explanations for market behavior"
                ]
            }
            
        except Exception as e:
            print(f"     ✗ Error: {str(e)[:60]}")
            return {'success': False, 'error': str(e)}
    
    def _generate_summary(self, methods: Dict) -> Dict:
        """Generate summary insights from all methods"""
        summary = {
            'methods_run': len(methods),
            'methods_successful': sum(1 for m in methods.values() if m.get('success')),
            'key_insights': []
        }
        
        # Collect insights from all methods
        for method_name, result in methods.items():
            if result.get('success') and 'insights' in result:
                for insight in result['insights']:
                    summary['key_insights'].append({
                        'source': method_name,
                        'insight': insight,
                        'confidence': 0.8
                    })
        
        return summary


# Integration function for backward compatibility
def integrate_hf_analysis(pipeline_results: Dict) -> Dict:
    """Integration function to enhance existing pipeline with HF AI"""
    enhanced_results = pipeline_results.copy()
    
    articles = pipeline_results.get('sections', {}).get('news', {}).get('articles', [])
    symbols = pipeline_results.get('metadata', {}).get('symbols_analyzed', [])
    date = pipeline_results.get('metadata', {}).get('date', '')
    event_name = pipeline_results.get('metadata', {}).get('event_name')
    
    analyzer = EnhancedHFAnalyzer()
    if analyzer.load_models():
        hf_results = analyzer.analyze_comprehensive(articles, symbols, date, event_name)
        enhanced_results['sections']['hf_methods'] = hf_results
    
    return enhanced_results


if __name__ == "__main__":
    analyzer = EnhancedHFAnalyzer()
    analyzer.load_models()
    
    test_articles = [
        {
            'title': 'Fed Raises Rates, Markets Rally',
            'content': 'The Federal Reserve raised interest rates...',
            'date': '2024-11-01'
        },
        {
            'title': 'Strong Employment Data Boosts Stocks',
            'content': 'Non-farm payrolls exceeded expectations...',
            'date': '2024-11-01'
        }
    ]
    
    results = analyzer.analyze_comprehensive(
        articles=test_articles,
        symbols=['EURUSD=X', 'GC=F'],
        date='2024-11-01',
        event_name='Non-Farm Payrolls'
    )
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(json.dumps(results['summary'], indent=2))
