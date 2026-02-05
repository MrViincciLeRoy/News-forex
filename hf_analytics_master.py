"""
HF ANALYTICS MASTER - Orchestrates all Hugging Face AI methods
Manages 10 AI-powered enhancement methods for trading analysis
Integrates with existing data sources and analysis pipelines
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import json
import os
import sys


class HFAnalyticsMaster:
    """
    Master orchestrator for all Hugging Face analytics methods
    Coordinates 10 AI-powered enhancement techniques
    """
    
    def __init__(self):
        """Initialize HF Analytics Master"""
        self.methods = {}
        self.results_cache = {}
        self.config = self._load_config()
        
        print("="*80)
        print("HF ANALYTICS MASTER - AI-Powered Trading Analysis")
        print("="*80)
        print("\nAvailable Methods:")
        print("  1. Financial Sentiment Analysis (FinBERT)")
        print("  2. Named Entity Recognition (NER)")
        print("  3. Time Series Forecasting (Chronos)")
        print("  4. Event Impact Classification (Zero-Shot)")
        print("  5. Correlation Discovery (Embeddings)")
        print("  6. Market Intelligence Q&A")
        print("  7. Anomaly Detection")
        print("  8. Multi-Modal Analysis (Text + Charts)")
        print("  9. Dynamic Event Categorization")
        print(" 10. Causal Inference & Explanation")
        print("="*80 + "\n")
    
    def _load_config(self) -> Dict:
        """Load configuration"""
        config_path = 'hf_analytics_config.json'
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration
        default_config = {
            'methods': {
                'sentiment': {
                    'enabled': True,
                    'model': 'ProsusAI/finbert',
                    'description': 'Financial sentiment analysis'
                },
                'ner': {
                    'enabled': True,
                    'model': 'dslim/bert-base-NER',
                    'description': 'Entity extraction from news'
                },
                'forecasting': {
                    'enabled': True,
                    'model': 'amazon/chronos-t5-small',
                    'description': 'Time series forecasting'
                },
                'classification': {
                    'enabled': True,
                    'model': 'facebook/bart-large-mnli',
                    'description': 'Zero-shot event classification'
                },
                'embeddings': {
                    'enabled': True,
                    'model': 'sentence-transformers/all-MiniLM-L6-v2',
                    'description': 'Semantic similarity & correlation'
                },
                'qa': {
                    'enabled': True,
                    'model': 'deepset/roberta-base-squad2',
                    'description': 'Question answering system'
                },
                'anomaly': {
                    'enabled': True,
                    'model': 'microsoft/deberta-v3-base',
                    'description': 'Anomaly detection'
                },
                'multimodal': {
                    'enabled': False,
                    'model': 'microsoft/git-base',
                    'description': 'Chart + text analysis'
                },
                'zeroshot': {
                    'enabled': True,
                    'model': 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli',
                    'description': 'Dynamic categorization'
                },
                'causal': {
                    'enabled': True,
                    'model': 'google/flan-t5-base',
                    'description': 'Causal reasoning & explanations'
                }
            },
            'data_sources': {
                'news_fetcher': 'news_fetcher.py',
                'economic_calendar': 'economic_calendar_generator.py',
                'cot_data': 'cot_data_fetcher.py',
                'technical_indicators': 'symbol_indicators.py',
                'market_structure': 'market_structure_analyzer.py',
                'correlation': 'correlation_analyzer.py'
            },
            'output_dir': 'hf_analytics_output',
            'cache_enabled': True
        }
        
        # Save default config
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        return default_config
    
    def initialize_method(self, method_name: str) -> bool:
        """Initialize specific HF method"""
        if method_name not in self.config['methods']:
            print(f"✗ Unknown method: {method_name}")
            return False
        
        if not self.config['methods'][method_name]['enabled']:
            print(f"⚠️  Method disabled: {method_name}")
            return False
        
        try:
            # Import method module
            if method_name == 'sentiment':
                from hf_method1_sentiment import HFSentimentAnalyzer
                self.methods[method_name] = HFSentimentAnalyzer()
                
            elif method_name == 'ner':
                from hf_method2_ner import HFEntityExtractor
                self.methods[method_name] = HFEntityExtractor()
                
            elif method_name == 'forecasting':
                from hf_method3_forecasting import HFTimeSeriesForecaster
                self.methods[method_name] = HFTimeSeriesForecaster()
            
            # Additional methods would be imported similarly
            # elif method_name == 'classification':
            #     from hf_method4_classification import HFEventClassifier
            #     self.methods[method_name] = HFEventClassifier()
            
            else:
                print(f"⚠️  Method not yet implemented: {method_name}")
                return False
            
            # Load model
            if hasattr(self.methods[method_name], 'load_model'):
                success = self.methods[method_name].load_model()
                if success:
                    print(f"✓ Initialized: {method_name}")
                    return True
                else:
                    print(f"✗ Failed to load model: {method_name}")
                    return False
            
            return True
            
        except ImportError as e:
            print(f"✗ Import error for {method_name}: {e}")
            return False
        except Exception as e:
            print(f"✗ Initialization error for {method_name}: {e}")
            return False
    
    def initialize_all(self) -> Dict[str, bool]:
        """Initialize all enabled methods"""
        results = {}
        
        print("\nInitializing HF Analytics Methods...")
        print("-" * 80)
        
        for method_name, method_config in self.config['methods'].items():
            if method_config['enabled']:
                print(f"\n{method_name.upper()}: {method_config['description']}")
                results[method_name] = self.initialize_method(method_name)
            else:
                print(f"\n{method_name.upper()}: Disabled")
                results[method_name] = False
        
        print("\n" + "-" * 80)
        successful = sum(1 for v in results.values() if v)
        print(f"✓ Initialized {successful}/{len(results)} methods")
        
        return results
    
    def run_sentiment_analysis(self, news_articles: List[Dict]) -> Dict:
        """Run sentiment analysis on news"""
        if 'sentiment' not in self.methods:
            print("⚠️  Sentiment analyzer not initialized")
            return {}
        
        print(f"\nRunning sentiment analysis on {len(news_articles)} articles...")
        
        analyzer = self.methods['sentiment']
        enhanced_articles = analyzer.analyze_news_articles(news_articles)
        aggregated = analyzer.aggregate_sentiment(enhanced_articles)
        
        return {
            'method': 'sentiment_analysis',
            'articles': enhanced_articles,
            'aggregated': aggregated,
            'timestamp': datetime.now().isoformat()
        }
    
    def run_entity_extraction(self, news_articles: List[Dict]) -> Dict:
        """Extract entities and symbols from news"""
        if 'ner' not in self.methods:
            print("⚠️  NER not initialized")
            return {}
        
        print(f"\nExtracting entities from {len(news_articles)} articles...")
        
        extractor = self.methods['ner']
        analyzed = extractor.analyze_batch(news_articles)
        aggregated = extractor.aggregate_symbols(analyzed)
        
        return {
            'method': 'entity_extraction',
            'analyzed_articles': analyzed,
            'aggregated_symbols': aggregated,
            'timestamp': datetime.now().isoformat()
        }
    
    def run_price_forecast(self, symbol: str, historical_data: pd.Series, 
                          horizon: int = 7) -> Dict:
        """Forecast future prices"""
        if 'forecasting' not in self.methods:
            print("⚠️  Forecaster not initialized")
            return {}
        
        print(f"\nForecasting {symbol} for next {horizon} periods...")
        
        forecaster = self.methods['forecasting']
        forecast = forecaster.forecast(historical_data, horizon)
        
        return {
            'method': 'price_forecasting',
            'symbol': symbol,
            'forecast': forecast,
            'timestamp': datetime.now().isoformat()
        }
    
    def run_comprehensive_analysis(self, 
                                   news_articles: List[Dict],
                                   price_data: Optional[Dict[str, pd.Series]] = None,
                                   economic_events: Optional[List[Dict]] = None) -> Dict:
        """
        Run comprehensive multi-method analysis
        Combines multiple HF methods for complete market intelligence
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE HF ANALYSIS")
        print("="*80)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'methods_used': [],
            'results': {}
        }
        
        # 1. Sentiment Analysis
        if 'sentiment' in self.methods and news_articles:
            sentiment_results = self.run_sentiment_analysis(news_articles)
            results['results']['sentiment'] = sentiment_results
            results['methods_used'].append('sentiment')
        
        # 2. Entity Extraction
        if 'ner' in self.methods and news_articles:
            ner_results = self.run_entity_extraction(news_articles)
            results['results']['entities'] = ner_results
            results['methods_used'].append('ner')
        
        # 3. Price Forecasting
        if 'forecasting' in self.methods and price_data:
            forecasts = {}
            for symbol, data in price_data.items():
                forecast = self.run_price_forecast(symbol, data)
                forecasts[symbol] = forecast
            results['results']['forecasts'] = forecasts
            results['methods_used'].append('forecasting')
        
        # Generate insights
        results['insights'] = self._generate_insights(results['results'])
        
        return results
    
    def _generate_insights(self, analysis_results: Dict) -> List[str]:
        """Generate actionable insights from analysis"""
        insights = []
        
        # Sentiment insights
        if 'sentiment' in analysis_results:
            agg = analysis_results['sentiment'].get('aggregated', {})
            sentiment = agg.get('overall_sentiment', 'neutral')
            confidence = agg.get('confidence', 0)
            
            if sentiment == 'positive' and confidence > 0.6:
                insights.append(f"📈 Strong bullish sentiment detected (confidence: {confidence:.2%})")
            elif sentiment == 'negative' and confidence > 0.6:
                insights.append(f"📉 Strong bearish sentiment detected (confidence: {confidence:.2%})")
        
        # Entity insights
        if 'entities' in analysis_results:
            symbols = analysis_results['entities'].get('aggregated_symbols', {})
            top_symbols = symbols.get('top_symbols', [])
            if top_symbols:
                insights.append(f"🎯 Most mentioned symbols: {', '.join(top_symbols[:5])}")
        
        # Forecast insights
        if 'forecasts' in analysis_results:
            for symbol, forecast in analysis_results['forecasts'].items():
                forecast_data = forecast.get('forecast', {})
                mean_forecast = forecast_data.get('forecast_mean', [])
                if mean_forecast:
                    trend = "upward" if mean_forecast[-1] > mean_forecast[0] else "downward"
                    insights.append(f"📊 {symbol}: {trend} trend predicted")
        
        if not insights:
            insights.append("ℹ️  Insufficient data for insights")
        
        return insights
    
    def save_results(self, results: Dict, filename: str = 'hf_analytics_results.json'):
        """Save analysis results"""
        output_dir = self.config['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n✓ Saved results to: {filepath}")
        return filepath
    
    def generate_report(self, results: Dict) -> str:
        """Generate human-readable report"""
        report_lines = [
            "="*80,
            "HF ANALYTICS REPORT",
            "="*80,
            f"\nGenerated: {results.get('timestamp', 'N/A')}",
            f"Methods Used: {', '.join(results.get('methods_used', []))}",
            "\n" + "="*80,
            "INSIGHTS",
            "="*80
        ]
        
        for insight in results.get('insights', []):
            report_lines.append(f"  {insight}")
        
        report_lines.extend([
            "\n" + "="*80,
            "DETAILED RESULTS",
            "="*80
        ])
        
        # Add sentiment summary
        if 'sentiment' in results.get('results', {}):
            agg = results['results']['sentiment'].get('aggregated', {})
            report_lines.extend([
                "\nSENTIMENT ANALYSIS:",
                f"  Overall: {agg.get('overall_sentiment', 'N/A').upper()}",
                f"  Bullish: {agg.get('bullish_score', 0):.2%}",
                f"  Bearish: {agg.get('bearish_score', 0):.2%}",
                f"  Articles: {agg.get('article_count', 0)}"
            ])
        
        # Add entity summary
        if 'entities' in results.get('results', {}):
            symbols = results['results']['entities'].get('aggregated_symbols', {})
            report_lines.extend([
                "\nENTITY EXTRACTION:",
                f"  Unique Symbols: {symbols.get('symbol_count', 0)}",
                f"  Top Mentions: {', '.join(symbols.get('top_symbols', [])[:5])}"
            ])
        
        report_lines.append("\n" + "="*80)
        
        return "\n".join(report_lines)


if __name__ == "__main__":
    print("Initializing HF Analytics Master...")
    
    master = HFAnalyticsMaster()
    
    # Initialize available methods
    init_results = master.initialize_all()
    
    # Test with sample data
    test_articles = [
        {'title': 'Gold prices surge on inflation fears', 'date': '2024-11-01'},
        {'title': 'Fed signals dovish stance on rates', 'date': '2024-11-01'},
        {'title': 'Tech stocks rally as earnings beat', 'date': '2024-11-01'}
    ]
    
    # Run analysis if methods initialized
    if any(init_results.values()):
        print("\n" + "="*80)
        print("RUNNING TEST ANALYSIS")
        print("="*80)
        
        results = master.run_comprehensive_analysis(
            news_articles=test_articles,
            price_data=None,  # Would include actual price data
            economic_events=None
        )
        
        # Generate and print report
        report = master.generate_report(results)
        print("\n" + report)
        
        # Save results
        master.save_results(results)
    else:
        print("\n⚠️  No methods initialized successfully")
        print("   Ensure transformers/torch installed: pip install transformers torch")
