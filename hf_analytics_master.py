"""
HF ANALYTICS MASTER - Complete orchestration of all 10 HF methods
Production-ready master controller with comprehensive error handling
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
import os


class HFAnalyticsMaster:
    """Master orchestrator for all Hugging Face analytics methods"""
    
    def __init__(self, config_path: str = 'hf_analytics_config.json'):
        self.methods = {}
        self.results_cache = {}
        self.config = self._load_config(config_path)
        
        print("="*80)
        print("HF ANALYTICS MASTER - AI-Powered Trading Analysis")
        print("="*80)
        print("\n10 Methods Available:")
        print("  1. Sentiment Analysis  2. Entity Extraction  3. Forecasting")
        print("  4. Classification      5. Correlation        6. Q&A System")
        print("  7. Anomaly Detection   8. Multi-Modal        9. Zero-Shot")
        print(" 10. Causal Explanations")
        print("="*80 + "\n")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load or create configuration"""
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        config = {
            'methods': {
                'sentiment': {'enabled': True, 'model': 'ProsusAI/finbert'},
                'ner': {'enabled': True, 'model': 'dslim/bert-base-NER'},
                'forecasting': {'enabled': True, 'model': 'amazon/chronos-t5-small'},
                'classification': {'enabled': True, 'model': 'facebook/bart-large-mnli'},
                'embeddings': {'enabled': True, 'model': 'sentence-transformers/all-MiniLM-L6-v2'},
                'qa': {'enabled': True, 'model': 'deepset/roberta-base-squad2'},
                'anomaly': {'enabled': True, 'model': 'statistical'},
                'multimodal': {'enabled': True, 'model': 'ProsusAI/finbert'},
                'zeroshot': {'enabled': True, 'model': 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'},
                'causal': {'enabled': True, 'model': 'google/flan-t5-base'}
            },
            'output_dir': 'hf_analytics_output',
            'cache_enabled': True
        }
        
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except:
            pass
        
        return config
    
    def initialize_method(self, method_name: str) -> bool:
        """Initialize specific method with error handling"""
        if method_name not in self.config['methods']:
            return False
        
        if not self.config['methods'][method_name]['enabled']:
            return False
        
        try:
            if method_name == 'sentiment':
                from hf_method1_sentiment import HFSentimentAnalyzer
                self.methods[method_name] = HFSentimentAnalyzer()
            elif method_name == 'ner':
                from hf_method2_ner import HFEntityExtractor
                self.methods[method_name] = HFEntityExtractor()
            elif method_name == 'forecasting':
                from hf_method3_forecasting import HFTimeSeriesForecaster
                self.methods[method_name] = HFTimeSeriesForecaster()
            elif method_name == 'classification':
                from hf_method4_classification import HFEventClassifier
                self.methods[method_name] = HFEventClassifier()
            elif method_name == 'embeddings':
                from hf_method5_embeddings import HFCorrelationDiscovery
                self.methods[method_name] = HFCorrelationDiscovery()
            elif method_name == 'qa':
                from hf_method6_qa import HFMarketQA
                self.methods[method_name] = HFMarketQA()
            elif method_name == 'anomaly':
                from hf_method7_anomaly import HFAnomalyDetector
                self.methods[method_name] = HFAnomalyDetector()
            elif method_name == 'multimodal':
                from hf_method8_multimodal import HFMultiModalAnalyzer
                self.methods[method_name] = HFMultiModalAnalyzer()
            elif method_name == 'zeroshot':
                from hf_method9_zeroshot import HFZeroShotCategorizer
                self.methods[method_name] = HFZeroShotCategorizer()
            elif method_name == 'causal':
                from hf_method10_causal import HFCausalExplainer
                self.methods[method_name] = HFCausalExplainer()
            else:
                return False
            
            if hasattr(self.methods[method_name], 'load_model'):
                self.methods[method_name].load_model()
            
            print(f"✓ {method_name}")
            return True
            
        except Exception as e:
            print(f"✗ {method_name}: {str(e)[:50]}")
            return False
    
    def initialize_all(self) -> Dict[str, bool]:
        """Initialize all enabled methods"""
        print("Initializing methods...")
        results = {}
        
        for method_name in self.config['methods'].keys():
            if self.config['methods'][method_name]['enabled']:
                results[method_name] = self.initialize_method(method_name)
        
        successful = sum(1 for v in results.values() if v)
        print(f"\n✓ {successful}/{len(results)} methods ready\n")
        
        return results
    
    def run_comprehensive_analysis(self,
                                   news_articles: Optional[List[Dict]] = None,
                                   price_data: Optional[Dict[str, pd.Series]] = None,
                                   economic_events: Optional[List[Dict]] = None) -> Dict:
        """Run all applicable analyses"""
        print("="*80)
        print("COMPREHENSIVE ANALYSIS")
        print("="*80)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'methods_used': [],
            'results': {},
            'errors': []
        }
        
        # Sentiment
        if 'sentiment' in self.methods and news_articles:
            try:
                analyzer = self.methods['sentiment']
                enhanced = analyzer.analyze_news_articles(news_articles)
                aggregated = analyzer.aggregate_sentiment(enhanced)
                results['results']['sentiment'] = {'articles': enhanced, 'aggregated': aggregated}
                results['methods_used'].append('sentiment')
            except Exception as e:
                results['errors'].append(f"Sentiment: {str(e)[:100]}")
        
        # Entity Extraction
        if 'ner' in self.methods and news_articles:
            try:
                extractor = self.methods['ner']
                analyzed = extractor.analyze_batch(news_articles)
                aggregated = extractor.aggregate_symbols(analyzed)
                results['results']['entities'] = {'analyzed': analyzed, 'aggregated': aggregated}
                results['methods_used'].append('ner')
            except Exception as e:
                results['errors'].append(f"NER: {str(e)[:100]}")
        
        # Forecasting
        if 'forecasting' in self.methods and price_data:
            forecasts = {}
            for symbol, data in price_data.items():
                try:
                    forecaster = self.methods['forecasting']
                    forecast = forecaster.forecast(data, forecast_horizon=7)
                    forecasts[symbol] = forecast
                except Exception as e:
                    results['errors'].append(f"Forecast {symbol}: {str(e)[:100]}")
            
            if forecasts:
                results['results']['forecasts'] = forecasts
                results['methods_used'].append('forecasting')
        
        # Event Classification
        if 'classification' in self.methods and economic_events:
            try:
                classifier = self.methods['classification']
                classified = classifier.classify_batch(economic_events)
                results['results']['classification'] = classified
                results['methods_used'].append('classification')
            except Exception as e:
                results['errors'].append(f"Classification: {str(e)[:100]}")
        
        # Generate insights
        results['insights'] = self._generate_insights(results['results'])
        
        # Generate explanation
        if 'causal' in self.methods and results['results']:
            try:
                explainer = self.methods['causal']
                explanation = explainer.generate_comprehensive_report(results['results'])
                results['explanation'] = explanation
            except Exception as e:
                results['errors'].append(f"Explanation: {str(e)[:100]}")
        
        return results
    
    def _generate_insights(self, analysis_results: Dict) -> List[str]:
        """Generate actionable insights"""
        insights = []
        
        if 'sentiment' in analysis_results:
            agg = analysis_results['sentiment'].get('aggregated', {})
            sentiment = agg.get('overall_sentiment', 'neutral')
            confidence = agg.get('confidence', 0)
            
            if sentiment == 'positive' and confidence > 0.6:
                insights.append(f"📈 Bullish sentiment (confidence: {confidence:.1%})")
            elif sentiment == 'negative' and confidence > 0.6:
                insights.append(f"📉 Bearish sentiment (confidence: {confidence:.1%})")
        
        if 'entities' in analysis_results:
            symbols = analysis_results['entities'].get('aggregated', {})
            top = symbols.get('top_symbols', [])
            if top:
                insights.append(f"🎯 Top symbols: {', '.join(top[:3])}")
        
        if 'forecasts' in analysis_results:
            for symbol, forecast in list(analysis_results['forecasts'].items())[:3]:
                if 'forecast_mean' in forecast and forecast['forecast_mean']:
                    trend = "↗" if forecast['forecast_mean'][-1] > forecast['forecast_mean'][0] else "↘"
                    insights.append(f"{trend} {symbol}: {forecast['method']} forecast")
        
        if not insights:
            insights.append("ℹ️  Analysis complete")
        
        return insights
    
    def save_results(self, results: Dict, filename: str = 'hf_analytics_results.json'):
        """Save results to file"""
        output_dir = self.config['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(output_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            print(f"✓ Saved: {filepath}")
            return filepath
        except Exception as e:
            print(f"✗ Save error: {e}")
            return None
    
    def generate_report(self, results: Dict) -> str:
        """Generate human-readable report"""
        lines = [
            "="*80,
            "HF ANALYTICS REPORT",
            "="*80,
            f"Generated: {results.get('timestamp', 'N/A')}",
            f"Methods: {', '.join(results.get('methods_used', []))}",
            "\n" + "="*80,
            "INSIGHTS",
            "="*80
        ]
        
        for insight in results.get('insights', []):
            lines.append(f"  {insight}")
        
        if results.get('errors'):
            lines.extend(["\n" + "="*80, "ERRORS", "="*80])
            for error in results['errors']:
                lines.append(f"  ⚠️  {error}")
        
        lines.extend(["\n" + "="*80, "DETAILED RESULTS", "="*80])
        
        if 'sentiment' in results.get('results', {}):
            agg = results['results']['sentiment'].get('aggregated', {})
            lines.extend([
                "\nSentiment:",
                f"  {agg.get('overall_sentiment', 'N/A').upper()}",
                f"  Bullish: {agg.get('bullish_score', 0):.1%}",
                f"  Bearish: {agg.get('bearish_score', 0):.1%}"
            ])
        
        if 'entities' in results.get('results', {}):
            symbols = results['results']['entities'].get('aggregated', {})
            lines.extend([
                "\nEntities:",
                f"  Symbols found: {symbols.get('symbol_count', 0)}",
                f"  Top: {', '.join(symbols.get('top_symbols', [])[:5])}"
            ])
        
        lines.append("\n" + "="*80)
        return "\n".join(lines)


if __name__ == "__main__":
    master = HFAnalyticsMaster()
    init_results = master.initialize_all()
    
    test_articles = [
        {'title': 'Gold prices surge on inflation fears'},
        {'title': 'Fed signals dovish stance'},
        {'title': 'Tech stocks rally on earnings'}
    ]
    
    test_events = [
        {'event': 'Non-Farm Payrolls', 'date': '2024-11-01'},
        {'event': 'Consumer Price Index', 'date': '2024-10-10'}
    ]
    
    if any(init_results.values()):
        print("Running test analysis...")
        
        results = master.run_comprehensive_analysis(
            news_articles=test_articles,
            economic_events=test_events
        )
        
        report = master.generate_report(results)
        print(report)
        
        master.save_results(results)
