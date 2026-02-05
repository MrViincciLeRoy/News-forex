"""
HF Analytics Method 4: Event Impact Classification
Zero-shot classification for economic events and market impact
Models: facebook/bart-large-mnli, MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
Enhances: news_impact_analyzer.py, economic_calendar_generator.py
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional


class HFEventClassifier:
    """
    Zero-shot classification for economic events
    Automatically categorizes events by impact without manual mapping
    """
    
    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        self.model_name = model_name
        self.pipeline = None
        
        self.impact_categories = [
            'high_impact_inflation',
            'high_impact_employment',
            'high_impact_monetary_policy',
            'medium_impact_growth',
            'medium_impact_sentiment',
            'low_impact_regional',
            'low_impact_technical'
        ]
        
        self.asset_categories = [
            'affects_currencies',
            'affects_commodities',
            'affects_equities',
            'affects_bonds',
            'affects_crypto'
        ]
        
        print(f"Initializing HF Event Classifier: {model_name}")
    
    def load_model(self):
        """Load zero-shot classification pipeline"""
        try:
            from transformers import pipeline
            
            print(f"Loading model: {self.model_name}")
            self.pipeline = pipeline(
                "zero-shot-classification",
                model=self.model_name
            )
            
            print("✓ Model loaded successfully")
            return True
            
        except ImportError:
            print("⚠️  transformers not installed")
            return False
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
    
    def classify_event(self, event_name: str, event_description: str = "") -> Dict:
        """Classify event impact and affected assets"""
        if not self.pipeline:
            return self._fallback_classification(event_name)
        
        try:
            text = f"{event_name}. {event_description}"
            
            impact_result = self.pipeline(
                text,
                candidate_labels=self.impact_categories,
                multi_label=True
            )
            
            asset_result = self.pipeline(
                text,
                candidate_labels=self.asset_categories,
                multi_label=True
            )
            
            impact_level = self._determine_impact_level(impact_result)
            
            return {
                'event_name': event_name,
                'impact_level': impact_level,
                'impact_scores': dict(zip(impact_result['labels'][:3], 
                                        impact_result['scores'][:3])),
                'affected_assets': dict(zip(asset_result['labels'][:3],
                                           asset_result['scores'][:3])),
                'primary_category': impact_result['labels'][0],
                'confidence': round(impact_result['scores'][0], 4),
                'method': 'zero_shot'
            }
            
        except Exception as e:
            print(f"Classification error: {e}")
            return self._fallback_classification(event_name)
    
    def _determine_impact_level(self, result: Dict) -> str:
        """Determine impact level from classification scores"""
        top_label = result['labels'][0]
        top_score = result['scores'][0]
        
        if 'high_impact' in top_label and top_score > 0.5:
            return 'HIGH'
        elif 'medium_impact' in top_label and top_score > 0.4:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _fallback_classification(self, event_name: str) -> Dict:
        """Keyword-based fallback classification"""
        event_lower = event_name.lower()
        
        high_impact_keywords = ['nfp', 'payroll', 'cpi', 'inflation', 'fomc', 
                                'fed', 'gdp', 'rate decision']
        medium_impact_keywords = ['ppi', 'retail', 'jobless', 'ism', 'sentiment']
        
        if any(kw in event_lower for kw in high_impact_keywords):
            impact = 'HIGH'
            category = 'high_impact_employment'
        elif any(kw in event_lower for kw in medium_impact_keywords):
            impact = 'MEDIUM'
            category = 'medium_impact_growth'
        else:
            impact = 'LOW'
            category = 'low_impact_regional'
        
        return {
            'event_name': event_name,
            'impact_level': impact,
            'impact_scores': {category: 0.8},
            'affected_assets': {'affects_currencies': 0.7},
            'primary_category': category,
            'confidence': 0.5,
            'method': 'fallback'
        }
    
    def classify_batch(self, events: List[Dict]) -> List[Dict]:
        """Classify multiple events"""
        results = []
        
        print(f"Classifying {len(events)} events...")
        
        for i, event in enumerate(events, 1):
            event_name = event.get('event', event.get('title', ''))
            description = event.get('description', '')
            
            result = self.classify_event(event_name, description)
            
            result['event_date'] = event.get('date', '')
            results.append(result)
            
            if i % 10 == 0:
                print(f"  Processed {i}/{len(events)}")
        
        return results
    
    def get_affected_symbols(self, classification: Dict) -> List[str]:
        """Map classification to trading symbols"""
        symbol_mapping = {
            'affects_currencies': ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'DX-Y.NYB'],
            'affects_commodities': ['GC=F', 'CL=F', 'SI=F'],
            'affects_equities': ['^GSPC', '^IXIC', '^DJI'],
            'affects_bonds': ['TLT', 'IEF', 'SHY'],
            'affects_crypto': ['BTC-USD', 'ETH-USD']
        }
        
        symbols = set()
        
        for asset_type, score in classification['affected_assets'].items():
            if score > 0.3:
                symbols.update(symbol_mapping.get(asset_type, []))
        
        return list(symbols)
    
    def save_results(self, results: List[Dict], filepath: str):
        """Save classification results"""
        import json
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved: {filepath}")


if __name__ == "__main__":
    print("="*80)
    print("HF METHOD 4: EVENT IMPACT CLASSIFICATION")
    print("="*80)
    
    classifier = HFEventClassifier()
    model_loaded = classifier.load_model()
    
    test_events = [
        {
            'event': 'Non-Farm Payrolls',
            'description': 'Monthly employment report showing job creation',
            'date': '2024-11-01'
        },
        {
            'event': 'Consumer Price Index',
            'description': 'Inflation measure tracking consumer prices',
            'date': '2024-10-10'
        },
        {
            'event': 'FOMC Rate Decision',
            'description': 'Federal Reserve interest rate policy announcement',
            'date': '2024-09-18'
        },
        {
            'event': 'ISM Manufacturing PMI',
            'description': 'Manufacturing sector activity index',
            'date': '2024-12-01'
        },
        {
            'event': 'Retail Sales',
            'description': 'Consumer spending on retail goods',
            'date': '2024-11-15'
        }
    ]
    
    results = classifier.classify_batch(test_events)
    
    print("\n" + "="*80)
    print("CLASSIFICATION RESULTS")
    print("="*80)
    
    for result in results:
        print(f"\n📅 {result['event_name']} ({result['event_date']})")
        print(f"   Impact: {result['impact_level']} (confidence: {result['confidence']:.2%})")
        print(f"   Category: {result['primary_category']}")
        
        print("   Top Impact Types:")
        for category, score in list(result['impact_scores'].items())[:2]:
            print(f"     {category}: {score:.2%}")
        
        print("   Affected Assets:")
        for asset, score in list(result['affected_assets'].items())[:2]:
            print(f"     {asset}: {score:.2%}")
        
        symbols = classifier.get_affected_symbols(result)
        print(f"   ➜ Symbols: {', '.join(symbols[:5])}")
    
    classifier.save_results(results, 'hf_event_classification_results.json')
    
    print("\n" + "="*80)
    print("✓ Event classification complete")
    print("="*80)
