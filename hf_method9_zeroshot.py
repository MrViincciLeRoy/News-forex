"""
HF Analytics Method 9: Dynamic Zero-Shot Categorization
Categorize new event types without retraining
Models: MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
Enhances: news_impact_analyzer.py - automatic category expansion
"""

import pandas as pd
from typing import List, Dict, Set
import json


class HFZeroShotCategorizer:
    """
    Zero-shot categorization for dynamic event classification
    Handles new event types without manual mapping
    """
    
    def __init__(self, model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"):
        self.model_name = model_name
        self.pipeline = None
        self.category_definitions = {}
        
        print(f"Initializing HF Zero-Shot Categorizer: {model_name}")
    
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
    
    def define_categories(self, categories: Dict[str, str]):
        """Define categories with descriptions for classification"""
        self.category_definitions = categories
        print(f"✓ Defined {len(categories)} categories")
    
    def categorize_event(self, event_text: str, 
                        candidate_categories: List[str] = None,
                        multi_label: bool = True) -> Dict:
        """Categorize event into defined categories"""
        if not self.pipeline:
            return self._fallback_categorization(event_text, candidate_categories)
        
        if candidate_categories is None:
            candidate_categories = list(self.category_definitions.keys())
        
        if not candidate_categories:
            return {'categories': [], 'scores': {}, 'method': 'no_categories'}
        
        try:
            result = self.pipeline(
                event_text,
                candidate_labels=candidate_categories,
                multi_label=multi_label,
                hypothesis_template="This event is related to {}."
            )
            
            categories = result['labels']
            scores = result['scores']
            
            return {
                'event_text': event_text[:100],
                'primary_category': categories[0],
                'primary_score': round(scores[0], 4),
                'all_categories': dict(zip(categories, 
                                          [round(s, 4) for s in scores])),
                'multi_label': multi_label,
                'method': 'zero_shot'
            }
            
        except Exception as e:
            print(f"Categorization error: {e}")
            return self._fallback_categorization(event_text, candidate_categories)
    
    def _fallback_categorization(self, event_text: str,
                                 categories: List[str]) -> Dict:
        """Keyword-based fallback categorization"""
        text_lower = event_text.lower()
        
        keyword_map = {
            'employment': ['job', 'employment', 'payroll', 'unemployment', 'labor'],
            'inflation': ['inflation', 'cpi', 'ppi', 'price'],
            'monetary_policy': ['fed', 'fomc', 'rate', 'central bank', 'monetary'],
            'growth': ['gdp', 'growth', 'economic activity', 'expansion'],
            'trade': ['trade', 'import', 'export', 'tariff'],
            'consumer': ['retail', 'consumer', 'spending', 'sales'],
            'manufacturing': ['manufacturing', 'industrial', 'production', 'pmi'],
            'housing': ['housing', 'home sales', 'construction', 'mortgage']
        }
        
        scores = {}
        for category in categories:
            keywords = keyword_map.get(category, [])
            score = sum(1 for kw in keywords if kw in text_lower) / max(len(keywords), 1)
            scores[category] = score
        
        sorted_cats = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'event_text': event_text[:100],
            'primary_category': sorted_cats[0][0] if sorted_cats else 'unknown',
            'primary_score': sorted_cats[0][1] if sorted_cats else 0.0,
            'all_categories': dict(sorted_cats),
            'method': 'fallback'
        }
    
    def categorize_batch(self, events: List[Dict],
                        text_field: str = 'event') -> List[Dict]:
        """Categorize multiple events"""
        results = []
        
        print(f"Categorizing {len(events)} events...")
        
        for i, event in enumerate(events, 1):
            event_text = event.get(text_field, '')
            if not event_text:
                continue
            
            category_result = self.categorize_event(event_text)
            
            result = {
                **event,
                'categorization': category_result
            }
            
            results.append(result)
            
            if i % 10 == 0:
                print(f"  Processed {i}/{len(events)}")
        
        return results
    
    def discover_new_categories(self, events: List[Dict],
                               text_field: str = 'event',
                               min_similarity: float = 0.7) -> List[str]:
        """Discover potential new categories from event patterns"""
        unique_events = set([e.get(text_field, '') for e in events])
        
        categorized = []
        for event_text in unique_events:
            if not event_text:
                continue
            
            result = self.categorize_event(event_text)
            
            if result.get('primary_score', 0) < min_similarity:
                categorized.append({
                    'text': event_text,
                    'best_match': result.get('primary_category'),
                    'score': result.get('primary_score')
                })
        
        if categorized:
            print(f"\n⚠️  Found {len(categorized)} events with weak categorization")
            print("   Consider adding new categories:")
            for item in categorized[:5]:
                print(f"     • {item['text']} (best: {item['best_match']}, {item['score']:.2f})")
        
        return [item['text'] for item in categorized]
    
    def update_symbol_mapping(self, category: str, symbols: List[str]):
        """Update symbol mapping for a category"""
        if category not in self.category_definitions:
            print(f"⚠️  Category '{category}' not defined")
            return
        
        if not hasattr(self, 'symbol_mappings'):
            self.symbol_mappings = {}
        
        self.symbol_mappings[category] = symbols
        print(f"✓ Updated symbols for '{category}': {', '.join(symbols[:5])}")
    
    def get_affected_symbols(self, categorization: Dict) -> List[str]:
        """Get affected symbols from categorization"""
        if not hasattr(self, 'symbol_mappings'):
            return []
        
        primary_category = categorization.get('primary_category')
        all_categories = categorization.get('all_categories', {})
        
        symbols = set()
        
        if primary_category and primary_category in self.symbol_mappings:
            symbols.update(self.symbol_mappings[primary_category])
        
        for category, score in all_categories.items():
            if score > 0.5 and category in self.symbol_mappings:
                symbols.update(self.symbol_mappings[category])
        
        return list(symbols)
    
    def save_results(self, results: List[Dict], filepath: str):
        """Save categorization results"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved: {filepath}")


if __name__ == "__main__":
    print("="*80)
    print("HF METHOD 9: DYNAMIC ZERO-SHOT CATEGORIZATION")
    print("="*80)
    
    categorizer = HFZeroShotCategorizer()
    model_loaded = categorizer.load_model()
    
    category_definitions = {
        'employment': 'Jobs, unemployment, labor market data',
        'inflation': 'Price levels, CPI, PPI, cost of living',
        'monetary_policy': 'Federal Reserve, interest rates, central bank policy',
        'growth': 'GDP, economic growth, expansion indicators',
        'trade': 'International trade, imports, exports',
        'consumer': 'Consumer spending, retail sales',
        'manufacturing': 'Industrial production, PMI indices',
        'housing': 'Real estate, home sales, construction'
    }
    
    categorizer.define_categories(category_definitions)
    
    categorizer.update_symbol_mapping('employment', 
                                     ['EURUSD=X', 'DX-Y.NYB', 'GC=F', '^GSPC'])
    categorizer.update_symbol_mapping('inflation',
                                     ['GC=F', 'TLT', 'BTC-USD', 'DX-Y.NYB'])
    categorizer.update_symbol_mapping('monetary_policy',
                                     ['EURUSD=X', 'GBPUSD=X', 'TLT', '^GSPC'])
    
    test_events = [
        {'event': 'Non-Farm Payrolls Report', 'date': '2024-11-01'},
        {'event': 'Consumer Price Index Release', 'date': '2024-10-10'},
        {'event': 'FOMC Interest Rate Decision', 'date': '2024-09-18'},
        {'event': 'Retail Sales Monthly Data', 'date': '2024-11-15'},
        {'event': 'ISM Manufacturing PMI', 'date': '2024-12-01'},
        {'event': 'Cryptocurrency Regulation Announcement', 'date': '2024-11-20'}
    ]
    
    print("\n" + "="*80)
    print("CATEGORIZING EVENTS")
    print("="*80)
    
    results = categorizer.categorize_batch(test_events)
    
    for result in results:
        cat = result['categorization']
        print(f"\n📅 {result['event']}")
        print(f"   Category: {cat['primary_category']} ({cat['primary_score']:.2%})")
        print(f"   Method: {cat['method']}")
        
        symbols = categorizer.get_affected_symbols(cat)
        if symbols:
            print(f"   ➜ Symbols: {', '.join(symbols[:5])}")
    
    print("\n" + "="*80)
    print("DISCOVERING NEW CATEGORIES")
    print("="*80)
    
    new_cats = categorizer.discover_new_categories(test_events, min_similarity=0.6)
    
    categorizer.save_results(results, 'hf_zeroshot_categorization_results.json')
    
    print("\n" + "="*80)
    print("✓ Zero-shot categorization complete")
    print("="*80)
