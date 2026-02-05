"""
HF Analytics Method 2: Named Entity Recognition (NER)
Extracts companies, currencies, commodities, indicators from news
Models: dslim/bert-base-NER, Jean-Baptiste/roberta-large-ner-english
Enhances: news_impact_analyzer.py - automatic symbol mapping
"""

import pandas as pd
from typing import List, Dict, Set, Tuple
import json
import re


class HFEntityExtractor:
    """
    Extract financial entities from text using NER models
    Identifies: Companies, Currencies, Commodities, Economic Indicators
    """
    
    def __init__(self, model_name: str = "dslim/bert-base-NER"):
        """
        Initialize NER model
        
        Args:
            model_name: HuggingFace NER model
        """
        self.model_name = model_name
        self.pipeline = None
        
        # Financial entity mappings
        self.entity_mappings = {
            'companies': {
                'apple': 'AAPL', 'microsoft': 'MSFT', 'google': 'GOOGL', 
                'alphabet': 'GOOGL', 'amazon': 'AMZN', 'tesla': 'TSLA',
                'meta': 'META', 'facebook': 'META', 'nvidia': 'NVDA',
                'jpmorgan': 'JPM', 'goldman sachs': 'GS', 'morgan stanley': 'MS',
                'bank of america': 'BAC', 'wells fargo': 'WFC'
            },
            'currencies': {
                'dollar': 'DX-Y.NYB', 'euro': 'EURUSD=X', 'pound': 'GBPUSD=X',
                'yen': 'USDJPY=X', 'yuan': 'CNYUSD=X', 'franc': 'USDCHF=X',
                'australian dollar': 'AUDUSD=X', 'canadian dollar': 'USDCAD=X'
            },
            'commodities': {
                'gold': 'GC=F', 'silver': 'SI=F', 'crude': 'CL=F', 'oil': 'CL=F',
                'brent': 'BZ=F', 'natural gas': 'NG=F', 'copper': 'HG=F',
                'platinum': 'PL=F', 'palladium': 'PA=F'
            },
            'indices': {
                's&p': '^GSPC', 's&p 500': '^GSPC', 'nasdaq': '^IXIC',
                'dow': '^DJI', 'dow jones': '^DJI', 'russell': '^RUT',
                'vix': '^VIX', 'volatility': '^VIX'
            },
            'economic_indicators': {
                'cpi': 'CPI', 'inflation': 'CPI', 'ppi': 'PPI',
                'nfp': 'NFP', 'payrolls': 'NFP', 'employment': 'NFP',
                'gdp': 'GDP', 'retail sales': 'RETAIL', 'jobless': 'CLAIMS',
                'fomc': 'FOMC', 'federal reserve': 'FED', 'fed': 'FED'
            }
        }
        
        print(f"Initializing HF Entity Extractor: {model_name}")
    
    def load_model(self):
        """Load NER pipeline from HuggingFace"""
        try:
            from transformers import pipeline
            
            print(f"Loading NER model: {self.model_name}")
            self.pipeline = pipeline("ner", model=self.model_name, aggregation_strategy="simple")
            
            print(f"✓ NER model loaded successfully")
            return True
            
        except ImportError:
            print("⚠️  transformers not installed")
            print("   Install: pip install transformers torch")
            return False
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return False
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text"""
        if self.pipeline:
            return self._extract_with_model(text)
        else:
            return self._extract_with_rules(text)
    
    def _extract_with_model(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using NER model"""
        try:
            results = self.pipeline(text)
            
            entities = {
                'organizations': [],
                'persons': [],
                'locations': [],
                'miscellaneous': []
            }
            
            for entity in results:
                entity_type = entity['entity_group']
                entity_text = entity['word']
                score = entity['score']
                
                if score > 0.5:  # Confidence threshold
                    if entity_type in ['ORG', 'ORGANIZATION']:
                        entities['organizations'].append(entity_text)
                    elif entity_type in ['PER', 'PERSON']:
                        entities['persons'].append(entity_text)
                    elif entity_type in ['LOC', 'LOCATION', 'GPE']:
                        entities['locations'].append(entity_text)
                    else:
                        entities['miscellaneous'].append(entity_text)
            
            # Deduplicate
            for key in entities:
                entities[key] = list(set(entities[key]))
            
            return entities
            
        except Exception as e:
            print(f"Model extraction error: {e}")
            return self._extract_with_rules(text)
    
    def _extract_with_rules(self, text: str) -> Dict[str, List[str]]:
        """Fallback rule-based extraction"""
        text_lower = text.lower()
        
        found_entities = {
            'companies': [],
            'currencies': [],
            'commodities': [],
            'indices': [],
            'economic_indicators': []
        }
        
        # Search for known entities
        for category, mappings in self.entity_mappings.items():
            for entity_name in mappings.keys():
                if entity_name in text_lower:
                    found_entities[category].append(entity_name)
        
        # Extract ticker symbols (e.g., AAPL, MSFT)
        ticker_pattern = r'\b[A-Z]{1,5}\b'
        tickers = re.findall(ticker_pattern, text)
        if tickers:
            found_entities['tickers'] = tickers
        
        # Deduplicate
        for key in found_entities:
            found_entities[key] = list(set(found_entities[key]))
        
        return found_entities
    
    def map_to_symbols(self, entities: Dict[str, List[str]]) -> Set[str]:
        """Map extracted entities to trading symbols"""
        symbols = set()
        
        for category, entity_list in entities.items():
            if category in self.entity_mappings:
                for entity in entity_list:
                    entity_lower = entity.lower()
                    if entity_lower in self.entity_mappings[category]:
                        symbols.add(self.entity_mappings[category][entity_lower])
        
        # Add any explicit tickers found
        if 'tickers' in entities:
            symbols.update(entities['tickers'])
        
        return symbols
    
    def analyze_article(self, article: Dict) -> Dict:
        """Extract entities and symbols from article"""
        title = article.get('title', '')
        content = article.get('content', '')
        
        # Combine title and content
        full_text = f"{title}. {content}"
        
        # Extract entities
        entities = self.extract_entities(full_text)
        
        # Map to symbols
        symbols = self.map_to_symbols(entities)
        
        return {
            'article': article,
            'entities': entities,
            'mapped_symbols': list(symbols),
            'symbol_count': len(symbols)
        }
    
    def analyze_batch(self, articles: List[Dict]) -> List[Dict]:
        """Analyze multiple articles"""
        results = []
        
        print(f"Analyzing {len(articles)} articles for entities...")
        
        for i, article in enumerate(articles, 1):
            result = self.analyze_article(article)
            results.append(result)
            
            if i % 10 == 0:
                print(f"  Processed {i}/{len(articles)}")
        
        return results
    
    def aggregate_symbols(self, analyzed_articles: List[Dict]) -> Dict:
        """Aggregate symbols across all articles"""
        all_symbols = set()
        symbol_frequency = {}
        
        for article in analyzed_articles:
            symbols = article.get('mapped_symbols', [])
            all_symbols.update(symbols)
            
            for symbol in symbols:
                symbol_frequency[symbol] = symbol_frequency.get(symbol, 0) + 1
        
        # Sort by frequency
        sorted_symbols = sorted(symbol_frequency.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'unique_symbols': list(all_symbols),
            'symbol_count': len(all_symbols),
            'symbol_frequency': dict(sorted_symbols),
            'top_symbols': [s[0] for s in sorted_symbols[:10]]
        }
    
    def save_results(self, results: Dict, filepath: str):
        """Save extraction results"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved: {filepath}")


if __name__ == "__main__":
    print("="*80)
    print("HF METHOD 2: NAMED ENTITY RECOGNITION")
    print("="*80)
    
    extractor = HFEntityExtractor()
    model_loaded = extractor.load_model()
    
    test_articles = [
        {
            'title': 'Apple and Microsoft lead tech rally as Nasdaq surges',
            'content': 'Major tech stocks rallied today...',
            'date': '2024-11-01'
        },
        {
            'title': 'Gold prices jump on dollar weakness and inflation fears',
            'content': 'Gold futures climbed...',
            'date': '2024-11-01'
        },
        {
            'title': 'Fed signals pause in rate hikes, S&P 500 responds positively',
            'content': 'The Federal Reserve indicated...',
            'date': '2024-11-01'
        },
        {
            'title': 'EUR/USD rises as European Central Bank holds rates steady',
            'content': 'The euro strengthened...',
            'date': '2024-11-01'
        },
        {
            'title': 'Non-Farm Payrolls beat expectations, unemployment falls to 3.7%',
            'content': 'Employment data released today...',
            'date': '2024-11-01'
        }
    ]
    
    print(f"\nAnalyzing {len(test_articles)} articles...")
    results = extractor.analyze_batch(test_articles)
    
    print("\n" + "="*80)
    print("ENTITY EXTRACTION RESULTS")
    print("="*80)
    
    for result in results:
        article = result['article']
        entities = result['entities']
        symbols = result['mapped_symbols']
        
        print(f"\n📰 {article['title']}")
        print(f"   Entities found: {sum(len(v) for v in entities.values())}")
        
        for category, items in entities.items():
            if items:
                print(f"   {category.capitalize()}: {', '.join(items[:3])}")
        
        if symbols:
            print(f"   ➜ Mapped Symbols: {', '.join(symbols)}")
        else:
            print(f"   ➜ No symbols mapped")
    
    # Aggregate
    aggregated = extractor.aggregate_symbols(results)
    
    print("\n" + "="*80)
    print("AGGREGATED SYMBOL ANALYSIS")
    print("="*80)
    print(f"\nTotal unique symbols: {aggregated['symbol_count']}")
    print(f"\nTop symbols by frequency:")
    for i, symbol in enumerate(aggregated['top_symbols'][:10], 1):
        freq = aggregated['symbol_frequency'][symbol]
        print(f"  {i}. {symbol}: {freq} mentions")
    
    # Save
    output = {
        'analyzed_articles': results,
        'aggregated_symbols': aggregated,
        'timestamp': datetime.now().isoformat()
    }
    
    extractor.save_results(output, 'hf_entity_extraction_results.json')
    
    print("\n" + "="*80)
    print("✓ Entity extraction complete")
    print("="*80)
