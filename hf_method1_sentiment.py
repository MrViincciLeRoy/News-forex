"""
HF Analytics Method 1: Financial Sentiment Analyzer
Uses FinBERT to analyze sentiment of financial news and market commentary
Models: ProsusAI/finbert, yiyanghkust/finbert-tone
Enhances: news_impact_analyzer.py, cot_news_integration.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
import os


class HFSentimentAnalyzer:
    """
    Financial Sentiment Analysis using Hugging Face FinBERT
    Analyzes news articles, headlines, and market commentary
    """
    
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initialize sentiment analyzer
        
        Args:
            model_name: Hugging Face model identifier
                       Options: "ProsusAI/finbert", "yiyanghkust/finbert-tone"
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = None
        self.sentiment_cache = {}
        
        print(f"Initializing HF Sentiment Analyzer: {model_name}")
    
    def load_model(self):
        """Load FinBERT model from Hugging Face"""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            
            print(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✓ Model loaded successfully on {self.device}")
            return True
            
        except ImportError:
            print("⚠️  transformers/torch not installed")
            print("   Install: pip install transformers torch")
            return False
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return False
    
    def analyze_text(self, text: str) -> Dict:
        """Analyze sentiment of text"""
        if not self.model:
            return self._fallback_sentiment(text)
        
        if text in self.sentiment_cache:
            return self.sentiment_cache[text]
        
        try:
            import torch
            
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            scores = predictions[0].cpu().numpy()
            labels = ['positive', 'negative', 'neutral']
            sentiment_scores = {label: float(score) for label, score in zip(labels, scores)}
            
            max_label = max(sentiment_scores, key=sentiment_scores.get)
            
            result = {
                'text': text[:100] + '...' if len(text) > 100 else text,
                'sentiment': max_label,
                'confidence': round(sentiment_scores[max_label], 4),
                'scores': {k: round(v, 4) for k, v in sentiment_scores.items()},
                'bullish_score': round(sentiment_scores.get('positive', 0), 4),
                'bearish_score': round(sentiment_scores.get('negative', 0), 4),
                'neutral_score': round(sentiment_scores.get('neutral', 0), 4),
                'method': 'finbert'
            }
            
            self.sentiment_cache[text] = result
            return result
            
        except Exception as e:
            print(f"Error: {e}")
            return self._fallback_sentiment(text)
    
    def _fallback_sentiment(self, text: str) -> Dict:
        """Fallback keyword-based sentiment"""
        text_lower = text.lower()
        
        bullish_words = ['gain', 'rise', 'up', 'growth', 'strong', 'beat', 'exceed', 
                         'positive', 'surge', 'rally', 'boost', 'improve', 'recovery']
        bearish_words = ['fall', 'drop', 'down', 'weak', 'miss', 'decline', 'negative',
                         'plunge', 'crash', 'loss', 'concern', 'risk', 'fear']
        
        bullish_count = sum(1 for word in bullish_words if word in text_lower)
        bearish_count = sum(1 for word in bearish_words if word in text_lower)
        
        total = bullish_count + bearish_count
        if total == 0:
            return {
                'text': text[:100] + '...' if len(text) > 100 else text,
                'sentiment': 'neutral',
                'confidence': 0.5,
                'scores': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34},
                'bullish_score': 0.33,
                'bearish_score': 0.33,
                'neutral_score': 0.34,
                'method': 'fallback'
            }
        
        bullish_score = bullish_count / total
        bearish_score = bearish_count / total
        
        sentiment = 'positive' if bullish_score > bearish_score else ('negative' if bearish_score > bullish_score else 'neutral')
        
        return {
            'text': text[:100] + '...' if len(text) > 100 else text,
            'sentiment': sentiment,
            'confidence': round(max(bullish_score, bearish_score), 4),
            'scores': {
                'positive': round(bullish_score, 4),
                'negative': round(bearish_score, 4),
                'neutral': round(1 - bullish_score - bearish_score, 4)
            },
            'bullish_score': round(bullish_score, 4),
            'bearish_score': round(bearish_score, 4),
            'neutral_score': round(1 - bullish_score - bearish_score, 4),
            'method': 'fallback'
        }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Analyze multiple texts"""
        return [self.analyze_text(text) for text in texts]
    
    def analyze_news_articles(self, articles: List[Dict]) -> List[Dict]:
        """Analyze news articles with sentiment"""
        enhanced = []
        for article in articles:
            title = article.get('title', '')
            sentiment = self.analyze_text(title)
            
            enhanced_article = article.copy()
            enhanced_article['sentiment_analysis'] = sentiment
            enhanced.append(enhanced_article)
        
        return enhanced
    
    def aggregate_sentiment(self, articles: List[Dict]) -> Dict:
        """Aggregate sentiment across articles"""
        if not articles:
            return {
                'overall_sentiment': 'neutral',
                'bullish_score': 0.33,
                'bearish_score': 0.33,
                'article_count': 0
            }
        
        bullish_scores = [a.get('sentiment_analysis', {}).get('bullish_score', 0) for a in articles]
        bearish_scores = [a.get('sentiment_analysis', {}).get('bearish_score', 0) for a in articles]
        
        avg_bullish = np.mean(bullish_scores)
        avg_bearish = np.mean(bearish_scores)
        
        overall = 'positive' if avg_bullish > avg_bearish + 0.1 else ('negative' if avg_bearish > avg_bullish + 0.1 else 'neutral')
        
        return {
            'overall_sentiment': overall,
            'bullish_score': round(avg_bullish, 4),
            'bearish_score': round(avg_bearish, 4),
            'confidence': round(max(avg_bullish, avg_bearish), 4),
            'article_count': len(articles)
        }
    
    def save_results(self, results: Dict, filepath: str):
        """Save analysis to JSON"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved: {filepath}")


if __name__ == "__main__":
    print("="*80)
    print("HF METHOD 1: FINANCIAL SENTIMENT ANALYSIS")
    print("="*80)
    
    analyzer = HFSentimentAnalyzer()
    model_loaded = analyzer.load_model()
    
    test_articles = [
        {'title': 'Gold prices surge as inflation fears mount', 'date': '2024-11-01'},
        {'title': 'Fed signals dovish stance on rates', 'date': '2024-11-01'},
        {'title': 'Stock market plunges on weak data', 'date': '2024-11-01'},
        {'title': 'Dollar strengthens against major currencies', 'date': '2024-11-01'}
    ]
    
    enhanced = analyzer.analyze_news_articles(test_articles)
    
    for article in enhanced:
        sent = article['sentiment_analysis']
        print(f"\n{article['title']}")
        print(f"  Sentiment: {sent['sentiment'].upper()}")
        print(f"  Bullish: {sent['bullish_score']:.2%} | Bearish: {sent['bearish_score']:.2%}")
    
    aggregated = analyzer.aggregate_sentiment(enhanced)
    print(f"\n{'='*80}")
    print(f"Overall: {aggregated['overall_sentiment'].upper()}")
    print(f"Bullish: {aggregated['bullish_score']:.2%} | Bearish: {aggregated['bearish_score']:.2%}")
    
    analyzer.save_results({'articles': enhanced, 'aggregated': aggregated}, 'hf_sentiment_results.json')
