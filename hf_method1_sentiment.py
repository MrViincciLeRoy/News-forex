"""
HF Method 1: Sentiment Analysis - CORRECTED VERSION
Uses FinBERT for financial sentiment analysis with clean output
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import logging as transformers_logging
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

# Suppress warnings globally
transformers_logging.set_verbosity_error()
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class HFSentimentAnalyzer:
    """
    Financial sentiment analyzer using FinBERT
    Clean loading without warnings
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.labels = ['negative', 'neutral', 'positive']
    
    def load_model(self):
        """Load FinBERT model quietly (no warnings)"""
        
        # Suppress logging during load
        old_level = transformers_logging.get_verbosity()
        transformers_logging.set_verbosity_error()
        
        try:
            print("Loading FinBERT...")
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                'ProsusAI/finbert',
                num_labels=3
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
            
            # Set device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.model.eval()
            
            print("✓ Text model loaded")
            
        finally:
            # Restore logging level
            transformers_logging.set_verbosity(old_level)
    
    def analyze_text(self, text):
        """
        Analyze sentiment of a single text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        if self.model is None:
            self.load_model()
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[0]
        
        # Convert to dict
        sentiment_scores = {
            label: float(prob) 
            for label, prob in zip(self.labels, probs)
        }
        
        # Determine overall sentiment
        max_label = max(sentiment_scores, key=sentiment_scores.get)
        
        return {
            'sentiment': max_label,
            'confidence': sentiment_scores[max_label],
            'scores': sentiment_scores
        }
    
    def analyze_news_articles(self, articles):
        """
        Analyze sentiment of multiple news articles
        
        Args:
            articles: List of article dictionaries with 'title' and 'content'
            
        Returns:
            List of sentiment analyses
        """
        if self.model is None:
            self.load_model()
        
        results = []
        
        for article in articles:
            # Combine title and content
            text = f"{article.get('title', '')} {article.get('content', '')}"
            
            # Analyze
            sentiment = self.analyze_text(text)
            
            results.append({
                'title': article.get('title', ''),
                'date': article.get('date', ''),
                'sentiment': sentiment['sentiment'],
                'confidence': sentiment['confidence'],
                'scores': sentiment['scores']
            })
        
        return results
    
    def aggregate_sentiment(self, sentiment_results):
        """
        Aggregate sentiment from multiple analyses
        
        Args:
            sentiment_results: List of sentiment analysis results
            
        Returns:
            Dictionary with aggregated sentiment
        """
        if not sentiment_results:
            return {
                'overall_sentiment': 'neutral',
                'confidence': 0.0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'total_articles': 0
            }
        
        # Count sentiments
        positive_count = sum(1 for r in sentiment_results if r['sentiment'] == 'positive')
        negative_count = sum(1 for r in sentiment_results if r['sentiment'] == 'negative')
        neutral_count = sum(1 for r in sentiment_results if r['sentiment'] == 'neutral')
        
        # Calculate average confidence
        avg_confidence = np.mean([r['confidence'] for r in sentiment_results])
        
        # Determine overall sentiment
        if positive_count > negative_count and positive_count > neutral_count:
            overall = 'positive'
        elif negative_count > positive_count and negative_count > neutral_count:
            overall = 'negative'
        else:
            overall = 'neutral'
        
        # Calculate weighted score
        total_positive = sum(r['scores']['positive'] for r in sentiment_results)
        total_negative = sum(r['scores']['negative'] for r in sentiment_results)
        total_neutral = sum(r['scores']['neutral'] for r in sentiment_results)
        
        total_score = total_positive + total_negative + total_neutral
        
        return {
            'overall_sentiment': overall,
            'confidence': float(avg_confidence),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'total_articles': len(sentiment_results),
            'positive_ratio': positive_count / len(sentiment_results),
            'negative_ratio': negative_count / len(sentiment_results),
            'weighted_scores': {
                'positive': float(total_positive / total_score) if total_score > 0 else 0,
                'negative': float(total_negative / total_score) if total_score > 0 else 0,
                'neutral': float(total_neutral / total_score) if total_score > 0 else 0
            }
        }
    
    def batch_analyze(self, texts, batch_size=8):
        """
        Analyze multiple texts in batches for efficiency
        
        Args:
            texts: List of texts
            batch_size: Batch size for processing
            
        Returns:
            List of sentiment results
        """
        if self.model is None:
            self.load_model()
        
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
            
            # Process results
            for j, prob in enumerate(probs):
                sentiment_scores = {
                    label: float(p) 
                    for label, p in zip(self.labels, prob)
                }
                
                max_label = max(sentiment_scores, key=sentiment_scores.get)
                
                results.append({
                    'text': batch[j],
                    'sentiment': max_label,
                    'confidence': sentiment_scores[max_label],
                    'scores': sentiment_scores
                })
        
        return results


if __name__ == "__main__":
    """Test sentiment analyzer"""
    
    print("="*80)
    print("SENTIMENT ANALYZER TEST")
    print("="*80)
    
    # Create analyzer
    analyzer = HFSentimentAnalyzer()
    analyzer.load_model()
    
    # Test articles
    test_articles = [
        {
            'title': 'Stock Market Rallies on Strong Jobs Report',
            'content': 'Markets surged today following better than expected employment data.',
            'date': '2024-11-01'
        },
        {
            'title': 'Fed Signals Rate Cuts Ahead',
            'content': 'The Federal Reserve indicated potential rate cuts in coming months.',
            'date': '2024-11-01'
        },
        {
            'title': 'Recession Fears Mount Amid Economic Data',
            'content': 'Economists warn of potential recession as indicators weaken.',
            'date': '2024-11-01'
        }
    ]
    
    print("\nAnalyzing articles...")
    results = analyzer.analyze_news_articles(test_articles)
    
    print("\nResults:")
    for r in results:
        print(f"\n  Title: {r['title']}")
        print(f"  Sentiment: {r['sentiment'].upper()}")
        print(f"  Confidence: {r['confidence']:.2%}")
    
    print("\n\nAggregated Sentiment:")
    aggregated = analyzer.aggregate_sentiment(results)
    print(f"  Overall: {aggregated['overall_sentiment'].upper()}")
    print(f"  Confidence: {aggregated['confidence']:.2%}")
    print(f"  Positive: {aggregated['positive_count']}")
    print(f"  Negative: {aggregated['negative_count']}")
    print(f"  Neutral: {aggregated['neutral_count']}")
    
    # Save results
    import json
    with open('hf_sentiment_results.json', 'w') as f:
        json.dump({
            'individual_results': results,
            'aggregated': aggregated
        }, f, indent=2)
    
    print("\n✓ Results saved to hf_sentiment_results.json")
    print("="*80)
