"""
HF Analytics Method 8: Multi-Modal Analysis
Combine text (news) + visual (charts) analysis
Models: microsoft/git-base, Salesforce/blip-image-captioning-base
Enhances: market_structure_analyzer.py + news_impact_analyzer.py
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
import json


class HFMultiModalAnalyzer:
    """
    Multi-modal analysis combining text and technical charts
    Provides holistic market view beyond single-mode analysis
    """
    
    def __init__(self, text_model: str = "ProsusAI/finbert",
                 image_model: str = "microsoft/git-base"):
        self.text_model_name = text_model
        self.image_model_name = image_model
        self.text_pipeline = None
        self.image_pipeline = None
        
        print(f"Initializing HF Multi-Modal Analyzer")
        print(f"  Text: {text_model}")
        print(f"  Image: {image_model}")
    
    def load_models(self):
        """Load both text and image models"""
        success = True
        
        try:
            from transformers import pipeline
            
            print("Loading text model...")
            self.text_pipeline = pipeline(
                "sentiment-analysis",
                model=self.text_model_name
            )
            print("✓ Text model loaded")
            
        except Exception as e:
            print(f"✗ Text model error: {e}")
            success = False
        
        print("\n⚠️  Image analysis requires PIL and additional setup")
        print("   Multimodal features limited to text in this demo")
        
        return success
    
    def analyze_text_context(self, news_articles: List[Dict]) -> Dict:
        """Analyze news text for market context"""
        if not self.text_pipeline:
            return self._fallback_text_analysis(news_articles)
        
        sentiments = []
        
        for article in news_articles:
            title = article.get('title', '')
            if not title:
                continue
            
            try:
                result = self.text_pipeline(title)[0]
                sentiments.append(result)
            except:
                continue
        
        if not sentiments:
            return {'sentiment': 'neutral', 'confidence': 0.0}
        
        positive_count = sum(1 for s in sentiments if s['label'] == 'positive')
        negative_count = sum(1 for s in sentiments if s['label'] == 'negative')
        
        overall = 'positive' if positive_count > negative_count else (
            'negative' if negative_count > positive_count else 'neutral'
        )
        
        avg_confidence = np.mean([s['score'] for s in sentiments])
        
        return {
            'sentiment': overall,
            'confidence': round(avg_confidence, 4),
            'article_count': len(news_articles),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'method': 'finbert'
        }
    
    def _fallback_text_analysis(self, news_articles: List[Dict]) -> Dict:
        """Fallback keyword-based text analysis"""
        bullish_words = ['gain', 'rise', 'up', 'growth', 'strong', 'positive']
        bearish_words = ['fall', 'drop', 'down', 'weak', 'negative', 'decline']
        
        bullish_count = 0
        bearish_count = 0
        
        for article in news_articles:
            title_lower = article.get('title', '').lower()
            
            bullish_count += sum(1 for word in bullish_words if word in title_lower)
            bearish_count += sum(1 for word in bearish_words if word in title_lower)
        
        overall = 'positive' if bullish_count > bearish_count else (
            'negative' if bearish_count > bullish_count else 'neutral'
        )
        
        return {
            'sentiment': overall,
            'confidence': 0.6,
            'article_count': len(news_articles),
            'method': 'fallback'
        }
    
    def analyze_technical_structure(self, price_data: pd.DataFrame) -> Dict:
        """Analyze technical chart patterns (text-based description)"""
        if price_data.empty:
            return {'structure': 'unknown', 'signals': []}
        
        signals = []
        
        recent = price_data.tail(20)
        
        sma_20 = recent['close'].mean()
        current_price = recent['close'].iloc[-1]
        
        if current_price > sma_20 * 1.02:
            signals.append('Price above 20-day average (bullish)')
        elif current_price < sma_20 * 0.98:
            signals.append('Price below 20-day average (bearish)')
        
        price_change = ((recent['close'].iloc[-1] - recent['close'].iloc[0]) /
                       recent['close'].iloc[0]) * 100
        
        if price_change > 5:
            signals.append('Strong uptrend (+5% over 20 days)')
        elif price_change < -5:
            signals.append('Strong downtrend (-5% over 20 days)')
        
        volatility = recent['close'].pct_change().std() * 100
        
        if volatility > 2:
            signals.append('High volatility detected')
        
        structure = 'bullish' if price_change > 2 else (
            'bearish' if price_change < -2 else 'neutral'
        )
        
        return {
            'structure': structure,
            'signals': signals,
            'price_change_20d': round(price_change, 2),
            'volatility': round(volatility, 2),
            'current_price': round(current_price, 2),
            'sma_20': round(sma_20, 2)
        }
    
    def combine_signals(self, text_analysis: Dict, 
                       technical_analysis: Dict) -> Dict:
        """Combine text sentiment and technical signals"""
        text_sentiment = text_analysis.get('sentiment', 'neutral')
        technical_structure = technical_analysis.get('structure', 'neutral')
        
        sentiment_score = {
            'positive': 1,
            'neutral': 0,
            'negative': -1,
            'bullish': 1,
            'bearish': -1
        }
        
        text_score = sentiment_score.get(text_sentiment, 0)
        tech_score = sentiment_score.get(technical_structure, 0)
        
        combined_score = (text_score + tech_score) / 2
        
        if combined_score > 0.5:
            combined_signal = 'STRONG_BUY'
        elif combined_score > 0:
            combined_signal = 'BUY'
        elif combined_score < -0.5:
            combined_signal = 'STRONG_SELL'
        elif combined_score < 0:
            combined_signal = 'SELL'
        else:
            combined_signal = 'NEUTRAL'
        
        confluence = text_sentiment == technical_structure
        
        return {
            'combined_signal': combined_signal,
            'text_sentiment': text_sentiment,
            'technical_structure': technical_structure,
            'confluence': confluence,
            'confidence': 'HIGH' if confluence else 'MODERATE',
            'combined_score': round(combined_score, 2),
            'text_confidence': text_analysis.get('confidence', 0.0),
            'signals': technical_analysis.get('signals', [])
        }
    
    def analyze_event_impact(self, event: Dict, news: List[Dict],
                            price_data: pd.DataFrame) -> Dict:
        """Comprehensive multi-modal event analysis"""
        text_analysis = self.analyze_text_context(news)
        
        technical_analysis = self.analyze_technical_structure(price_data)
        
        combined = self.combine_signals(text_analysis, technical_analysis)
        
        return {
            'event': event.get('event', 'Unknown'),
            'date': event.get('date', ''),
            'text_analysis': text_analysis,
            'technical_analysis': technical_analysis,
            'combined_analysis': combined,
            'recommendation': self._generate_recommendation(combined)
        }
    
    def _generate_recommendation(self, combined: Dict) -> str:
        """Generate trading recommendation"""
        signal = combined['combined_signal']
        confidence = combined['confidence']
        
        if signal == 'STRONG_BUY' and confidence == 'HIGH':
            return "Strong buy opportunity - both fundamentals and technicals align"
        elif signal == 'BUY' and confidence == 'HIGH':
            return "Buy signal - favorable conditions"
        elif signal == 'STRONG_SELL' and confidence == 'HIGH':
            return "Strong sell signal - risk elevated"
        elif signal == 'SELL' and confidence == 'HIGH':
            return "Sell signal - caution advised"
        elif confidence == 'MODERATE':
            return "Mixed signals - wait for clearer confluence"
        else:
            return "Neutral - no clear directional bias"
    
    def save_results(self, results: Dict, filepath: str):
        """Save analysis results"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"✓ Saved: {filepath}")


if __name__ == "__main__":
    import yfinance as yf
    
    print("="*80)
    print("HF METHOD 8: MULTI-MODAL ANALYSIS")
    print("="*80)
    
    analyzer = HFMultiModalAnalyzer()
    analyzer.load_models()
    
    test_event = {
        'event': 'Non-Farm Payrolls',
        'date': '2024-11-01'
    }
    
    test_news = [
        {'title': 'Strong jobs report exceeds expectations'},
        {'title': 'Employment growth accelerates'},
        {'title': 'Unemployment falls to new low'}
    ]
    
    symbol = 'GC=F'
    ticker = yf.Ticker(symbol)
    df = ticker.history(period='1mo')
    
    if not df.empty:
        df = df.rename(columns={
            'Close': 'close', 'High': 'high', 'Low': 'low'
        })
        
        print("\n" + "="*80)
        print("MULTI-MODAL ANALYSIS")
        print("="*80)
        
        result = analyzer.analyze_event_impact(test_event, test_news, df)
        
        print(f"\nEvent: {result['event']} ({result['date']})")
        
        print("\n📰 Text Analysis:")
        text = result['text_analysis']
        print(f"  Sentiment: {text['sentiment'].upper()}")
        print(f"  Confidence: {text['confidence']:.2%}")
        print(f"  Articles: {text['article_count']}")
        
        print("\n📊 Technical Analysis:")
        tech = result['technical_analysis']
        print(f"  Structure: {tech['structure'].upper()}")
        print(f"  20-day Change: {tech['price_change_20d']}%")
        print(f"  Signals:")
        for signal in tech['signals']:
            print(f"    • {signal}")
        
        print("\n🎯 Combined Analysis:")
        combined = result['combined_analysis']
        print(f"  Signal: {combined['combined_signal']}")
        print(f"  Confluence: {combined['confluence']}")
        print(f"  Confidence: {combined['confidence']}")
        
        print(f"\n💡 Recommendation:")
        print(f"  {result['recommendation']}")
        
        analyzer.save_results(result, 'hf_multimodal_results.json')
    
    print("\n" + "="*80)
    print("✓ Multi-modal analysis complete")
    print("="*80)
