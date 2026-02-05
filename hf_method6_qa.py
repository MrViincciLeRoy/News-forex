"""
HF Analytics Method 6: Market Intelligence Q&A
Answer questions about economic data using QA models
Models: deepset/roberta-base-squad2, Intel/dynamic_tinybert
Enhances: All analysis modules with natural language queries
"""

import pandas as pd
import json
from typing import List, Dict, Optional
from datetime import datetime


class HFMarketQA:
    """
    Question answering system for market intelligence
    Query economic data, COT positioning, correlations using natural language
    """
    
    def __init__(self, model_name: str = "deepset/roberta-base-squad2"):
        self.model_name = model_name
        self.pipeline = None
        self.knowledge_base = {}
        
        print(f"Initializing HF Market Q&A: {model_name}")
    
    def load_model(self):
        """Load question answering pipeline"""
        try:
            from transformers import pipeline
            
            print(f"Loading model: {self.model_name}")
            self.pipeline = pipeline(
                "question-answering",
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
    
    def index_economic_calendar(self, calendar_data: List[Dict]):
        """Index economic calendar for Q&A"""
        context_parts = []
        
        for event in calendar_data:
            event_text = f"On {event.get('date', 'unknown date')}, there was {event.get('event', 'an event')}. "
            
            if event.get('indicators'):
                ind = event['indicators']
                event_text += f"Gold price was ${ind.get('price', 0)}. "
                event_text += f"The overall signal was {ind.get('overall_signal', 'neutral')}. "
                
                if ind.get('indicators'):
                    for name, data in list(ind['indicators'].items())[:3]:
                        event_text += f"{name} showed {data.get('signal', 'neutral')} signal. "
            
            if event.get('news'):
                event_text += f"Related news: {' '.join([n.get('title', '') for n in event['news'][:2]])}. "
            
            context_parts.append(event_text)
        
        self.knowledge_base['economic_calendar'] = ' '.join(context_parts)
        print(f"✓ Indexed {len(calendar_data)} economic events")
    
    def index_cot_data(self, cot_data: List[Dict]):
        """Index COT positioning data for Q&A"""
        context_parts = []
        
        for data in cot_data:
            text = f"For {data.get('symbol', 'unknown')} on {data.get('report_date', 'unknown')}, "
            text += f"the sentiment was {data.get('sentiment', 'neutral')}. "
            
            if data.get('dealer'):
                text += f"Dealers had {data['dealer'].get('net', 0)} net positions. "
            
            if data.get('leveraged'):
                text += f"Hedge funds had {data['leveraged'].get('net', 0)} net positions. "
            
            context_parts.append(text)
        
        self.knowledge_base['cot_positioning'] = ' '.join(context_parts)
        print(f"✓ Indexed {len(cot_data)} COT reports")
    
    def index_correlations(self, correlation_data: Dict):
        """Index correlation analysis for Q&A"""
        context_parts = []
        
        for symbol, data in correlation_data.items():
            text = f"{symbol} analysis: "
            
            if data.get('top_positive_correlations'):
                text += "Positively correlated with: "
                text += ", ".join([f"{k} ({v:.2f})" for k, v in 
                                 list(data['top_positive_correlations'].items())[:3]])
                text += ". "
            
            if data.get('top_negative_correlations'):
                text += "Negatively correlated with: "
                text += ", ".join([f"{k} ({v:.2f})" for k, v in 
                                 list(data['top_negative_correlations'].items())[:3]])
                text += ". "
            
            context_parts.append(text)
        
        self.knowledge_base['correlations'] = ' '.join(context_parts)
        print(f"✓ Indexed {len(correlation_data)} correlation analyses")
    
    def index_news_impact(self, news_impact_data: List[Dict]):
        """Index news impact analysis for Q&A"""
        context_parts = []
        
        for analysis in news_impact_data:
            text = f"Event: {analysis.get('event_name', 'Unknown')} on {analysis.get('event_date', 'unknown')}. "
            
            if analysis.get('high_impact_symbols'):
                text += f"High impact symbols: {', '.join(analysis['high_impact_symbols'][:5])}. "
            
            if analysis.get('symbols'):
                for symbol, data in list(analysis['symbols'].items())[:3]:
                    if data.get('price_change_pct'):
                        text += f"{symbol} changed {data['price_change_pct'].get('total', 0)}%. "
            
            context_parts.append(text)
        
        self.knowledge_base['news_impact'] = ' '.join(context_parts)
        print(f"✓ Indexed {len(news_impact_data)} news impact analyses")
    
    def ask(self, question: str, context_type: str = 'all') -> Dict:
        """Ask a question about the indexed data"""
        if not self.pipeline:
            return {
                'question': question,
                'answer': 'Q&A model not loaded',
                'confidence': 0.0,
                'method': 'fallback'
            }
        
        if context_type == 'all':
            context = ' '.join(self.knowledge_base.values())
        else:
            context = self.knowledge_base.get(context_type, '')
        
        if not context:
            return {
                'question': question,
                'answer': 'No relevant context indexed',
                'confidence': 0.0,
                'method': 'fallback'
            }
        
        try:
            result = self.pipeline(
                question=question,
                context=context[:4000]  # Limit context length
            )
            
            return {
                'question': question,
                'answer': result['answer'],
                'confidence': round(result['score'], 4),
                'context_type': context_type,
                'method': 'qa_model'
            }
            
        except Exception as e:
            print(f"Q&A error: {e}")
            return {
                'question': question,
                'answer': f'Error: {str(e)}',
                'confidence': 0.0,
                'method': 'error'
            }
    
    def batch_ask(self, questions: List[str], context_type: str = 'all') -> List[Dict]:
        """Answer multiple questions"""
        results = []
        
        print(f"Answering {len(questions)} questions...")
        
        for i, question in enumerate(questions, 1):
            result = self.ask(question, context_type)
            results.append(result)
            
            if i % 5 == 0:
                print(f"  Processed {i}/{len(questions)}")
        
        return results
    
    def save_results(self, results: List[Dict], filepath: str):
        """Save Q&A results"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved: {filepath}")


if __name__ == "__main__":
    print("="*80)
    print("HF METHOD 6: MARKET INTELLIGENCE Q&A")
    print("="*80)
    
    qa_system = HFMarketQA()
    model_loaded = qa_system.load_model()
    
    sample_calendar = [
        {
            'date': '2024-11-01',
            'event': 'Non-Farm Payrolls',
            'indicators': {
                'price': 2650,
                'overall_signal': 'BUY',
                'indicators': {
                    'RSI': {'signal': 'NEUTRAL'},
                    'MACD': {'signal': 'BUY'}
                }
            },
            'news': [
                {'title': 'Strong jobs growth exceeds expectations'},
                {'title': 'Unemployment rate falls to 3.7%'}
            ]
        },
        {
            'date': '2024-10-10',
            'event': 'Consumer Price Index',
            'indicators': {
                'price': 2700,
                'overall_signal': 'SELL'
            }
        }
    ]
    
    sample_cot = [
        {
            'symbol': 'EUR',
            'report_date': '2024-10-29',
            'sentiment': 'BULLISH',
            'dealer': {'net': 50000},
            'leveraged': {'net': -30000}
        }
    ]
    
    qa_system.index_economic_calendar(sample_calendar)
    qa_system.index_cot_data(sample_cot)
    
    if model_loaded:
        test_questions = [
            "What was the gold price on November 1, 2024?",
            "What was the overall signal for Non-Farm Payrolls?",
            "What is the EUR sentiment?",
            "Which event had a BUY signal?",
            "What were the news headlines for NFP?"
        ]
        
        print("\n" + "="*80)
        print("TESTING Q&A SYSTEM")
        print("="*80)
        
        results = qa_system.batch_ask(test_questions)
        
        for result in results:
            print(f"\nQ: {result['question']}")
            print(f"A: {result['answer']} (confidence: {result['confidence']:.2%})")
        
        qa_system.save_results(results, 'hf_market_qa_results.json')
    
    print("\n" + "="*80)
    print("✓ Market Q&A complete")
    print("="*80)
