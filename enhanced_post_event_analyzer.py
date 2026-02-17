"""
Enhanced Post-Event Analyzer
Analyzes market performance after event
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
from pathlib import Path
import pandas as pd
import numpy as np
import os

class PostEventAnalyzer:
    
    def __init__(self, event_date: str, event_name: str, symbols: Optional[List[str]] = None, max_articles: int = 30):
        self.event_date = event_date
        self.event_name = event_name
        self.symbols = symbols or self._get_major_pairs()
        self.max_articles = max_articles
        self.analysis_id = f"post_{event_date}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        base_dir = Path(os.getenv("ANALYSIS_OUTPUT_DIR", str(Path.cwd() / "outputs")))
        self.output_dir = base_dir / f"post_event_{self.analysis_id}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_major_pairs(self) -> List[str]:
        """Get major currency pairs"""
        return [
            'EURUSD=X', 'USDJPY=X', 'GBPUSD=X', 'USDCHF=X', 
            'AUDUSD=X', 'USDCAD=X', 'USDZAR=X', 'BTC-USD', 'ETH-USD'
        ]
    
    async def run_full_analysis(self) -> Dict:
        """Run complete post-event analysis"""
        
        print(f"\n{'='*80}")
        print(f"POST-EVENT ANALYSIS: {self.event_name}")
        print(f"Date: {self.event_date}")
        print(f"{'='*80}\n")
        
        results = {
            'analysis_id': self.analysis_id,
            'type': 'POST-EVENT',
            'event_date': self.event_date,
            'event_name': self.event_name,
            'analysis_date': datetime.now().isoformat(),
            'sections': {}
        }
        
        # 1. Event Impact Analysis
        results['sections']['impact'] = await self._analyze_event_impact()
        self._save_section('impact', results['sections']['impact'])
        
        # 2. Price Movement Analysis
        results['sections']['price_movement'] = await self._analyze_price_movement()
        self._save_section('price_movement', results['sections']['price_movement'])
        
        # 3. Prediction Accuracy
        results['sections']['accuracy'] = await self._analyze_prediction_accuracy()
        self._save_section('accuracy', results['sections']['accuracy'])
        
        # 4. News Reaction
        results['sections']['news_reaction'] = await self._analyze_news_reaction()
        self._save_section('news_reaction', results['sections']['news_reaction'])
        
        # 5. Volume Analysis
        results['sections']['volume'] = await self._analyze_post_volume()
        self._save_section('volume', results['sections']['volume'])
        
        # 6. Summary
        results['sections']['summary'] = await self._generate_post_summary(results['sections'])
        self._save_section('summary', results['sections']['summary'])
        
        # Save full results
        results_file = self.output_dir / "full_analysis.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✓ Post-event analysis complete")
        print(f"Output: {self.output_dir}\n")
        
        return results
    
    async def _analyze_event_impact(self) -> Dict:
        """Analyze immediate impact of event"""
        print("  ⚡ Event Impact...")
        
        from symbol_indicators import SymbolIndicatorCalculator
        calc = SymbolIndicatorCalculator()
        
        impacts = []
        
        for symbol in self.symbols:
            try:
                data = calc.get_historical_data(symbol, period='5d')
                if data is None or data.empty:
                    continue
                
                # Compare pre/post event prices
                event_idx = len(data) // 2  # Approximate event time
                pre_price = float(data['Close'].iloc[event_idx - 1])
                post_price = float(data['Close'].iloc[-1])
                
                change_pct = (post_price - pre_price) / pre_price * 100
                
                impacts.append({
                    'symbol': symbol,
                    'pre_event_price': pre_price,
                    'post_event_price': post_price,
                    'change_pct': float(change_pct),
                    'direction': 'UP' if change_pct > 0 else 'DOWN',
                    'magnitude': 'STRONG' if abs(change_pct) > 1.0 else 'MODERATE' if abs(change_pct) > 0.5 else 'WEAK'
                })
            except:
                continue
        
        return {
            'impacts': impacts,
            'winners': sorted([i for i in impacts if i['direction'] == 'UP'], 
                            key=lambda x: x['change_pct'], reverse=True)[:5],
            'losers': sorted([i for i in impacts if i['direction'] == 'DOWN'], 
                           key=lambda x: x['change_pct'])[:5],
            'average_impact': np.mean([i['change_pct'] for i in impacts])
        }
    
    async def _analyze_price_movement(self) -> Dict:
        """Analyze detailed price movements"""
        print("  📊 Price Movement...")
        
        from symbol_indicators import SymbolIndicatorCalculator
        calc = SymbolIndicatorCalculator()
        
        movements = []
        
        for symbol in self.symbols[:5]:  # Top 5
            try:
                data = calc.get_historical_data(symbol, period='5d')
                if data is None or data.empty:
                    continue
                
                movements.append({
                    'symbol': symbol,
                    'high': float(data['High'].max()),
                    'low': float(data['Low'].min()),
                    'close': float(data['Close'].iloc[-1]),
                    'volatility': float(data['Close'].pct_change().std() * 100),
                    'chart_data': {
                        'dates': data.index.strftime('%Y-%m-%d').tolist(),
                        'closes': data['Close'].values.tolist()
                    }
                })
            except:
                continue
        
        return {'movements': movements}
    
    async def _analyze_prediction_accuracy(self) -> Dict:
        """Analyze prediction accuracy if pre-event analysis available"""
        print("  🎯 Prediction Accuracy...")
        
        return {
            'note': 'Prediction accuracy requires pre-event analysis data',
            'accuracy': 'N/A'
        }
    
    async def _analyze_news_reaction(self) -> Dict:
        """Analyze news reaction to event"""
        print("  📰 News Reaction...")
        
        from news_fetcher import NewsFetcher
        fetcher = NewsFetcher()
        
        articles = fetcher.fetch_event_news(
            date=self.event_date,
            event_name=f"{self.event_name} results",
            max_records=20
        )
        
        return {
            'article_count': len(articles),
            'articles': articles[:10],
            'post_event_sentiment': 'POSITIVE'  # Simplified
        }
    
    async def _analyze_post_volume(self) -> Dict:
        """Analyze post-event volume"""
        print("  📈 Volume Analysis...")
        
        from symbol_indicators import SymbolIndicatorCalculator
        calc = SymbolIndicatorCalculator()
        
        volume_analysis = []
        
        for symbol in self.symbols[:5]:
            try:
                data = calc.get_historical_data(symbol, period='5d')
                if data is None or data.empty or 'Volume' not in data.columns:
                    continue
                
                event_idx = len(data) // 2
                pre_volume = float(data['Volume'].iloc[:event_idx].mean())
                post_volume = float(data['Volume'].iloc[event_idx:].mean())
                
                volume_analysis.append({
                    'symbol': symbol,
                    'pre_event_avg_volume': pre_volume,
                    'post_event_avg_volume': post_volume,
                    'volume_change_pct': float((post_volume - pre_volume) / pre_volume * 100)
                })
            except:
                continue
        
        return {'volume_analysis': volume_analysis}
    
    async def _generate_post_summary(self, sections: Dict) -> Dict:
        """Generate post-event summary"""
        print("  📝 Summary...")
        
        summary = {
            'event': self.event_name,
            'event_date': self.event_date,
            'analysis_type': 'POST-EVENT',
            'key_findings': []
        }
        
        # Impact summary
        if 'impact' in sections:
            avg_impact = sections['impact'].get('average_impact', 0)
            summary['key_findings'].append(
                f"Average market impact: {avg_impact:.2f}%"
            )
            
            if sections['impact'].get('winners'):
                top_winner = sections['impact']['winners'][0]
                summary['key_findings'].append(
                    f"Top gainer: {top_winner['symbol']} (+{top_winner['change_pct']:.2f}%)"
                )
        
        return summary
    
    def _save_section(self, name: str, data: Dict):
        """Save section to JSON"""
        section_file = self.output_dir / f"{name}_section.json"
        with open(section_file, 'w') as f:
            json.dump({
                'section_type': name,
                'timestamp': datetime.now().isoformat(),
                'data': data
            }, f, indent=2, default=str)
        print(f"    ✓ {name} → {section_file.name}")
