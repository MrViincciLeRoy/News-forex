"""
HF Analytics Method 10: Causal Inference & Explanations
Generate natural language explanations for market behavior
Uses rule-based system (HF models optional)
"""

import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime
import json


class HFCausalExplainer:
    """
    Generate causal explanations for market movements
    Explains WHY correlations exist and WHAT drives patterns
    """
    
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.pipeline = None
        self.explanation_cache = {}
        
        print(f"Initializing HF Causal Explainer: {model_name}")
    
    def load_model(self):
        """Load text generation model (optional - falls back to rules)"""
        try:
            from transformers import pipeline
            
            print(f"Loading model: {self.model_name}")
            
            self.pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                max_length=150,
                truncation=True
            )
            
            print("✓ Model loaded successfully")
            return True
            
        except ImportError:
            print("⚠️  transformers not installed, using rule-based explanations")
            return False
        except Exception as e:
            print(f"⚠️  Model load failed ({e}), using rule-based explanations")
            return False
    
    def explain_correlation(self, asset1: str, asset2: str,
                           correlation: float, context: Dict = None) -> str:
        """Explain why two assets are correlated"""
        if abs(correlation) < 0.3:
            relationship = "weakly related"
        elif correlation > 0.7:
            relationship = "strongly positively correlated"
        elif correlation < -0.7:
            relationship = "strongly inversely correlated"
        elif correlation > 0:
            relationship = "moderately positively correlated"
        else:
            relationship = "moderately inversely correlated"
        
        if self.pipeline:
            prompt = f"Explain why {asset1} and {asset2} are {relationship} in financial markets:"
            
            try:
                result = self.pipeline(prompt, max_length=150, num_return_sequences=1)[0]['generated_text']
                explanation = result.replace(prompt, '').strip()
                if explanation:
                    return explanation
            except:
                pass
        
        return self._generate_correlation_explanation(asset1, asset2, correlation)
    
    def _generate_correlation_explanation(self, asset1: str, asset2: str,
                                         correlation: float) -> str:
        """Rule-based correlation explanation"""
        explanations = {
            ('GOLD', 'DX-Y.NYB'): "Gold and the US Dollar typically move inversely because gold is priced in dollars. When the dollar strengthens, gold becomes more expensive for foreign buyers, reducing demand.",
            ('GOLD', 'TLT'): "Gold and Treasury bonds both serve as safe-haven assets. When investors seek safety, they buy both, creating positive correlation.",
            ('^GSPC', '^VIX'): "The S&P 500 and VIX (fear index) are inversely correlated. When stocks fall, volatility and fear rise.",
            ('EURUSD=X', 'DX-Y.NYB'): "EUR/USD and the US Dollar Index move inversely since the Dollar Index measures USD strength against a basket including the Euro.",
            ('CL=F', 'EURUSD=X'): "Oil and EUR/USD often correlate because oil is priced in dollars. Rising oil prices can weaken the dollar."
        }
        
        key = (asset1, asset2)
        reverse_key = (asset2, asset1)
        
        if key in explanations:
            return explanations[key]
        elif reverse_key in explanations:
            return explanations[reverse_key]
        else:
            if correlation > 0:
                return f"{asset1} and {asset2} tend to move in the same direction, possibly due to shared economic drivers or market sentiment."
            else:
                return f"{asset1} and {asset2} typically move in opposite directions, suggesting they respond differently to economic conditions."
    
    def explain_event_impact(self, event: str, affected_symbols: List[str],
                            impact_data: Dict = None) -> str:
        """Explain why an event affects specific symbols"""
        if self.pipeline:
            symbols_str = ', '.join(affected_symbols[:5])
            prompt = f"Explain how {event} affects {symbols_str}:"
            
            try:
                result = self.pipeline(prompt, max_length=150, num_return_sequences=1)[0]['generated_text']
                explanation = result.replace(prompt, '').strip()
                if explanation:
                    return explanation
            except:
                pass
        
        return self._generate_event_explanation(event, affected_symbols)
    
    def _generate_event_explanation(self, event: str, symbols: List[str]) -> str:
        """Rule-based event explanation"""
        event_lower = event.lower()
        
        if 'payroll' in event_lower or 'employment' in event_lower:
            return "Employment data affects currency markets because strong job growth indicates economic strength, influencing central bank policy. It impacts gold as investors assess inflation risks, and affects stocks through consumer spending expectations."
        
        elif 'cpi' in event_lower or 'inflation' in event_lower:
            return "Inflation data is crucial because it influences central bank decisions on interest rates. Higher inflation typically strengthens the dollar (as rates may rise), pressures gold higher as an inflation hedge, and can hurt stocks by compressing valuations."
        
        elif 'fomc' in event_lower or 'fed' in event_lower:
            return "Federal Reserve decisions directly impact interest rates, which affect all asset classes. Rate hikes strengthen the dollar, pressure gold (which pays no yield), and influence stock valuations through discount rates."
        
        elif 'gdp' in event_lower:
            return "GDP reports measure overall economic health, affecting investor risk appetite. Strong growth supports stocks and currencies but may pressure bonds. Weak growth often benefits safe havens like gold."
        
        else:
            return f"This event affects multiple markets through shifts in economic expectations, risk sentiment, and capital flows across global asset classes."
    
    def explain_seasonality(self, asset: str, pattern: Dict) -> str:
        """Explain seasonal patterns"""
        best_month = pattern.get('best_month', 'unknown')
        worst_month = pattern.get('worst_month', 'unknown')
        
        return self._generate_seasonality_explanation(asset, best_month, worst_month)
    
    def _generate_seasonality_explanation(self, asset: str, best: str, worst: str) -> str:
        """Rule-based seasonality explanation"""
        if 'GOLD' in asset.upper():
            if best in ['Jan', 'Feb', 'Aug', 'Sep']:
                return f"Gold often strengthens in {best} due to jewelry demand (wedding seasons), central bank buying, and safe-haven flows during summer uncertainty."
            return "Gold seasonality is influenced by jewelry demand cycles, central bank buying patterns, and traditional investment flows."
        
        elif '^GSPC' in asset or 'SPY' in asset:
            return "Stock market seasonality is driven by institutional rebalancing, tax considerations (year-end), the 'Santa Claus rally,' and the 'sell in May' pattern."
        
        else:
            return f"{asset} seasonality reflects recurring patterns in supply/demand, institutional flows, and economic cycles throughout the year."
    
    def explain_anomaly(self, anomaly_data: Dict) -> str:
        """Explain why an anomaly occurred"""
        anomaly_type = anomaly_data.get('type', 'unknown')
        
        return self._get_anomaly_explanation(anomaly_type)
    
    def _get_anomaly_explanation(self, anomaly_type: str) -> str:
        """Get predefined anomaly explanation"""
        explanations = {
            'PRICE_ANOMALY': "Significant price deviation often results from unexpected news, large institutional orders, algorithmic trading cascades, or sudden shifts in market sentiment.",
            'VOLUME_ANOMALY': "Unusual volume spikes typically occur around major news events, earnings releases, derivative expiration, or when large institutional positions are established or unwound.",
            'VOLATILITY_REGIME_CHANGE': "Volatility regime shifts happen when market uncertainty changes dramatically, often due to geopolitical events, policy changes, or economic data surprises.",
            'PRICE_GAP': "Price gaps occur when markets open significantly away from the previous close, usually due to overnight news, earnings announcements, or developments during market closure."
        }
        
        return explanations.get(anomaly_type, "Market anomalies typically result from unexpected events, structural breaks, or shifts in investor behavior patterns.")
    
    def generate_comprehensive_report(self, analysis_data: Dict) -> str:
        """Generate comprehensive narrative report"""
        sections = []
        
        sections.append("=== MARKET ANALYSIS REPORT ===\n")
        sections.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        if 'correlations' in analysis_data:
            sections.append("\n--- CORRELATION ANALYSIS ---")
            for item in analysis_data['correlations'][:3]:
                asset1 = item.get('asset1', 'Asset 1')
                asset2 = item.get('asset2', 'Asset 2')
                corr = item.get('correlation', 0)
                
                explanation = self.explain_correlation(asset1, asset2, corr)
                sections.append(f"\n{asset1} vs {asset2} ({corr:.2f}):")
                sections.append(f"  {explanation}")
        
        if 'events' in analysis_data:
            sections.append("\n\n--- EVENT IMPACT ANALYSIS ---")
            for event in analysis_data['events'][:2]:
                event_name = event.get('event_name', 'Event')
                symbols = event.get('affected_symbols', [])
                
                explanation = self.explain_event_impact(event_name, symbols)
                sections.append(f"\n{event_name}:")
                sections.append(f"  {explanation}")
        
        if 'anomalies' in analysis_data:
            sections.append("\n\n--- ANOMALY DETECTION ---")
            for anomaly in analysis_data['anomalies'][:2]:
                explanation = self.explain_anomaly(anomaly)
                sections.append(f"\n{anomaly.get('type', 'Anomaly')}:")
                sections.append(f"  {explanation}")
        
        if 'seasonality' in analysis_data:
            sections.append("\n\n--- SEASONAL PATTERNS ---")
            for symbol, pattern in list(analysis_data['seasonality'].items())[:2]:
                explanation = self.explain_seasonality(symbol, pattern)
                sections.append(f"\n{symbol}:")
                sections.append(f"  {explanation}")
        
        sections.append("\n\n=== END OF REPORT ===")
        
        return '\n'.join(sections)
    
    def save_explanation(self, explanation: str, filepath: str):
        """Save explanation to file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(explanation)
        print(f"✓ Saved: {filepath}")


if __name__ == "__main__":
    print("="*80)
    print("HF METHOD 10: CAUSAL INFERENCE & EXPLANATIONS")
    print("="*80)
    
    explainer = HFCausalExplainer()
    model_loaded = explainer.load_model()
    
    print("\n" + "="*80)
    print("TEST 1: CORRELATION EXPLANATION")
    print("="*80)
    
    explanation1 = explainer.explain_correlation('GOLD', 'DX-Y.NYB', -0.75)
    print(f"\nGold vs US Dollar (correlation: -0.75):")
    print(f"  {explanation1}")
    
    print("\n" + "="*80)
    print("TEST 2: EVENT IMPACT EXPLANATION")
    print("="*80)
    
    explanation2 = explainer.explain_event_impact(
        'Non-Farm Payrolls',
        ['EURUSD=X', 'GOLD', '^GSPC']
    )
    print(f"\nNon-Farm Payrolls Impact:")
    print(f"  {explanation2}")
    
    print("\n" + "="*80)
    print("TEST 3: SEASONALITY EXPLANATION")
    print("="*80)
    
    explanation3 = explainer.explain_seasonality(
        'GOLD',
        {'best_month': 'September', 'worst_month': 'March'}
    )
    print(f"\nGold Seasonality:")
    print(f"  {explanation3}")
    
    print("\n" + "="*80)
    print("TEST 4: ANOMALY EXPLANATION")
    print("="*80)
    
    explanation4 = explainer.explain_anomaly({
        'type': 'VOLUME_ANOMALY',
        'severity': 'HIGH'
    })
    print(f"\nVolume Anomaly:")
    print(f"  {explanation4}")
    
    print("\n" + "="*80)
    print("TEST 5: COMPREHENSIVE REPORT")
    print("="*80)
    
    test_data = {
        'correlations': [
            {'asset1': 'GOLD', 'asset2': 'DX-Y.NYB', 'correlation': -0.75},
            {'asset1': '^GSPC', 'asset2': '^VIX', 'correlation': -0.85}
        ],
        'events': [
            {
                'event_name': 'Consumer Price Index',
                'affected_symbols': ['GOLD', 'EURUSD=X', 'TLT']
            }
        ],
        'anomalies': [
            {'type': 'PRICE_ANOMALY', 'severity': 'HIGH'}
        ],
        'seasonality': {
            'GOLD': {'best_month': 'September', 'worst_month': 'March'}
        }
    }
    
    report = explainer.generate_comprehensive_report(test_data)
    print(report)
    
    explainer.save_explanation(report, 'hf_causal_explanation_report.txt')
    
    print("\n" + "="*80)
    print("✓ Causal explanation complete")
    print("="*80)
