"""
Analysis Synthesis Module
Generates insights, signals, and recommendations from pipeline results
"""

from typing import Dict, List, Any
from datetime import datetime


class AnalysisSynthesizer:
    """Synthesizes insights from comprehensive analysis results"""
    
    def __init__(self):
        self.signal_weights = {
            'sentiment': 0.25,
            'technical': 0.30,
            'economic': 0.20,
            'cot': 0.15,
            'volume': 0.10
        }
    
    def synthesize(self, results: Dict) -> Dict:
        """
        Generate comprehensive insights from all analysis sections
        
        Args:
            results: Full pipeline results dictionary
            
        Returns:
            Dictionary with summary, findings, signals, risks, opportunities
        """
        insights = {
            'summary': [],
            'key_findings': [],
            'signals': [],
            'risks': [],
            'opportunities': [],
            'confidence_scores': {}
        }
        
        meta = results.get('metadata', {})
        sections = results.get('sections', {})
        
        # Basic summary
        insights['summary'].append(
            f"Analyzed {meta.get('event_name', 'market')} on {meta.get('date', 'unknown date')}"
        )
        insights['summary'].append(
            f"Processed {len(meta.get('symbols_analyzed', []))} symbols across {len(sections)} analysis types"
        )
        
        # HF AI Insights
        hf_insights = self._analyze_hf_methods(sections.get('hf_methods', {}))
        insights['key_findings'].extend(hf_insights['findings'])
        insights['signals'].extend(hf_insights['signals'])
        insights['risks'].extend(hf_insights['risks'])
        insights['opportunities'].extend(hf_insights['opportunities'])
        insights['confidence_scores']['sentiment'] = hf_insights.get('confidence', 0)
        
        # Technical Indicators
        tech_insights = self._analyze_technical(sections.get('indicators', {}))
        insights['signals'].extend(tech_insights['signals'])
        if tech_insights['bias'] == 'bullish':
            insights['opportunities'].extend(tech_insights['details'])
        elif tech_insights['bias'] == 'bearish':
            insights['risks'].extend(tech_insights['details'])
        insights['confidence_scores']['technical'] = tech_insights.get('confidence', 0)
        
        # Economic Indicators
        econ_insights = self._analyze_economic(sections.get('economic'))
        insights['key_findings'].extend(econ_insights['findings'])
        insights['risks'].extend(econ_insights['risks'])
        insights['opportunities'].extend(econ_insights['opportunities'])
        insights['confidence_scores']['economic'] = econ_insights.get('confidence', 0)
        
        # COT Positioning
        cot_insights = self._analyze_cot(sections.get('cot', {}))
        insights['signals'].extend(cot_insights['signals'])
        insights['risks'].extend(cot_insights['risks'])
        
        # Volume Analysis
        volume_insights = self._analyze_volume(sections.get('volume', {}))
        insights['signals'].extend(volume_insights['signals'])
        
        # Market Structure
        structure_insights = self._analyze_structure(sections.get('structure', {}))
        insights['key_findings'].extend(structure_insights['findings'])
        
        # Overall confidence
        insights['overall_confidence'] = self._calculate_overall_confidence(
            insights['confidence_scores']
        )
        
        return insights
    
    def _analyze_hf_methods(self, hf_data: Dict) -> Dict:
        """Analyze HF AI method results"""
        insights = {
            'findings': [],
            'signals': [],
            'risks': [],
            'opportunities': [],
            'confidence': 0
        }
        
        if not hf_data:
            return insights
        
        # Sentiment Analysis
        if 'sentiment' in hf_data:
            sent = hf_data['sentiment'].get('aggregated', {})
            sentiment = sent.get('overall_sentiment', 'N/A').upper()
            confidence = sent.get('confidence', 0)
            
            insights['confidence'] = confidence
            
            insights['findings'].append(
                f"AI Sentiment: {sentiment} ({confidence:.1%} confidence)"
            )
            
            if sentiment == 'NEGATIVE' and confidence > 0.6:
                insights['risks'].append("Strong negative market sentiment detected")
                insights['signals'].append("BEARISH: Negative sentiment")
            elif sentiment == 'POSITIVE' and confidence > 0.6:
                insights['opportunities'].append("Strong positive market sentiment")
                insights['signals'].append("BULLISH: Positive sentiment")
            elif confidence < 0.5:
                insights['risks'].append("Low sentiment confidence - market uncertainty")
        
        # Entity Extraction
        if 'entities' in hf_data:
            entities = hf_data['entities'].get('aggregated', {})
            symbol_count = entities.get('symbol_count', 0)
            
            if symbol_count > 0:
                insights['findings'].append(
                    f"Extracted {symbol_count} financial symbols from news"
                )
        
        # Event Classification
        if 'classification' in hf_data:
            classification = hf_data['classification']
            impact = classification.get('impact_level', 'N/A')
            
            insights['findings'].append(f"Event Impact: {impact}")
            
            if impact == 'HIGH':
                insights['risks'].append("High-impact event - expect volatility")
            elif impact == 'MEDIUM':
                insights['findings'].append("Medium-impact event")
        
        # Q&A Insights
        if 'qa' in hf_data:
            qa_results = hf_data['qa']
            if qa_results:
                insights['findings'].append(
                    f"AI answered {len(qa_results)} market questions"
                )
        
        return insights
    
    def _analyze_technical(self, indicators: Dict) -> Dict:
        """Analyze technical indicator signals"""
        insights = {
            'signals': [],
            'details': [],
            'bias': 'neutral',
            'confidence': 0
        }
        
        if not indicators:
            return insights
        
        buy_count = sum(1 for ind in indicators.values() if ind.get('overall_signal') == 'BUY')
        sell_count = sum(1 for ind in indicators.values() if ind.get('overall_signal') == 'SELL')
        neutral_count = len(indicators) - buy_count - sell_count
        
        total = len(indicators)
        
        if buy_count > sell_count * 1.5:
            insights['bias'] = 'bullish'
            insights['signals'].append(f"BULLISH: {buy_count}/{total} symbols showing buy signals")
            insights['details'].append(f"{buy_count} bullish technical signals")
            insights['confidence'] = buy_count / total
        
        elif sell_count > buy_count * 1.5:
            insights['bias'] = 'bearish'
            insights['signals'].append(f"BEARISH: {sell_count}/{total} symbols showing sell signals")
            insights['details'].append(f"{sell_count} bearish technical signals")
            insights['confidence'] = sell_count / total
        
        else:
            insights['signals'].append(f"MIXED: {buy_count} BUY, {sell_count} SELL, {neutral_count} NEUTRAL")
            insights['confidence'] = 0.3
        
        return insights
    
    def _analyze_economic(self, econ_data: Dict) -> Dict:
        """Analyze economic indicator data"""
        insights = {
            'findings': [],
            'risks': [],
            'opportunities': [],
            'confidence': 0
        }
        
        if not econ_data:
            return insights
        
        status = econ_data.get('overall_economic_status', 'N/A')
        insights['findings'].append(f"Economic Status: {status}")
        
        # Check if we have actual data
        has_data = (
            econ_data.get('interest_rates') is not None or
            econ_data.get('inflation') is not None or
            econ_data.get('employment') is not None
        )
        
        if not has_data:
            insights['confidence'] = 0
            return insights
        
        insights['confidence'] = 0.7
        
        if 'RECESSION' in status.upper():
            insights['risks'].append("Recessionary economic indicators")
        elif 'EXPANSION' in status.upper():
            insights['opportunities'].append("Expansionary economic conditions")
        elif 'STAGFLATION' in status.upper():
            insights['risks'].append("Stagflation risk detected")
        
        # Interest rate insights
        if econ_data.get('interest_rates'):
            rates = econ_data['interest_rates']
            if rates.get('trend') == 'rising':
                insights['risks'].append("Rising interest rate environment")
            elif rates.get('trend') == 'falling':
                insights['opportunities'].append("Declining interest rates")
        
        # Inflation insights
        if econ_data.get('inflation'):
            inflation = econ_data['inflation']
            if inflation.get('level', 0) > 3:
                insights['risks'].append(f"Elevated inflation at {inflation.get('level')}%")
        
        return insights
    
    def _analyze_cot(self, cot_data: Dict) -> Dict:
        """Analyze COT positioning data"""
        insights = {
            'signals': [],
            'risks': []
        }
        
        if not cot_data:
            return insights
        
        for symbol, positioning in cot_data.items():
            net_position = positioning.get('net_positioning', 0)
            
            if abs(net_position) > 50000:
                if net_position > 0:
                    insights['signals'].append(f"{symbol}: Strong NET LONG positioning")
                else:
                    insights['signals'].append(f"{symbol}: Strong NET SHORT positioning")
            
            extreme = positioning.get('extreme_positioning', False)
            if extreme:
                insights['risks'].append(f"{symbol}: Extreme positioning - reversal risk")
        
        return insights
    
    def _analyze_volume(self, volume_data: Dict) -> Dict:
        """Analyze volume patterns"""
        insights = {'signals': []}
        
        if not volume_data:
            return insights
        
        for symbol, vol in volume_data.items():
            if vol.get('volume_spike', False):
                insights['signals'].append(f"{symbol}: Volume spike detected")
            
            if vol.get('unusual_volume', False):
                insights['signals'].append(f"{symbol}: Unusual volume activity")
        
        return insights
    
    def _analyze_structure(self, structure_data: Dict) -> Dict:
        """Analyze market structure"""
        insights = {'findings': []}
        
        if not structure_data:
            return insights
        
        for symbol, structure in structure_data.items():
            trend = structure.get('trend', 'N/A')
            strength = structure.get('trend_strength', 0)
            
            if strength > 0.7:
                insights['findings'].append(
                    f"{symbol}: Strong {trend} trend (strength: {strength:.2f})"
                )
        
        return insights
    
    def _calculate_overall_confidence(self, scores: Dict) -> float:
        """Calculate weighted overall confidence score"""
        total_weight = 0
        weighted_sum = 0
        
        for component, score in scores.items():
            weight = self.signal_weights.get(component, 0.1)
            weighted_sum += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight
    
    def generate_markdown_summary(self, insights: Dict, metadata: Dict) -> str:
        """Generate markdown summary of insights"""
        lines = [
            "# Analysis Insights",
            "",
            f"**Date:** {metadata.get('date', 'N/A')}",
            f"**Event:** {metadata.get('event_name', 'Market Analysis')}",
            f"**Overall Confidence:** {insights.get('overall_confidence', 0):.1%}",
            "",
            "---",
            "",
            "## Executive Summary",
            ""
        ]
        
        for item in insights.get('summary', []):
            lines.append(f"- {item}")
        
        lines.extend(["", "## Key Findings", ""])
        for finding in insights.get('key_findings', []):
            lines.append(f"- **{finding}**")
        
        if insights.get('signals'):
            lines.extend(["", "## Signals", ""])
            for signal in insights['signals']:
                lines.append(f"- 📊 {signal}")
        
        if insights.get('opportunities'):
            lines.extend(["", "## Opportunities", ""])
            for opp in insights['opportunities']:
                lines.append(f"- 💡 {opp}")
        
        if insights.get('risks'):
            lines.extend(["", "## Risks", ""])
            for risk in insights['risks']:
                lines.append(f"- ⚠️ {risk}")
        
        if insights.get('confidence_scores'):
            lines.extend(["", "## Confidence Scores", ""])
            for component, score in insights['confidence_scores'].items():
                lines.append(f"- {component.title()}: {score:.1%}")
        
        return '\n'.join(lines)


if __name__ == "__main__":
    print("="*80)
    print("SYNTHESIS MODULE TEST")
    print("="*80)
    
    # Mock results
    mock_results = {
        'metadata': {
            'date': '2024-11-01',
            'event_name': 'Non-Farm Payrolls',
            'symbols_analyzed': ['EURUSD=X', 'GC=F', '^GSPC']
        },
        'sections': {
            'hf_methods': {
                'sentiment': {
                    'aggregated': {
                        'overall_sentiment': 'positive',
                        'confidence': 0.85
                    }
                }
            },
            'indicators': {
                'EURUSD=X': {'overall_signal': 'BUY'},
                'GC=F': {'overall_signal': 'BUY'},
                '^GSPC': {'overall_signal': 'NEUTRAL'}
            }
        }
    }
    
    synthesizer = AnalysisSynthesizer()
    insights = synthesizer.synthesize(mock_results)
    
    print("\nGenerated Insights:")
    print("-"*80)
    print(f"Summary: {len(insights['summary'])} items")
    print(f"Findings: {len(insights['key_findings'])} items")
    print(f"Signals: {len(insights['signals'])} items")
    print(f"Overall Confidence: {insights['overall_confidence']:.1%}")
    
    markdown = synthesizer.generate_markdown_summary(insights, mock_results['metadata'])
    print("\nMarkdown Preview:")
    print("-"*80)
    print(markdown[:500])
    
    print("\n" + "="*80)
    print("✓ TEST COMPLETE")
    print("="*80)
