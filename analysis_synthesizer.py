import json
from typing import Dict, List, Optional
from datetime import datetime

class AnalysisSynthesizer:
    """Synthesizes insights from multiple analysis sections"""
    
    def synthesize(self, results: Dict) -> Dict:
        """Generate synthesis from all sections"""
        try:
            sections = results.get('sections', {})
            metadata = results.get('metadata', {})
            
            synthesis = {
                'key_findings': [],
                'overall_confidence': 0.5,
                'market_sentiment': 'NEUTRAL',
                'risk_level': 'MODERATE',
                'recommendations': [],
                'data_quality': self._assess_data_quality(sections)
            }
            
            # Extract findings from news
            news = sections.get('news', {})
            if news and news.get('article_count', 0) > 0:
                synthesis['key_findings'].append({
                    'finding': f"Analyzed {news['article_count']} news articles",
                    'importance': 'high',
                    'source': 'news_analysis'
                })
                
                if news.get('key_themes'):
                    themes = ', '.join(news['key_themes'][:3])
                    synthesis['key_findings'].append({
                        'finding': f"Key themes identified: {themes}",
                        'importance': 'medium',
                        'source': 'news_analysis'
                    })
            
            # Extract findings from technical indicators
            indicators = sections.get('indicators', {})
            if indicators and indicators.get('symbols_analyzed', 0) > 0:
                bias = indicators.get('overall_bias', 'NEUTRAL')
                buy_signals = indicators.get('buy_signals', 0)
                sell_signals = indicators.get('sell_signals', 0)
                
                synthesis['key_findings'].append({
                    'finding': f"Technical indicators show {bias} bias ({buy_signals} BUY vs {sell_signals} SELL)",
                    'importance': 'high',
                    'source': 'technical_analysis'
                })
                
                if bias == 'BULLISH':
                    synthesis['market_sentiment'] = 'BULLISH'
                elif bias == 'BEARISH':
                    synthesis['market_sentiment'] = 'BEARISH'
            
            # Extract findings from COT
            cot = sections.get('cot', {})
            if cot and cot.get('symbols_analyzed', 0) > 0:
                synthesis['key_findings'].append({
                    'finding': f"COT data analyzed for {cot['symbols_analyzed']} instruments",
                    'importance': 'medium',
                    'source': 'positioning'
                })
            
            # Extract findings from correlations
            corr = sections.get('correlations', {})
            if corr and corr.get('key_relationships'):
                strong_corrs = [k for k, v in corr['key_relationships'].items() 
                               if v.get('strength') == 'STRONG']
                if strong_corrs:
                    synthesis['key_findings'].append({
                        'finding': f"Strong correlations found: {', '.join(strong_corrs[:2])}",
                        'importance': 'medium',
                        'source': 'correlation_analysis'
                    })
            
            # Extract findings from economic indicators
            econ = sections.get('economic', {})
            if econ:
                status = econ.get('overall_status', 'MODERATE')
                synthesis['key_findings'].append({
                    'finding': f"Economic environment: {status}",
                    'importance': 'high',
                    'source': 'economic_data'
                })
                
                if status == 'WEAK':
                    synthesis['risk_level'] = 'HIGH'
                elif status == 'STRONG':
                    synthesis['risk_level'] = 'LOW'
            
            # Extract findings from market structure
            structure = sections.get('structure', {})
            if structure and structure.get('symbols_analyzed', 0) > 0:
                regime = structure.get('overall_market_regime', 'MIXED')
                synthesis['key_findings'].append({
                    'finding': f"Market regime: {regime}",
                    'importance': 'medium',
                    'source': 'market_structure'
                })
            
            # Extract findings from HF methods
            hf = sections.get('hf_methods', {})
            if hf and hf.get('methods_run', 0) > 0:
                synthesis['key_findings'].append({
                    'finding': f"AI analysis completed: {hf['methods_run']} methods, {hf.get('insights_count', 0)} insights",
                    'importance': 'high',
                    'source': 'ai_analysis'
                })
            
            # Generate recommendations
            synthesis['recommendations'] = self._generate_recommendations(synthesis, sections)
            
            # Calculate overall confidence
            synthesis['overall_confidence'] = self._calculate_confidence(sections)
            
            return synthesis
            
        except Exception as e:
            print(f"Error in synthesis: {e}")
            return {
                'key_findings': [],
                'overall_confidence': 0.0,
                'market_sentiment': 'UNKNOWN',
                'risk_level': 'UNKNOWN',
                'recommendations': [],
                'error': str(e)
            }
    
    def _assess_data_quality(self, sections: Dict) -> Dict:
        """Assess quality of data across sections"""
        quality = {
            'sections_completed': 0,
            'sections_total': 8,
            'data_completeness': 0.0,
            'issues': []
        }
        
        expected_sections = ['news', 'cot', 'indicators', 'correlations', 
                           'economic', 'structure', 'seasonality', 'volume']
        
        for section in expected_sections:
            if sections.get(section):
                quality['sections_completed'] += 1
        
        quality['data_completeness'] = quality['sections_completed'] / quality['sections_total']
        
        if quality['data_completeness'] < 0.5:
            quality['issues'].append("Less than 50% of data sections available")
        
        return quality
    
    def _generate_recommendations(self, synthesis: Dict, sections: Dict) -> List[Dict]:
        """Generate actionable recommendations"""
        recommendations = []
        
        sentiment = synthesis.get('market_sentiment', 'NEUTRAL')
        risk = synthesis.get('risk_level', 'MODERATE')
        
        if sentiment == 'BULLISH' and risk == 'LOW':
            recommendations.append({
                'action': 'Consider long positions',
                'confidence': 'moderate',
                'timeframe': 'short-term'
            })
        elif sentiment == 'BEARISH' and risk == 'HIGH':
            recommendations.append({
                'action': 'Consider defensive positioning',
                'confidence': 'moderate',
                'timeframe': 'short-term'
            })
        else:
            recommendations.append({
                'action': 'Maintain balanced portfolio',
                'confidence': 'high',
                'timeframe': 'short-term'
            })
        
        # Add specific recommendations based on correlations
        corr = sections.get('correlations', {})
        if corr and corr.get('key_relationships'):
            for rel_name, rel_data in list(corr['key_relationships'].items())[:2]:
                if rel_data.get('strength') == 'STRONG':
                    recommendations.append({
                        'action': f"Monitor {rel_name} relationship",
                        'confidence': 'high',
                        'timeframe': 'ongoing'
                    })
        
        return recommendations
    
    def _calculate_confidence(self, sections: Dict) -> float:
        """Calculate overall confidence score"""
        weights = {
            'news': 0.2,
            'indicators': 0.2,
            'cot': 0.15,
            'economic': 0.15,
            'correlations': 0.1,
            'structure': 0.1,
            'hf_methods': 0.1
        }
        
        confidence = 0.0
        
        for section, weight in weights.items():
            if sections.get(section):
                data = sections[section]
                
                # Check data quality for each section
                if section == 'news' and data.get('article_count', 0) > 5:
                    confidence += weight
                elif section == 'indicators' and data.get('symbols_analyzed', 0) > 3:
                    confidence += weight
                elif section == 'cot' and data.get('symbols_analyzed', 0) > 0:
                    confidence += weight
                elif section in ['economic', 'correlations', 'structure'] and data:
                    confidence += weight * 0.8
                elif section == 'hf_methods' and data.get('methods_run', 0) > 3:
                    confidence += weight
        
        return round(confidence, 2)
    
    def generate_markdown_summary(self, synthesis: Dict, metadata: Dict) -> str:
        """Generate markdown summary"""
        md = f"""# Analysis Summary
        
**Date:** {metadata.get('date')}  
**Event:** {metadata.get('event_name', 'Market Analysis')}  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

**Market Sentiment:** {synthesis.get('market_sentiment', 'NEUTRAL')}  
**Risk Level:** {synthesis.get('risk_level', 'MODERATE')}  
**Confidence:** {synthesis.get('overall_confidence', 0.5):.0%}

## Key Findings

"""
        
        for i, finding in enumerate(synthesis.get('key_findings', []), 1):
            importance = finding.get('importance', 'medium')
            emoji = '🔴' if importance == 'high' else '🟡' if importance == 'medium' else '🟢'
            md += f"{i}. {emoji} {finding.get('finding', '')}\n"
        
        md += "\n## Recommendations\n\n"
        
        for i, rec in enumerate(synthesis.get('recommendations', []), 1):
            md += f"{i}. **{rec.get('action')}** (Confidence: {rec.get('confidence')}, Timeframe: {rec.get('timeframe')})\n"
        
        md += "\n## Data Quality\n\n"
        quality = synthesis.get('data_quality', {})
        md += f"- Sections Completed: {quality.get('sections_completed', 0)}/{quality.get('sections_total', 8)}\n"
        md += f"- Completeness: {quality.get('data_completeness', 0):.0%}\n"
        
        return md
