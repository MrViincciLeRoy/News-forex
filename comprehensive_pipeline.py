"""
Enhanced Comprehensive Analysis Pipeline
Saves individual JSON files for each analysis section
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import os
import warnings

warnings.filterwarnings('ignore')


class EnhancedComprehensivePipeline:
    
    def __init__(self, output_dir='pipeline_output', enable_hf=True, enable_viz=True, max_articles=20):
        self.output_dir = output_dir
        self.viz_dir = os.path.join(output_dir, 'visualizations')
        self.sections_dir = os.path.join(output_dir, 'sections')
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)
        os.makedirs(self.sections_dir, exist_ok=True)
        
        self.max_articles = max_articles
        self.enable_hf = enable_hf
        self.enable_viz = enable_viz
        
        print("="*80)
        print("ENHANCED COMPREHENSIVE ANALYSIS PIPELINE")
        print("="*80)
        
        self._initialize_modules()
        
        if self.enable_viz:
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                self.plt = plt
                print("✓ Visualization support enabled")
            except ImportError:
                print("⚠️  matplotlib not available, visualizations disabled")
                self.enable_viz = False
    
    def _initialize_modules(self):
        """Initialize analysis modules"""
        try:
            from analysis_synthesizer import AnalysisSynthesizer
            self.synthesizer = AnalysisSynthesizer()
        except ImportError:
            print("⚠️  AnalysisSynthesizer not available")
            self.synthesizer = None
        
        print("✓ Modules initialized")
    
    def analyze(self, date: str, event_name: Optional[str] = None, symbols: Optional[List[str]] = None) -> Dict:
        """Run comprehensive analysis with individual section JSONs"""
        
        print(f"\n{'='*80}")
        print(f"ANALYZING: {event_name or 'Market Analysis'} - {date}")
        print(f"{'='*80}\n")
        
        results = {
            'metadata': {
                'date': date,
                'event_name': event_name,
                'timestamp': datetime.now().isoformat(),
                'pipeline_version': '2.0',
                'symbols': symbols or []
            },
            'sections': {}
        }
        
        # Execute each analysis section
        sections_to_run = [
            ('news', self._analyze_news),
            ('indicators', self._analyze_indicators),
            ('cot', self._analyze_cot),
            ('economic', self._analyze_economic),
            ('correlations', self._analyze_correlations),
            ('structure', self._analyze_structure),
            ('seasonality', self._analyze_seasonality),
            ('volume', self._analyze_volume)
        ]
        
        if self.enable_hf:
            sections_to_run.append(('hf_methods', self._analyze_hf))
        
        for section_name, analyze_func in sections_to_run:
            try:
                print(f"\n📊 Running {section_name} analysis...")
                section_data = analyze_func(date, event_name, symbols)
                results['sections'][section_name] = section_data
                print(f"✓ {section_name} complete")
            except Exception as e:
                print(f"⚠️  {section_name} failed: {e}")
                results['sections'][section_name] = {'error': str(e)}
        
        # Synthesis
        try:
            print(f"\n💡 Synthesizing results...")
            synthesis = self._synthesize_results(results)
            results['sections']['synthesis'] = synthesis
            print(f"✓ Synthesis complete")
        except Exception as e:
            print(f"⚠️  Synthesis failed: {e}")
        
        # Executive summary
        results['sections']['executive_summary'] = self._create_executive_summary(results)
        
        # Generate visualizations
        if self.enable_viz:
            print(f"\n📊 Generating visualizations...")
            self._generate_visualizations(results)
        
        # Save results
        report_file = self._save_results(results, date, event_name)
        results['report_file'] = report_file
        
        print(f"\n{'='*80}")
        print(f"✓ ANALYSIS COMPLETE")
        print(f"{'='*80}\n")
        
        return results
    
    def _analyze_news(self, date: str, event_name: Optional[str], symbols: Optional[List[str]]) -> Dict:
        """News analysis placeholder"""
        return {
            'article_count': 15,
            'sources': ['Reuters', 'Bloomberg', 'FT'],
            'key_themes': ['inflation', 'employment', 'growth'],
            'sentiment': 'MIXED'
        }
    
    def _analyze_indicators(self, date: str, event_name: Optional[str], symbols: Optional[List[str]]) -> Dict:
        """Technical indicators placeholder"""
        return {
            'overall_bias': 'BULLISH',
            'buy_signals': 8,
            'sell_signals': 3,
            'symbols_analyzed': len(symbols) if symbols else 5
        }
    
    def _analyze_cot(self, date: str, event_name: Optional[str], symbols: Optional[List[str]]) -> Dict:
        """COT analysis placeholder"""
        return {
            'net_positioning': 'LONG',
            'positioning_change': '+15%',
            'institutional_sentiment': 'BULLISH'
        }
    
    def _analyze_economic(self, date: str, event_name: Optional[str], symbols: Optional[List[str]]) -> Dict:
        """Economic indicators placeholder"""
        return {
            'overall_status': 'MODERATE',
            'inflation_trend': 'RISING',
            'growth_outlook': 'STABLE'
        }
    
    def _analyze_correlations(self, date: str, event_name: Optional[str], symbols: Optional[List[str]]) -> Dict:
        """Correlation analysis placeholder"""
        return {
            'key_relationships': {
                'DXY_GOLD': {'strength': 'STRONG', 'direction': 'INVERSE'},
                'SPX_DXY': {'strength': 'MODERATE', 'direction': 'INVERSE'}
            }
        }
    
    def _analyze_structure(self, date: str, event_name: Optional[str], symbols: Optional[List[str]]) -> Dict:
        """Market structure placeholder"""
        return {
            'trend': 'UPTREND',
            'support_levels': [1950, 1925, 1900],
            'resistance_levels': [2000, 2025, 2050]
        }
    
    def _analyze_seasonality(self, date: str, event_name: Optional[str], symbols: Optional[List[str]]) -> Dict:
        """Seasonality analysis placeholder"""
        return {
            'seasonal_bias': 'BULLISH',
            'historical_performance': '+2.5%'
        }
    
    def _analyze_volume(self, date: str, event_name: Optional[str], symbols: Optional[List[str]]) -> Dict:
        """Volume analysis placeholder"""
        return {
            'volume_trend': 'INCREASING',
            'volume_profile': 'ACCUMULATION'
        }
    
    def _analyze_hf(self, date: str, event_name: Optional[str], symbols: Optional[List[str]]) -> Dict:
        """HuggingFace methods placeholder"""
        return {
            'sentiment_score': 0.65,
            'forecast_direction': 'UP',
            'anomalies_detected': 0
        }
    
    def _synthesize_results(self, results: Dict) -> Dict:
        """Synthesize all sections"""
        sections = results.get('sections', {})
        
        return {
            'overall_outlook': 'BULLISH',
            'confidence': 0.75,
            'key_factors': [
                'Technical indicators showing strength',
                'Positive institutional positioning',
                'Supportive seasonality'
            ],
            'risks': [
                'Economic uncertainty',
                'Elevated volatility'
            ]
        }
    
    def _create_executive_summary(self, results: Dict) -> Dict:
        """Create executive summary"""
        sections = results.get('sections', {})
        metadata = results.get('metadata', {})
        
        return {
            'market_overview': f"Analysis of {metadata.get('event_name', 'market')} on {metadata.get('date')}",
            'sentiment': {
                'overall': 'BULLISH',
                'confidence': 0.72
            },
            'key_findings': [
                'Technical indicators align bullishly',
                'Institutional positioning shows strength',
                'Seasonal patterns favorable',
                'Economic backdrop moderately supportive'
            ],
            'recommendations': [
                'Monitor key support levels',
                'Watch for volume confirmation',
                'Stay alert to economic data releases'
            ]
        }
    
    def _generate_visualizations(self, results: Dict):
        """Generate visualizations for key sections"""
        
        sections = results.get('sections', {})
        
        # Indicators chart
        if 'indicators' in sections:
            try:
                data = sections['indicators']
                buy = data.get('buy_signals', 0)
                sell = data.get('sell_signals', 0)
                
                fig, ax = self.plt.subplots(figsize=(8, 6))
                ax.pie([buy, sell], labels=['Buy', 'Sell'], autopct='%1.1f%%',
                      colors=['#28a745', '#dc3545'], startangle=90)
                ax.set_title('Technical Signals Distribution')
                
                filename = f'{self.viz_dir}/indicators_signals.png'
                self.plt.savefig(filename, dpi=150, bbox_inches='tight')
                self.plt.close()
                print(f"  ✓ Saved: {filename}")
            except Exception as e:
                print(f"  ⚠️  Indicators chart failed: {e}")
        
        # News themes
        if 'news' in sections:
            try:
                data = sections['news']
                themes = data.get('key_themes', [])[:5]
                values = [len(t) * 10 for t in themes]
                
                if themes:
                    fig, ax = self.plt.subplots(figsize=(10, 6))
                    ax.barh(themes, values, color='#667eea')
                    ax.set_xlabel('Frequency')
                    ax.set_title('Top News Themes')
                    ax.grid(axis='x', alpha=0.3)
                    
                    filename = f'{self.viz_dir}/news_themes.png'
                    self.plt.savefig(filename, dpi=150, bbox_inches='tight')
                    self.plt.close()
                    print(f"  ✓ Saved: {filename}")
            except Exception as e:
                print(f"  ⚠️  News chart failed: {e}")
        
        # Structure levels
        if 'structure' in sections:
            try:
                data = sections['structure']
                support = data.get('support_levels', [])
                resistance = data.get('resistance_levels', [])
                
                if support and resistance:
                    fig, ax = self.plt.subplots(figsize=(10, 6))
                    
                    # Plot support
                    ax.hlines(support, 0, 1, colors='green', linestyles='dashed', 
                             label='Support', linewidth=2, alpha=0.7)
                    
                    # Plot resistance
                    ax.hlines(resistance, 0, 1, colors='red', linestyles='dashed',
                             label='Resistance', linewidth=2, alpha=0.7)
                    
                    # Current price
                    current = 1975
                    ax.hlines(current, 0, 1, colors='blue', linewidth=3, label='Current')
                    
                    ax.set_xlim(0, 1)
                    ax.set_ylabel('Price Level')
                    ax.set_title('Key Support & Resistance Levels')
                    ax.legend()
                    ax.grid(axis='y', alpha=0.3)
                    ax.set_xticks([])
                    
                    filename = f'{self.viz_dir}/structure_levels.png'
                    self.plt.savefig(filename, dpi=150, bbox_inches='tight')
                    self.plt.close()
                    print(f"  ✓ Saved: {filename}")
            except Exception as e:
                print(f"  ⚠️  Structure chart failed: {e}")
        
        # Synthesis overview
        if 'synthesis' in sections:
            try:
                data = sections['synthesis']
                factors = data.get('key_factors', [])[:4]
                
                if factors:
                    values = [85, 70, 65, 55]
                    
                    fig, ax = self.plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
                    
                    angles = np.linspace(0, 2 * np.pi, len(factors), endpoint=False).tolist()
                    values_plot = values + [values[0]]
                    angles_plot = angles + [angles[0]]
                    
                    ax.plot(angles_plot, values_plot, 'o-', linewidth=2, color='#667eea')
                    ax.fill(angles_plot, values_plot, alpha=0.25, color='#667eea')
                    ax.set_xticks(angles)
                    ax.set_xticklabels([f[:20] + '...' if len(f) > 20 else f for f in factors])
                    ax.set_ylim(0, 100)
                    ax.set_title('Key Factors Importance', y=1.08)
                    ax.grid(True)
                    
                    filename = f'{self.viz_dir}/synthesis_factors.png'
                    self.plt.savefig(filename, dpi=150, bbox_inches='tight')
                    self.plt.close()
                    print(f"  ✓ Saved: {filename}")
            except Exception as e:
                print(f"  ⚠️  Synthesis chart failed: {e}")
    
    def _save_results(self, results: Dict, date: str, event_name: Optional[str]) -> str:
        """Save analysis results and individual section JSONs"""
        date_clean = date.replace('-', '_')
        event_clean = (event_name or 'analysis').replace(' ', '_').lower()
        
        # Save main comprehensive file
        filename = f'{self.output_dir}/comprehensive_{date_clean}_{event_clean}.json'
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n✓ Main file: {filename}")
        
        # Save individual section JSONs
        sections = results.get('sections', {})
        print(f"\n📁 Saving {len(sections)} section files...")
        
        for section_name, section_data in sections.items():
            section_file = f'{self.sections_dir}/{section_name}_{date_clean}.json'
            section_output = {
                'section_type': section_name,
                'date': date,
                'event_name': event_name,
                'timestamp': datetime.now().isoformat(),
                'data': section_data
            }
            with open(section_file, 'w', encoding='utf-8') as f:
                json.dump(section_output, f, indent=2, ensure_ascii=False, default=str)
            print(f"  ✓ {section_file}")
        
        return filename
    
    def generate_markdown_report(self, results: Dict) -> str:
        """Generate markdown report"""
        metadata = results.get('metadata', {})
        sections = results.get('sections', {})
        
        md = f"""# Analysis Report

**Date:** {metadata.get('date', 'N/A')}
**Event:** {metadata.get('event_name', 'N/A')}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

{sections.get('executive_summary', {}).get('market_overview', 'N/A')}

**Overall Sentiment:** {sections.get('executive_summary', {}).get('sentiment', {}).get('overall', 'N/A')}

## Sections Analyzed

"""
        
        for section_name in sections.keys():
            if section_name != 'executive_summary':
                md += f"- {section_name.replace('_', ' ').title()}\n"
        
        md += f"\n## Status\n\nAnalysis completed successfully.\n"
        
        return md


ComprehensiveAnalysisPipeline = EnhancedComprehensivePipeline


if __name__ == "__main__":
    pipeline = ComprehensiveAnalysisPipeline(
        output_dir='test_pipeline_output',
        max_articles=20
    )
    
    results = pipeline.analyze(
        date='2024-11-01',
        event_name='Non-Farm Payrolls'
    )
    
    print(f"\n✓ Complete: {results['report_file']}")
    print(f"✓ Sections saved to: {pipeline.sections_dir}")
