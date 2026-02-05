"""
Pipeline Visualizer Module
Handles all visualization generation for comprehensive pipeline results
"""

import os
from typing import Dict, List, Optional
from pathlib import Path


class PipelineVisualizer:
    """Generates visualizations from pipeline results"""
    
    def __init__(self, output_dir: str = 'visualizations'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            self.plt = plt
            self.sns = sns
            self.available = True
            
            sns.set_style("whitegrid")
            sns.set_palette("husl")
        
        except ImportError:
            self.available = False
            print("⊘ Matplotlib/Seaborn not available - visualizations disabled")
    
    def generate_all(self, results: Dict) -> Dict:
        """
        Generate all visualizations from pipeline results
        
        Args:
            results: Full pipeline results dictionary
            
        Returns:
            Dictionary with chart paths and metadata
        """
        if not self.available:
            return {'charts_created': 0, 'chart_files': []}
        
        viz_results = {
            'charts_created': 0,
            'chart_files': [],
            'charts_by_type': {}
        }
        
        sections = results.get('sections', {})
        metadata = results.get('metadata', {})
        
        # 1. Sentiment Distribution
        if sections.get('hf_methods', {}).get('sentiment'):
            chart = self.plot_sentiment_distribution(
                sections['hf_methods']['sentiment'],
                metadata
            )
            if chart:
                viz_results['charts_created'] += 1
                viz_results['chart_files'].append(chart)
                viz_results['charts_by_type']['sentiment'] = chart
        
        # 2. Technical Signals
        if sections.get('indicators'):
            chart = self.plot_technical_signals(
                sections['indicators'],
                metadata
            )
            if chart:
                viz_results['charts_created'] += 1
                viz_results['chart_files'].append(chart)
                viz_results['charts_by_type']['technical'] = chart
        
        # 3. COT Positioning
        if sections.get('cot'):
            chart = self.plot_cot_positioning(
                sections['cot'],
                metadata
            )
            if chart:
                viz_results['charts_created'] += 1
                viz_results['chart_files'].append(chart)
                viz_results['charts_by_type']['cot'] = chart
        
        # 4. Economic Dashboard
        if sections.get('economic'):
            chart = self.plot_economic_dashboard(
                sections['economic'],
                metadata
            )
            if chart:
                viz_results['charts_created'] += 1
                viz_results['chart_files'].append(chart)
                viz_results['charts_by_type']['economic'] = chart
        
        # 5. Correlation Heatmap
        if sections.get('correlations'):
            chart = self.plot_correlation_heatmap(
                sections['correlations'],
                metadata
            )
            if chart:
                viz_results['charts_created'] += 1
                viz_results['chart_files'].append(chart)
                viz_results['charts_by_type']['correlations'] = chart
        
        # 6. Volume Analysis
        if sections.get('volume'):
            chart = self.plot_volume_analysis(
                sections['volume'],
                metadata
            )
            if chart:
                viz_results['charts_created'] += 1
                viz_results['chart_files'].append(chart)
                viz_results['charts_by_type']['volume'] = chart
        
        # 7. Overall Summary
        chart = self.plot_summary_dashboard(results)
        if chart:
            viz_results['charts_created'] += 1
            viz_results['chart_files'].append(chart)
            viz_results['charts_by_type']['summary'] = chart
        
        return viz_results
    
    def plot_sentiment_distribution(self, sentiment_data: Dict, metadata: Dict) -> Optional[str]:
        """Plot sentiment distribution"""
        try:
            aggregated = sentiment_data.get('aggregated', {})
            
            fig, ax = self.plt.subplots(figsize=(10, 6))
            
            sentiments = ['Positive', 'Negative', 'Neutral']
            values = [
                aggregated.get('positive_count', 0),
                aggregated.get('negative_count', 0),
                aggregated.get('neutral_count', 0)
            ]
            
            colors = ['#2ecc71', '#e74c3c', '#95a5a6']
            bars = ax.bar(sentiments, values, color=colors, alpha=0.8, edgecolor='black')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontweight='bold')
            
            ax.set_title(
                f'Sentiment Distribution\n{metadata.get("event_name", "Analysis")}',
                fontsize=14,
                fontweight='bold'
            )
            ax.set_ylabel('Number of Articles', fontsize=12)
            ax.set_xlabel('Sentiment', fontsize=12)
            
            overall = aggregated.get('overall_sentiment', 'N/A').upper()
            confidence = aggregated.get('confidence', 0)
            ax.text(
                0.98, 0.98,
                f'Overall: {overall}\nConfidence: {confidence:.1%}',
                transform=ax.transAxes,
                ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
            
            chart_file = os.path.join(self.output_dir, 'sentiment_distribution.png')
            self.plt.tight_layout()
            self.plt.savefig(chart_file, dpi=150, bbox_inches='tight')
            self.plt.close()
            
            return chart_file
        
        except Exception as e:
            print(f"  ✗ Sentiment plot error: {str(e)[:50]}")
            return None
    
    def plot_technical_signals(self, indicators: Dict, metadata: Dict) -> Optional[str]:
        """Plot technical indicator signals"""
        try:
            fig, ax = self.plt.subplots(figsize=(12, max(6, len(indicators) * 0.3)))
            
            symbols = list(indicators.keys())[:20]
            signals = [indicators[s].get('overall_signal', 'NEUTRAL') for s in symbols]
            
            signal_colors = {
                'BUY': '#2ecc71',
                'SELL': '#e74c3c',
                'NEUTRAL': '#95a5a6',
                'STRONG_BUY': '#27ae60',
                'STRONG_SELL': '#c0392b'
            }
            
            colors = [signal_colors.get(s, '#95a5a6') for s in signals]
            
            y_pos = range(len(symbols))
            ax.barh(y_pos, [1]*len(symbols), color=colors, alpha=0.8, edgecolor='black')
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(symbols)
            ax.set_xlim(0, 1)
            ax.set_xticks([])
            ax.set_title(
                f'Technical Signals - {metadata.get("event_name", "Analysis")}',
                fontsize=14,
                fontweight='bold'
            )
            
            for i, (symbol, signal) in enumerate(zip(symbols, signals)):
                ax.text(0.5, i, signal, ha='center', va='center',
                       fontweight='bold', color='white', fontsize=9)
            
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#2ecc71', label='BUY'),
                Patch(facecolor='#e74c3c', label='SELL'),
                Patch(facecolor='#95a5a6', label='NEUTRAL')
            ]
            ax.legend(handles=legend_elements, loc='lower right')
            
            chart_file = os.path.join(self.output_dir, 'technical_signals.png')
            self.plt.tight_layout()
            self.plt.savefig(chart_file, dpi=150, bbox_inches='tight')
            self.plt.close()
            
            return chart_file
        
        except Exception as e:
            print(f"  ✗ Technical signals plot error: {str(e)[:50]}")
            return None
    
    def plot_cot_positioning(self, cot_data: Dict, metadata: Dict) -> Optional[str]:
        """Plot COT positioning"""
        try:
            if not cot_data:
                return None
            
            fig, ax = self.plt.subplots(figsize=(12, 6))
            
            symbols = list(cot_data.keys())[:10]
            net_positions = [cot_data[s].get('net_positioning', 0) for s in symbols]
            
            colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in net_positions]
            
            bars = ax.barh(symbols, net_positions, color=colors, alpha=0.8, edgecolor='black')
            
            ax.axvline(0, color='black', linewidth=1, linestyle='--')
            ax.set_xlabel('Net Positioning', fontsize=12)
            ax.set_title(
                f'COT Net Positioning - {metadata.get("event_name", "Analysis")}',
                fontsize=14,
                fontweight='bold'
            )
            
            ax.text(
                0.98, 0.98,
                'Green = Net Long\nRed = Net Short',
                transform=ax.transAxes,
                ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
            
            chart_file = os.path.join(self.output_dir, 'cot_positioning.png')
            self.plt.tight_layout()
            self.plt.savefig(chart_file, dpi=150, bbox_inches='tight')
            self.plt.close()
            
            return chart_file
        
        except Exception as e:
            print(f"  ✗ COT plot error: {str(e)[:50]}")
            return None
    
    def plot_economic_dashboard(self, econ_data: Dict, metadata: Dict) -> Optional[str]:
        """Plot economic indicators dashboard"""
        try:
            if not econ_data:
                return None
            
            fig, ((ax1, ax2), (ax3, ax4)) = self.plt.subplots(2, 2, figsize=(14, 10))
            
            status = econ_data.get('overall_economic_status', 'N/A')
            
            fig.suptitle(
                f'Economic Dashboard - {status}\n{metadata.get("event_name", "Analysis")}',
                fontsize=16,
                fontweight='bold'
            )
            
            # Interest Rates
            rates = econ_data.get('interest_rates', {})
            if rates:
                ax1.text(0.5, 0.5, f'{rates.get("current_rate", 0):.2f}%',
                        ha='center', va='center', fontsize=40, fontweight='bold')
                ax1.set_title('Fed Funds Rate', fontweight='bold')
                ax1.axis('off')
            
            # Inflation
            inflation = econ_data.get('inflation', {})
            if inflation:
                ax2.text(0.5, 0.5, f'{inflation.get("level", 0):.2f}%',
                        ha='center', va='center', fontsize=40, fontweight='bold')
                ax2.set_title('CPI Inflation', fontweight='bold')
                ax2.axis('off')
            
            # Employment
            employment = econ_data.get('employment', {})
            if employment:
                ax3.text(0.5, 0.5, f'{employment.get("unemployment_rate", 0):.1f}%',
                        ha='center', va='center', fontsize=40, fontweight='bold')
                ax3.set_title('Unemployment Rate', fontweight='bold')
                ax3.axis('off')
            
            # GDP
            gdp = econ_data.get('gdp', {})
            if gdp:
                ax4.text(0.5, 0.5, f'{gdp.get("growth_rate", 0):.1f}%',
                        ha='center', va='center', fontsize=40, fontweight='bold')
                ax4.set_title('GDP Growth', fontweight='bold')
                ax4.axis('off')
            
            chart_file = os.path.join(self.output_dir, 'economic_dashboard.png')
            self.plt.tight_layout()
            self.plt.savefig(chart_file, dpi=150, bbox_inches='tight')
            self.plt.close()
            
            return chart_file
        
        except Exception as e:
            print(f"  ✗ Economic dashboard error: {str(e)[:50]}")
            return None
    
    def plot_correlation_heatmap(self, corr_data: Dict, metadata: Dict) -> Optional[str]:
        """Plot correlation heatmap"""
        try:
            import numpy as np
            
            symbols = list(corr_data.keys())[:8]
            
            if len(symbols) < 2:
                return None
            
            corr_matrix = np.zeros((len(symbols), len(symbols)))
            
            for i, sym1 in enumerate(symbols):
                for j, sym2 in enumerate(symbols):
                    if i == j:
                        corr_matrix[i][j] = 1.0
                    elif sym2 in corr_data.get(sym1, {}).get('correlations', {}):
                        corr_matrix[i][j] = corr_data[sym1]['correlations'][sym2]
            
            fig, ax = self.plt.subplots(figsize=(10, 8))
            
            im = ax.imshow(corr_matrix, cmap='RdYlGn', vmin=-1, vmax=1)
            
            ax.set_xticks(range(len(symbols)))
            ax.set_yticks(range(len(symbols)))
            ax.set_xticklabels(symbols, rotation=45, ha='right')
            ax.set_yticklabels(symbols)
            
            for i in range(len(symbols)):
                for j in range(len(symbols)):
                    text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=9)
            
            ax.set_title(
                f'Correlation Heatmap\n{metadata.get("event_name", "Analysis")}',
                fontsize=14,
                fontweight='bold'
            )
            
            self.plt.colorbar(im, ax=ax, label='Correlation')
            
            chart_file = os.path.join(self.output_dir, 'correlation_heatmap.png')
            self.plt.tight_layout()
            self.plt.savefig(chart_file, dpi=150, bbox_inches='tight')
            self.plt.close()
            
            return chart_file
        
        except Exception as e:
            print(f"  ✗ Correlation heatmap error: {str(e)[:50]}")
            return None
    
    def plot_volume_analysis(self, volume_data: Dict, metadata: Dict) -> Optional[str]:
        """Plot volume analysis"""
        try:
            symbols = list(volume_data.keys())[:10]
            
            if not symbols:
                return None
            
            fig, ax = self.plt.subplots(figsize=(12, 6))
            
            relative_volumes = []
            colors = []
            
            for symbol in symbols:
                vol = volume_data[symbol]
                rel_vol = vol.get('relative_volume', 1.0)
                relative_volumes.append(rel_vol)
                
                if vol.get('volume_spike'):
                    colors.append('#e74c3c')
                elif rel_vol > 1.2:
                    colors.append('#f39c12')
                else:
                    colors.append('#3498db')
            
            bars = ax.bar(symbols, relative_volumes, color=colors, alpha=0.8, edgecolor='black')
            
            ax.axhline(1.0, color='black', linewidth=1, linestyle='--', label='Average')
            ax.set_ylabel('Relative Volume', fontsize=12)
            ax.set_title(
                f'Volume Analysis\n{metadata.get("event_name", "Analysis")}',
                fontsize=14,
                fontweight='bold'
            )
            ax.legend()
            
            self.plt.xticks(rotation=45, ha='right')
            
            chart_file = os.path.join(self.output_dir, 'volume_analysis.png')
            self.plt.tight_layout()
            self.plt.savefig(chart_file, dpi=150, bbox_inches='tight')
            self.plt.close()
            
            return chart_file
        
        except Exception as e:
            print(f"  ✗ Volume analysis error: {str(e)[:50]}")
            return None
    
    def plot_summary_dashboard(self, results: Dict) -> Optional[str]:
        """Plot overall summary dashboard"""
        try:
            synthesis = results.get('sections', {}).get('synthesis', {})
            
            if not synthesis:
                return None
            
            fig, ((ax1, ax2), (ax3, ax4)) = self.plt.subplots(2, 2, figsize=(14, 10))
            
            metadata = results.get('metadata', {})
            
            fig.suptitle(
                f'Analysis Summary Dashboard\n{metadata.get("event_name", "Analysis")} - {metadata.get("date", "N/A")}',
                fontsize=16,
                fontweight='bold'
            )
            
            # Key Findings
            findings = synthesis.get('key_findings', [])[:5]
            ax1.axis('off')
            ax1.set_title('Key Findings', fontweight='bold', fontsize=12)
            y = 0.9
            for finding in findings:
                ax1.text(0.05, y, f'• {finding[:60]}...', fontsize=9, va='top')
                y -= 0.18
            
            # Signals
            signals = synthesis.get('signals', [])[:5]
            ax2.axis('off')
            ax2.set_title('Signals', fontweight='bold', fontsize=12)
            y = 0.9
            for signal in signals:
                ax2.text(0.05, y, f'• {signal[:60]}...', fontsize=9, va='top')
                y -= 0.18
            
            # Risks vs Opportunities
            risks = len(synthesis.get('risks', []))
            opps = len(synthesis.get('opportunities', []))
            
            ax3.bar(['Risks', 'Opportunities'], [risks, opps],
                   color=['#e74c3c', '#2ecc71'], alpha=0.8, edgecolor='black')
            ax3.set_title('Risks vs Opportunities', fontweight='bold')
            ax3.set_ylabel('Count')
            
            # Overall Confidence
            confidence = synthesis.get('overall_confidence', 0)
            ax4.text(0.5, 0.5, f'{confidence:.1%}',
                    ha='center', va='center', fontsize=50, fontweight='bold')
            ax4.set_title('Overall Confidence', fontweight='bold')
            ax4.axis('off')
            
            chart_file = os.path.join(self.output_dir, 'summary_dashboard.png')
            self.plt.tight_layout()
            self.plt.savefig(chart_file, dpi=150, bbox_inches='tight')
            self.plt.close()
            
            return chart_file
        
        except Exception as e:
            print(f"  ✗ Summary dashboard error: {str(e)[:50]}")
            return None


if __name__ == "__main__":
    print("="*80)
    print("PIPELINE VISUALIZER TEST")
    print("="*80)
    
    visualizer = PipelineVisualizer('test_viz_output')
    
    if visualizer.available:
        print("\n✓ Matplotlib/Seaborn available")
        print(f"✓ Output directory: {visualizer.output_dir}")
    else:
        print("\n✗ Visualization libraries not available")
    
    print("\n" + "="*80)
