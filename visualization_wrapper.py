"""
Advanced Visualization Wrapper
Generates informative images for any analysis file
Automatically detects data types and creates appropriate visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class VisualizationWrapper:
    """
    Universal visualization wrapper for all analysis files
    Detects data type and generates appropriate charts
    """
    
    def __init__(self, output_dir='visualizations'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.color_scheme = {
            'bullish': '#2ecc71',
            'bearish': '#e74c3c',
            'neutral': '#95a5a6',
            'primary': '#3498db',
            'secondary': '#9b59b6',
            'warning': '#f39c12',
            'info': '#1abc9c'
        }
        
        print(f"Visualization Wrapper initialized | Output: {output_dir}/")
    
    def detect_data_type(self, data):
        """Detect what type of data we're working with"""
        if isinstance(data, dict):
            if 'forecast_mean' in data or 'forecast_median' in data:
                return 'forecast'
            elif 'sentiment' in data or 'bullish_score' in data:
                return 'sentiment'
            elif 'entities' in data or 'mapped_symbols' in data:
                return 'entities'
            elif 'correlations' in data or 'correlation_matrix' in data:
                return 'correlation'
            elif 'anomalies' in data or 'anomaly' in str(data):
                return 'anomaly'
            elif 'event' in data and 'indicators' in data:
                return 'calendar'
            elif 'clusters' in data or 'cluster' in str(data):
                return 'clustering'
            elif any(k in data for k in ['buy_count', 'sell_count', 'indicators']):
                return 'indicators'
            elif 'cot' in str(data).lower() or 'positioning' in data:
                return 'cot'
        
        elif isinstance(data, pd.DataFrame):
            if 'close' in data.columns or 'price' in data.columns:
                return 'timeseries'
            elif 'correlation' in str(data.columns).lower():
                return 'correlation_matrix'
        
        return 'generic'
    
    def visualize(self, data, data_name='data', file_type=None):
        """Main visualization dispatcher"""
        detected_type = file_type or self.detect_data_type(data)
        
        print(f"Visualizing {data_name} (type: {detected_type})")
        
        visualizers = {
            'forecast': self.visualize_forecast,
            'sentiment': self.visualize_sentiment,
            'entities': self.visualize_entities,
            'correlation': self.visualize_correlation,
            'anomaly': self.visualize_anomaly,
            'calendar': self.visualize_calendar,
            'clustering': self.visualize_clustering,
            'indicators': self.visualize_indicators,
            'cot': self.visualize_cot,
            'timeseries': self.visualize_timeseries,
            'correlation_matrix': self.visualize_correlation_matrix
        }
        
        visualizer = visualizers.get(detected_type, self.visualize_generic)
        return visualizer(data, data_name)
    
    def visualize_forecast(self, data, name):
        """Visualize time series forecast with confidence intervals"""
        files = []
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        forecast_mean = data.get('forecast_mean', [])
        forecast_dates = data.get('forecast_dates', range(len(forecast_mean)))
        
        ax.plot(forecast_dates, forecast_mean, 
               linewidth=2.5, color=self.color_scheme['primary'], 
               label='Forecast', marker='o', markersize=6)
        
        if 'confidence_80' in data:
            lower = data['confidence_80']['lower']
            upper = data['confidence_80']['upper']
            ax.fill_between(forecast_dates, lower, upper, 
                           alpha=0.3, color=self.color_scheme['primary'],
                           label='80% Confidence')
        
        if 'confidence_95' in data:
            lower = data['confidence_95']['lower']
            upper = data['confidence_95']['upper']
            ax.fill_between(forecast_dates, lower, upper, 
                           alpha=0.15, color=self.color_scheme['primary'],
                           label='95% Confidence')
        
        ax.set_title(f'Time Series Forecast - {name}', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Time Period', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        filepath = f'{self.output_dir}/{name}_forecast.png'
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        files.append(filepath)
        
        return files
    
    def visualize_sentiment(self, data, name):
        """Visualize sentiment analysis results"""
        files = []
        
        # Sentiment distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        if 'articles' in data:
            articles = data['articles']
            sentiments = [a.get('sentiment_analysis', {}).get('sentiment', 'neutral') 
                         for a in articles]
            
            sentiment_counts = pd.Series(sentiments).value_counts()
            colors = [self.color_scheme.get(s, self.color_scheme['neutral']) 
                     for s in sentiment_counts.index]
            
            ax1.bar(sentiment_counts.index, sentiment_counts.values, color=colors, alpha=0.8)
            ax1.set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Sentiment', fontsize=11)
            ax1.set_ylabel('Count', fontsize=11)
            ax1.grid(True, alpha=0.3, axis='y')
        
        # Aggregated scores
        if 'aggregated' in data:
            agg = data['aggregated']
            scores = {
                'Bullish': agg.get('bullish_score', 0),
                'Bearish': agg.get('bearish_score', 0),
                'Neutral': 1 - agg.get('bullish_score', 0) - agg.get('bearish_score', 0)
            }
            
            colors_pie = [self.color_scheme['bullish'], 
                         self.color_scheme['bearish'],
                         self.color_scheme['neutral']]
            
            ax2.pie(scores.values(), labels=scores.keys(), autopct='%1.1f%%',
                   colors=colors_pie, startangle=90)
            ax2.set_title('Overall Sentiment Score', fontsize=14, fontweight='bold')
        
        filepath = f'{self.output_dir}/{name}_sentiment.png'
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        files.append(filepath)
        
        return files
    
    def visualize_entities(self, data, name):
        """Visualize named entity extraction results"""
        files = []
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Symbol frequency
        if 'aggregated_symbols' in data or 'aggregated' in data:
            agg = data.get('aggregated_symbols') or data.get('aggregated', {})
            symbol_freq = agg.get('symbol_frequency', {})
            
            if symbol_freq:
                top_symbols = dict(sorted(symbol_freq.items(), 
                                        key=lambda x: x[1], reverse=True)[:15])
                
                ax1.barh(list(top_symbols.keys()), list(top_symbols.values()),
                        color=self.color_scheme['primary'], alpha=0.8)
                ax1.set_title('Top Mentioned Symbols', fontsize=14, fontweight='bold')
                ax1.set_xlabel('Frequency', fontsize=11)
                ax1.grid(True, alpha=0.3, axis='x')
                ax1.invert_yaxis()
        
        # Entity types distribution
        if 'analyzed_articles' in data:
            entity_types = {}
            for article in data['analyzed_articles']:
                entities = article.get('entities', {})
                for etype, items in entities.items():
                    entity_types[etype] = entity_types.get(etype, 0) + len(items)
            
            if entity_types:
                ax2.bar(entity_types.keys(), entity_types.values(),
                       color=self.color_scheme['secondary'], alpha=0.8)
                ax2.set_title('Entity Types Distribution', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Entity Type', fontsize=11)
                ax2.set_ylabel('Count', fontsize=11)
                ax2.grid(True, alpha=0.3, axis='y')
                plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        filepath = f'{self.output_dir}/{name}_entities.png'
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        files.append(filepath)
        
        return files
    
    def visualize_correlation(self, data, name):
        """Visualize correlation analysis"""
        files = []
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if 'top_positive_correlations' in data and 'top_negative_correlations' in data:
            pos_corr = data['top_positive_correlations']
            neg_corr = data['top_negative_correlations']
            
            all_corr = {**pos_corr, **neg_corr}
            symbols = list(all_corr.keys())[:20]
            values = [all_corr[s] for s in symbols]
            colors = [self.color_scheme['bullish'] if v > 0 else self.color_scheme['bearish'] 
                     for v in values]
            
            ax.barh(symbols, values, color=colors, alpha=0.8)
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
            ax.set_title(f'Correlation Analysis - {name}', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Correlation Coefficient', fontsize=12)
            ax.grid(True, alpha=0.3, axis='x')
            ax.invert_yaxis()
        
        filepath = f'{self.output_dir}/{name}_correlation.png'
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        files.append(filepath)
        
        return files
    
    def visualize_anomaly(self, data, name):
        """Visualize anomaly detection results"""
        files = []
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        anomalies = data.get('anomalies', [])
        
        if anomalies:
            # Anomaly types
            anomaly_types = {}
            for a in anomalies:
                atype = a.get('type', 'UNKNOWN')
                anomaly_types[atype] = anomaly_types.get(atype, 0) + 1
            
            axes[0].bar(anomaly_types.keys(), anomaly_types.values(),
                       color=self.color_scheme['warning'], alpha=0.8)
            axes[0].set_title('Anomaly Types', fontsize=14, fontweight='bold')
            axes[0].set_ylabel('Count', fontsize=11)
            axes[0].grid(True, alpha=0.3, axis='y')
            plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Severity distribution
            severities = [a.get('severity', 'NORMAL') for a in anomalies]
            severity_counts = pd.Series(severities).value_counts()
            
            severity_colors = {
                'EXTREME': self.color_scheme['bearish'],
                'HIGH': self.color_scheme['warning'],
                'NORMAL': self.color_scheme['neutral']
            }
            colors = [severity_colors.get(s, self.color_scheme['neutral']) 
                     for s in severity_counts.index]
            
            axes[1].bar(severity_counts.index, severity_counts.values,
                       color=colors, alpha=0.8)
            axes[1].set_title('Severity Distribution', fontsize=14, fontweight='bold')
            axes[1].set_ylabel('Count', fontsize=11)
            axes[1].grid(True, alpha=0.3, axis='y')
            
            # Z-scores if available
            z_scores = [a.get('z_score', 0) for a in anomalies if 'z_score' in a]
            if z_scores:
                axes[2].hist(z_scores, bins=15, color=self.color_scheme['primary'], alpha=0.7)
                axes[2].axvline(x=3, color=self.color_scheme['warning'], 
                              linestyle='--', label='3σ threshold')
                axes[2].set_title('Z-Score Distribution', fontsize=14, fontweight='bold')
                axes[2].set_xlabel('Z-Score', fontsize=11)
                axes[2].set_ylabel('Frequency', fontsize=11)
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)
        
        # Hide unused subplot
        axes[3].axis('off')
        
        filepath = f'{self.output_dir}/{name}_anomaly.png'
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        files.append(filepath)
        
        return files
    
    def visualize_calendar(self, data, name):
        """Visualize economic calendar with indicators"""
        files = []
        
        if not isinstance(data, list):
            data = [data]
        
        # Extract price and signal data
        dates = []
        prices = []
        signals = []
        
        for event in data:
            if 'date' in event and 'indicators' in event:
                dates.append(event['date'])
                indicators = event['indicators']
                prices.append(indicators.get('price', 0))
                signals.append(indicators.get('overall_signal', 'NEUTRAL'))
        
        if dates and prices:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), 
                                          gridspec_kw={'height_ratios': [2, 1]})
            
            # Price chart with signals
            dates_dt = pd.to_datetime(dates)
            ax1.plot(dates_dt, prices, linewidth=2, color=self.color_scheme['primary'],
                    marker='o', markersize=8)
            
            # Color-code points by signal
            for i, (date, price, signal) in enumerate(zip(dates_dt, prices, signals)):
                color = self.color_scheme.get(signal.lower(), self.color_scheme['neutral'])
                ax1.scatter(date, price, s=150, color=color, alpha=0.6, zorder=5)
            
            ax1.set_title('Economic Calendar - Price & Signals', 
                         fontsize=16, fontweight='bold', pad=20)
            ax1.set_ylabel('Price', fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            # Signal distribution
            signal_counts = pd.Series(signals).value_counts()
            colors = [self.color_scheme.get(s.lower(), self.color_scheme['neutral']) 
                     for s in signal_counts.index]
            
            ax2.bar(signal_counts.index, signal_counts.values, color=colors, alpha=0.8)
            ax2.set_title('Signal Distribution', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Count', fontsize=11)
            ax2.grid(True, alpha=0.3, axis='y')
            
            filepath = f'{self.output_dir}/{name}_calendar.png'
            plt.tight_layout()
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            files.append(filepath)
        
        return files
    
    def visualize_clustering(self, data, name):
        """Visualize clustering results"""
        files = []
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        if 'clusters' in data:
            clusters = data['clusters']
            
            cluster_sizes = {f"Cluster {cid}": cdata['event_count'] 
                           for cid, cdata in clusters.items()}
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_sizes)))
            
            ax.bar(cluster_sizes.keys(), cluster_sizes.values(), 
                  color=colors, alpha=0.8)
            ax.set_title(f'Cluster Distribution - {name}', 
                        fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Cluster', fontsize=12)
            ax.set_ylabel('Number of Events', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        filepath = f'{self.output_dir}/{name}_clustering.png'
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        files.append(filepath)
        
        return files
    
    def visualize_indicators(self, data, name):
        """Visualize technical indicators"""
        files = []
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        if 'indicators' in data:
            indicators = data['indicators']
            
            # Signal distribution
            signals = [ind.get('signal', 'NEUTRAL') for ind in indicators.values()]
            signal_counts = pd.Series(signals).value_counts()
            
            colors = []
            for sig in signal_counts.index:
                if 'BUY' in sig:
                    colors.append(self.color_scheme['bullish'])
                elif 'SELL' in sig:
                    colors.append(self.color_scheme['bearish'])
                else:
                    colors.append(self.color_scheme['neutral'])
            
            ax1.bar(signal_counts.index, signal_counts.values, color=colors, alpha=0.8)
            ax1.set_title('Indicator Signals Distribution', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Count', fontsize=11)
            ax1.grid(True, alpha=0.3, axis='y')
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Overall signal summary
            buy_count = data.get('buy_count', 0)
            sell_count = data.get('sell_count', 0)
            neutral_count = len(indicators) - buy_count - sell_count
            
            summary = {'BUY': buy_count, 'SELL': sell_count, 'NEUTRAL': neutral_count}
            colors_pie = [self.color_scheme['bullish'], 
                         self.color_scheme['bearish'],
                         self.color_scheme['neutral']]
            
            ax2.pie(summary.values(), labels=summary.keys(), autopct='%1.1f%%',
                   colors=colors_pie, startangle=90)
            ax2.set_title('Overall Signal Summary', fontsize=14, fontweight='bold')
        
        filepath = f'{self.output_dir}/{name}_indicators.png'
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        files.append(filepath)
        
        return files
    
    def visualize_cot(self, data, name):
        """Visualize COT (Commitment of Traders) data"""
        files = []
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Net positioning
        categories = []
        net_positions = []
        
        for key in ['dealer', 'asset_manager', 'leveraged', 'other']:
            if key in data:
                categories.append(key.replace('_', ' ').title())
                net_positions.append(data[key].get('net', 0))
        
        if categories:
            colors = [self.color_scheme['bullish'] if p > 0 else self.color_scheme['bearish'] 
                     for p in net_positions]
            
            ax1.barh(categories, net_positions, color=colors, alpha=0.8)
            ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
            ax1.set_title('Net Positions by Category', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Net Contracts', fontsize=11)
            ax1.grid(True, alpha=0.3, axis='x')
        
        # Long vs Short
        if 'leveraged' in data:
            lev = data['leveraged']
            values = [lev.get('long', 0), lev.get('short', 0)]
            labels = ['Long', 'Short']
            colors = [self.color_scheme['bullish'], self.color_scheme['bearish']]
            
            ax2.pie(values, labels=labels, autopct='%1.1f%%',
                   colors=colors, startangle=90)
            ax2.set_title('Hedge Funds Long/Short', fontsize=14, fontweight='bold')
        
        filepath = f'{self.output_dir}/{name}_cot.png'
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        files.append(filepath)
        
        return files
    
    def visualize_timeseries(self, df, name):
        """Visualize time series data"""
        files = []
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        price_col = 'close' if 'close' in df.columns else 'price'
        
        if price_col in df.columns:
            ax.plot(df.index, df[price_col], linewidth=2, 
                   color=self.color_scheme['primary'])
            
            # Add moving averages if enough data
            if len(df) >= 20:
                ma20 = df[price_col].rolling(20).mean()
                ax.plot(df.index, ma20, linewidth=1.5, alpha=0.7,
                       color=self.color_scheme['warning'], label='MA20')
            
            if len(df) >= 50:
                ma50 = df[price_col].rolling(50).mean()
                ax.plot(df.index, ma50, linewidth=1.5, alpha=0.7,
                       color=self.color_scheme['bearish'], label='MA50')
            
            ax.set_title(f'Price Chart - {name}', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Price', fontsize=12)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        
        filepath = f'{self.output_dir}/{name}_timeseries.png'
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        files.append(filepath)
        
        return files
    
    def visualize_correlation_matrix(self, df, name):
        """Visualize correlation matrix as heatmap"""
        files = []
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(df, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                   ax=ax)
        
        ax.set_title(f'Correlation Matrix - {name}', 
                    fontsize=16, fontweight='bold', pad=20)
        
        filepath = f'{self.output_dir}/{name}_correlation_matrix.png'
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        files.append(filepath)
        
        return files
    
    def visualize_generic(self, data, name):
        """Generic visualization for unknown data types"""
        files = []
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ax.text(0.5, 0.5, f'Data Type: {type(data).__name__}\n\n' + 
                          'Custom visualization not implemented\n' +
                          'Use specific visualization methods',
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        
        filepath = f'{self.output_dir}/{name}_generic.png'
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        files.append(filepath)
        
        return files
    
    def visualize_from_file(self, filepath, output_name=None):
        """Load data from file and visualize"""
        if output_name is None:
            output_name = Path(filepath).stem
        
        print(f"\nLoading: {filepath}")
        
        # Detect file type and load
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                data = json.load(f)
        elif filepath.endswith('.csv'):
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        else:
            print(f"Unsupported file type: {filepath}")
            return []
        
        return self.visualize(data, output_name)
    
    def batch_visualize(self, file_list):
        """Visualize multiple files"""
        all_files = []
        
        for filepath in file_list:
            files = self.visualize_from_file(filepath)
            all_files.extend(files)
        
        return all_files


if __name__ == "__main__":
    print("="*80)
    print("ADVANCED VISUALIZATION WRAPPER - TEST")
    print("="*80)
    
    viz = VisualizationWrapper(output_dir='test_visualizations')
    
    # Test data examples
    test_data = {
        'forecast': {
            'forecast_mean': [100, 102, 104, 106, 108, 110, 112],
            'forecast_dates': ['2024-01-01', '2024-01-02', '2024-01-03', 
                             '2024-01-04', '2024-01-05', '2024-01-06', '2024-01-07'],
            'confidence_80': {
                'lower': [98, 100, 101, 103, 104, 106, 108],
                'upper': [102, 104, 107, 109, 112, 114, 116]
            },
            'confidence_95': {
                'lower': [96, 98, 99, 101, 102, 104, 106],
                'upper': [104, 106, 109, 111, 114, 116, 118]
            }
        },
        'sentiment': {
            'articles': [
                {'sentiment_analysis': {'sentiment': 'positive'}},
                {'sentiment_analysis': {'sentiment': 'positive'}},
                {'sentiment_analysis': {'sentiment': 'negative'}},
                {'sentiment_analysis': {'sentiment': 'neutral'}},
                {'sentiment_analysis': {'sentiment': 'positive'}}
            ],
            'aggregated': {
                'bullish_score': 0.6,
                'bearish_score': 0.2,
                'overall_sentiment': 'positive'
            }
        },
        'indicators': {
            'indicators': {
                'RSI': {'value': 45, 'signal': 'NEUTRAL'},
                'MACD': {'value': 0.5, 'signal': 'BUY'},
                'Stochastic': {'value': 25, 'signal': 'BUY'},
                'Bollinger': {'value': '100-110', 'signal': 'NEUTRAL'}
            },
            'buy_count': 2,
            'sell_count': 0,
            'overall_signal': 'BUY'
        }
    }
    
    # Test each visualization type
    for data_type, data in test_data.items():
        print(f"\nTesting {data_type} visualization...")
        files = viz.visualize(data, f'test_{data_type}', file_type=data_type)
        print(f"Generated: {', '.join(files)}")
    
    print("\n" + "="*80)
    print(f"✓ Test complete | Files in: {viz.output_dir}/")
    print("="*80)
