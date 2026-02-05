"""
PDF-Optimized Report Generator
Generates HTML reports specifically designed for clean PDF conversion
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any


class PDFOptimizedReportGenerator:
    """Generate HTML reports optimized for PDF conversion"""
    
    def __init__(self):
        pass
    
    def generate_report(self, json_file: str, output_file: Optional[str] = None) -> str:
        """Generate PDF-optimized HTML report from analysis JSON"""
        
        print(f"\n{'='*80}")
        print("GENERATING PDF-OPTIMIZED REPORT")
        print(f"{'='*80}")
        print(f"Input: {json_file}")
        
        # Load data
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        metadata = data.get('metadata', {})
        sections = data.get('sections', {})
        
        # Generate report filename
        if output_file is None:
            base_name = Path(json_file).stem
            output_file = f"{base_name}_pdf_optimized.html"
        
        # Generate HTML
        html_content = self._generate_html(metadata, sections)
        
        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\n✓ Report saved: {output_file}")
        print(f"Size: {os.path.getsize(output_file) / 1024:.1f} KB")
        print(f"{'='*80}\n")
        
        return output_file
    
    def _generate_insights(self, section_name: str, section_data: Dict[str, Any]) -> str:
        """Generate insights for a section"""
        
        insights = []
        
        if section_name == 'executive_summary':
            sentiment = section_data.get('sentiment', {})
            insights.append(f"Overall sentiment: {sentiment.get('overall', 'N/A')}")
            findings = section_data.get('key_findings', [])
            if findings:
                insights.append(f"Key finding: {findings[0]}")
        
        elif section_name == 'news':
            count = section_data.get('article_count', 0)
            sentiment = section_data.get('sentiment', 'NEUTRAL')
            insights.append(f"Analyzed {count} articles with {sentiment} sentiment")
            themes = section_data.get('key_themes', [])
            if themes:
                insights.append(f"Key themes: {', '.join(themes[:3])}")
        
        elif section_name == 'indicators':
            bias = section_data.get('overall_bias', 'NEUTRAL')
            buy = section_data.get('buy_signals', 0)
            sell = section_data.get('sell_signals', 0)
            insights.append(f"Technical bias: {bias}")
            insights.append(f"Signal ratio: {buy} BUY vs {sell} SELL")
        
        elif section_name == 'cot':
            positioning = section_data.get('net_positioning', 'NEUTRAL')
            change = section_data.get('positioning_change', 'N/A')
            insights.append(f"Net positioning: {positioning} ({change})")
        
        elif section_name == 'economic':
            status = section_data.get('overall_status', 'MODERATE')
            insights.append(f"Economic environment: {status}")
        
        elif section_name == 'correlations':
            relationships = section_data.get('key_relationships', {})
            strong = [k for k, v in relationships.items() if v.get('strength') == 'STRONG']
            if strong:
                insights.append(f"Strong correlations: {', '.join(strong)}")
        
        elif section_name == 'structure':
            trend = section_data.get('trend', 'NEUTRAL')
            insights.append(f"Market structure: {trend}")
        
        elif section_name == 'seasonality':
            bias = section_data.get('seasonal_bias', 'NEUTRAL')
            insights.append(f"Seasonal bias: {bias}")
        
        elif section_name == 'volume':
            trend = section_data.get('volume_trend', 'NEUTRAL')
            insights.append(f"Volume trend: {trend}")
        
        elif section_name == 'hf_methods':
            sentiment = section_data.get('sentiment_score', 0.5)
            insights.append(f"AI sentiment score: {sentiment:.2f}")
        
        elif section_name == 'synthesis':
            outlook = section_data.get('overall_outlook', 'NEUTRAL')
            confidence = section_data.get('confidence', 0.5)
            insights.append(f"Overall outlook: {outlook} (confidence: {confidence:.0%})")
        
        if not insights:
            insights.append(f"{section_name.replace('_', ' ').title()} analysis completed")
        
        return ' • '.join(insights)
    
    def _generate_html(self, metadata: Dict[str, Any], sections: Dict[str, Any]) -> str:
        """Generate PDF-optimized HTML with inline CSS"""
        
        date = metadata.get('date', 'N/A')
        event_name = metadata.get('event_name', 'Market Analysis')
        
        # Start HTML with embedded CSS
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{event_name} - Financial Analysis Report</title>
    
    <style>
        /* PDF-Optimized Styles */
        @page {{
            size: A4;
            margin: 2cm 1.5cm;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            font-size: 11pt;
            line-height: 1.6;
            color: #333;
            background: white;
        }}
        
        /* Header */
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            margin-bottom: 30px;
            page-break-after: avoid;
        }}
        
        .header h1 {{
            font-size: 28pt;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        
        .header .subtitle {{
            font-size: 14pt;
            opacity: 0.9;
        }}
        
        .header .meta {{
            margin-top: 15px;
            font-size: 10pt;
        }}
        
        .header .badge {{
            display: inline-block;
            background: rgba(255,255,255,0.2);
            padding: 5px 15px;
            border-radius: 4px;
            margin-right: 10px;
        }}
        
        /* Sections */
        .section {{
            margin-bottom: 25px;
            page-break-inside: avoid;
            border-left: 4px solid #667eea;
            padding-left: 15px;
        }}
        
        .section-title {{
            font-size: 16pt;
            font-weight: 700;
            color: #667eea;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
        }}
        
        .section-title .icon {{
            margin-right: 10px;
        }}
        
        /* AI Insights Box */
        .insights {{
            background: #e7f3ff;
            border-left: 4px solid #2196F3;
            padding: 15px;
            margin-bottom: 15px;
            page-break-inside: avoid;
        }}
        
        .insights-title {{
            font-weight: 700;
            color: #2196F3;
            margin-bottom: 8px;
            font-size: 11pt;
        }}
        
        .insights-content {{
            color: #555;
            font-size: 10pt;
        }}
        
        /* Metrics Grid */
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 15px;
        }}
        
        .metric-card {{
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 12px;
            text-align: center;
            page-break-inside: avoid;
        }}
        
        .metric-label {{
            font-size: 9pt;
            color: #666;
            margin-bottom: 5px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .metric-value {{
            font-size: 20pt;
            font-weight: 700;
            color: #333;
        }}
        
        .metric-value.positive {{ color: #28a745; }}
        .metric-value.negative {{ color: #dc3545; }}
        .metric-value.neutral {{ color: #ffc107; }}
        
        /* Tables */
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            page-break-inside: avoid;
        }}
        
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }}
        
        th {{
            background: #f5f5f5;
            font-weight: 700;
            font-size: 10pt;
        }}
        
        td {{
            font-size: 10pt;
        }}
        
        /* Lists */
        ul {{
            margin: 10px 0 10px 25px;
        }}
        
        li {{
            margin-bottom: 5px;
            font-size: 10pt;
        }}
        
        /* Data Display */
        .data-grid {{
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 8px;
            margin: 10px 0;
        }}
        
        .data-label {{
            font-weight: 600;
            color: #555;
            font-size: 10pt;
        }}
        
        .data-value {{
            color: #333;
            font-size: 10pt;
        }}
        
        /* Footer */
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #667eea;
            text-align: center;
            font-size: 9pt;
            color: #666;
        }}
        
        /* Print-specific */
        @media print {{
            body {{
                background: white;
            }}
            
            .section {{
                page-break-inside: avoid;
            }}
            
            .no-print {{
                display: none;
            }}
        }}
        
        /* Status badges */
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 9pt;
            font-weight: 600;
        }}
        
        .badge.success {{ background: #d4edda; color: #155724; }}
        .badge.danger {{ background: #f8d7da; color: #721c24; }}
        .badge.warning {{ background: #fff3cd; color: #856404; }}
        .badge.info {{ background: #d1ecf1; color: #0c5460; }}
    </style>
</head>
<body>
    <!-- Header -->
    <div class="header">
        <h1>📊 {event_name}</h1>
        <div class="subtitle">AI-Powered Financial Analysis Report</div>
        <div class="meta">
            <span class="badge">📅 {date}</span>
            <span class="badge">🤖 BIST-Financial-Qwen-7B</span>
            <span class="badge">⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
        </div>
    </div>
    
    <!-- Content -->
"""
        
        # Render sections
        section_order = [
            ('executive_summary', 'Executive Summary', '⭐'),
            ('news', 'News Analysis', '📰'),
            ('indicators', 'Technical Indicators', '📈'),
            ('cot', 'COT Report', '📊'),
            ('economic', 'Economic Indicators', '💰'),
            ('correlations', 'Correlations', '🔗'),
            ('structure', 'Market Structure', '🏗️'),
            ('seasonality', 'Seasonality', '📅'),
            ('volume', 'Volume Analysis', '📊'),
            ('hf_methods', 'AI Methods', '🤖'),
            ('synthesis', 'Synthesis', '💡')
        ]
        
        for section_key, section_title, icon in section_order:
            if section_key in sections:
                html += self._render_section(
                    section_key, 
                    section_title, 
                    icon,
                    sections[section_key]
                )
        
        # Footer
        html += f"""
    <div class="footer">
        <p>🤖 Powered by BIST-Financial-Qwen-7B Local LLM</p>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Financial Analysis Pipeline v2.0</p>
    </div>
</body>
</html>
"""
        
        return html
    
    def _render_section(self, key: str, title: str, icon: str, data: Dict[str, Any]) -> str:
        """Render a section with PDF-friendly layout"""
        
        # Extract actual data if nested
        actual_data = data.get('data', data)
        
        # Generate insights
        insights = self._generate_insights(key, actual_data)
        
        html = f"""
    <div class="section">
        <div class="section-title">
            <span class="icon">{icon}</span>
            <span>{title}</span>
        </div>
        
        <div class="insights">
            <div class="insights-title">🤖 AI Insights</div>
            <div class="insights-content">{insights}</div>
        </div>
"""
        
        # Add section-specific content
        if key == 'executive_summary':
            html += self._render_executive_summary(actual_data)
        elif key == 'news':
            html += self._render_news(actual_data)
        elif key == 'indicators':
            html += self._render_indicators(actual_data)
        elif key == 'cot':
            html += self._render_cot(actual_data)
        elif key == 'economic':
            html += self._render_economic(actual_data)
        elif key == 'correlations':
            html += self._render_correlations(actual_data)
        elif key == 'structure':
            html += self._render_structure(actual_data)
        elif key == 'seasonality':
            html += self._render_seasonality(actual_data)
        elif key == 'volume':
            html += self._render_volume(actual_data)
        elif key == 'hf_methods':
            html += self._render_hf_methods(actual_data)
        elif key == 'synthesis':
            html += self._render_synthesis(actual_data)
        
        html += """
    </div>
"""
        
        return html
    
    def _render_executive_summary(self, data: Dict) -> str:
        sentiment = data.get('sentiment', {})
        findings = data.get('key_findings', [])
        recommendations = data.get('recommendations', [])
        
        html = f"""
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-label">Overall Sentiment</div>
                <div class="metric-value positive">{sentiment.get('overall', 'N/A')}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Confidence</div>
                <div class="metric-value">{sentiment.get('confidence', 0):.0%}</div>
            </div>
        </div>
"""
        
        if findings:
            html += "<h4>Key Findings:</h4><ul>"
            for finding in findings[:5]:
                html += f"<li>{finding}</li>"
            html += "</ul>"
        
        if recommendations:
            html += "<h4>Recommendations:</h4><ul>"
            for rec in recommendations[:5]:
                html += f"<li>{rec}</li>"
            html += "</ul>"
        
        return html
    
    def _render_news(self, data: Dict) -> str:
        html = f"""
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-label">Articles</div>
                <div class="metric-value">{data.get('article_count', 0)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Sources</div>
                <div class="metric-value">{len(data.get('sources', []))}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Sentiment</div>
                <div class="metric-value neutral">{data.get('sentiment', 'N/A')}</div>
            </div>
        </div>
"""
        
        themes = data.get('key_themes', [])
        if themes:
            html += "<h4>Key Themes:</h4><ul>"
            for theme in themes[:5]:
                html += f"<li>{theme}</li>"
            html += "</ul>"
        
        return html
    
    def _render_indicators(self, data: Dict) -> str:
        buy = data.get('buy_signals', 0)
        sell = data.get('sell_signals', 0)
        total = buy + sell
        buy_pct = (buy / total * 100) if total > 0 else 0
        
        html = f"""
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-label">Overall Bias</div>
                <div class="metric-value positive">{data.get('overall_bias', 'N/A')}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Buy Signals</div>
                <div class="metric-value positive">{buy} ({buy_pct:.0f}%)</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Sell Signals</div>
                <div class="metric-value negative">{sell} ({100-buy_pct:.0f}%)</div>
            </div>
        </div>
"""
        return html
    
    def _render_cot(self, data: Dict) -> str:
        html = f"""
        <div class="data-grid">
            <div class="data-label">Net Positioning:</div>
            <div class="data-value"><span class="badge success">{data.get('net_positioning', 'N/A')}</span></div>
            
            <div class="data-label">Position Change:</div>
            <div class="data-value">{data.get('positioning_change', 'N/A')}</div>
            
            <div class="data-label">Institutional Sentiment:</div>
            <div class="data-value"><span class="badge info">{data.get('institutional_sentiment', 'N/A')}</span></div>
        </div>
"""
        return html
    
    def _render_economic(self, data: Dict) -> str:
        html = f"""
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-label">Overall Status</div>
                <div class="metric-value neutral">{data.get('overall_status', 'N/A')}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Inflation</div>
                <div class="metric-value">{data.get('inflation_trend', 'N/A')}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Growth</div>
                <div class="metric-value">{data.get('growth_outlook', 'N/A')}</div>
            </div>
        </div>
"""
        return html
    
    def _render_correlations(self, data: Dict) -> str:
        relationships = data.get('key_relationships', {})
        
        if relationships:
            html = "<table><thead><tr><th>Pair</th><th>Strength</th><th>Direction</th></tr></thead><tbody>"
            for pair, details in list(relationships.items())[:5]:
                html += f"""
                <tr>
                    <td>{pair.replace('_', ' / ')}</td>
                    <td><span class="badge info">{details.get('strength', 'N/A')}</span></td>
                    <td>{details.get('direction', 'N/A')}</td>
                </tr>
"""
            html += "</tbody></table>"
        else:
            html = "<p>No correlation data available</p>"
        
        return html
    
    def _render_structure(self, data: Dict) -> str:
        support = data.get('support_levels', [])
        resistance = data.get('resistance_levels', [])
        
        html = f"""
        <div class="data-grid">
            <div class="data-label">Trend:</div>
            <div class="data-value"><span class="badge success">{data.get('trend', 'N/A')}</span></div>
            
            <div class="data-label">Support Levels:</div>
            <div class="data-value">{', '.join(map(str, support[:3])) if support else 'N/A'}</div>
            
            <div class="data-label">Resistance Levels:</div>
            <div class="data-value">{', '.join(map(str, resistance[:3])) if resistance else 'N/A'}</div>
        </div>
"""
        return html
    
    def _render_seasonality(self, data: Dict) -> str:
        html = f"""
        <div class="data-grid">
            <div class="data-label">Seasonal Bias:</div>
            <div class="data-value"><span class="badge success">{data.get('seasonal_bias', 'N/A')}</span></div>
            
            <div class="data-label">Historical Performance:</div>
            <div class="data-value">{data.get('historical_performance', 'N/A')}</div>
        </div>
"""
        return html
    
    def _render_volume(self, data: Dict) -> str:
        html = f"""
        <div class="data-grid">
            <div class="data-label">Volume Trend:</div>
            <div class="data-value"><span class="badge info">{data.get('volume_trend', 'N/A')}</span></div>
            
            <div class="data-label">Volume Profile:</div>
            <div class="data-value">{data.get('volume_profile', 'N/A')}</div>
        </div>
"""
        return html
    
    def _render_hf_methods(self, data: Dict) -> str:
        html = f"""
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-label">Sentiment Score</div>
                <div class="metric-value">{data.get('sentiment_score', 0):.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Forecast</div>
                <div class="metric-value positive">{data.get('forecast_direction', 'N/A')}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Anomalies</div>
                <div class="metric-value">{data.get('anomalies_detected', 0)}</div>
            </div>
        </div>
"""
        return html
    
    def _render_synthesis(self, data: Dict) -> str:
        factors = data.get('key_factors', [])
        risks = data.get('risks', [])
        
        html = f"""
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-label">Overall Outlook</div>
                <div class="metric-value positive">{data.get('overall_outlook', 'N/A')}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Confidence</div>
                <div class="metric-value">{data.get('confidence', 0):.0%}</div>
            </div>
        </div>
"""
        
        if factors:
            html += "<h4>Key Factors:</h4><ul>"
            for factor in factors[:5]:
                html += f"<li>{factor}</li>"
            html += "</ul>"
        
        if risks:
            html += "<h4>Risks:</h4><ul>"
            for risk in risks[:3]:
                html += f"<li>{risk}</li>"
            html += "</ul>"
        
        return html


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pdf_optimized_report_generator.py <json_file>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    if not os.path.exists(json_file):
        print(f"Error: File not found: {json_file}")
        sys.exit(1)
    
    generator = PDFOptimizedReportGenerator()
    output_file = generator.generate_report(json_file)
    
    print(f"\n✓ Success! Open: {output_file}")
    print(f"\n💡 To convert to PDF:")
    print(f"   python html_to_pdf.py {output_file}")


if __name__ == "__main__":
    main()
