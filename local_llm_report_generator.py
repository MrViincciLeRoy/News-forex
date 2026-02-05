"""
Local LLM Report Generator - Fixed Version
Generates AI-powered HTML reports using local LLM (no API keys needed)
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import sys

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("⚠️  llama-cpp-python not available, using fallback mode")


class LocalLLMReportGenerator:
    """Generate comprehensive HTML reports using local LLM"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        
        if LLAMA_CPP_AVAILABLE and model_path and os.path.exists(model_path):
            try:
                print(f"Loading model from: {model_path}")
                self.model = Llama(
                    model_path=model_path,
                    n_ctx=2048,
                    n_threads=4,
                    n_gpu_layers=0
                )
                print("✓ Model loaded successfully")
            except Exception as e:
                print(f"⚠️  Could not load model: {e}")
                self.model = None
        else:
            print("⚠️  Using fallback mode (no LLM)")
    
    def generate_report(self, json_file: str, output_file: Optional[str] = None) -> str:
        """Generate HTML report from analysis JSON"""
        
        print(f"\n{'='*80}")
        print("GENERATING AI-POWERED REPORT")
        print(f"{'='*80}")
        print(f"Input: {json_file}")
        
        # Load main data
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        metadata = data.get('metadata', {})
        sections = data.get('sections', {})
        
        # Try to load section JSONs if available
        sections_dir = Path(json_file).parent / 'sections'
        if sections_dir.exists():
            print(f"Loading section JSONs from: {sections_dir}")
            sections = self._load_section_jsons(sections_dir, data)
        
        # Generate report filename
        if output_file is None:
            base_name = Path(json_file).stem
            output_file = f"{base_name}_ai_report.html"
        
        # Generate HTML
        html_content = self._generate_html(metadata, sections)
        
        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\n✓ Report saved: {output_file}")
        print(f"Size: {os.path.getsize(output_file) / 1024:.1f} KB")
        print(f"{'='*80}\n")
        
        return output_file
    
    def _load_section_jsons(self, sections_dir: Path, main_data: Dict[str, Any]) -> Dict[str, Any]:
        """Load individual section JSON files"""
        sections = main_data.get('sections', {})
        
        for section_file in sections_dir.glob('*.json'):
            try:
                with open(section_file, 'r') as f:
                    section_data = json.load(f)
                    section_type = section_data.get('section_type', section_file.stem.split('_')[0])
                    sections[section_type] = section_data.get('data', section_data)
            except Exception as e:
                print(f"⚠️  Could not load {section_file.name}: {e}")
        
        return sections
    
    def _generate_insights(self, section_name: str, section_data: Dict[str, Any]) -> str:
        """Generate AI insights for a section"""
        
        if not self.model:
            return self._generate_fallback_insights(section_name, section_data)
        
        # Create prompt
        prompt = f"""Analyze this {section_name} data and provide 2-3 key insights:

{json.dumps(section_data, indent=2)[:500]}

Insights:"""
        
        try:
            response = self.model(
                prompt,
                max_tokens=200,
                temperature=0.7,
                stop=["###", "\n\n\n"]
            )
            
            insights = response['choices'][0]['text'].strip()
            return insights or self._generate_fallback_insights(section_name, section_data)
            
        except Exception as e:
            print(f"⚠️  LLM generation failed: {e}")
            return self._generate_fallback_insights(section_name, section_data)
    
    def _generate_fallback_insights(self, section_name: str, section_data: Dict[str, Any]) -> str:
        """Generate insights without LLM"""
        
        insights = []
        
        if section_name == 'news':
            count = section_data.get('article_count', 0)
            themes = section_data.get('key_themes', [])
            insights.append(f"• Analyzed {count} news articles")
            if themes:
                insights.append(f"• Key themes: {', '.join(themes[:3])}")
        
        elif section_name == 'indicators':
            bias = section_data.get('overall_bias', 'NEUTRAL')
            buy = section_data.get('buy_signals', 0)
            sell = section_data.get('sell_signals', 0)
            insights.append(f"• Technical bias: {bias}")
            insights.append(f"• Signals: {buy} BUY vs {sell} SELL")
        
        elif section_name == 'economic':
            status = section_data.get('overall_status', 'MODERATE')
            insights.append(f"• Economic environment: {status}")
        
        elif section_name == 'correlations':
            relationships = section_data.get('key_relationships', {})
            strong = [k for k, v in relationships.items() if v.get('strength') == 'STRONG']
            if strong:
                insights.append(f"• Strong correlations found in: {', '.join(strong[:2])}")
        
        if not insights:
            insights.append(f"• {section_name.title()} analysis completed")
        
        return '\n'.join(insights)
    
    def _generate_html(self, metadata: Dict[str, Any], sections: Dict[str, Any]) -> str:
        """Generate complete HTML report"""
        
        date = metadata.get('date', 'N/A')
        event_name = metadata.get('event_name', 'Market Analysis')
        timestamp = metadata.get('timestamp', datetime.now().isoformat())
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{event_name} - Analysis Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #2d3748;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        }}
        h1 {{
            color: #667eea;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .meta {{
            color: #718096;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid #e2e8f0;
        }}
        .section {{
            margin: 30px 0;
            padding: 25px;
            background: #f7fafc;
            border-radius: 12px;
            border-left: 4px solid #667eea;
        }}
        .section-title {{
            font-size: 1.5em;
            color: #2d3748;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .insights {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin: 15px 0;
            line-height: 1.8;
            white-space: pre-line;
        }}
        .stat {{
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.9em;
            margin: 5px;
            font-weight: 600;
        }}
        .data-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 15px 0;
        }}
        .data-card {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            border-left: 3px solid #48bb78;
        }}
        .data-label {{
            color: #718096;
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .data-value {{
            font-size: 1.3em;
            font-weight: bold;
            color: #2d3748;
            margin-top: 5px;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #e2e8f0;
            text-align: center;
            color: #718096;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 {event_name}</h1>
        <div class="meta">
            <strong>Date:</strong> {date}<br>
            <strong>Generated:</strong> {timestamp[:19].replace('T', ' ')}<br>
            <strong>Sections:</strong> {len(sections)}
        </div>
"""
        
        # Add executive summary if available
        exec_summary = sections.get('executive_summary', {})
        if exec_summary:
            html += self._render_executive_summary(exec_summary)
        
        # Render each section
        section_order = ['news', 'indicators', 'cot', 'economic', 'correlations', 
                        'structure', 'seasonality', 'volume', 'hf_methods', 'synthesis']
        
        for section_name in section_order:
            if section_name in sections:
                html += self._render_section(section_name, sections[section_name])
        
        # Render any remaining sections
        for section_name, section_data in sections.items():
            if section_name not in section_order and section_name != 'executive_summary':
                html += self._render_section(section_name, section_data)
        
        html += f"""
        <div class="footer">
            Generated by Local LLM Report Generator<br>
            {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>
"""
        
        return html
    
    def _render_executive_summary(self, summary: Dict[str, Any]) -> str:
        """Render executive summary section"""
        
        html = """
        <div class="section">
            <div class="section-title">⭐ Executive Summary</div>
            <div class="insights">
"""
        
        if summary.get('market_overview'):
            html += f"<strong>Overview:</strong> {summary['market_overview']}<br><br>"
        
        sentiment = summary.get('sentiment', {})
        html += f"""
            <div class="data-grid">
                <div class="data-card">
                    <div class="data-label">Overall Sentiment</div>
                    <div class="data-value">{sentiment.get('overall', 'NEUTRAL')}</div>
                </div>
                <div class="data-card">
                    <div class="data-label">Confidence</div>
                    <div class="data-value">{sentiment.get('confidence', 0.5):.0%}</div>
                </div>
            </div>
"""
        
        findings = summary.get('key_findings', [])
        if findings:
            html += "<strong>Key Findings:</strong><ul>"
            for finding in findings[:5]:
                if isinstance(finding, dict):
                    html += f"<li>{finding.get('finding', str(finding))}</li>"
                else:
                    html += f"<li>{finding}</li>"
            html += "</ul>"
        
        html += """
            </div>
        </div>
"""
        return html
    
    def _render_section(self, section_name: str, section_data: Dict[str, Any]) -> str:
        """Render a section"""
        
        # Get actual data if wrapped in metadata
        if 'data' in section_data and isinstance(section_data['data'], dict):
            actual_data = section_data['data']
        else:
            actual_data = section_data
        
        icons = {
            'news': '📰',
            'indicators': '📈',
            'cot': '📊',
            'economic': '💹',
            'correlations': '🔗',
            'structure': '🏗️',
            'seasonality': '📅',
            'volume': '📊',
            'hf_methods': '🤖',
            'synthesis': '💡'
        }
        
        icon = icons.get(section_name, '📋')
        title = section_name.replace('_', ' ').title()
        
        html = f"""
        <div class="section">
            <div class="section-title">{icon} {title}</div>
"""
        
        # Generate insights
        insights = self._generate_insights(section_name, actual_data)
        html += f'<div class="insights">{insights}</div>'
        
        # Add key metrics
        html += self._render_metrics(section_name, actual_data)
        
        html += """
        </div>
"""
        return html
    
    def _render_metrics(self, section_name: str, data: Dict[str, Any]) -> str:
        """Render key metrics for a section"""
        
        html = '<div class="data-grid">'
        
        if section_name == 'news':
            html += f"""
                <div class="data-card">
                    <div class="data-label">Articles</div>
                    <div class="data-value">{data.get('article_count', 0)}</div>
                </div>
                <div class="data-card">
                    <div class="data-label">Sources</div>
                    <div class="data-value">{len(data.get('sources', []))}</div>
                </div>
            """
        
        elif section_name == 'indicators':
            html += f"""
                <div class="data-card">
                    <div class="data-label">Overall Bias</div>
                    <div class="data-value">{data.get('overall_bias', 'NEUTRAL')}</div>
                </div>
                <div class="data-card">
                    <div class="data-label">Symbols Analyzed</div>
                    <div class="data-value">{data.get('symbols_analyzed', 0)}</div>
                </div>
            """
        
        elif section_name == 'economic':
            html += f"""
                <div class="data-card">
                    <div class="data-label">Economic Status</div>
                    <div class="data-value">{data.get('overall_status', 'MODERATE')}</div>
                </div>
            """
        
        html += '</div>'
        return html


def main():
    """Main entry point"""
    
    if len(sys.argv) < 2:
        print("Usage: python local_llm_report_generator.py <json_file> [model_path]")
        sys.exit(1)
    
    json_file = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(json_file):
        print(f"Error: File not found: {json_file}")
        sys.exit(1)
    
    generator = LocalLLMReportGenerator(model_path=model_path)
    output_file = generator.generate_report(json_file)
    
    print(f"\n✓ Success! Open: {output_file}")


if __name__ == "__main__":
    main()
