"""
Local LLM Report Generator using BIST-Financial-Qwen-7B
Generates AI-powered HTML reports using financial-tuned LLM
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
    print("⚠️  llama-cpp-python not available")
    print("Install with: pip install llama-cpp-python")


class LocalLLMReportGenerator:
    """Generate comprehensive HTML reports using BIST-Financial-Qwen-7B"""
    
    def __init__(self):
        self.model = None
        
        if LLAMA_CPP_AVAILABLE:
            try:
                print("Loading BIST-Financial-Qwen-7B model...")
                self.model = Llama.from_pretrained(
                    repo_id="bist-quant/BIST-Financial-Qwen-7B",
                    filename="gguf/qwen-kap-final-Q4_K_M.gguf",
                    n_ctx=2048,
                    n_threads=4,
                    n_gpu_layers=0
                )
                print("✓ BIST-Financial-Qwen-7B loaded successfully")
            except Exception as e:
                print(f"⚠️  Could not load model: {e}")
                print("⚠️  Using fallback mode")
                self.model = None
        else:
            print("⚠️  llama-cpp-python not available, using fallback mode")
    
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
        
        # Load section JSONs
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
                    print(f"  ✓ Loaded {section_type}")
            except Exception as e:
                print(f"  ⚠️  Could not load {section_file.name}: {e}")
        
        return sections
    
    def _generate_insights(self, section_name: str, section_data: Dict[str, Any]) -> str:
        """Generate AI insights using BIST-Financial-Qwen-7B"""
        
        if not self.model:
            return self._generate_fallback_insights(section_name, section_data)
        
        # Create financial analysis prompt
        data_summary = json.dumps(section_data, indent=2)[:1000]
        
        prompt = f"""Analyze this {section_name} financial data and provide 3 key insights:

{data_summary}

Provide concise, actionable insights in bullet points."""
        
        try:
            response = self.model.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial analyst providing clear, data-driven insights."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            insights = response['choices'][0]['message']['content'].strip()
            return insights if insights else self._generate_fallback_insights(section_name, section_data)
            
        except Exception as e:
            print(f"  ⚠️  LLM generation failed for {section_name}: {e}")
            return self._generate_fallback_insights(section_name, section_data)
    
    def _generate_fallback_insights(self, section_name: str, section_data: Dict[str, Any]) -> str:
        """Generate insights without LLM"""
        
        insights = []
        
        if section_name == 'news':
            count = section_data.get('article_count', 0)
            themes = section_data.get('key_themes', [])
            sentiment = section_data.get('sentiment', 'NEUTRAL')
            insights.append(f"• Analyzed {count} news articles with {sentiment} sentiment")
            if themes:
                insights.append(f"• Key themes: {', '.join(themes[:3])}")
        
        elif section_name == 'indicators':
            bias = section_data.get('overall_bias', 'NEUTRAL')
            buy = section_data.get('buy_signals', 0)
            sell = section_data.get('sell_signals', 0)
            insights.append(f"• Technical bias: {bias}")
            insights.append(f"• Signal ratio: {buy} BUY vs {sell} SELL")
            if buy > sell * 1.5:
                insights.append(f"• Strong bullish momentum detected")
        
        elif section_name == 'economic':
            status = section_data.get('overall_status', 'MODERATE')
            insights.append(f"• Economic environment: {status}")
            if 'inflation_trend' in section_data:
                insights.append(f"• Inflation trend: {section_data['inflation_trend']}")
        
        elif section_name == 'cot':
            positioning = section_data.get('net_positioning', 'NEUTRAL')
            change = section_data.get('positioning_change', 'N/A')
            insights.append(f"• Net positioning: {positioning} ({change})")
            insights.append(f"• Institutional sentiment: {section_data.get('institutional_sentiment', 'N/A')}")
        
        elif section_name == 'correlations':
            relationships = section_data.get('key_relationships', {})
            strong = [k for k, v in relationships.items() if v.get('strength') == 'STRONG']
            if strong:
                insights.append(f"• Strong correlations: {', '.join(strong[:2])}")
        
        elif section_name == 'structure':
            trend = section_data.get('trend', 'NEUTRAL')
            insights.append(f"• Market structure: {trend}")
            support = section_data.get('support_levels', [])
            if support:
                insights.append(f"• Key support: {support[0]}")
        
        elif section_name == 'seasonality':
            bias = section_data.get('seasonal_bias', 'NEUTRAL')
            perf = section_data.get('historical_performance', 'N/A')
            insights.append(f"• Seasonal bias: {bias}")
            insights.append(f"• Historical performance: {perf}")
        
        elif section_name == 'volume':
            trend = section_data.get('volume_trend', 'NEUTRAL')
            profile = section_data.get('volume_profile', 'NEUTRAL')
            insights.append(f"• Volume trend: {trend}")
            insights.append(f"• Profile: {profile}")
        
        elif section_name == 'hf_methods':
            sentiment = section_data.get('sentiment_score', 0.5)
            forecast = section_data.get('forecast_direction', 'NEUTRAL')
            insights.append(f"• AI sentiment score: {sentiment:.2f}")
            insights.append(f"• Forecast direction: {forecast}")
        
        elif section_name == 'synthesis':
            outlook = section_data.get('overall_outlook', 'NEUTRAL')
            confidence = section_data.get('confidence', 0.5)
            insights.append(f"• Overall outlook: {outlook} (confidence: {confidence:.0%})")
            factors = section_data.get('key_factors', [])
            if factors:
                insights.append(f"• Key factors: {factors[0]}")
        
        elif section_name == 'executive_summary':
            sentiment = section_data.get('sentiment', {})
            insights.append(f"• Overall: {sentiment.get('overall', 'N/A')}")
            findings = section_data.get('key_findings', [])
            if findings:
                insights.append(f"• {findings[0]}")
        
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
    <title>{event_name} - AI Analysis Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #2d3748;
            line-height: 1.6;
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
        .ai-badge {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.6em;
            font-weight: bold;
        }}
        .insights {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin: 15px 0;
            line-height: 1.8;
            white-space: pre-line;
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
            <strong>AI Model:</strong> BIST-Financial-Qwen-7B<br>
            <strong>Sections:</strong> {len(sections)}
        </div>
"""
        
        # Executive summary first
        if 'executive_summary' in sections:
            html += self._render_section('executive_summary', sections['executive_summary'], True)
        
        # Render sections in order
        section_order = [
            'news', 'indicators', 'cot', 'economic', 
            'correlations', 'structure', 'seasonality', 'volume',
            'hf_methods', 'synthesis'
        ]
        
        for section_name in section_order:
            if section_name in sections:
                html += self._render_section(section_name, sections[section_name])
        
        html += f"""
        <div class="footer">
            Powered by BIST-Financial-Qwen-7B<br>
            {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>
"""
        
        return html
    
    def _render_section(self, section_name: str, section_data: Dict[str, Any], is_summary: bool = False) -> str:
        """Render a section with AI insights"""
        
        if 'data' in section_data:
            actual_data = section_data['data']
        else:
            actual_data = section_data
        
        icons = {
            'executive_summary': '⭐',
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
            <div class="section-title">
                {icon} {title}
                <span class="ai-badge">AI</span>
            </div>
"""
        
        # Generate AI insights
        print(f"  🤖 Generating insights for {section_name}...")
        insights = self._generate_insights(section_name, actual_data)
        html += f'<div class="insights">{insights}</div>'
        
        # Add metrics
        if not is_summary:
            html += self._render_metrics(section_name, actual_data)
        
        html += """
        </div>
"""
        return html
    
    def _render_metrics(self, section_name: str, data: Dict[str, Any]) -> str:
        """Render key metrics"""
        
        html = '<div class="data-grid">'
        
        if section_name == 'news':
            html += f"""
                <div class="data-card">
                    <div class="data-label">Articles</div>
                    <div class="data-value">{data.get('article_count', 0)}</div>
                </div>
                <div class="data-card">
                    <div class="data-label">Sentiment</div>
                    <div class="data-value">{data.get('sentiment', 'N/A')}</div>
                </div>
            """
        
        elif section_name == 'indicators':
            html += f"""
                <div class="data-card">
                    <div class="data-label">Bias</div>
                    <div class="data-value">{data.get('overall_bias', 'N/A')}</div>
                </div>
                <div class="data-card">
                    <div class="data-label">Buy/Sell</div>
                    <div class="data-value">{data.get('buy_signals', 0)}/{data.get('sell_signals', 0)}</div>
                </div>
            """
        
        html += '</div>'
        return html


def main():
    """Main entry point"""
    
    if len(sys.argv) < 2:
        print("Usage: python local_llm_report_generator.py <json_file>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    if not os.path.exists(json_file):
        print(f"Error: File not found: {json_file}")
        sys.exit(1)
    
    generator = LocalLLMReportGenerator()
    output_file = generator.generate_report(json_file)
    
    print(f"\n✓ Success! Open: {output_file}")


if __name__ == "__main__":
    main()
