"""
AI-Powered Comprehensive Report Generator using Groq API
Uses Qwen2.5-32B model to analyze pipeline results and generate detailed HTML reports
FIXED: Corrected model name format for Groq API
"""

import json
import os
import base64
from datetime import datetime
from pathlib import Path
import requests


class GroqAIReportGenerator:
    """
    Generate comprehensive HTML reports using Groq's free tier Qwen model
    Analyzes all pipeline data and creates detailed, interactive reports
    """
    
    def __init__(self, groq_api_key=None):
        self.api_key = groq_api_key or os.environ.get('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found. Set environment variable or pass as parameter.")
        
        self.api_url = "https://api.groq.com/v1/chat/completions"
        # FIXED: Use correct Groq API model name format
        self.model = "qwen/qwen-2.5-32b"  # Changed from "qwen/qwen3-32b"
        
        self.max_tokens_per_request = 8000
        self.requests_per_minute = 30
    
    def _call_groq_api(self, messages, max_tokens=8000):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except requests.exceptions.RequestException as e:
            print(f"API Error: {str(e)}")
            return None
    
    def analyze_section(self, section_name, section_data, context=""):
        prompt = f"""You are a financial market analyst reviewing comprehensive market analysis data.

Analyze the following {section_name.upper()} section data and provide:
1. A detailed summary of key findings
2. Important patterns or trends identified
3. Implications for traders and investors
4. Risk factors or opportunities highlighted by this data
5. Specific numerical insights with context

Context: {context}

Data:
{json.dumps(section_data, indent=2)[:4000]}

Provide your analysis in well-structured HTML format using Bootstrap classes:
- Use <h4> for subsection headers
- Use <p class="lead"> for key insights
- Use <ul> or <ol> for lists
- Use <div class="alert alert-info/warning/success"> for important callouts
- Include specific numbers and percentages from the data
- Be detailed and thorough but concise

Generate ONLY the HTML content, no markdown wrappers."""

        messages = [
            {
                "role": "system",
                "content": "You are an expert financial analyst specializing in market data interpretation. Provide detailed, actionable insights in HTML format."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        analysis = self._call_groq_api(messages)
        
        if not analysis:
            return f"<p>Analysis unavailable for {section_name}</p>"
        
        return analysis
    
    def generate_comprehensive_report(self, json_file, output_html=None, viz_dir=None):
        print("="*80)
        print("AI-POWERED COMPREHENSIVE REPORT GENERATOR")
        print("="*80)
        print(f"Model: {self.model}")
        print(f"Input: {json_file}")
        print("="*80)
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if output_html is None:
            base_name = Path(json_file).stem
            output_html = str(Path(json_file).parent / f"{base_name}_ai_report.html")
        
        if viz_dir is None:
            viz_dir = Path(json_file).parent / 'visualizations'
        
        metadata = data.get('metadata', {})
        event_name = metadata.get('event_name', 'Market Analysis')
        analysis_date = metadata.get('date', 'N/A')
        
        print(f"\n📊 Analyzing: {event_name}")
        print(f"📅 Date: {analysis_date}")
        
        html_parts = [self._get_html_header(event_name, analysis_date)]
        
        print("\n🤖 Generating Executive Summary...")
        exec_summary = self._generate_executive_summary(data)
        html_parts.append(exec_summary)
        
        sections = data.get('sections', {})
        
        section_order = [
            ('news', '📰 News Analysis'),
            ('hf_methods', '🤖 AI Analysis'),
            ('indicators', '📈 Technical Indicators'),
            ('cot', '📊 COT Positioning'),
            ('economic', '💹 Economic Indicators'),
            ('correlations', '🔗 Correlations'),
            ('structure', '🏗️ Market Structure'),
            ('volume', '📊 Volume Analysis'),
            ('seasonality', '📅 Seasonality'),
            ('synthesis', '💡 Synthesis & Insights')
        ]
        
        for section_key, section_title in section_order:
            if section_key in sections and sections[section_key]:
                print(f"\n🔍 Analyzing: {section_title}")
                section_html = self._generate_section_html(
                    section_key, 
                    section_title, 
                    sections[section_key],
                    metadata
                )
                html_parts.append(section_html)
        
        if viz_dir and Path(viz_dir).exists():
            print(f"\n🖼️ Adding visualizations...")
            viz_html = self._generate_visualizations_section(viz_dir)
            html_parts.append(viz_html)
        
        print("\n📋 Adding raw data section...")
        html_parts.append(self._generate_raw_data_section(data))
        
        html_parts.append(self._get_html_footer())
        
        final_html = '\n'.join(html_parts)
        with open(output_html, 'w', encoding='utf-8') as f:
            f.write(final_html)
        
        print(f"\n✅ Report generated: {output_html}")
        print("="*80)
        
        return output_html
    
    def _generate_executive_summary(self, data):
        summary_data = {
            'metadata': data.get('metadata', {}),
            'synthesis': data.get('sections', {}).get('synthesis', {}),
            'hf_sentiment': data.get('sections', {}).get('hf_methods', {}).get('sentiment', {}).get('aggregated', {}),
            'indicators_count': len(data.get('sections', {}).get('indicators', {})),
            'symbols_analyzed': data.get('metadata', {}).get('symbols_count', 0)
        }
        
        prompt = f"""You are a senior financial analyst preparing an executive summary for stakeholders.

Based on this comprehensive market analysis data:
{json.dumps(summary_data, indent=2)}

Create a compelling executive summary that includes:
1. **Market Overview**: What happened and why it matters
2. **Key Findings**: 3-5 most important insights with specific data points
3. **AI Sentiment Analysis**: Summary of sentiment findings
4. **Risk Assessment**: Top 3 risks identified
5. **Opportunities**: Top 3 opportunities highlighted
6. **Bottom Line**: Clear actionable conclusion

Format as HTML with Bootstrap classes:
- Use <div class="alert alert-primary"> for the overview
- Use <div class="row"> and <div class="col-md-6"> for side-by-side risk/opportunity
- Use <span class="badge bg-success/danger/warning"> for sentiment indicators
- Use icons: ⬆️ for bullish, ⬇️ for bearish, ➡️ for neutral
- Be specific with numbers and percentages
- Make it visually appealing and easy to scan

Generate ONLY the HTML content for the executive summary section."""

        messages = [
            {
                "role": "system",
                "content": "You are a top-tier financial analyst writing executive summaries for institutional investors."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        summary = self._call_groq_api(messages, max_tokens=2000)
        
        if not summary:
            summary = "<p>Executive summary unavailable</p>"
        
        return f"""
        <section id="executive-summary" class="mb-5">
            <h2 class="mb-4">
                <i class="bi bi-star-fill text-warning"></i> Executive Summary
            </h2>
            {summary}
        </section>
        """
    
    def _generate_section_html(self, section_key, section_title, section_data, metadata):
        context = f"Analysis Date: {metadata.get('date')}. Event: {metadata.get('event_name', 'General Market Analysis')}."
        
        ai_analysis = self.analyze_section(section_key, section_data, context)
        chart_html = self._create_section_charts(section_key, section_data)
        
        return f"""
        <section id="section-{section_key}" class="mb-5">
            <h2 class="mb-4">
                {section_title}
            </h2>
            
            <div class="row">
                <div class="col-lg-8">
                    <div class="card mb-4">
                        <div class="card-header bg-primary text-white">
                            <h5 class="mb-0">AI Analysis</h5>
                        </div>
                        <div class="card-body">
                            {ai_analysis}
                        </div>
                    </div>
                </div>
                
                <div class="col-lg-4">
                    {chart_html}
                </div>
            </div>
            
            <div class="mt-3">
                <button class="btn btn-outline-secondary btn-sm" type="button" 
                        data-bs-toggle="collapse" data-bs-target="#raw-{section_key}">
                    <i class="bi bi-code-square"></i> View Raw Data
                </button>
                <div class="collapse mt-2" id="raw-{section_key}">
                    <pre class="bg-light p-3 rounded"><code>{json.dumps(section_data, indent=2)[:2000]}</code></pre>
                </div>
            </div>
        </section>
        """
    
    def _create_section_charts(self, section_key, section_data):
        if section_key == 'hf_methods':
            return self._create_sentiment_chart(section_data)
        elif section_key == 'indicators':
            return self._create_signals_chart(section_data)
        elif section_key == 'economic':
            return self._create_economic_chart(section_data)
        else:
            return ""
    
    def _create_sentiment_chart(self, hf_data):
        sentiment = hf_data.get('sentiment', {}).get('aggregated', {})
        
        if not sentiment:
            return ""
        
        pos = sentiment.get('positive_count', 0)
        neg = sentiment.get('negative_count', 0)
        neu = sentiment.get('neutral_count', 0)
        
        chart_id = f"chart-{abs(hash('sentiment')) % 10000}"
        
        return f"""
        <div class="card">
            <div class="card-header">
                <h6 class="mb-0">Sentiment Distribution</h6>
            </div>
            <div class="card-body">
                <canvas id="{chart_id}" style="max-height: 250px;"></canvas>
            </div>
        </div>
        
        <script>
        document.addEventListener('DOMContentLoaded', function() {{
            new Chart(document.getElementById('{chart_id}'), {{
                type: 'doughnut',
                data: {{
                    labels: ['Positive', 'Negative', 'Neutral'],
                    datasets: [{{
                        data: [{pos}, {neg}, {neu}],
                        backgroundColor: ['#28a745', '#dc3545', '#6c757d']
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            position: 'bottom'
                        }}
                    }}
                }}
            }});
        }});
        </script>
        """
    
    def _create_signals_chart(self, indicators_data):
        if not indicators_data:
            return ""
        
        buy = sum(1 for ind in indicators_data.values() if ind.get('overall_signal') == 'BUY')
        sell = sum(1 for ind in indicators_data.values() if ind.get('overall_signal') == 'SELL')
        neutral = len(indicators_data) - buy - sell
        
        chart_id = f"chart-{abs(hash('signals')) % 10000}"
        
        return f"""
        <div class="card">
            <div class="card-header">
                <h6 class="mb-0">Technical Signals</h6>
            </div>
            <div class="card-body">
                <canvas id="{chart_id}" style="max-height: 250px;"></canvas>
            </div>
        </div>
        
        <script>
        document.addEventListener('DOMContentLoaded', function() {{
            new Chart(document.getElementById('{chart_id}'), {{
                type: 'bar',
                data: {{
                    labels: ['BUY', 'SELL', 'NEUTRAL'],
                    datasets: [{{
                        label: 'Signals',
                        data: [{buy}, {sell}, {neutral}],
                        backgroundColor: ['#28a745', '#dc3545', '#ffc107']
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            ticks: {{
                                stepSize: 1
                            }}
                        }}
                    }},
                    plugins: {{
                        legend: {{
                            display: false
                        }}
                    }}
                }}
            }});
        }});
        </script>
        """
    
    def _create_economic_chart(self, econ_data):
        if not econ_data or 'indicator_statuses' not in econ_data:
            return ""
        
        statuses = list(econ_data['indicator_statuses'].values())
        strong = statuses.count('STRONG_GROWTH')
        moderate = statuses.count('MODERATE')
        weak = statuses.count('WEAK')
        recession = statuses.count('RECESSION_WARNING')
        
        chart_id = f"chart-{abs(hash('economic')) % 10000}"
        
        return f"""
        <div class="card">
            <div class="card-header">
                <h6 class="mb-0">Economic Health</h6>
            </div>
            <div class="card-body">
                <canvas id="{chart_id}" style="max-height: 250px;"></canvas>
            </div>
        </div>
        
        <script>
        document.addEventListener('DOMContentLoaded', function() {{
            new Chart(document.getElementById('{chart_id}'), {{
                type: 'pie',
                data: {{
                    labels: ['Strong', 'Moderate', 'Weak', 'Warning'],
                    datasets: [{{
                        data: [{strong}, {moderate}, {weak}, {recession}],
                        backgroundColor: ['#28a745', '#ffc107', '#fd7e14', '#dc3545']
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            position: 'bottom'
                        }}
                    }}
                }}
            }});
        }});
        </script>
        """
    
    def _generate_visualizations_section(self, viz_dir):
        viz_path = Path(viz_dir)
        images = list(viz_path.glob('*.png'))
        
        if not images:
            return ""
        
        images_html = []
        for img in images:
            with open(img, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode()
            
            img_name = img.stem.replace('_', ' ').title()
            
            images_html.append(f"""
                <div class="col-md-6 mb-4">
                    <div class="card">
                        <div class="card-header">
                            <h6 class="mb-0">{img_name}</h6>
                        </div>
                        <div class="card-body text-center">
                            <img src="data:image/png;base64,{img_data}" 
                                 class="img-fluid rounded" 
                                 alt="{img_name}">
                        </div>
                    </div>
                </div>
            """)
        
        return f"""
        <section id="visualizations" class="mb-5">
            <h2 class="mb-4">
                <i class="bi bi-image"></i> Visualizations
            </h2>
            <div class="row">
                {''.join(images_html)}
            </div>
        </section>
        """
    
    def _generate_raw_data_section(self, data):
        return f"""
        <section id="raw-data" class="mb-5">
            <h2 class="mb-4">
                <i class="bi bi-code-square"></i> Raw Data
            </h2>
            <button class="btn btn-outline-primary" type="button" 
                    data-bs-toggle="collapse" data-bs-target="#fullRawData">
                <i class="bi bi-eye"></i> Show/Hide Full JSON Data
            </button>
            <div class="collapse mt-3" id="fullRawData">
                <pre class="bg-light p-3 rounded" style="max-height: 600px; overflow-y: auto;"><code>{json.dumps(data, indent=2)}</code></pre>
            </div>
        </section>
        """
    
    def _get_html_header(self, title, date):
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - Comprehensive Analysis Report</title>
    
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    
    <style>
        body {{
            background-color: #f8f9fa;
            padding-top: 20px;
        }}
        .hero {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 60px 0;
            margin-bottom: 40px;
            border-radius: 10px;
        }}
        .card {{
            box-shadow: 0 0.125rem 0.25rem rgba(0,0,0,0.075);
            border: none;
            margin-bottom: 20px;
        }}
        .section-divider {{
            height: 2px;
            background: linear-gradient(to right, #667eea, #764ba2);
            margin: 40px 0;
        }}
        pre {{
            font-size: 0.85rem;
        }}
        .badge {{
            font-size: 0.9rem;
        }}
        h2 {{
            color: #495057;
            font-weight: 600;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        .ai-badge {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="hero text-center">
            <h1 class="display-4 mb-3">
                <i class="bi bi-graph-up-arrow"></i> Comprehensive Market Analysis
            </h1>
            <h2 class="mb-4">{title}</h2>
            <p class="lead mb-2">
                <i class="bi bi-calendar3"></i> Analysis Date: {date}
            </p>
            <p class="mb-0">
                <span class="ai-badge">
                    <i class="bi bi-robot"></i> AI-Powered Report by Qwen2.5-32B
                </span>
            </p>
            <p class="mt-2 mb-0">
                <small>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small>
            </p>
        </div>
        
        <div class="card mb-5">
            <div class="card-header bg-dark text-white">
                <h5 class="mb-0"><i class="bi bi-list-ul"></i> Table of Contents</h5>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-6">
                        <ul class="list-unstyled">
                            <li><a href="#executive-summary">Executive Summary</a></li>
                            <li><a href="#section-news">News Analysis</a></li>
                            <li><a href="#section-hf_methods">AI Analysis</a></li>
                            <li><a href="#section-indicators">Technical Indicators</a></li>
                            <li><a href="#section-cot">COT Positioning</a></li>
                        </ul>
                    </div>
                    <div class="col-md-6">
                        <ul class="list-unstyled">
                            <li><a href="#section-economic">Economic Indicators</a></li>
                            <li><a href="#section-structure">Market Structure</a></li>
                            <li><a href="#section-synthesis">Synthesis & Insights</a></li>
                            <li><a href="#visualizations">Visualizations</a></li>
                            <li><a href="#raw-data">Raw Data</a></li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
"""
    
    def _get_html_footer(self):
        return """
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    
    <script>
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            });
        });
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    import sys
    
    if not os.environ.get('GROQ_API_KEY'):
        print("ERROR: GROQ_API_KEY environment variable not set")
        print("\nTo use this tool:")
        print("1. Get a free API key from https://console.groq.com")
        print("2. Set environment variable: export GROQ_API_KEY='your-key-here'")
        print("3. Run this script again")
        sys.exit(1)
    
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        import glob
        json_files = glob.glob('**/comprehensive_*.json', recursive=True)
        
        if not json_files:
            print("ERROR: No comprehensive JSON files found")
            print("\nPlease run the comprehensive pipeline first:")
            print("  python comprehensive_pipeline.py")
            sys.exit(1)
        
        json_file = max(json_files, key=os.path.getctime)
        print(f"Using most recent file: {json_file}")
    
    generator = GroqAIReportGenerator()
    
    try:
        html_file = generator.generate_comprehensive_report(json_file)
        
        print(f"\n✅ SUCCESS!")
        print(f"📄 HTML Report: {html_file}")
        print(f"\nOpen in browser:")
        print(f"  file://{os.path.abspath(html_file)}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
