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
        """Generate complete HTML report with Bootstrap 5 and Chart.js"""
        
        date = metadata.get('date', 'N/A')
        event_name = metadata.get('event_name', 'Market Analysis')
        timestamp = metadata.get('timestamp', datetime.now().isoformat())
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{event_name} - AI Analysis Report</title>
    
    <!-- Bootstrap 5 CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css" rel="stylesheet">
    
    <style>
        body {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .main-container {{
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
        }}
        .section {{
            margin: 20px 0;
            border-left: 4px solid #667eea;
            background: #f8f9fa;
        }}
        .ai-badge {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }}
        .chart-container {{
            position: relative;
            height: 300px;
            margin: 20px 0;
        }}
        .raw-data {{
            background: #1e1e1e;
            color: #d4d4d4;
            border-radius: 8px;
            padding: 15px;
            font-family: 'Courier New', monospace;
            font-size: 0.85em;
            max-height: 400px;
            overflow-y: auto;
        }}
        .image-container {{
            margin: 20px 0;
        }}
        .image-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <div class="container-fluid px-3 px-lg-5">
        <div class="main-container">
            <!-- Header -->
            <div class="header">
                <div class="row align-items-center">
                    <div class="col-lg-8">
                        <h1 class="display-4 mb-3">
                            <i class="bi bi-graph-up"></i> {event_name}
                        </h1>
                        <p class="lead mb-0">AI-Powered Financial Analysis Report</p>
                    </div>
                    <div class="col-lg-4 text-lg-end mt-3 mt-lg-0">
                        <div class="badge bg-light text-dark p-3 mb-2">
                            <i class="bi bi-calendar3"></i> {date}
                        </div><br>
                        <div class="badge bg-light text-dark p-3">
                            <i class="bi bi-robot"></i> BIST-Financial-Qwen-7B
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Navigation -->
            <nav class="navbar navbar-expand-lg navbar-light bg-light sticky-top">
                <div class="container-fluid">
                    <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                        <span class="navbar-toggler-icon"></span>
                    </button>
                    <div class="collapse navbar-collapse" id="navbarNav">
                        <ul class="navbar-nav">
                            <li class="nav-item"><a class="nav-link" href="#executive">Executive Summary</a></li>
                            <li class="nav-item"><a class="nav-link" href="#news">News</a></li>
                            <li class="nav-item"><a class="nav-link" href="#indicators">Indicators</a></li>
                            <li class="nav-item"><a class="nav-link" href="#cot">COT</a></li>
                            <li class="nav-item"><a class="nav-link" href="#economic">Economic</a></li>
                            <li class="nav-item"><a class="nav-link" href="#correlations">Correlations</a></li>
                            <li class="nav-item"><a class="nav-link" href="#structure">Structure</a></li>
                            <li class="nav-item"><a class="nav-link" href="#synthesis">Synthesis</a></li>
                        </ul>
                    </div>
                </div>
            </nav>
            
            <!-- Content -->
            <div class="p-4">
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
            </div>
            
            <!-- Footer -->
            <footer class="bg-light text-center p-4 mt-4">
                <p class="mb-0">
                    <i class="bi bi-robot"></i> Powered by BIST-Financial-Qwen-7B<br>
                    <small class="text-muted">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small>
                </p>
            </footer>
        </div>
    </div>
    
    <!-- Bootstrap 5 JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    
    <script>
        // Smooth scroll for navigation
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {{
            anchor.addEventListener('click', function (e) {{
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {{
                    target.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
                }}
            }});
        }});
        
        // Initialize all charts after page load
        window.addEventListener('load', function() {{
            console.log('Charts initialized');
        }});
    </script>
</body>
</html>
"""
        
        return html
    
    def _render_section(self, section_name: str, section_data: Dict[str, Any], is_summary: bool = False) -> str:
        """Render a section with AI insights, charts, images, and raw data"""
        
        if 'data' in section_data:
            actual_data = section_data['data']
        else:
            actual_data = section_data
        
        icons = {
            'executive_summary': 'stars',
            'news': 'newspaper',
            'indicators': 'graph-up-arrow',
            'cot': 'pie-chart',
            'economic': 'currency-dollar',
            'correlations': 'diagram-3',
            'structure': 'bricks',
            'seasonality': 'calendar3',
            'volume': 'bar-chart',
            'hf_methods': 'robot',
            'synthesis': 'lightbulb'
        }
        
        icon = icons.get(section_name, 'file-text')
        title = section_name.replace('_', ' ').title()
        section_id = section_name.replace('_', '-')
        
        html = f"""
        <div id="{section_id}" class="section card mb-4">
            <div class="card-header">
                <h3>
                    <i class="bi bi-{icon}"></i> {title}
                    <span class="badge ai-badge text-white float-end">AI Powered</span>
                </h3>
            </div>
            <div class="card-body">
                
                <!-- AI Insights -->
                <div class="alert alert-primary">
                    <h5><i class="bi bi-robot"></i> AI Insights</h5>
"""
        
        # Generate AI insights
        print(f"  🤖 Generating insights for {section_name}...")
        insights = self._generate_insights(section_name, actual_data)
        html += f'<div>{insights}</div>'
        
        html += """
                </div>
"""
        
        # Add metrics
        if not is_summary:
            html += self._render_metrics(section_name, actual_data)
        
        # Add visualizations
        html += self._render_visualizations(section_name, actual_data)
        
        # Add images if available
        html += self._render_images(section_name)
        
        # Add raw data accordion
        html += f"""
                <!-- Raw Data -->
                <div class="accordion mt-4" id="accordion-{section_id}">
                    <div class="accordion-item">
                        <h2 class="accordion-header">
                            <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapse-{section_id}">
                                <i class="bi bi-code-square"></i> View Raw Data
                            </button>
                        </h2>
                        <div id="collapse-{section_id}" class="accordion-collapse collapse" data-bs-parent="#accordion-{section_id}">
                            <div class="accordion-body">
                                <pre class="raw-data"><code>{json.dumps(actual_data, indent=2)}</code></pre>
                            </div>
                        </div>
                    </div>
                </div>
                
            </div>
        </div>
"""
        
        return html
    
    def _render_metrics(self, section_name: str, data: Dict[str, Any]) -> str:
        """Render key metrics with Bootstrap cards"""
        
        html = '<div class="row g-3 my-3">'
        
        if section_name == 'news':
            html += f"""
                <div class="col-md-3">
                    <div class="card text-center border-primary">
                        <div class="card-body">
                            <h6 class="card-subtitle text-muted mb-2">Articles</h6>
                            <h3 class="card-title text-primary">{data.get('article_count', 0)}</h3>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card text-center border-info">
                        <div class="card-body">
                            <h6 class="card-subtitle text-muted mb-2">Sources</h6>
                            <h3 class="card-title text-info">{len(data.get('sources', []))}</h3>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card text-center border-warning">
                        <div class="card-body">
                            <h6 class="card-subtitle text-muted mb-2">Sentiment</h6>
                            <h3 class="card-title text-warning">{data.get('sentiment', 'N/A')}</h3>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card text-center border-success">
                        <div class="card-body">
                            <h6 class="card-subtitle text-muted mb-2">Themes</h6>
                            <h3 class="card-title text-success">{len(data.get('key_themes', []))}</h3>
                        </div>
                    </div>
                </div>
            """
        
        elif section_name == 'indicators':
            buy = data.get('buy_signals', 0)
            sell = data.get('sell_signals', 0)
            total = buy + sell
            buy_pct = (buy / total * 100) if total > 0 else 0
            
            html += f"""
                <div class="col-md-4">
                    <div class="card text-center border-success">
                        <div class="card-body">
                            <h6 class="card-subtitle text-muted mb-2">Overall Bias</h6>
                            <h3 class="card-title text-success">{data.get('overall_bias', 'N/A')}</h3>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card text-center border-primary">
                        <div class="card-body">
                            <h6 class="card-subtitle text-muted mb-2">Buy Signals</h6>
                            <h3 class="card-title text-primary">{buy} <small>({buy_pct:.0f}%)</small></h3>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card text-center border-danger">
                        <div class="card-body">
                            <h6 class="card-subtitle text-muted mb-2">Sell Signals</h6>
                            <h3 class="card-title text-danger">{sell} <small>({100-buy_pct:.0f}%)</small></h3>
                        </div>
                    </div>
                </div>
            """
        
        elif section_name == 'cot':
            html += f"""
                <div class="col-md-4">
                    <div class="card text-center border-primary">
                        <div class="card-body">
                            <h6 class="card-subtitle text-muted mb-2">Net Positioning</h6>
                            <h3 class="card-title text-primary">{data.get('net_positioning', 'N/A')}</h3>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card text-center border-info">
                        <div class="card-body">
                            <h6 class="card-subtitle text-muted mb-2">Change</h6>
                            <h3 class="card-title text-info">{data.get('positioning_change', 'N/A')}</h3>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card text-center border-success">
                        <div class="card-body">
                            <h6 class="card-subtitle text-muted mb-2">Sentiment</h6>
                            <h3 class="card-title text-success">{data.get('institutional_sentiment', 'N/A')}</h3>
                        </div>
                    </div>
                </div>
            """
        
        elif section_name == 'economic':
            html += f"""
                <div class="col-md-4">
                    <div class="card text-center border-warning">
                        <div class="card-body">
                            <h6 class="card-subtitle text-muted mb-2">Overall Status</h6>
                            <h3 class="card-title text-warning">{data.get('overall_status', 'N/A')}</h3>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card text-center border-danger">
                        <div class="card-body">
                            <h6 class="card-subtitle text-muted mb-2">Inflation</h6>
                            <h3 class="card-title text-danger">{data.get('inflation_trend', 'N/A')}</h3>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card text-center border-success">
                        <div class="card-body">
                            <h6 class="card-subtitle text-muted mb-2">Growth</h6>
                            <h3 class="card-title text-success">{data.get('growth_outlook', 'N/A')}</h3>
                        </div>
                    </div>
                </div>
            """
        
        elif section_name == 'synthesis':
            confidence = data.get('confidence', 0.5)
            html += f"""
                <div class="col-md-6">
                    <div class="card text-center border-primary">
                        <div class="card-body">
                            <h6 class="card-subtitle text-muted mb-2">Overall Outlook</h6>
                            <h3 class="card-title text-primary">{data.get('overall_outlook', 'N/A')}</h3>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card text-center border-success">
                        <div class="card-body">
                            <h6 class="card-subtitle text-muted mb-2">Confidence</h6>
                            <h3 class="card-title text-success">{confidence:.0%}</h3>
                            <div class="progress mt-2">
                                <div class="progress-bar bg-success" style="width: {confidence*100}%"></div>
                            </div>
                        </div>
                    </div>
                </div>
            """
        
        html += '</div>'
        return html


    def _render_visualizations(self, section_name: str, data: Dict[str, Any]) -> str:
        """Render Chart.js visualizations"""
        
        html = ""
        chart_id = f"chart-{section_name}"
        
        if section_name == 'indicators':
            buy = data.get('buy_signals', 0)
            sell = data.get('sell_signals', 0)
            
            html += f"""
                <div class="chart-container">
                    <canvas id="{chart_id}"></canvas>
                </div>
                <script>
                    new Chart(document.getElementById('{chart_id}'), {{
                        type: 'doughnut',
                        data: {{
                            labels: ['Buy Signals', 'Sell Signals'],
                            datasets: [{{
                                data: [{buy}, {sell}],
                                backgroundColor: ['#28a745', '#dc3545']
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {{
                                title: {{
                                    display: true,
                                    text: 'Signal Distribution'
                                }}
                            }}
                        }}
                    }});
                </script>
            """
        
        elif section_name == 'news':
            themes = data.get('key_themes', [])[:5]
            if themes:
                values = [len(t) * 10 for t in themes]  # Mock values
                html += f"""
                    <div class="chart-container">
                        <canvas id="{chart_id}"></canvas>
                    </div>
                    <script>
                        new Chart(document.getElementById('{chart_id}'), {{
                            type: 'bar',
                            data: {{
                                labels: {json.dumps(themes)},
                                datasets: [{{
                                    label: 'Theme Frequency',
                                    data: {json.dumps(values)},
                                    backgroundColor: '#667eea'
                                }}]
                            }},
                            options: {{
                                responsive: true,
                                maintainAspectRatio: false,
                                plugins: {{
                                    title: {{
                                        display: true,
                                        text: 'Top News Themes'
                                    }}
                                }}
                            }}
                        }});
                    </script>
                """
        
        elif section_name == 'structure':
            support = data.get('support_levels', [])
            resistance = data.get('resistance_levels', [])
            
            if support or resistance:
                html += f"""
                    <div class="chart-container">
                        <canvas id="{chart_id}"></canvas>
                    </div>
                    <script>
                        new Chart(document.getElementById('{chart_id}'), {{
                            type: 'line',
                            data: {{
                                labels: ['S3', 'S2', 'S1', 'Current', 'R1', 'R2', 'R3'],
                                datasets: [{{
                                    label: 'Support/Resistance Levels',
                                    data: [{','.join(map(str, support[:3] + [1975] + resistance[:3]))}],
                                    borderColor: '#667eea',
                                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                                    fill: true
                                }}]
                            }},
                            options: {{
                                responsive: true,
                                maintainAspectRatio: false,
                                plugins: {{
                                    title: {{
                                        display: true,
                                        text: 'Key Price Levels'
                                    }}
                                }}
                            }}
                        }});
                    </script>
                """
        
        elif section_name == 'synthesis':
            factors = data.get('key_factors', [])[:4]
            if factors:
                values = [85, 70, 65, 55]  # Mock importance scores
                html += f"""
                    <div class="chart-container">
                        <canvas id="{chart_id}"></canvas>
                    </div>
                    <script>
                        new Chart(document.getElementById('{chart_id}'), {{
                            type: 'radar',
                            data: {{
                                labels: {json.dumps([f[:30] + '...' if len(f) > 30 else f for f in factors])},
                                datasets: [{{
                                    label: 'Factor Importance',
                                    data: {json.dumps(values)},
                                    backgroundColor: 'rgba(102, 126, 234, 0.2)',
                                    borderColor: '#667eea'
                                }}]
                            }},
                            options: {{
                                responsive: true,
                                maintainAspectRatio: false,
                                scales: {{
                                    r: {{
                                        beginAtZero: true,
                                        max: 100
                                    }}
                                }}
                            }}
                        }});
                    </script>
                """
        
        return html
    
    def _render_images(self, section_name: str) -> str:
        """Render images from visualizations directory"""
        
        html = ""
        
        # Look for images in visualizations directory
        viz_dir = Path('visualizations') if Path('visualizations').exists() else None
        if not viz_dir:
            # Try relative to json file
            viz_dir = Path('pipeline_output/visualizations') if Path('pipeline_output/visualizations').exists() else None
        
        if viz_dir and viz_dir.exists():
            # Look for section-specific images
            patterns = [
                f"{section_name}_*.png",
                f"{section_name}_*.jpg",
                f"*{section_name}*.png"
            ]
            
            images = []
            for pattern in patterns:
                images.extend(list(viz_dir.glob(pattern)))
            
            if images:
                html += '<div class="row g-3 my-3">'
                for img in images[:3]:  # Max 3 images per section
                    # Convert to base64 for embedding
                    try:
                        import base64
                        with open(img, 'rb') as f:
                            img_data = base64.b64encode(f.read()).decode()
                        
                        html += f"""
                            <div class="col-md-4">
                                <div class="image-container">
                                    <img src="data:image/png;base64,{img_data}" alt="{img.stem}" class="img-fluid">
                                    <p class="text-center mt-2 text-muted small">{img.stem}</p>
                                </div>
                            </div>
                        """
                    except Exception as e:
                        print(f"  ⚠️  Could not embed image {img}: {e}")
                
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
