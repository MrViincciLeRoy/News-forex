"""
Enhanced Local LLM Report Generator
Comprehensive HTML reports with detailed data, images, and analytics
"""

import json
import os
import base64
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import sys

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False


class EnhancedLLMReportGenerator:
    """Generate detailed HTML reports with comprehensive data visualization"""
    
    def __init__(self, include_raw_data=True, embed_images=True):
        self.include_raw_data = include_raw_data
        self.embed_images = embed_images
        self.model = None
        
        if LLAMA_CPP_AVAILABLE:
            try:
                print("Loading BIST-Financial-Qwen-7B...")
                self.model = Llama.from_pretrained(
                    repo_id="bist-quant/BIST-Financial-Qwen-7B",
                    filename="gguf/qwen-kap-final-Q4_K_M.gguf",
                    n_ctx=2048,
                    n_threads=4,
                    n_gpu_layers=0
                )
                print("✓ Model loaded")
            except Exception as e:
                print(f"⚠️  Model not available: {e}")
                self.model = None
    
    def generate_report(self, json_file: str, output_file: Optional[str] = None) -> str:
        """Generate comprehensive HTML report"""
        
        print(f"\n{'='*80}")
        print("ENHANCED AI-POWERED REPORT GENERATION")
        print(f"{'='*80}")
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        metadata = data.get('metadata', {})
        sections = data.get('sections', {})
        
        # Load section JSONs
        sections_dir = Path(json_file).parent / 'sections'
        if sections_dir.exists():
            sections = self._load_section_jsons(sections_dir, data)
        
        # Find visualization images
        viz_dir = self._find_viz_directory(json_file)
        
        if output_file is None:
            output_file = f"{Path(json_file).stem}_enhanced_report.html"
        
        html_content = self._generate_html(metadata, sections, viz_dir)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\n✓ Report: {output_file}")
        print(f"Size: {os.path.getsize(output_file) / 1024:.1f} KB")
        print(f"{'='*80}\n")
        
        return output_file
    
    def _load_section_jsons(self, sections_dir: Path, main_data: Dict) -> Dict:
        """Load individual section files"""
        sections = main_data.get('sections', {})
        
        for section_file in sections_dir.glob('*.json'):
            try:
                with open(section_file, 'r') as f:
                    section_data = json.load(f)
                    section_type = section_data.get('section_type', section_file.stem.split('_')[0])
                    sections[section_type] = section_data.get('data', section_data)
                    print(f"  ✓ Loaded {section_type}")
            except Exception as e:
                print(f"  ⚠️  {section_file.name}: {e}")
        
        return sections
    
    def _find_viz_directory(self, json_file: str) -> Optional[Path]:
        """Find visualization directory"""
        base_dir = Path(json_file).parent
        
        candidates = [
            base_dir / 'visualizations',
            base_dir / 'pipeline_output' / 'visualizations',
            base_dir / 'test_pipeline_output' / 'visualizations',
            Path('visualizations'),
            Path('pipeline_output/visualizations')
        ]
        
        for candidate in candidates:
            if candidate.exists() and list(candidate.glob('*.png')):
                print(f"  ✓ Found visualizations: {candidate}")
                return candidate
        
        return None
    
    def _embed_image(self, image_path: Path) -> str:
        """Convert image to base64 for embedding"""
        try:
            with open(image_path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode()
            return f"data:image/png;base64,{img_data}"
        except Exception as e:
            print(f"  ⚠️  Image embed failed: {e}")
            return ""
    
    def _generate_insights(self, section_name: str, data: Dict) -> str:
        """Generate AI insights"""
        
        if self.model:
            try:
                data_str = json.dumps(data, indent=2)[:800]
                prompt = f"""Analyze {section_name} data and provide 3-5 key insights:

{data_str}

Insights (bullet points):"""
                
                response = self.model.create_chat_completion(
                    messages=[
                        {"role": "system", "content": "Financial analyst providing data-driven insights."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=250,
                    temperature=0.7
                )
                
                insights = response['choices'][0]['message']['content'].strip()
                return insights if insights else self._fallback_insights(section_name, data)
            except Exception as e:
                print(f"  ⚠️  AI generation failed: {e}")
        
        return self._fallback_insights(section_name, data)
    
    def _fallback_insights(self, section: str, data: Dict) -> str:
        """Generate insights without LLM"""
        insights = []
        
        if section == 'executive_summary':
            sentiment = data.get('sentiment', {})
            findings = data.get('key_findings', [])
            insights.append(f"• Overall: {sentiment.get('overall', 'N/A')} (Confidence: {sentiment.get('confidence', 0):.0%})")
            if findings:
                insights.extend([f"• {f}" for f in findings[:3]])
        
        elif section == 'news':
            count = data.get('article_count', 0)
            themes = data.get('key_themes', [])
            sentiment = data.get('sentiment', 'N/A')
            sources = data.get('sources', [])
            insights.append(f"• {count} articles analyzed from {len(sources)} sources")
            insights.append(f"• Dominant sentiment: {sentiment}")
            if themes:
                insights.append(f"• Top themes: {', '.join(themes[:3])}")
        
        elif section == 'indicators':
            bias = data.get('overall_bias', 'N/A')
            buy = data.get('buy_signals', 0)
            sell = data.get('sell_signals', 0)
            total = buy + sell
            insights.append(f"• Technical bias: {bias}")
            insights.append(f"• {buy}/{total} signals bullish ({buy/total*100 if total else 0:.0f}%)")
            if buy > sell * 2:
                insights.append("• Strong bullish momentum confirmed")
        
        elif section == 'cot':
            pos = data.get('net_positioning', 'N/A')
            change = data.get('positioning_change', 'N/A')
            sentiment = data.get('institutional_sentiment', 'N/A')
            insights.append(f"• Net positioning: {pos} ({change})")
            insights.append(f"• Institutional sentiment: {sentiment}")
            if 'commercials_long' in data:
                comm_long = data.get('commercials_long', 0)
                comm_short = data.get('commercials_short', 0)
                insights.append(f"• Commercials: {comm_long:,} long vs {comm_short:,} short")
        
        elif section == 'economic':
            status = data.get('overall_status', 'N/A')
            inflation = data.get('inflation_trend', 'N/A')
            growth = data.get('growth_outlook', 'N/A')
            insights.append(f"• Economic status: {status}")
            insights.append(f"• Inflation: {inflation}")
            insights.append(f"• Growth: {growth}")
            if 'unemployment_rate' in data:
                insights.append(f"• Unemployment: {data['unemployment_rate']}%")
        
        elif section == 'correlations':
            rels = data.get('key_relationships', {})
            strong = [(k, v) for k, v in rels.items() if v.get('strength') == 'STRONG']
            if strong:
                insights.append(f"• {len(strong)} strong correlations identified")
                for pair, details in strong[:2]:
                    insights.append(f"• {pair}: {details.get('direction', 'N/A')} ({details.get('correlation', 'N/A')})")
        
        elif section == 'structure':
            trend = data.get('trend', 'N/A')
            support = data.get('support_levels', [])
            resistance = data.get('resistance_levels', [])
            insights.append(f"• Market structure: {trend}")
            if support:
                insights.append(f"• Nearest support: {support[0]}")
            if resistance:
                insights.append(f"• Nearest resistance: {resistance[0]}")
        
        elif section == 'seasonality':
            bias = data.get('seasonal_bias', 'N/A')
            perf = data.get('historical_performance', 'N/A')
            insights.append(f"• Seasonal bias: {bias}")
            insights.append(f"• Historical avg: {perf}")
            if 'win_rate' in data:
                insights.append(f"• Win rate: {data['win_rate']}%")
        
        elif section == 'volume':
            trend = data.get('volume_trend', 'N/A')
            profile = data.get('volume_profile', 'N/A')
            insights.append(f"• Volume trend: {trend}")
            insights.append(f"• Profile: {profile}")
            if 'volume_change' in data:
                insights.append(f"• Change: {data['volume_change']}")
        
        elif section == 'hf_methods':
            sentiment = data.get('sentiment_score', 0.5)
            forecast = data.get('forecast_direction', 'N/A')
            confidence = data.get('forecast_confidence', 0.5)
            insights.append(f"• AI sentiment: {sentiment:.2f}")
            insights.append(f"• Forecast: {forecast} (confidence: {confidence:.0%})")
            insights.append(f"• Anomalies detected: {data.get('anomalies_detected', 0)}")
        
        elif section == 'synthesis':
            outlook = data.get('overall_outlook', 'N/A')
            conf = data.get('confidence', 0.5)
            factors = data.get('key_factors', [])
            insights.append(f"• Outlook: {outlook} (confidence: {conf:.0%})")
            if factors:
                insights.extend([f"• {f}" for f in factors[:3]])
        
        return '\n'.join(insights) if insights else f"• {section.replace('_', ' ').title()} analysis complete"
    
    def _generate_html(self, metadata: Dict, sections: Dict, viz_dir: Optional[Path]) -> str:
        """Generate comprehensive HTML"""
        
        date = metadata.get('date', 'N/A')
        event = metadata.get('event_name', 'Analysis')
        symbols = metadata.get('symbols', [])
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{event} - Comprehensive Analysis</title>
    
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css" rel="stylesheet">
    
    <style>
        :root {{
            --primary: #667eea;
            --secondary: #764ba2;
            --success: #28a745;
            --danger: #dc3545;
            --warning: #ffc107;
            --info: #17a2b8;
        }}
        
        body {{
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            min-height: 100vh;
            padding: 20px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }}
        
        .main-container {{
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .header {{
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: white;
            padding: 50px 40px;
            position: relative;
        }}
        
        .header::after {{
            content: '';
            position: absolute;
            bottom: -30px;
            left: 0;
            right: 0;
            height: 30px;
            background: white;
            clip-path: ellipse(70% 100% at 50% 0%);
        }}
        
        .header h1 {{
            font-size: 3rem;
            font-weight: 700;
            margin-bottom: 15px;
        }}
        
        .header .lead {{
            font-size: 1.3rem;
            opacity: 0.95;
        }}
        
        .header .badges {{
            margin-top: 25px;
        }}
        
        .header .badge {{
            padding: 10px 20px;
            font-size: 1rem;
            margin-right: 10px;
            background: rgba(255,255,255,0.2);
            backdrop-filter: blur(10px);
        }}
        
        .navbar {{
            background: #f8f9fa;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            position: sticky;
            top: 0;
            z-index: 1000;
        }}
        
        .section {{
            margin: 30px 0;
            border-left: 5px solid var(--primary);
            background: #f8f9fa;
            border-radius: 10px;
            overflow: hidden;
        }}
        
        .section-header {{
            background: linear-gradient(90deg, rgba(102,126,234,0.1) 0%, transparent 100%);
            padding: 25px 30px;
            border-bottom: 2px solid #e9ecef;
        }}
        
        .section-title {{
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--primary);
            margin: 0;
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        
        .section-body {{
            padding: 30px;
        }}
        
        .insights-box {{
            background: linear-gradient(135deg, #e7f3ff 0%, #f0f7ff 100%);
            border-left: 5px solid var(--info);
            padding: 25px;
            margin-bottom: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}
        
        .insights-title {{
            font-size: 1.2rem;
            font-weight: 700;
            color: var(--info);
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .insights-content {{
            color: #2c5aa0;
            white-space: pre-line;
            line-height: 1.8;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 25px 0;
        }}
        
        .metric-card {{
            background: white;
            border: 2px solid #e9ecef;
            border-radius: 12px;
            padding: 25px;
            text-align: center;
            transition: all 0.3s;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        }}
        
        .metric-label {{
            font-size: 0.85rem;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
            font-weight: 600;
        }}
        
        .metric-value {{
            font-size: 2.5rem;
            font-weight: 700;
            color: #333;
            line-height: 1;
        }}
        
        .metric-value.positive {{ color: var(--success); }}
        .metric-value.negative {{ color: var(--danger); }}
        .metric-value.neutral {{ color: var(--warning); }}
        
        .metric-subtext {{
            font-size: 0.9rem;
            color: #6c757d;
            margin-top: 8px;
        }}
        
        .detail-table {{
            width: 100%;
            margin: 25px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}
        
        .detail-table th {{
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: white;
            padding: 15px;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85rem;
            letter-spacing: 0.5px;
        }}
        
        .detail-table td {{
            padding: 15px;
            border-bottom: 1px solid #e9ecef;
        }}
        
        .detail-table tr:last-child td {{
            border-bottom: none;
        }}
        
        .detail-table tr:hover {{
            background: #f8f9fa;
        }}
        
        .data-grid {{
            display: grid;
            grid-template-columns: 200px 1fr;
            gap: 15px;
            margin: 20px 0;
            background: white;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}
        
        .data-label {{
            font-weight: 700;
            color: #495057;
            padding: 8px 0;
        }}
        
        .data-value {{
            color: #333;
            padding: 8px 0;
        }}
        
        .image-gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 25px;
            margin: 25px 0;
        }}
        
        .image-card {{
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }}
        
        .image-card:hover {{
            transform: scale(1.02);
        }}
        
        .image-card img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        
        .image-caption {{
            padding: 15px;
            background: #f8f9fa;
            text-align: center;
            font-weight: 600;
            color: #495057;
        }}
        
        .raw-data {{
            background: #1e1e1e;
            color: #d4d4d4;
            border-radius: 8px;
            padding: 20px;
            font-family: 'Courier New', monospace;
            font-size: 0.85rem;
            max-height: 500px;
            overflow-y: auto;
            line-height: 1.6;
            box-shadow: inset 0 2px 8px rgba(0,0,0,0.5);
        }}
        
        .raw-data .key {{
            color: #9cdcfe;
        }}
        
        .raw-data .string {{
            color: #ce9178;
        }}
        
        .raw-data .number {{
            color: #b5cea8;
        }}
        
        .raw-data .boolean {{
            color: #569cd6;
        }}
        
        .badge-custom {{
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.9rem;
        }}
        
        .badge-bullish {{
            background: #d4edda;
            color: #155724;
        }}
        
        .badge-bearish {{
            background: #f8d7da;
            color: #721c24;
        }}
        
        .badge-neutral {{
            background: #fff3cd;
            color: #856404;
        }}
        
        .list-detailed {{
            list-style: none;
            padding: 0;
        }}
        
        .list-detailed li {{
            padding: 12px 20px;
            margin: 8px 0;
            background: white;
            border-left: 4px solid var(--primary);
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        
        .list-detailed li:before {{
            content: '▸';
            color: var(--primary);
            font-weight: bold;
            margin-right: 10px;
        }}
        
        .chart-container {{
            position: relative;
            height: 400px;
            margin: 25px 0;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}
        
        .footer {{
            background: #f8f9fa;
            padding: 40px;
            text-align: center;
            border-top: 3px solid var(--primary);
            margin-top: 50px;
        }}
        
        .footer .copyright {{
            color: #6c757d;
            margin-top: 15px;
            font-size: 0.9rem;
        }}
        
        @media print {{
            .navbar, .no-print {{
                display: none;
            }}
            .section {{
                page-break-inside: avoid;
            }}
            /* Keep accordions expanded in PDF */
            .accordion-collapse {{
                display: block !important;
                height: auto !important;
            }}
            .accordion-button::after {{
                display: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="main-container">
        <!-- Header -->
        <div class="header">
            <div class="container-fluid">
                <div class="row align-items-center">
                    <div class="col-lg-8">
                        <h1><i class="bi bi-graph-up-arrow"></i> {event}</h1>
                        <p class="lead">Comprehensive Financial Analysis Report</p>
                    </div>
                    <div class="col-lg-4 text-lg-end">
                        <div class="badges">
                            <span class="badge"><i class="bi bi-calendar3"></i> {date}</span><br>
                            <span class="badge"><i class="bi bi-robot"></i> AI-Powered</span>
                        </div>
                    </div>
                </div>
                {f'<div class="mt-3"><small>Symbols: {", ".join(symbols)}</small></div>' if symbols else ''}
            </div>
        </div>
        
        <!-- Navigation -->
        <nav class="navbar navbar-expand-lg navbar-light">
            <div class="container-fluid">
                <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navContent">
                    <span class="navbar-toggler-icon"></span>
                </button>
                <div class="collapse navbar-collapse" id="navContent">
                    <ul class="navbar-nav">
                        <li class="nav-item"><a class="nav-link" href="#executive">Executive</a></li>
                        <li class="nav-item"><a class="nav-link" href="#news">News</a></li>
                        <li class="nav-item"><a class="nav-link" href="#indicators">Indicators</a></li>
                        <li class="nav-item"><a class="nav-link" href="#cot">COT</a></li>
                        <li class="nav-item"><a class="nav-link" href="#economic">Economic</a></li>
                        <li class="nav-item"><a class="nav-link" href="#correlations">Correlations</a></li>
                        <li class="nav-item"><a class="nav-link" href="#structure">Structure</a></li>
                        <li class="nav-item"><a class="nav-link" href="#seasonality">Seasonality</a></li>
                        <li class="nav-item"><a class="nav-link" href="#volume">Volume</a></li>
                        <li class="nav-item"><a class="nav-link" href="#hf">AI Methods</a></li>
                        <li class="nav-item"><a class="nav-link" href="#synthesis">Synthesis</a></li>
                    </ul>
                </div>
            </div>
        </nav>
        
        <!-- Content -->
        <div class="p-4">
"""
        
        # Render sections
        section_configs = [
            ('executive_summary', 'Executive Summary', 'stars', 'executive'),
            ('news', 'News Analysis', 'newspaper', 'news'),
            ('indicators', 'Technical Indicators', 'graph-up-arrow', 'indicators'),
            ('cot', 'COT Report', 'pie-chart-fill', 'cot'),
            ('economic', 'Economic Indicators', 'currency-dollar', 'economic'),
            ('correlations', 'Correlations', 'diagram-3-fill', 'correlations'),
            ('structure', 'Market Structure', 'bricks', 'structure'),
            ('seasonality', 'Seasonality', 'calendar3', 'seasonality'),
            ('volume', 'Volume Analysis', 'bar-chart-fill', 'volume'),
            ('hf_methods', 'AI Methods', 'robot', 'hf'),
            ('synthesis', 'Synthesis', 'lightbulb-fill', 'synthesis')
        ]
        
        for section_key, title, icon, anchor in section_configs:
            if section_key in sections:
                html += self._render_section(
                    section_key, title, icon, anchor,
                    sections[section_key], viz_dir
                )
        
        html += f"""
        </div>
        
        <!-- Footer -->
        <div class="footer">
            <h5><i class="bi bi-robot"></i> Powered by BIST-Financial-Qwen-7B</h5>
            <p class="mt-3">Advanced Financial Analysis Pipeline v2.0</p>
            <p class="copyright">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    
    <script>
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {{
            anchor.addEventListener('click', function (e) {{
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {{
                    target.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
                }}
            }});
        }});
        
        // Syntax highlighting for JSON
        document.querySelectorAll('.raw-data code').forEach(block => {{
            const json = block.textContent;
            block.innerHTML = json
                .replace(/"([^"]+)":/g, '<span class="key">"$1"</span>:')
                .replace(/: "([^"]+)"/g, ': <span class="string">"$1"</span>')
                .replace(/: (\\d+\\.?\\d*)/g, ': <span class="number">$1</span>')
                .replace(/: (true|false)/g, ': <span class="boolean">$1</span>');
        }});
    </script>
</body>
</html>
"""
        
        return html
    
    def _render_section(self, key: str, title: str, icon: str, anchor: str, 
                       data: Dict, viz_dir: Optional[Path]) -> str:
        """Render comprehensive section"""
        
        actual_data = data.get('data', data)
        
        print(f"  📊 Rendering {key}...")
        insights = self._generate_insights(key, actual_data)
        
        html = f"""
        <div id="{anchor}" class="section">
            <div class="section-header">
                <h2 class="section-title">
                    <i class="bi bi-{icon}"></i>
                    <span>{title}</span>
                    <span class="badge bg-primary ms-auto">AI-Powered</span>
                </h2>
            </div>
            <div class="section-body">
                
                <!-- AI Insights -->
                <div class="insights-box">
                    <div class="insights-title">
                        <i class="bi bi-robot"></i>
                        <span>AI Insights</span>
                    </div>
                    <div class="insights-content">{insights}</div>
                </div>
"""
        
        # Section-specific content
        html += self._render_section_details(key, actual_data)
        
        # Images
        if viz_dir and self.embed_images:
            html += self._render_section_images(key, viz_dir)
        
        # Raw data
        if self.include_raw_data:
            html += f"""
                <div class="accordion mt-4">
                    <div class="accordion-item">
                        <h2 class="accordion-header">
                            <button class="accordion-button" type="button" 
                                    data-bs-toggle="collapse" data-bs-target="#raw-{key}">
                                <i class="bi bi-code-square"></i>&nbsp; Raw Data
                            </button>
                        </h2>
                        <div id="raw-{key}" class="accordion-collapse collapse show">
                            <div class="accordion-body">
                                <pre class="raw-data"><code>{json.dumps(actual_data, indent=2)}</code></pre>
                            </div>
                        </div>
                    </div>
                </div>
"""
        
        html += """
            </div>
        </div>
"""
        
        return html
    
    def _render_section_details(self, key: str, data: Dict) -> str:
        """Render detailed section content"""
        
        if key == 'executive_summary':
            return self._render_executive_details(data)
        elif key == 'news':
            return self._render_news_details(data)
        elif key == 'indicators':
            return self._render_indicators_details(data)
        elif key == 'cot':
            return self._render_cot_details(data)
        elif key == 'economic':
            return self._render_economic_details(data)
        elif key == 'correlations':
            return self._render_correlations_details(data)
        elif key == 'structure':
            return self._render_structure_details(data)
        elif key == 'seasonality':
            return self._render_seasonality_details(data)
        elif key == 'volume':
            return self._render_volume_details(data)
        elif key == 'hf_methods':
            return self._render_hf_details(data)
        elif key == 'synthesis':
            return self._render_synthesis_details(data)
        
        return ""
    
    def _render_executive_details(self, data: Dict) -> str:
        sentiment = data.get('sentiment', {})
        findings = data.get('key_findings', [])
        recs = data.get('recommendations', [])
        
        html = f"""
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Overall Sentiment</div>
                <div class="metric-value positive">{sentiment.get('overall', 'N/A')}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Confidence Level</div>
                <div class="metric-value">{sentiment.get('confidence', 0):.0%}</div>
                <div class="metric-subtext">High confidence</div>
            </div>
        </div>
        
        <div class="row mt-4">
            <div class="col-md-6">
                <h4 class="mb-3"><i class="bi bi-check-circle-fill text-success"></i> Key Findings</h4>
                <ul class="list-detailed">
"""
        for finding in findings:
            html += f"<li>{finding}</li>"
        
        html += f"""
                </ul>
            </div>
            <div class="col-md-6">
                <h4 class="mb-3"><i class="bi bi-lightbulb-fill text-warning"></i> Recommendations</h4>
                <ul class="list-detailed">
"""
        for rec in recs:
            html += f"<li>{rec}</li>"
        
        html += """
                </ul>
            </div>
        </div>
"""
        
        if 'market_overview' in data:
            html += f"""
        <div class="alert alert-info mt-4">
            <h5><i class="bi bi-info-circle-fill"></i> Market Overview</h5>
            <p class="mb-0">{data['market_overview']}</p>
        </div>
"""
        
        return html
    
    def _render_news_details(self, data: Dict) -> str:
        count = data.get('article_count', 0)
        sources = data.get('sources', [])
        themes = data.get('key_themes', [])
        sentiment = data.get('sentiment', 'N/A')
        
        html = f"""
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Articles Analyzed</div>
                <div class="metric-value">{count}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">News Sources</div>
                <div class="metric-value">{len(sources)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Overall Sentiment</div>
                <div class="metric-value neutral">{sentiment}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Key Themes</div>
                <div class="metric-value">{len(themes)}</div>
            </div>
        </div>
"""
        
        if sources:
            html += """
        <h4 class="mt-4 mb-3">Sources</h4>
        <div class="d-flex flex-wrap gap-2">
"""
            for source in sources:
                html += f'<span class="badge bg-secondary badge-custom">{source}</span>'
            html += "</div>"
        
        if themes:
            html += """
        <h4 class="mt-4 mb-3">Top Themes</h4>
        <table class="detail-table">
            <thead>
                <tr>
                    <th>Theme</th>
                    <th>Relevance</th>
                </tr>
            </thead>
            <tbody>
"""
            for theme in themes[:10]:
                relevance = "High" if len(theme) > 10 else "Medium"
                html += f"""
                <tr>
                    <td><strong>{theme}</strong></td>
                    <td><span class="badge bg-info">{relevance}</span></td>
                </tr>
"""
            html += """
            </tbody>
        </table>
"""
        
        return html
    
    def _render_indicators_details(self, data: Dict) -> str:
        buy = data.get('buy_signals', 0)
        sell = data.get('sell_signals', 0)
        neutral = data.get('neutral_signals', 0)
        total = buy + sell + neutral
        
        html = f"""
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Overall Bias</div>
                <div class="metric-value positive">{data.get('overall_bias', 'N/A')}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Buy Signals</div>
                <div class="metric-value positive">{buy}</div>
                <div class="metric-subtext">{buy/total*100 if total else 0:.1f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Sell Signals</div>
                <div class="metric-value negative">{sell}</div>
                <div class="metric-subtext">{sell/total*100 if total else 0:.1f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Neutral Signals</div>
                <div class="metric-value neutral">{neutral}</div>
                <div class="metric-subtext">{neutral/total*100 if total else 0:.1f}%</div>
            </div>
        </div>
        
        <div class="chart-container">
            <canvas id="chart-indicators"></canvas>
        </div>
        
        <script>
            new Chart(document.getElementById('chart-indicators'), {{
                type: 'doughnut',
                data: {{
                    labels: ['Buy Signals', 'Sell Signals', 'Neutral'],
                    datasets: [{{
                        data: [{buy}, {sell}, {neutral}],
                        backgroundColor: ['#28a745', '#dc3545', '#ffc107']
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        title: {{
                            display: true,
                            text: 'Signal Distribution',
                            font: {{ size: 16, weight: 'bold' }}
                        }},
                        legend: {{ position: 'bottom' }}
                    }}
                }}
            }});
        </script>
"""
        
        if 'strongest_signal' in data or 'symbols_analyzed' in data:
            html += '<div class="data-grid mt-4">'
            if 'symbols_analyzed' in data:
                html += f'''
                <div class="data-label">Symbols Analyzed:</div>
                <div class="data-value">{data['symbols_analyzed']}</div>
'''
            if 'strongest_signal' in data:
                html += f'''
                <div class="data-label">Strongest Signal:</div>
                <div class="data-value"><span class="badge badge-bullish">{data['strongest_signal']}</span></div>
'''
            if 'weakest_signal' in data:
                html += f'''
                <div class="data-label">Weakest Signal:</div>
                <div class="data-value"><span class="badge badge-bearish">{data['weakest_signal']}</span></div>
'''
            html += '</div>'
        
        return html
    
    def _render_cot_details(self, data: Dict) -> str:
        html = f"""
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Net Positioning</div>
                <div class="metric-value positive">{data.get('net_positioning', 'N/A')}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Position Change</div>
                <div class="metric-value">{data.get('positioning_change', 'N/A')}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Institutional Sentiment</div>
                <div class="metric-value positive">{data.get('institutional_sentiment', 'N/A')}</div>
            </div>
        </div>
"""
        
        if any(k in data for k in ['commercials_long', 'non_commercials_long']):
            html += """
        <h4 class="mt-4 mb-3">Position Breakdown</h4>
        <table class="detail-table">
            <thead>
                <tr>
                    <th>Category</th>
                    <th>Long</th>
                    <th>Short</th>
                    <th>Net</th>
                </tr>
            </thead>
            <tbody>
"""
            if 'commercials_long' in data:
                comm_long = data.get('commercials_long', 0)
                comm_short = data.get('commercials_short', 0)
                html += f"""
                <tr>
                    <td><strong>Commercials</strong></td>
                    <td>{comm_long:,}</td>
                    <td>{comm_short:,}</td>
                    <td class="{'text-success' if comm_long > comm_short else 'text-danger'}">{comm_long - comm_short:+,}</td>
                </tr>
"""
            if 'non_commercials_long' in data:
                non_comm_long = data.get('non_commercials_long', 0)
                non_comm_short = data.get('non_commercials_short', 0)
                html += f"""
                <tr>
                    <td><strong>Non-Commercials</strong></td>
                    <td>{non_comm_long:,}</td>
                    <td>{non_comm_short:,}</td>
                    <td class="{'text-success' if non_comm_long > non_comm_short else 'text-danger'}">{non_comm_long - non_comm_short:+,}</td>
                </tr>
"""
            html += """
            </tbody>
        </table>
"""
        
        return html
    
    def _render_economic_details(self, data: Dict) -> str:
        html = f"""
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Overall Status</div>
                <div class="metric-value neutral">{data.get('overall_status', 'N/A')}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Inflation Trend</div>
                <div class="metric-value">{data.get('inflation_trend', 'N/A')}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Growth Outlook</div>
                <div class="metric-value positive">{data.get('growth_outlook', 'N/A')}</div>
            </div>
        </div>
"""
        
        if any(k in data for k in ['unemployment_rate', 'gdp_growth', 'cpi_yoy']):
            html += '<div class="data-grid mt-4">'
            if 'unemployment_rate' in data:
                html += f'''
                <div class="data-label">Unemployment Rate:</div>
                <div class="data-value">{data['unemployment_rate']}%</div>
'''
            if 'gdp_growth' in data:
                html += f'''
                <div class="data-label">GDP Growth:</div>
                <div class="data-value">{data['gdp_growth']}%</div>
'''
            if 'cpi_yoy' in data:
                html += f'''
                <div class="data-label">CPI (YoY):</div>
                <div class="data-value">{data['cpi_yoy']}%</div>
'''
            if 'fed_funds_rate' in data:
                html += f'''
                <div class="data-label">Fed Funds Rate:</div>
                <div class="data-value">{data['fed_funds_rate']}%</div>
'''
            html += '</div>'
        
        return html
    
    def _render_correlations_details(self, data: Dict) -> str:
        rels = data.get('key_relationships', {})
        
        if not rels:
            return '<p class="text-muted">No correlation data available</p>'
        
        html = """
        <h4 class="mb-3">Key Relationships</h4>
        <table class="detail-table">
            <thead>
                <tr>
                    <th>Asset Pair</th>
                    <th>Strength</th>
                    <th>Direction</th>
                    <th>Correlation</th>
                </tr>
            </thead>
            <tbody>
"""
        for pair, details in rels.items():
            strength = details.get('strength', 'N/A')
            direction = details.get('direction', 'N/A')
            corr = details.get('correlation', 'N/A')
            
            strength_badge = 'success' if strength == 'STRONG' else 'warning' if strength == 'MODERATE' else 'secondary'
            
            html += f"""
                <tr>
                    <td><strong>{pair.replace('_', ' / ')}</strong></td>
                    <td><span class="badge bg-{strength_badge}">{strength}</span></td>
                    <td>{direction}</td>
                    <td>{corr if isinstance(corr, str) else f'{corr:.2f}' if isinstance(corr, (int, float)) else 'N/A'}</td>
                </tr>
"""
        
        html += """
            </tbody>
        </table>
"""
        
        return html
    
    def _render_structure_details(self, data: Dict) -> str:
        support = data.get('support_levels', [])
        resistance = data.get('resistance_levels', [])
        
        html = f"""
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Market Trend</div>
                <div class="metric-value positive">{data.get('trend', 'N/A')}</div>
            </div>
"""
        
        if 'current_price' in data:
            html += f"""
            <div class="metric-card">
                <div class="metric-label">Current Price</div>
                <div class="metric-value">{data['current_price']}</div>
            </div>
"""
        
        if 'trend_strength' in data:
            html += f"""
            <div class="metric-card">
                <div class="metric-label">Trend Strength</div>
                <div class="metric-value">{data['trend_strength']}</div>
            </div>
"""
        
        html += "</div>"
        
        if support or resistance:
            html += """
        <div class="row mt-4">
            <div class="col-md-6">
                <h4 class="mb-3 text-success"><i class="bi bi-arrow-down-circle-fill"></i> Support Levels</h4>
                <ul class="list-group">
"""
            for level in support:
                html += f'<li class="list-group-item d-flex justify-content-between align-items-center"><strong>S:</strong> <span class="badge bg-success">{level}</span></li>'
            
            html += """
                </ul>
            </div>
            <div class="col-md-6">
                <h4 class="mb-3 text-danger"><i class="bi bi-arrow-up-circle-fill"></i> Resistance Levels</h4>
                <ul class="list-group">
"""
            for level in resistance:
                html += f'<li class="list-group-item d-flex justify-content-between align-items-center"><strong>R:</strong> <span class="badge bg-danger">{level}</span></li>'
            
            html += """
                </ul>
            </div>
        </div>
"""
        
        return html
    
    def _render_seasonality_details(self, data: Dict) -> str:
        html = f"""
        <div class="data-grid">
            <div class="data-label">Seasonal Bias:</div>
            <div class="data-value"><span class="badge badge-bullish">{data.get('seasonal_bias', 'N/A')}</span></div>
            
            <div class="data-label">Historical Performance:</div>
            <div class="data-value">{data.get('historical_performance', 'N/A')}</div>
"""
        
        if 'best_months' in data:
            html += f'''
            <div class="data-label">Best Months:</div>
            <div class="data-value">{', '.join(data['best_months'])}</div>
'''
        
        if 'current_month_avg' in data:
            html += f'''
            <div class="data-label">Current Month Avg:</div>
            <div class="data-value">{data['current_month_avg']}</div>
'''
        
        if 'win_rate' in data:
            html += f'''
            <div class="data-label">Win Rate:</div>
            <div class="data-value">{data['win_rate']}%</div>
'''
        
        html += '</div>'
        
        return html
    
    def _render_volume_details(self, data: Dict) -> str:
        html = f"""
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Volume Trend</div>
                <div class="metric-value positive">{data.get('volume_trend', 'N/A')}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Volume Profile</div>
                <div class="metric-value">{data.get('volume_profile', 'N/A')}</div>
            </div>
"""
        
        if 'avg_volume_20d' in data:
            html += f"""
            <div class="metric-card">
                <div class="metric-label">20-Day Average</div>
                <div class="metric-value">{data['avg_volume_20d']:,}</div>
            </div>
"""
        
        if 'volume_change' in data:
            html += f"""
            <div class="metric-card">
                <div class="metric-label">Volume Change</div>
                <div class="metric-value positive">{data['volume_change']}</div>
            </div>
"""
        
        html += "</div>"
        
        return html
    
    def _render_hf_details(self, data: Dict) -> str:
        html = f"""
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">AI Sentiment Score</div>
                <div class="metric-value">{data.get('sentiment_score', 0):.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Forecast Direction</div>
                <div class="metric-value positive">{data.get('forecast_direction', 'N/A')}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Forecast Confidence</div>
                <div class="metric-value">{data.get('forecast_confidence', 0):.0%}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Anomalies Detected</div>
                <div class="metric-value">{data.get('anomalies_detected', 0)}</div>
            </div>
        </div>
"""
        
        if any(k in data for k in ['pattern_recognition', 'ai_recommendation']):
            html += '<div class="data-grid mt-4">'
            if 'pattern_recognition' in data:
                html += f'''
                <div class="data-label">Pattern Recognition:</div>
                <div class="data-value"><span class="badge bg-info">{data['pattern_recognition']}</span></div>
'''
            if 'ai_recommendation' in data:
                html += f'''
                <div class="data-label">AI Recommendation:</div>
                <div class="data-value"><span class="badge badge-bullish">{data['ai_recommendation']}</span></div>
'''
            html += '</div>'
        
        return html
    
    def _render_synthesis_details(self, data: Dict) -> str:
        factors = data.get('key_factors', [])
        risks = data.get('risks', [])
        trades = data.get('trade_ideas', [])
        
        html = f"""
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Overall Outlook</div>
                <div class="metric-value positive">{data.get('overall_outlook', 'N/A')}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Confidence Level</div>
                <div class="metric-value">{data.get('confidence', 0):.0%}</div>
            </div>
        </div>
        
        <div class="row mt-4">
            <div class="col-md-6">
                <h4 class="mb-3 text-success"><i class="bi bi-check-circle-fill"></i> Key Factors</h4>
                <ul class="list-detailed">
"""
        
        for factor in factors:
            html += f"<li>{factor}</li>"
        
        html += f"""
                </ul>
            </div>
            <div class="col-md-6">
                <h4 class="mb-3 text-danger"><i class="bi bi-exclamation-triangle-fill"></i> Risk Factors</h4>
                <ul class="list-detailed">
"""
        
        for risk in risks:
            html += f"<li>{risk}</li>"
        
        html += """
                </ul>
            </div>
        </div>
"""
        
        if trades:
            html += """
        <div class="mt-4">
            <h4 class="mb-3"><i class="bi bi-graph-up"></i> Trade Ideas</h4>
            <ul class="list-detailed">
"""
            for trade in trades:
                html += f"<li>{trade}</li>"
            html += """
            </ul>
        </div>
"""
        
        return html
    
    def _render_section_images(self, section_key: str, viz_dir: Path) -> str:
        """Render embedded visualization images"""
        
        patterns = [
            f"{section_key}_*.png",
            f"*{section_key}*.png"
        ]
        
        images = []
        for pattern in patterns:
            images.extend(list(viz_dir.glob(pattern)))
        
        if not images:
            return ""
        
        html = '<div class="image-gallery mt-4">'
        
        for img in images[:4]:  # Max 4 images per section
            img_data = self._embed_image(img)
            if img_data:
                html += f"""
                <div class="image-card">
                    <img src="{img_data}" alt="{img.stem}">
                    <div class="image-caption">{img.stem.replace('_', ' ').title()}</div>
                </div>
"""
        
        html += '</div>'
        
        return html


def main():
    if len(sys.argv) < 2:
        print("Usage: python local_llm_report_generator_enhanced.py <json_file>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    if not os.path.exists(json_file):
        print(f"Error: {json_file} not found")
        sys.exit(1)
    
    generator = EnhancedLLMReportGenerator(
        include_raw_data=True,
        embed_images=True
    )
    
    output = generator.generate_report(json_file)
    print(f"\n✓ Success! Open: {output}")


if __name__ == "__main__":
    main()
