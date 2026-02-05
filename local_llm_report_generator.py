"""
AI-Powered Comprehensive Report Generator using Local LLM
Uses BIST-Financial-Qwen-7B via llama-cpp-python (no API required!)
Analyzes pipeline results and generates detailed HTML reports
Enhanced with template data support for missing sections
"""

import json
import os
import base64
from datetime import datetime
from pathlib import Path


class LocalLLMReportGenerator:
    """
    Generate comprehensive HTML reports using local BIST-Financial-Qwen-7B model
    Analyzes all pipeline data and creates detailed, interactive reports
    No API keys required - runs completely locally!
    Enhanced with template data to fill missing information
    """
    
    def __init__(self, model_path=None, template_data_path=None):
        print("🔄 Loading local LLM model...")
        print("This may take a minute on first run...")
        
        # Load template data for filling missing sections
        self.template_data = self._load_template_data(template_data_path)
        
        try:
            from llama_cpp import Llama
            
            # Use BIST-Financial-Qwen-7B - optimized for financial analysis
            self.llm = Llama.from_pretrained(
                repo_id="bist-quant/BIST-Financial-Qwen-7B",
                filename="gguf/qwen-kap-final-Q4_K_M.gguf",
                n_ctx=4096,  # Context window
                n_threads=4,  # CPU threads
                verbose=False
            )
            
            print("✅ Model loaded successfully!")
            
        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed. Install with:\n"
                "pip install llama-cpp-python"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def _load_template_data(self, template_path=None):
        """Load template data for filling missing sections"""
        if template_path is None:
            # Look for template in common locations
            possible_paths = [
                'sample_data_templates.json',
                '/home/claude/sample_data_templates.json',
                os.path.join(os.path.dirname(__file__), 'sample_data_templates.json'),
                os.path.join(os.getcwd(), 'sample_data_templates.json')
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    template_path = path
                    break
        
        if template_path and os.path.exists(template_path):
            try:
                with open(template_path, 'r') as f:
                    data = json.load(f)
                print(f"✅ Loaded template data from {template_path}")
                return data
            except Exception as e:
                print(f"⚠️ Could not load template data: {e}")
                return {}
        else:
            print("⚠️ No template data file found, using empty defaults")
            return {}
    
    def _merge_with_template(self, section_key, actual_data):
        """Merge actual data with template data to fill gaps"""
        if not self.template_data or section_key not in self.template_data:
            return actual_data
        
        template = self.template_data[section_key]
        
        # If actual data is empty or None, use template
        if not actual_data:
            print(f"  📋 Using template data for {section_key}")
            return template
        
        # If actual data exists but is sparse, merge intelligently
        if isinstance(actual_data, dict) and isinstance(template, dict):
            merged = template.copy()
            # Update with actual data, keeping template as fallback
            self._deep_merge(merged, actual_data)
            print(f"  🔄 Merged template + actual data for {section_key}")
            return merged
        
        # Otherwise return actual data
        return actual_data
    
    def _deep_merge(self, base, updates):
        """Recursively merge updates into base dictionary"""
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def _load_section_jsons(self, sections_dir: Path, main_data: Dict) -> Dict:
        """Load all section JSON files from enhanced pipeline output"""
        sections_dir = Path(sections_dir)
        section_data = {}
        
        # Map section files to main section keys
        section_mapping = {
            'news_analysis': 'news',
            'ai_analysis': 'hf_methods',
            'technical_indicators': 'indicators',
            'cot_positioning': 'cot',
            'economic_indicators': 'economic',
            'correlations': 'correlations',
            'market_structure': 'structure',
            'volume_analysis': 'volume',
            'seasonality': 'seasonality',
            'synthesis': 'synthesis'
        }
        
        # Extract date and event from main data for filename matching
        metadata = main_data.get('metadata', {})
        date = metadata.get('date', '').replace('-', '_')
        event = metadata.get('event_name', 'analysis').replace(' ', '_').lower()
        
        for section_file_prefix, section_key in section_mapping.items():
            # Try to find matching section file
            pattern = f"{section_file_prefix}_{date}_*.json"
            matches = list(sections_dir.glob(pattern))
            
            if not matches:
                # Try without event name
                pattern = f"{section_file_prefix}_*.json"
                matches = list(sections_dir.glob(pattern))
            
            if matches:
                # Use most recent file
                latest_file = max(matches, key=lambda p: p.stat().st_mtime)
                
                try:
                    with open(latest_file, 'r', encoding='utf-8') as f:
                        section_json = json.load(f)
                    
                    # Extract data from enhanced format
                    section_data[section_key] = section_json.get('data', section_json)
                    
                except Exception as e:
                    print(f"  ⚠️ Error loading {latest_file.name}: {str(e)[:50]}")
        
        return section_data
    
    def _generate_response(self, prompt, max_tokens=2000):
        """Generate response from local LLM."""
        try:
            response = self.llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert financial analyst specializing in market data interpretation. Provide detailed, actionable insights in HTML format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
            )
            
            return response['choices'][0]['message']['content']
            
        except Exception as e:
            print(f"⚠️ Generation Error: {str(e)}")
            return None
    
    def analyze_section(self, section_name, section_data, context=""):
        """Analyze a section of data with the local LLM."""
        
        # Merge with template data first
        section_data = self._merge_with_template(section_name, section_data)
        
        # Truncate data to fit context window
        data_str = json.dumps(section_data, indent=2)
        if len(data_str) > 3000:
            data_str = data_str[:3000] + "\n... (truncated)"
        
        prompt = f"""You are a financial market analyst reviewing comprehensive market analysis data.

Analyze the following {section_name.upper()} section data and provide:
1. A detailed summary of key findings
2. Important patterns or trends identified
3. Implications for traders and investors
4. Risk factors or opportunities highlighted by this data
5. Specific numerical insights with context

Context: {context}

Data:
{data_str}

Provide your analysis in well-structured HTML format using Bootstrap classes:
- Use <h4> for subsection headers
- Use <p class="lead"> for key insights
- Use <ul> or <ol> for lists
- Use <div class="alert alert-info/warning/success"> for important callouts
- Include specific numbers and percentages from the data
- Be detailed and thorough but concise

Generate ONLY the HTML content, no markdown wrappers."""

        print(f"  🤖 Analyzing {section_name}...")
        analysis = self._generate_response(prompt, max_tokens=1500)
        
        if not analysis:
            # Fallback to template-based summary
            return self._generate_template_summary(section_name, section_data)
        
        return analysis
    
    def _generate_template_summary(self, section_name, section_data):
        """Generate a basic HTML summary from template data when LLM fails"""
        if not section_data:
            return f"<p class='text-muted'>No data available for {section_name}</p>"
        
        html = f"<div class='alert alert-info'><strong>Analysis for {section_name}:</strong> Data loaded from template</div>"
        
        # Generate basic summary based on section type
        if isinstance(section_data, dict):
            html += "<ul>"
            for key, value in list(section_data.items())[:5]:
                if isinstance(value, (str, int, float)):
                    html += f"<li><strong>{key}:</strong> {value}</li>"
            html += "</ul>"
        
        return html
    
    def generate_comprehensive_report(self, json_file, output_html=None, viz_dir=None, sections_dir=None):
        """Generate the complete HTML report."""
        
        print("="*80)
        print("AI-POWERED COMPREHENSIVE REPORT GENERATOR (LOCAL LLM)")
        print("="*80)
        print(f"Model: BIST-Financial-Qwen-7B")
        print(f"Input: {json_file}")
        print(f"Template Data: {'Loaded' if self.template_data else 'Not available'}")
        print("="*80)
        
        # Load main data
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Set output paths
        if output_html is None:
            base_name = Path(json_file).stem
            output_html = str(Path(json_file).parent / f"{base_name}_ai_report.html")
        
        if viz_dir is None:
            viz_dir = Path(json_file).parent / 'visualizations'
        
        if sections_dir is None:
            sections_dir = Path(json_file).parent / 'sections'
        
        # Load section JSON files if available
        if Path(sections_dir).exists():
            print(f"\n📂 Loading section JSON files from: {sections_dir}")
            section_data = self._load_section_jsons(sections_dir, data)
            # Merge section data into main data
            for section_key, section_content in section_data.items():
                if section_content:
                    data['sections'][section_key] = section_content
                    print(f"  ✓ Loaded {section_key}")
        else:
            print(f"\n⊘ No sections directory found at: {sections_dir}")
        
        metadata = data.get('metadata', {})
        event_name = metadata.get('event_name', 'Market Analysis')
        analysis_date = metadata.get('date', 'N/A')
        
        print(f"\n📊 Analyzing: {event_name}")
        print(f"📅 Date: {analysis_date}")
        
        # Start building HTML
        html_parts = [self._get_html_header(event_name, analysis_date)]
        
        # Generate Executive Summary
        print("\n🤖 Generating Executive Summary...")
        exec_summary = self._generate_executive_summary(data)
        html_parts.append(exec_summary)
        
        # Process all sections with template fallback
        sections = data.get('sections', {})
        
        section_order = [
            ('news', '📰 News Analysis', 'news_analysis'),
            ('hf_methods', '🤖 AI Analysis', 'ai_analysis'),
            ('indicators', '📈 Technical Indicators', 'technical_indicators'),
            ('cot', '📊 COT Positioning', 'cot_positioning'),
            ('economic', '💹 Economic Indicators', 'economic_indicators'),
            ('correlations', '🔗 Correlations', 'correlations'),
            ('structure', '🏗️ Market Structure', 'market_structure'),
            ('volume', '📊 Volume Analysis', 'volume_analysis'),
            ('seasonality', '📅 Seasonality', 'seasonality'),
            ('synthesis', '💡 Synthesis & Insights', 'synthesis')
        ]
        
        for section_key, section_title, template_key in section_order:
            section_data = sections.get(section_key, {})
            
            # Merge with template if needed
            if not section_data and template_key in self.template_data:
                print(f"\n📋 {section_title}: Using template data")
                section_data = self.template_data[template_key]
            elif section_data:
                print(f"\n🔍 Processing: {section_title}")
            else:
                print(f"\n⊘ {section_title}: No data available")
                continue
            
            section_html = self._generate_section_html(
                section_key, 
                section_title, 
                section_data,
                metadata,
                template_key
            )
            html_parts.append(section_html)
        
        # Add visualizations if they exist
        if viz_dir and Path(viz_dir).exists():
            print(f"\n🖼️ Adding visualizations...")
            viz_html = self._generate_visualizations_section(viz_dir)
            html_parts.append(viz_html)
        
        # Add raw data section
        print("\n📋 Adding raw data section...")
        html_parts.append(self._generate_raw_data_section(data))
        
        # Close HTML
        html_parts.append(self._get_html_footer())
        
        # Write file
        final_html = '\n'.join(html_parts)
        with open(output_html, 'w', encoding='utf-8') as f:
            f.write(final_html)
        
        print(f"\n✅ Report generated: {output_html}")
        print("="*80)
        
        return output_html
    
    def _generate_executive_summary(self, data):
        """Generate executive summary section."""
        
        # Merge with template
        summary_data = {
            'metadata': data.get('metadata', {}),
            'synthesis': data.get('sections', {}).get('synthesis', {}),
            'hf_sentiment': data.get('sections', {}).get('hf_methods', {}).get('sentiment', {}).get('aggregated', {}),
            'indicators_count': len(data.get('sections', {}).get('indicators', {})),
            'symbols_analyzed': data.get('metadata', {}).get('symbols_count', 0)
        }
        
        summary_data = self._merge_with_template('executive_summary', summary_data)
        
        # Truncate for context window
        summary_str = json.dumps(summary_data, indent=2)
        if len(summary_str) > 2000:
            summary_str = summary_str[:2000] + "\n... (truncated)"
        
        prompt = f"""You are a senior financial analyst preparing an executive summary for stakeholders.

Based on this comprehensive market analysis data:
{summary_str}

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

        summary = self._generate_response(prompt, max_tokens=2000)
        
        if not summary:
            summary = self._generate_template_executive_summary(summary_data)
        
        return f"""
        <section id="executive-summary" class="mb-5">
            <h2 class="mb-4">
                <i class="bi bi-star-fill text-warning"></i> Executive Summary
            </h2>
            {summary}
        </section>
        """
    
    def _generate_template_executive_summary(self, data):
        """Generate executive summary from template when LLM unavailable"""
        html = """
        <div class="alert alert-primary">
            <h4>Market Overview</h4>
            <p class="lead">{overview}</p>
        </div>
        
        <div class="row">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header bg-success text-white">
                        <h5>Opportunities</h5>
                    </div>
                    <div class="card-body">
                        <ul>
                            {opportunities}
                        </ul>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header bg-warning">
                        <h5>Risks</h5>
                    </div>
                    <div class="card-body">
                        <ul>
                            {risks}
                        </ul>
                    </div>
                </div>
            </div>
        </div>
        """
        
        overview = data.get('market_overview', 'Market analysis in progress.')
        
        opportunities = ""
        for opp in data.get('opportunities', [])[:3]:
            opportunities += f"<li>{opp.get('opportunity', 'N/A')}</li>"
        
        risks = ""
        for risk in data.get('risks', [])[:3]:
            risks += f"<li>{risk.get('risk', 'N/A')}</li>"
        
        return html.format(overview=overview, opportunities=opportunities, risks=risks)
    
    def _generate_section_html(self, section_key, section_title, section_data, metadata, template_key):
        """Generate HTML for a section with AI analysis."""
        
        context = f"Analysis Date: {metadata.get('date')}. Event: {metadata.get('event_name', 'General Market Analysis')}."
        
        ai_analysis = self.analyze_section(template_key, section_data, context)
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
        """Create charts for specific sections."""
        if section_key == 'hf_methods':
            return self._create_sentiment_chart(section_data)
        elif section_key == 'indicators':
            return self._create_signals_chart(section_data)
        elif section_key == 'economic':
            return self._create_economic_chart(section_data)
        else:
            return ""
    
    def _create_sentiment_chart(self, hf_data):
        """Create sentiment distribution chart."""
        sentiment = hf_data.get('sentiment', {})
        if isinstance(sentiment, dict):
            sentiment = sentiment.get('aggregated', sentiment)
        
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
        """Create technical signals chart."""
        if not indicators_data:
            return ""
        
        buy = 0
        sell = 0
        neutral = 0
        
        for symbol_data in indicators_data.values():
            signal = symbol_data.get('overall_signal', 'NEUTRAL')
            if signal == 'BUY':
                buy += 1
            elif signal == 'SELL':
                sell += 1
            else:
                neutral += 1
        
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
        """Create economic health chart."""
        if not econ_data:
            return ""
        
        # Try to extract economic status information
        status = econ_data.get('overall_status', 'MODERATE')
        
        # Simple status indicator
        status_color = {
            'STRONG': 'success',
            'MODERATE': 'warning',
            'WEAK': 'danger'
        }.get(status, 'secondary')
        
        return f"""
        <div class="card">
            <div class="card-header">
                <h6 class="mb-0">Economic Health</h6>
            </div>
            <div class="card-body text-center">
                <div class="alert alert-{status_color}">
                    <h3 class="mb-0">{status}</h3>
                    <p class="mb-0 small">Overall Economic Status</p>
                </div>
            </div>
        </div>
        """
    
    def _generate_visualizations_section(self, viz_dir):
        """Add existing visualizations to report."""
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
        """Add raw data section."""
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
        """Generate HTML header."""
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
                    <i class="bi bi-robot"></i> AI-Powered Report by BIST-Financial-Qwen-7B (Local)
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
        """Generate HTML footer."""
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
    
    print("="*80)
    print("LOCAL LLM REPORT GENERATOR WITH TEMPLATE SUPPORT")
    print("="*80)
    print("Using: BIST-Financial-Qwen-7B (Financial Analysis Specialist)")
    print("No API keys required - runs completely locally!")
    print("Template data fills missing sections automatically")
    print("="*80)
    
    # Find JSON file
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        import glob
        json_files = glob.glob('**/comprehensive_*.json', recursive=True)
        
        if not json_files:
            print("\n❌ ERROR: No comprehensive JSON files found")
            print("\nPlease run the comprehensive pipeline first:")
            print("  python comprehensive_pipeline.py")
            sys.exit(1)
        
        json_file = max(json_files, key=os.path.getctime)
        print(f"\n📁 Using most recent file: {json_file}")
    
    # Generate report
    try:
        generator = LocalLLMReportGenerator()
        html_file = generator.generate_comprehensive_report(json_file)
        
        print(f"\n✅ SUCCESS!")
        print(f"📄 HTML Report: {html_file}")
        print(f"\n🌐 Open in browser:")
        print(f"  file://{os.path.abspath(html_file)}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
