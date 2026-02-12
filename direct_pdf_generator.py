"""
Direct PDF Report Generator
Creates PDF reports directly from JSON data with images
No HTML intermediate step
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import sys

try:
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.lib.units import inch, cm
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
    from reportlab.platypus import KeepTogether
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


class DirectPDFGenerator:
    
    def __init__(self):
        if not REPORTLAB_AVAILABLE:
            raise ImportError("Install: pip install reportlab")
        
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Custom styles for report"""
        
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#667eea'),
            spaceBefore=20,
            spaceAfter=12,
            leftIndent=0
        ))
        
        self.styles.add(ParagraphStyle(
            name='SubHeader',
            parent=self.styles['Heading3'],
            fontSize=12,
            textColor=colors.HexColor('#2c5aa0'),
            spaceBefore=10,
            spaceAfter=8
        ))
        
        self.styles.add(ParagraphStyle(
            name='Insight',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#2c5aa0'),
            leftIndent=20,
            spaceBefore=5
        ))
        
        self.styles.add(ParagraphStyle(
            name='MetricLabel',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=colors.grey,
            alignment=TA_CENTER
        ))
        
        self.styles.add(ParagraphStyle(
            name='MetricValue',
            parent=self.styles['Normal'],
            fontSize=18,
            textColor=colors.HexColor('#333333'),
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
    
    def generate(self, json_file: str, output_file: Optional[str] = None) -> str:
        """Generate PDF report"""
        
        print(f"\n{'='*80}")
        print("DIRECT PDF GENERATION")
        print(f"{'='*80}\n")
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        metadata = data.get('metadata', {})
        sections = data.get('sections', {})
        
        if output_file is None:
            output_file = f"{Path(json_file).stem}_report.pdf"
        
        # Find visualizations
        viz_dir = self._find_viz_dir(json_file)
        
        # Create PDF
        doc = SimpleDocTemplate(
            output_file,
            pagesize=A4,
            topMargin=2*cm,
            bottomMargin=2*cm,
            leftMargin=2*cm,
            rightMargin=2*cm
        )
        
        story = []
        
        # Header
        story.extend(self._create_header(metadata))
        story.append(PageBreak())
        
        # Sections
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
        
        for section_key, title, icon in section_order:
            if section_key in sections:
                print(f"  📄 {title}")
                story.extend(self._create_section(
                    section_key, title, icon,
                    sections[section_key], viz_dir
                ))
        
        # Build PDF
        print(f"\n  🔨 Building PDF...")
        doc.build(story)
        
        size = os.path.getsize(output_file) / (1024 * 1024)
        print(f"\n✓ PDF: {output_file}")
        print(f"Size: {size:.2f} MB")
        print(f"{'='*80}\n")
        
        return output_file
    
    def _find_viz_dir(self, json_file: str) -> Optional[Path]:
        """Find visualization directory"""
        base = Path(json_file).parent
        
        candidates = [
            base / 'visualizations',
            base / 'pipeline_output' / 'visualizations',
            base / 'test_pipeline_output' / 'visualizations',
        ]
        
        for c in candidates:
            if c.exists() and list(c.glob('*.png')):
                print(f"  ✓ Images: {c}")
                return c
        
        return None
    
    def _create_header(self, metadata: Dict) -> List:
        """Create report header"""
        elements = []
        
        date = metadata.get('date', 'N/A')
        event = metadata.get('event_name', 'Analysis')
        symbols = metadata.get('symbols', [])
        
        elements.append(Paragraph(f"📊 {event}", self.styles['CustomTitle']))
        elements.append(Spacer(1, 0.3*cm))
        elements.append(Paragraph(
            "AI-Powered Financial Analysis Report",
            self.styles['Normal']
        ))
        elements.append(Spacer(1, 0.5*cm))
        
        # Metadata table
        data = [
            ['Date:', date],
            ['Generated:', datetime.now().strftime('%Y-%m-%d %H:%M')],
            ['AI Model:', 'BIST-Financial-Qwen-7B'],
        ]
        
        if symbols:
            data.append(['Symbols:', ', '.join(symbols)])
        
        table = Table(data, colWidths=[4*cm, 10*cm])
        table.setStyle(TableStyle([
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.grey),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 1*cm))
        
        return elements
    
    def _create_section(self, key: str, title: str, icon: str, 
                       data: Dict, viz_dir: Optional[Path]) -> List:
        """Create section with data and images"""
        elements = []
        
        actual_data = data.get('data', data)
        
        # Section header
        elements.append(Paragraph(f"{icon} {title}", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.3*cm))
        
        # AI Insights
        insights = self._generate_insights(key, actual_data)
        elements.append(Paragraph("🤖 AI Insights", self.styles['SubHeader']))
        for line in insights.split('\n'):
            if line.strip():
                elements.append(Paragraph(f"• {line.strip('• ')}", self.styles['Insight']))
        elements.append(Spacer(1, 0.5*cm))
        
        # Section content
        elements.extend(self._create_section_content(key, actual_data))
        
        # Images
        if viz_dir:
            images = self._find_section_images(key, viz_dir)
            for img_path in images[:2]:  # Max 2 images per section
                try:
                    img = Image(str(img_path), width=14*cm, height=10*cm)
                    elements.append(Spacer(1, 0.3*cm))
                    elements.append(img)
                    elements.append(Spacer(1, 0.2*cm))
                except Exception as e:
                    print(f"    ⚠️  Image failed: {e}")
        
        elements.append(Spacer(1, 0.5*cm))
        
        return elements
    
    def _generate_insights(self, section: str, data: Dict) -> str:
        """Generate insights text"""
        insights = []
        
        if section == 'executive_summary':
            sentiment = data.get('sentiment', {})
            insights.append(f"Overall: {sentiment.get('overall', 'N/A')} (Confidence: {sentiment.get('confidence', 0):.0%})")
            findings = data.get('key_findings', [])
            if findings:
                insights.extend(findings[:3])
        
        elif section == 'news':
            count = data.get('article_count', 0)
            sources = data.get('sources', [])
            themes = data.get('key_themes', [])
            sentiment = data.get('sentiment', 'N/A')
            insights.append(f"{count} articles from {len(sources)} sources")
            insights.append(f"Sentiment: {sentiment}")
            if themes:
                insights.append(f"Themes: {', '.join(themes[:3])}")
        
        elif section == 'indicators':
            bias = data.get('overall_bias', 'N/A')
            buy = data.get('buy_signals', 0)
            sell = data.get('sell_signals', 0)
            total = buy + sell
            insights.append(f"Bias: {bias}")
            insights.append(f"{buy}/{total} signals bullish ({buy/total*100 if total else 0:.0f}%)")
        
        elif section == 'cot':
            pos = data.get('net_positioning', 'N/A')
            change = data.get('positioning_change', 'N/A')
            insights.append(f"Net: {pos} ({change})")
            insights.append(f"Institutional: {data.get('institutional_sentiment', 'N/A')}")
        
        elif section == 'economic':
            insights.append(f"Status: {data.get('overall_status', 'N/A')}")
            insights.append(f"Inflation: {data.get('inflation_trend', 'N/A')}")
            insights.append(f"Growth: {data.get('growth_outlook', 'N/A')}")
        
        elif section == 'correlations':
            rels = data.get('key_relationships', {})
            strong = [k for k, v in rels.items() if v.get('strength') == 'STRONG']
            if strong:
                insights.append(f"Strong: {', '.join(strong)}")
        
        elif section == 'structure':
            insights.append(f"Trend: {data.get('trend', 'N/A')}")
            support = data.get('support_levels', [])
            resistance = data.get('resistance_levels', [])
            if support:
                insights.append(f"Support: {support[0]}")
            if resistance:
                insights.append(f"Resistance: {resistance[0]}")
        
        elif section == 'seasonality':
            insights.append(f"Bias: {data.get('seasonal_bias', 'N/A')}")
            insights.append(f"Historical: {data.get('historical_performance', 'N/A')}")
        
        elif section == 'volume':
            insights.append(f"Trend: {data.get('volume_trend', 'N/A')}")
            insights.append(f"Profile: {data.get('volume_profile', 'N/A')}")
        
        elif section == 'hf_methods':
            insights.append(f"Sentiment: {data.get('sentiment_score', 0.5):.2f}")
            insights.append(f"Forecast: {data.get('forecast_direction', 'N/A')}")
        
        elif section == 'synthesis':
            insights.append(f"Outlook: {data.get('overall_outlook', 'N/A')}")
            insights.append(f"Confidence: {data.get('confidence', 0.5):.0%}")
        
        return '\n'.join(insights) if insights else f"{section.replace('_', ' ').title()} complete"
    
    def _create_section_content(self, key: str, data: Dict) -> List:
        """Create section-specific content"""
        elements = []
        
        if key == 'executive_summary':
            elements.extend(self._content_executive(data))
        elif key == 'news':
            elements.extend(self._content_news(data))
        elif key == 'indicators':
            elements.extend(self._content_indicators(data))
        elif key == 'cot':
            elements.extend(self._content_cot(data))
        elif key == 'economic':
            elements.extend(self._content_economic(data))
        elif key == 'correlations':
            elements.extend(self._content_correlations(data))
        elif key == 'structure':
            elements.extend(self._content_structure(data))
        elif key == 'seasonality':
            elements.extend(self._content_seasonality(data))
        elif key == 'volume':
            elements.extend(self._content_volume(data))
        elif key == 'hf_methods':
            elements.extend(self._content_hf(data))
        elif key == 'synthesis':
            elements.extend(self._content_synthesis(data))
        
        return elements
    
    def _content_executive(self, data: Dict) -> List:
        elements = []
        
        sentiment = data.get('sentiment', {})
        findings = data.get('key_findings', [])
        recs = data.get('recommendations', [])
        
        # Metrics
        metric_data = [
            ['Overall Sentiment', sentiment.get('overall', 'N/A')],
            ['Confidence', f"{sentiment.get('confidence', 0):.0%}"]
        ]
        
        table = Table(metric_data, colWidths=[6*cm, 6*cm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f5f5f5')),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.grey),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#28a745')),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.white),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 0.5*cm))
        
        if findings:
            elements.append(Paragraph("Key Findings:", self.styles['SubHeader']))
            for f in findings[:5]:
                elements.append(Paragraph(f"• {f}", self.styles['Normal']))
            elements.append(Spacer(1, 0.3*cm))
        
        if recs:
            elements.append(Paragraph("Recommendations:", self.styles['SubHeader']))
            for r in recs[:5]:
                elements.append(Paragraph(f"• {r}", self.styles['Normal']))
        
        return elements
    
    def _content_news(self, data: Dict) -> List:
        elements = []
        
        metric_data = [
            ['Articles', str(data.get('article_count', 0))],
            ['Sources', str(len(data.get('sources', [])))],
            ['Sentiment', data.get('sentiment', 'N/A')]
        ]
        
        table = Table(metric_data, colWidths=[4*cm, 8*cm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f5f5f5')),
            ('GRID', (0, 0), (-1, -1), 1, colors.white),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
        ]))
        
        elements.append(table)
        
        themes = data.get('key_themes', [])
        if themes:
            elements.append(Spacer(1, 0.3*cm))
            elements.append(Paragraph(f"Themes: {', '.join(themes[:5])}", self.styles['Normal']))
        
        return elements
    
    def _content_indicators(self, data: Dict) -> List:
        elements = []
        
        buy = data.get('buy_signals', 0)
        sell = data.get('sell_signals', 0)
        neutral = data.get('neutral_signals', 0)
        
        metric_data = [
            ['Overall Bias', data.get('overall_bias', 'N/A')],
            ['Buy Signals', str(buy)],
            ['Sell Signals', str(sell)],
            ['Neutral', str(neutral)]
        ]
        
        table = Table(metric_data, colWidths=[5*cm, 7*cm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f5f5f5')),
            ('GRID', (0, 0), (-1, -1), 1, colors.white),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (1, 1), (1, 1), colors.HexColor('#28a745')),
            ('TEXTCOLOR', (1, 2), (1, 2), colors.HexColor('#dc3545')),
        ]))
        
        elements.append(table)
        return elements
    
    def _content_cot(self, data: Dict) -> List:
        elements = []
        
        metric_data = [
            ['Net Positioning', data.get('net_positioning', 'N/A')],
            ['Change', data.get('positioning_change', 'N/A')],
            ['Institutional', data.get('institutional_sentiment', 'N/A')]
        ]
        
        table = Table(metric_data, colWidths=[5*cm, 7*cm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f5f5f5')),
            ('GRID', (0, 0), (-1, -1), 1, colors.white),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
        ]))
        
        elements.append(table)
        return elements
    
    def _content_economic(self, data: Dict) -> List:
        elements = []
        
        metric_data = [
            ['Status', data.get('overall_status', 'N/A')],
            ['Inflation', data.get('inflation_trend', 'N/A')],
            ['Growth', data.get('growth_outlook', 'N/A')]
        ]
        
        table = Table(metric_data, colWidths=[5*cm, 7*cm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f5f5f5')),
            ('GRID', (0, 0), (-1, -1), 1, colors.white),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
        ]))
        
        elements.append(table)
        return elements
    
    def _content_correlations(self, data: Dict) -> List:
        elements = []
        
        rels = data.get('key_relationships', {})
        if rels:
            table_data = [['Pair', 'Strength', 'Direction']]
            for pair, details in list(rels.items())[:5]:
                table_data.append([
                    pair.replace('_', ' / '),
                    details.get('strength', 'N/A'),
                    details.get('direction', 'N/A')
                ])
            
            table = Table(table_data, colWidths=[4*cm, 4*cm, 4*cm])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f5f5f5')),
                ('GRID', (0, 0), (-1, -1), 1, colors.white),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ]))
            
            elements.append(table)
        
        return elements
    
    def _content_structure(self, data: Dict) -> List:
        elements = []
        
        support = data.get('support_levels', [])
        resistance = data.get('resistance_levels', [])
        
        metric_data = [
            ['Trend', data.get('trend', 'N/A')],
        ]
        
        if support:
            metric_data.append(['Support', ', '.join(map(str, support[:3]))])
        if resistance:
            metric_data.append(['Resistance', ', '.join(map(str, resistance[:3]))])
        
        table = Table(metric_data, colWidths=[5*cm, 7*cm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f5f5f5')),
            ('GRID', (0, 0), (-1, -1), 1, colors.white),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
        ]))
        
        elements.append(table)
        return elements
    
    def _content_seasonality(self, data: Dict) -> List:
        elements = []
        
        metric_data = [
            ['Seasonal Bias', data.get('seasonal_bias', 'N/A')],
            ['Historical Performance', data.get('historical_performance', 'N/A')]
        ]
        
        table = Table(metric_data, colWidths=[6*cm, 6*cm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f5f5f5')),
            ('GRID', (0, 0), (-1, -1), 1, colors.white),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
        ]))
        
        elements.append(table)
        return elements
    
    def _content_volume(self, data: Dict) -> List:
        elements = []
        
        metric_data = [
            ['Volume Trend', data.get('volume_trend', 'N/A')],
            ['Volume Profile', data.get('volume_profile', 'N/A')]
        ]
        
        table = Table(metric_data, colWidths=[6*cm, 6*cm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f5f5f5')),
            ('GRID', (0, 0), (-1, -1), 1, colors.white),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
        ]))
        
        elements.append(table)
        return elements
    
    def _content_hf(self, data: Dict) -> List:
        elements = []
        
        metric_data = [
            ['Sentiment Score', f"{data.get('sentiment_score', 0):.2f}"],
            ['Forecast', data.get('forecast_direction', 'N/A')],
            ['Confidence', f"{data.get('forecast_confidence', 0):.0%}"],
            ['Anomalies', str(data.get('anomalies_detected', 0))]
        ]
        
        table = Table(metric_data, colWidths=[5*cm, 7*cm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f5f5f5')),
            ('GRID', (0, 0), (-1, -1), 1, colors.white),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
        ]))
        
        elements.append(table)
        return elements
    
    def _content_synthesis(self, data: Dict) -> List:
        elements = []
        
        factors = data.get('key_factors', [])
        risks = data.get('risks', [])
        
        metric_data = [
            ['Outlook', data.get('overall_outlook', 'N/A')],
            ['Confidence', f"{data.get('confidence', 0):.0%}"]
        ]
        
        table = Table(metric_data, colWidths=[6*cm, 6*cm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f5f5f5')),
            ('GRID', (0, 0), (-1, -1), 1, colors.white),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 0.3*cm))
        
        if factors:
            elements.append(Paragraph("Key Factors:", self.styles['SubHeader']))
            for f in factors[:5]:
                elements.append(Paragraph(f"• {f}", self.styles['Normal']))
        
        if risks:
            elements.append(Spacer(1, 0.2*cm))
            elements.append(Paragraph("Risks:", self.styles['SubHeader']))
            for r in risks[:3]:
                elements.append(Paragraph(f"• {r}", self.styles['Normal']))
        
        return elements
    
    def _find_section_images(self, section: str, viz_dir: Path) -> List[Path]:
        """Find images for section"""
        patterns = [f"{section}_*.png", f"*{section}*.png"]
        
        images = []
        for pattern in patterns:
            images.extend(list(viz_dir.glob(pattern)))
        
        return images


def main():
    if len(sys.argv) < 2:
        print("Usage: python direct_pdf_generator.py <json_file> [output.pdf]")
        print("\nInstall: pip install reportlab")
        sys.exit(1)
    
    json_file = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(json_file):
        print(f"Error: {json_file} not found")
        sys.exit(1)
    
    try:
        generator = DirectPDFGenerator()
        result = generator.generate(json_file, output)
        print(f"✓ Success: {result}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
