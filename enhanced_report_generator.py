"""
Enhanced Report Generator
Generate comprehensive PDF reports from analysis data
"""

import os
from pathlib import Path
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from datetime import datetime
import json
from typing import Dict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

class EnhancedReportGenerator:
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2c3e50'),
            spaceBefore=20,
            spaceAfter=12,
            borderColor=colors.HexColor('#3498db'),
            borderWidth=0,
            borderPadding=5
        ))
        
        self.styles.add(ParagraphStyle(
            name='SubHeader',
            parent=self.styles['Heading3'],
            fontSize=12,
            textColor=colors.HexColor('#34495e'),
            spaceBefore=10,
            spaceAfter=8
        ))
    
    def generate_pre_event_pdf(self, results: Dict) -> str:
        """Generate pre-event PDF report"""
        
        output_dir = Path(os.getenv('ANALYSIS_OUTPUT_DIR', str(Path.cwd() / 'outputs')))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = str(output_dir / f"pre_event_{results['event_name'].replace(' ', '_')}_{results['event_date']}.pdf")
        
        doc = SimpleDocTemplate(
            output_file,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        story = []
        
        # Title Page
        story.append(Paragraph("PRE-EVENT ANALYSIS REPORT", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph(f"<b>{results['event_name']}</b>", self.styles['Title']))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(f"Event Date: {results['event_date']}", self.styles['Normal']))
        story.append(Paragraph(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d')}", self.styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        story.append(PageBreak())
        
        sections = results.get('sections', {})
        
        # Executive Summary
        if 'executive' in sections:
            story.extend(self._add_executive_summary(sections['executive']))
            story.append(PageBreak())
        
        # News Analysis
        if 'news' in sections:
            story.extend(self._add_news_section(sections['news']))
            story.append(PageBreak())
        
        # Technical Indicators
        if 'indicators' in sections:
            story.extend(self._add_indicators_section(sections['indicators']))
            story.append(PageBreak())
        
        # COT Analysis
        if 'cot' in sections:
            story.extend(self._add_cot_section(sections['cot']))
            story.append(PageBreak())
        
        # Synthesis
        if 'synthesis' in sections:
            story.extend(self._add_synthesis_section(sections['synthesis']))
        
        # Build PDF
        doc.build(story)
        
        return output_file
    
    def generate_post_event_pdf(self, results: Dict) -> str:
        """Generate post-event PDF report"""
        
        output_dir = Path(os.getenv('ANALYSIS_OUTPUT_DIR', str(Path.cwd() / 'outputs')))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = str(output_dir / f"post_event_{results['event_name'].replace(' ', '_')}_{results['event_date']}.pdf")
        
        doc = SimpleDocTemplate(
            output_file,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        story = []
        
        # Title Page
        story.append(Paragraph("POST-EVENT ANALYSIS REPORT", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph(f"<b>{results['event_name']}</b>", self.styles['Title']))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(f"Event Date: {results['event_date']}", self.styles['Normal']))
        story.append(PageBreak())
        
        sections = results.get('sections', {})
        
        # Impact Analysis
        if 'impact' in sections:
            story.extend(self._add_impact_section(sections['impact']))
        
        # Summary
        if 'summary' in sections:
            story.extend(self._add_post_summary_section(sections['summary']))
        
        doc.build(story)
        
        return output_file
    
    def _add_executive_summary(self, data: Dict) -> list:
        """Add executive summary section"""
        elements = []
        
        elements.append(Paragraph("EXECUTIVE SUMMARY", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.1*inch))
        
        # Overall Outlook
        outlook = data.get('overall_outlook', 'NEUTRAL')
        confidence = data.get('confidence_level', 50)
        
        color_map = {
            'BULLISH': colors.green,
            'BEARISH': colors.red,
            'NEUTRAL': colors.gray
        }
        
        outlook_text = f"<b>Overall Outlook:</b> <font color='{color_map.get(outlook, colors.black).hexval()}'>{outlook}</font> (Confidence: {confidence:.0f}%)"
        elements.append(Paragraph(outlook_text, self.styles['Normal']))
        elements.append(Spacer(1, 0.1*inch))
        
        # Recommendation
        recommendation = data.get('recommendation', '')
        elements.append(Paragraph(f"<b>Recommendation:</b><br/>{recommendation}", self.styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Section Summaries
        elements.append(Paragraph("<b>Section Highlights:</b>", self.styles['SubHeader']))
        
        for section in data.get('section_summaries', [])[:6]:
            elements.append(Paragraph(f"<b>{section['section']}:</b> {section['summary']}", self.styles['Normal']))
            elements.append(Spacer(1, 0.05*inch))
        
        return elements
    
    def _add_news_section(self, data: Dict) -> list:
        """Add news analysis section"""
        elements = []
        
        elements.append(Paragraph("NEWS ANALYSIS", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.1*inch))
        
        # Summary stats
        elements.append(Paragraph(f"<b>Articles Analyzed:</b> {data.get('article_count', 0)}", self.styles['Normal']))
        elements.append(Paragraph(f"<b>Dominant Sentiment:</b> {data.get('dominant_sentiment', 'N/A').upper()}", self.styles['Normal']))
        elements.append(Spacer(1, 0.1*inch))
        
        # Top articles
        elements.append(Paragraph("<b>Key Articles:</b>", self.styles['SubHeader']))
        
        for article in data.get('articles', [])[:10]:
            article_text = f"<b>{article['title']}</b><br/>"
            article_text += f"<i>{article.get('source', 'Unknown')}</i> | "
            article_text += f"Sentiment: {article.get('sentiment', 'N/A')} | "
            article_text += f"Theme: {article.get('theme', 'general')}<br/>"
            article_text += f"{article.get('snippet', '')[:200]}..."
            
            elements.append(Paragraph(article_text, self.styles['Normal']))
            elements.append(Spacer(1, 0.1*inch))
        
        return elements
    
    def _add_indicators_section(self, data: Dict) -> list:
        """Add technical indicators section"""
        elements = []
        
        elements.append(Paragraph("TECHNICAL INDICATORS", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.1*inch))
        
        summary = data.get('summary', {})
        
        # Create table of indicator signals
        table_data = [['Pair', 'RSI', 'MACD', 'Trend', 'Overall']]
        
        for symbol, indicators in list(summary.items())[:10]:
            if isinstance(indicators, dict):
                table_data.append([
                    symbol,
                    indicators.get('rsi_signal', 'N/A'),
                    indicators.get('macd_signal', 'N/A'),
                    indicators.get('trend', 'N/A'),
                    indicators.get('signal', 'N/A')
                ])
        
        table = Table(table_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1*inch, 1*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 0.2*inch))
        
        return elements
    
    def _add_cot_section(self, data: Dict) -> list:
        """Add COT analysis section"""
        elements = []
        
        elements.append(Paragraph("INSTITUTIONAL POSITIONING (COT)", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.1*inch))
        
        elements.append(Paragraph(f"<b>Overall Strategy:</b> {data.get('overall_strategy', 'MIXED')}", self.styles['Normal']))
        elements.append(Paragraph(f"<b>Smart Money Bias:</b> {data.get('smart_money_bias', 'NEUTRAL')}", self.styles['Normal']))
        elements.append(Spacer(1, 0.1*inch))
        
        # Institutional positions
        elements.append(Paragraph("<b>Key Institutional Positions:</b>", self.styles['SubHeader']))
        
        for position in data.get('positions', [])[:5]:
            pos_text = f"<b>{position.get('currency', 'N/A')}:</b> {position.get('strategy', 'N/A')}<br/>"
            pos_text += f"Smart Money: {position.get('smart_money_direction', 'N/A')} | "
            pos_text += f"Hedge Funds: {position.get('hedge_fund_direction', 'N/A')}"
            
            elements.append(Paragraph(pos_text, self.styles['Normal']))
            elements.append(Spacer(1, 0.05*inch))
        
        return elements
    
    def _add_synthesis_section(self, data: Dict) -> list:
        """Add synthesis section"""
        elements = []
        
        elements.append(Paragraph("SYNTHESIS", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.1*inch))
        
        outlook = data.get('overall_outlook', 'NEUTRAL')
        confidence = data.get('confidence', 50)
        
        elements.append(Paragraph(f"<b>Overall Outlook:</b> {outlook} ({confidence:.1f}% confidence)", self.styles['Normal']))
        elements.append(Spacer(1, 0.1*inch))
        
        # Explanation
        explanation = data.get('explanation', '')
        elements.append(Paragraph(explanation, self.styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Signal breakdown
        signals = data.get('signals', {})
        
        elements.append(Paragraph("<b>Signal Breakdown:</b>", self.styles['SubHeader']))
        elements.append(Paragraph(f"Bullish Signals: {len(signals.get('bullish', []))}", self.styles['Normal']))
        elements.append(Paragraph(f"Bearish Signals: {len(signals.get('bearish', []))}", self.styles['Normal']))
        
        return elements
    
    def _add_impact_section(self, data: Dict) -> list:
        """Add impact analysis section"""
        elements = []
        
        elements.append(Paragraph("EVENT IMPACT ANALYSIS", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.1*inch))
        
        # Winners and Losers
        elements.append(Paragraph("<b>Top Gainers:</b>", self.styles['SubHeader']))
        
        for winner in data.get('winners', [])[:5]:
            elements.append(Paragraph(f"{winner['symbol']}: +{winner['change_pct']:.2f}%", self.styles['Normal']))
        
        elements.append(Spacer(1, 0.1*inch))
        elements.append(Paragraph("<b>Top Losers:</b>", self.styles['SubHeader']))
        
        for loser in data.get('losers', [])[:5]:
            elements.append(Paragraph(f"{loser['symbol']}: {loser['change_pct']:.2f}%", self.styles['Normal']))
        
        return elements
    
    def _add_post_summary_section(self, data: Dict) -> list:
        """Add post-event summary section"""
        elements = []
        
        elements.append(Paragraph("SUMMARY", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.1*inch))
        
        for finding in data.get('key_findings', []):
            elements.append(Paragraph(f"• {finding}", self.styles['Normal']))
            elements.append(Spacer(1, 0.05*inch))
        
        return elements
