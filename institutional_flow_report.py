"""
Institutional Flow Report Generator
Creates comprehensive PDF reports with institutional positioning,
currency flows, and crypto market analysis
"""

from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Optional
import sys


class InstitutionalFlowReportGenerator:
    """
    Generate detailed PDF reports showing:
    - Top 10 institutional players positioning
    - Major currency pair flows (EUR/USD, USD/JPY, etc.)
    - COT report positioning
    - Crypto market correlations
    """
    
    def __init__(self):
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.units import cm
            from reportlab.lib import colors
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
            from reportlab.lib.enums import TA_CENTER, TA_LEFT
            
            self.available = True
            self.A4 = A4
            self.cm = cm
            self.colors = colors
            self.SimpleDocTemplate = SimpleDocTemplate
            self.Paragraph = Paragraph
            self.Spacer = Spacer
            self.Table = Table
            self.TableStyle = TableStyle
            self.PageBreak = PageBreak
            self.TA_CENTER = TA_CENTER
            self.TA_LEFT = TA_LEFT
            
            self.styles = getSampleStyleSheet()
            self._setup_custom_styles()
            
        except ImportError:
            self.available = False
            print("⚠️  ReportLab not available")
            print("   Install: pip install reportlab")
    
    def _setup_custom_styles(self):
        """Setup custom PDF styles"""
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        
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
            spaceAfter=12
        ))
        
        self.styles.add(ParagraphStyle(
            name='Insight',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#2c5aa0'),
            leftIndent=20,
            spaceBefore=5
        ))
    
    def generate_report(self, 
                       pre_report_file: str,
                       post_report_file: str,
                       output_pdf: Optional[str] = None) -> str:
        """
        Generate comprehensive PDF from PRE and POST reports
        
        Args:
            pre_report_file: Path to PRE-event JSON
            post_report_file: Path to POST-event JSON  
            output_pdf: Output PDF filename
        """
        
        if not self.available:
            print("❌ ReportLab not available - cannot generate PDF")
            return None
        
        print(f"\n{'='*80}")
        print("INSTITUTIONAL FLOW REPORT GENERATION")
        print(f"{'='*80}\n")
        
        # Load reports
        with open(pre_report_file, 'r') as f:
            pre_data = json.load(f)
        
        with open(post_report_file, 'r') as f:
            post_data = json.load(f)
        
        # Generate output filename
        if output_pdf is None:
            event_key = Path(pre_report_file).parent.name.split('_')[0]
            event_date = Path(pre_report_file).parent.name.split('_')[1]
            output_pdf = f"institutional_flow_{event_key}_{event_date}.pdf"
        
        # Create PDF
        doc = self.SimpleDocTemplate(
            output_pdf,
            pagesize=self.A4,
            topMargin=2*self.cm,
            bottomMargin=2*self.cm,
            leftMargin=2*self.cm,
            rightMargin=2*self.cm
        )
        
        story = []
        
        # Title Page
        story.extend(self._create_title_page(pre_data, post_data))
        story.append(self.PageBreak())
        
        # Executive Summary
        story.extend(self._create_executive_summary(pre_data, post_data))
        story.append(self.PageBreak())
        
        # PRE-Event Analysis
        story.extend(self._create_pre_event_section(pre_data))
        story.append(self.PageBreak())
        
        # Institutional Positioning
        story.extend(self._create_institutional_section(pre_data))
        story.append(self.PageBreak())
        
        # Currency Flows
        story.extend(self._create_currency_flows_section(pre_data))
        story.append(self.PageBreak())
        
        # Crypto Analysis
        story.extend(self._create_crypto_section(pre_data))
        story.append(self.PageBreak())
        
        # POST-Event Analysis
        story.extend(self._create_post_event_section(post_data))
        story.append(self.PageBreak())
        
        # Impact Analysis
        story.extend(self._create_impact_section(post_data))
        
        # Build PDF
        print("  🔨 Building PDF...")
        doc.build(story)
        
        print(f"\n✓ PDF: {output_pdf}")
        print(f"{'='*80}\n")
        
        return output_pdf
    
    def _create_title_page(self, pre_data: Dict, post_data: Dict) -> list:
        """Create title page"""
        elements = []
        
        event_date = pre_data.get('event_date', 'N/A')
        event_type = pre_data.get('type', 'ANALYSIS')
        
        elements.append(self.Paragraph(
            "📊 INSTITUTIONAL FLOW ANALYSIS",
            self.styles['CustomTitle']
        ))
        
        elements.append(self.Spacer(1, 1*self.cm))
        
        elements.append(self.Paragraph(
            f"Event Date: {event_date}",
            self.styles['Normal']
        ))
        
        elements.append(self.Spacer(1, 0.5*self.cm))
        
        elements.append(self.Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            self.styles['Normal']
        ))
        
        return elements
    
    def _create_executive_summary(self, pre_data: Dict, post_data: Dict) -> list:
        """Create executive summary"""
        elements = []
        
        elements.append(self.Paragraph(
            "📋 Executive Summary",
            self.styles['SectionHeader']
        ))
        
        elements.append(self.Paragraph(
            "This report analyzes institutional positioning, major currency flows, "
            "and crypto market correlations before and after a major economic event.",
            self.styles['Normal']
        ))
        
        elements.append(self.Spacer(1, 0.5*self.cm))
        
        # Key metrics table
        metrics_data = [
            ['Metric', 'PRE-Event', 'POST-Event'],
            ['Analysis Date', pre_data.get('analysis_date', 'N/A'), post_data.get('analysis_date', 'N/A')],
            ['Type', 'Positioning & Setup', 'Impact & Reaction'],
        ]
        
        table = self.Table(metrics_data, colWidths=[6*self.cm, 5*self.cm, 5*self.cm])
        table.setStyle(self.TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), self.colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), self.colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, self.colors.black),
        ]))
        
        elements.append(table)
        
        return elements
    
    def _create_pre_event_section(self, pre_data: Dict) -> list:
        """Create PRE-event section"""
        elements = []
        
        elements.append(self.Paragraph(
            "🔍 PRE-EVENT ANALYSIS",
            self.styles['SectionHeader']
        ))
        
        elements.append(self.Paragraph(
            "Market positioning and setup ahead of the event.",
            self.styles['Normal']
        ))
        
        elements.append(self.Spacer(1, 0.5*self.cm))
        
        expectations = pre_data.get('market_expectations', {})
        
        elements.append(self.Paragraph(
            f"• Market Consensus: {expectations.get('consensus', 'N/A')}",
            self.styles['Insight']
        ))
        
        elements.append(self.Paragraph(
            f"• Expected Volatility: {expectations.get('volatility_expected', 'N/A')}",
            self.styles['Insight']
        ))
        
        return elements
    
    def _create_institutional_section(self, pre_data: Dict) -> list:
        """Create institutional positioning section"""
        elements = []
        
        elements.append(self.Paragraph(
            "🏦 INSTITUTIONAL POSITIONING",
            self.styles['SectionHeader']
        ))
        
        elements.append(self.Paragraph(
            "Top 10 Currency Traders - Market Share Analysis",
            self.styles['Normal']
        ))
        
        elements.append(self.Spacer(1, 0.5*self.cm))
        
        positioning = pre_data.get('institutional_positioning', {})
        context = positioning.get('institutional_context', {})
        major_players = context.get('major_players', [])
        
        if major_players:
            # Create table
            table_data = [['Rank', 'Institution', 'Country', 'Market Share']]
            
            for i, player in enumerate(major_players, 1):
                table_data.append([
                    str(i),
                    player['name'],
                    player['country'],
                    f"{player['share']:.2f}%"
                ])
            
            table = self.Table(table_data, colWidths=[2*self.cm, 6*self.cm, 3*self.cm, 3*self.cm])
            table.setStyle(self.TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), self.colors.HexColor('#667eea')),
                ('TEXTCOLOR', (0, 0), (-1, 0), self.colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, self.colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [self.colors.white, self.colors.lightgrey]),
            ]))
            
            elements.append(table)
            elements.append(self.Spacer(1, 0.5*self.cm))
            
            total_share = context.get('total_top10_share', 0)
            elements.append(self.Paragraph(
                f"Total Top 10 Market Share: {total_share:.2f}%",
                self.styles['Insight']
            ))
        
        # COT Positioning
        elements.append(self.Spacer(1, 0.5*self.cm))
        elements.append(self.Paragraph(
            "COT Report - Smart Money Positioning:",
            self.styles['Normal']
        ))
        
        for currency, pos in positioning.items():
            if currency != 'institutional_context' and isinstance(pos, dict):
                elements.append(self.Paragraph(
                    f"• {currency}: {pos.get('net_positioning', 'N/A')} "
                    f"(Smart Money: {pos.get('smart_money_net', 0):,})",
                    self.styles['Insight']
                ))
        
        return elements
    
    def _create_currency_flows_section(self, pre_data: Dict) -> list:
        """Create currency flows section"""
        elements = []
        
        elements.append(self.Paragraph(
            "💱 MAJOR CURRENCY FLOWS",
            self.styles['SectionHeader']
        ))
        
        elements.append(self.Paragraph(
            "Analysis of top currency pairs by trading volume",
            self.styles['Normal']
        ))
        
        elements.append(self.Spacer(1, 0.5*self.cm))
        
        flows = pre_data.get('currency_flows', {})
        
        if flows:
            table_data = [['Pair', 'Market Share', 'Price', 'Signal', 'Buy/Sell']]
            
            for symbol, data in flows.items():
                if isinstance(data, dict):
                    table_data.append([
                        data.get('pair_name', symbol),
                        f"{data.get('market_share', 0):.1f}%",
                        f"${data.get('price', 0):.4f}",
                        data.get('signal', 'N/A'),
                        f"{data.get('buy_signals', 0)}/{data.get('sell_signals', 0)}"
                    ])
            
            table = self.Table(table_data, colWidths=[3*self.cm, 2.5*self.cm, 3*self.cm, 2.5*self.cm, 2.5*self.cm])
            table.setStyle(self.TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), self.colors.HexColor('#667eea')),
                ('TEXTCOLOR', (0, 0), (-1, 0), self.colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, self.colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [self.colors.white, self.colors.lightgrey]),
            ]))
            
            elements.append(table)
        
        return elements
    
    def _create_crypto_section(self, pre_data: Dict) -> list:
        """Create crypto analysis section"""
        elements = []
        
        elements.append(self.Paragraph(
            "₿ CRYPTO MARKET ANALYSIS",
            self.styles['SectionHeader']
        ))
        
        elements.append(self.Paragraph(
            "Cryptocurrency correlations with traditional markets",
            self.styles['Normal']
        ))
        
        elements.append(self.Spacer(1, 0.5*self.cm))
        
        crypto = pre_data.get('crypto_analysis', {})
        
        elements.append(self.Paragraph(
            f"• BTC Dominance: {crypto.get('btc_dominance', 0):.1f}%",
            self.styles['Insight']
        ))
        
        elements.append(self.Paragraph(
            f"• Risk Sentiment: {crypto.get('risk_sentiment', 'N/A')}",
            self.styles['Insight']
        ))
        
        positions = crypto.get('crypto_positions', {})
        
        if positions:
            elements.append(self.Spacer(1, 0.3*self.cm))
            elements.append(self.Paragraph(
                "Top Cryptocurrencies:",
                self.styles['Normal']
            ))
            
            for symbol, data in list(positions.items())[:5]:
                elements.append(self.Paragraph(
                    f"• {symbol}: ${data.get('price', 0):,.2f} - {data.get('signal', 'N/A')}",
                    self.styles['Insight']
                ))
        
        return elements
    
    def _create_post_event_section(self, post_data: Dict) -> list:
        """Create POST-event section"""
        elements = []
        
        elements.append(self.Paragraph(
            "📊 POST-EVENT ANALYSIS",
            self.styles['SectionHeader']
        ))
        
        elements.append(self.Paragraph(
            "Market reaction and institutional moves following the event.",
            self.styles['Normal']
        ))
        
        elements.append(self.Spacer(1, 0.5*self.cm))
        
        # Currency reactions
        reactions = post_data.get('currency_reactions', {})
        
        if reactions:
            elements.append(self.Paragraph(
                "Currency Pair Reactions:",
                self.styles['Normal']
            ))
            
            table_data = [['Pair', 'Before', 'After', 'Change %', 'Signal Flip']]
            
            for pair, data in reactions.items():
                table_data.append([
                    pair,
                    f"${data.get('price_before', 0):.4f}",
                    f"${data.get('price_after', 0):.4f}",
                    f"{data.get('change_pct', 0):+.2f}%",
                    '✓' if data.get('signal_flip') else '✗'
                ])
            
            table = self.Table(table_data, colWidths=[3*self.cm, 3*self.cm, 3*self.cm, 2.5*self.cm, 2.5*self.cm])
            table.setStyle(self.TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), self.colors.HexColor('#667eea')),
                ('TEXTCOLOR', (0, 0), (-1, 0), self.colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, self.colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [self.colors.white, self.colors.lightgrey]),
            ]))
            
            elements.append(table)
        
        return elements
    
    def _create_impact_section(self, post_data: Dict) -> list:
        """Create impact analysis section"""
        elements = []
        
        elements.append(self.Paragraph(
            "💥 EVENT IMPACT",
            self.styles['SectionHeader']
        ))
        
        crypto_impact = post_data.get('crypto_impact', {})
        
        if crypto_impact:
            elements.append(self.Paragraph(
                "Cryptocurrency Market Impact:",
                self.styles['Normal']
            ))
            
            reactions = crypto_impact.get('crypto_reactions', {})
            
            for symbol, data in reactions.items():
                color_text = "green" if data.get('direction') == 'UP' else "red"
                elements.append(self.Paragraph(
                    f"• {symbol}: {data.get('change_pct', 0):+.2f}% "
                    f"({data.get('magnitude', 'N/A')} {data.get('direction', 'N/A')})",
                    self.styles['Insight']
                ))
            
            elements.append(self.Spacer(1, 0.3*self.cm))
            elements.append(self.Paragraph(
                f"Overall Crypto Sentiment: {crypto_impact.get('overall_crypto_sentiment', 'N/A')}",
                self.styles['Insight']
            ))
            
            elements.append(self.Paragraph(
                f"TradFi Correlation: {crypto_impact.get('correlation_with_tradfi', 'N/A')}",
                self.styles['Insight']
            ))
        
        return elements


def main():
    if len(sys.argv) < 3:
        print("Usage: python institutional_flow_report.py <pre_report.json> <post_report.json> [output.pdf]")
        sys.exit(1)
    
    pre_file = sys.argv[1]
    post_file = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) > 3 else None
    
    generator = InstitutionalFlowReportGenerator()
    
    if not generator.available:
        print("❌ ReportLab not installed")
        sys.exit(1)
    
    result = generator.generate_report(pre_file, post_file, output)
    
    if result:
        print(f"✓ Success: {result}")
    else:
        print("❌ Failed to generate report")
        sys.exit(1)


if __name__ == "__main__":
    main()
