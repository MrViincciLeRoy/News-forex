"""
Complete Pipeline: Analysis -> HTML Report -> PDF Export
Run comprehensive analysis and generate both HTML and PDF reports
"""

import sys
import os
from pathlib import Path
from typing import Optional
import argparse


def run_complete_pipeline(
    date: str,
    event_name: str,
    output_dir: str = 'output',
    symbols: Optional[list] = None,
    generate_pdf: bool = True,
    pdf_method: str = 'auto'
):
    """
    Run complete analysis pipeline with PDF export
    
    Args:
        date: Analysis date (YYYY-MM-DD)
        event_name: Event name
        output_dir: Output directory
        symbols: List of symbols to analyze
        generate_pdf: Whether to generate PDF
        pdf_method: PDF conversion method
    """
    
    print(f"\n{'='*80}")
    print("COMPLETE ANALYSIS PIPELINE")
    print(f"{'='*80}")
    print(f"Date: {date}")
    print(f"Event: {event_name}")
    print(f"Output: {output_dir}")
    print(f"PDF: {generate_pdf}")
    print(f"{'='*80}\n")
    
    # Step 1: Run analysis
    print("STEP 1: Running comprehensive analysis...")
    print("-" * 80)
    
    try:
        from comprehensive_pipeline import ComprehensiveAnalysisPipeline
        
        pipeline = ComprehensiveAnalysisPipeline(
            output_dir=output_dir,
            enable_viz=True,
            enable_hf=True,
            max_articles=20
        )
        
        results = pipeline.analyze(
            date=date,
            event_name=event_name,
            symbols=symbols
        )
        
        json_file = results['report_file']
        print(f"\n✓ Analysis complete: {json_file}\n")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        return None
    
    # Step 2: Generate HTML report
    print("\nSTEP 2: Generating AI-powered HTML report...")
    print("-" * 80)
    
    try:
        from local_llm_report_generator import LocalLLMReportGenerator
        
        generator = LocalLLMReportGenerator()
        html_file = generator.generate_report(json_file)
        
        print(f"✓ HTML report complete: {html_file}\n")
        
    except Exception as e:
        print(f"\n❌ HTML generation failed: {e}")
        return None
    
    # Step 3: Convert to PDF
    if generate_pdf:
        print("\nSTEP 3: Converting to PDF...")
        print("-" * 80)
        
        try:
            from html_to_pdf import HTMLtoPDFConverter
            
            converter = HTMLtoPDFConverter(method=pdf_method)
            pdf_file = converter.convert(html_file, optimize=True)
            
            print(f"✓ PDF export complete: {pdf_file}\n")
            
        except ImportError as e:
            print(f"\n⚠️  PDF conversion skipped: {e}")
            print("Install: pip install weasyprint")
            pdf_file = None
        
        except Exception as e:
            print(f"\n❌ PDF conversion failed: {e}")
            pdf_file = None
    else:
        pdf_file = None
    
    # Summary
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE")
    print(f"{'='*80}")
    print(f"\nGenerated files:")
    print(f"  📊 Data:  {json_file}")
    print(f"  🌐 HTML:  {html_file}")
    if pdf_file:
        print(f"  📄 PDF:   {pdf_file}")
    
    print(f"\n✓ All done!\n")
    
    return {
        'json': json_file,
        'html': html_file,
        'pdf': pdf_file
    }


def main():
    """Command-line interface"""
    
    parser = argparse.ArgumentParser(
        description='Complete analysis pipeline with PDF export',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Basic usage
  python pipeline_with_pdf.py --date 2024-11-01 --event "Non-Farm Payrolls"
  
  # Skip PDF generation
  python pipeline_with_pdf.py --date 2024-11-01 --event "FOMC Meeting" --no-pdf
  
  # Specify symbols
  python pipeline_with_pdf.py --date 2024-11-01 --event "NFP" --symbols GC DX ES
  
  # Custom output directory
  python pipeline_with_pdf.py --date 2024-11-01 --event "CPI" --output custom_output/
        '''
    )
    
    parser.add_argument('--date', required=True, help='Analysis date (YYYY-MM-DD)')
    parser.add_argument('--event', required=True, help='Event name')
    parser.add_argument('--output', default='output', help='Output directory')
    parser.add_argument('--symbols', nargs='+', help='Symbols to analyze')
    parser.add_argument('--no-pdf', action='store_true', help='Skip PDF generation')
    parser.add_argument('--pdf-method', choices=['auto', 'weasyprint', 'pdfkit'],
                       default='auto', help='PDF conversion method')
    
    args = parser.parse_args()
    
    try:
        results = run_complete_pipeline(
            date=args.date,
            event_name=args.event,
            output_dir=args.output,
            symbols=args.symbols,
            generate_pdf=not args.no_pdf,
            pdf_method=args.pdf_method
        )
        
        if results:
            sys.exit(0)
        else:
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(130)
    
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
