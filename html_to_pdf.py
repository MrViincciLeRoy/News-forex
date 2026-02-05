"""
HTML Report to PDF Converter
Converts AI-powered HTML reports to professional PDF documents
"""

import sys
import os
from pathlib import Path
from typing import Optional

try:
    from weasyprint import HTML, CSS
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False
    print("⚠️  WeasyPrint not available")

try:
    import pdfkit
    PDFKIT_AVAILABLE = True
except ImportError:
    PDFKIT_AVAILABLE = False
    print("⚠️  pdfkit not available")


class HTMLtoPDFConverter:
    """Convert HTML reports to PDF"""
    
    def __init__(self, method: str = 'auto'):
        """
        Initialize converter
        
        Args:
            method: 'weasyprint', 'pdfkit', or 'auto' (chooses best available)
        """
        self.method = method
        
        if method == 'auto':
            if WEASYPRINT_AVAILABLE:
                self.method = 'weasyprint'
                print("✓ Using WeasyPrint (recommended)")
            elif PDFKIT_AVAILABLE:
                self.method = 'pdfkit'
                print("✓ Using pdfkit")
            else:
                raise ImportError("No PDF converter available. Install: pip install weasyprint")
        
        elif method == 'weasyprint' and not WEASYPRINT_AVAILABLE:
            raise ImportError("WeasyPrint not available. Install: pip install weasyprint")
        
        elif method == 'pdfkit' and not PDFKIT_AVAILABLE:
            raise ImportError("pdfkit not available. Install: pip install pdfkit")
    
    def convert(self, html_file: str, pdf_file: Optional[str] = None, 
                optimize: bool = True) -> str:
        """
        Convert HTML to PDF
        
        Args:
            html_file: Path to HTML file
            pdf_file: Output PDF path (auto-generated if None)
            optimize: Apply PDF optimizations
        
        Returns:
            Path to generated PDF
        """
        
        if not os.path.exists(html_file):
            raise FileNotFoundError(f"HTML file not found: {html_file}")
        
        # Generate PDF filename
        if pdf_file is None:
            pdf_file = str(Path(html_file).with_suffix('.pdf'))
        
        print(f"\n{'='*80}")
        print("HTML TO PDF CONVERSION")
        print(f"{'='*80}")
        print(f"Input:  {html_file}")
        print(f"Output: {pdf_file}")
        print(f"Method: {self.method}")
        print(f"{'='*80}\n")
        
        if self.method == 'weasyprint':
            return self._convert_weasyprint(html_file, pdf_file, optimize)
        elif self.method == 'pdfkit':
            return self._convert_pdfkit(html_file, pdf_file)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _convert_weasyprint(self, html_file: str, pdf_file: str, 
                           optimize: bool) -> str:
        """Convert using WeasyPrint (best quality)"""
        
        print("Converting with WeasyPrint...")
        
        # Custom CSS for PDF optimization
        pdf_css = CSS(string='''
            @page {
                size: A4;
                margin: 1.5cm;
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }
            
            /* Prevent page breaks inside elements */
            .card, .section, .chart-container {
                page-break-inside: avoid;
            }
            
            /* Better image handling */
            img {
                max-width: 100%;
                height: auto;
                page-break-inside: avoid;
            }
            
            /* Table improvements */
            table {
                page-break-inside: avoid;
            }
            
            /* Header stays at top */
            .header {
                page-break-after: avoid;
            }
            
            /* Navigation not needed in PDF */
            .navbar {
                display: none;
            }
            
            /* Adjust accordion for PDF */
            .accordion-collapse {
                display: block !important;
                height: auto !important;
            }
            
            .accordion-button {
                pointer-events: none;
            }
            
            /* Print-friendly colors */
            @media print {
                body {
                    background: white !important;
                }
                
                .main-container {
                    box-shadow: none !important;
                }
            }
        ''')
        
        try:
            html = HTML(filename=html_file)
            html.write_pdf(
                pdf_file,
                stylesheets=[pdf_css],
                optimize_images=optimize,
                jpeg_quality=85
            )
            
            file_size = os.path.getsize(pdf_file) / (1024 * 1024)
            print(f"\n✓ PDF generated successfully")
            print(f"Size: {file_size:.2f} MB")
            
            return pdf_file
            
        except Exception as e:
            print(f"\n❌ Conversion failed: {e}")
            raise
    
    def _convert_pdfkit(self, html_file: str, pdf_file: str) -> str:
        """Convert using pdfkit (requires wkhtmltopdf)"""
        
        print("Converting with pdfkit...")
        
        options = {
            'page-size': 'A4',
            'margin-top': '1.5cm',
            'margin-right': '1.5cm',
            'margin-bottom': '1.5cm',
            'margin-left': '1.5cm',
            'encoding': 'UTF-8',
            'enable-local-file-access': None,
            'no-outline': None,
            'print-media-type': None,
            'disable-smart-shrinking': None,
            'image-quality': 85,
        }
        
        try:
            pdfkit.from_file(html_file, pdf_file, options=options)
            
            file_size = os.path.getsize(pdf_file) / (1024 * 1024)
            print(f"\n✓ PDF generated successfully")
            print(f"Size: {file_size:.2f} MB")
            
            return pdf_file
            
        except Exception as e:
            print(f"\n❌ Conversion failed: {e}")
            print("\nNote: pdfkit requires wkhtmltopdf to be installed:")
            print("  - Ubuntu: sudo apt-get install wkhtmltopdf")
            print("  - macOS: brew install wkhtmltopdf")
            print("  - Windows: Download from https://wkhtmltopdf.org/downloads.html")
            raise
    
    def batch_convert(self, html_files: list, output_dir: Optional[str] = None) -> list:
        """
        Convert multiple HTML files to PDF
        
        Args:
            html_files: List of HTML file paths
            output_dir: Output directory (same as source if None)
        
        Returns:
            List of generated PDF paths
        """
        
        pdf_files = []
        
        print(f"\n{'='*80}")
        print(f"BATCH CONVERSION: {len(html_files)} file(s)")
        print(f"{'='*80}\n")
        
        for i, html_file in enumerate(html_files, 1):
            print(f"[{i}/{len(html_files)}] Converting: {Path(html_file).name}")
            
            if output_dir:
                pdf_file = str(Path(output_dir) / Path(html_file).with_suffix('.pdf').name)
            else:
                pdf_file = None
            
            try:
                result = self.convert(html_file, pdf_file, optimize=True)
                pdf_files.append(result)
                print(f"  ✓ Success\n")
            except Exception as e:
                print(f"  ❌ Failed: {e}\n")
        
        print(f"{'='*80}")
        print(f"Batch complete: {len(pdf_files)}/{len(html_files)} successful")
        print(f"{'='*80}\n")
        
        return pdf_files


def main():
    """Main entry point"""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert HTML reports to PDF',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Convert single file
  python html_to_pdf.py report.html
  
  # Convert with custom output name
  python html_to_pdf.py report.html -o output.pdf
  
  # Convert multiple files
  python html_to_pdf.py report1.html report2.html report3.html
  
  # Specify converter method
  python html_to_pdf.py report.html --method weasyprint
  
  # Batch convert to directory
  python html_to_pdf.py *.html --output-dir pdfs/
        '''
    )
    
    parser.add_argument('html_files', nargs='+', help='HTML file(s) to convert')
    parser.add_argument('-o', '--output', help='Output PDF file (single file mode)')
    parser.add_argument('--output-dir', help='Output directory (batch mode)')
    parser.add_argument('--method', choices=['auto', 'weasyprint', 'pdfkit'], 
                       default='auto', help='Conversion method')
    parser.add_argument('--no-optimize', action='store_true', 
                       help='Disable PDF optimization')
    
    args = parser.parse_args()
    
    try:
        converter = HTMLtoPDFConverter(method=args.method)
        
        if len(args.html_files) == 1 and not args.output_dir:
            # Single file conversion
            pdf_file = converter.convert(
                args.html_files[0],
                args.output,
                optimize=not args.no_optimize
            )
            print(f"\n✓ Success! PDF saved to: {pdf_file}")
        
        else:
            # Batch conversion
            pdf_files = converter.batch_convert(args.html_files, args.output_dir)
            print(f"\n✓ Generated {len(pdf_files)} PDF(s)")
            for pdf in pdf_files:
                print(f"  - {pdf}")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
