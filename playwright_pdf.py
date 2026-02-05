"""
Lightweight HTML to PDF Converter using Playwright
Alternative to WeasyPrint/pdfkit - headless browser approach
"""

import sys
import os
from pathlib import Path
from typing import Optional

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


class PlaywrightPDFConverter:
    """Convert HTML to PDF using Playwright (headless Chrome)"""
    
    def __init__(self):
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError(
                "Playwright not available. Install with:\n"
                "  pip install playwright\n"
                "  playwright install chromium"
            )
    
    def convert(self, html_file: str, pdf_file: Optional[str] = None) -> str:
        """
        Convert HTML to PDF
        
        Args:
            html_file: Path to HTML file
            pdf_file: Output PDF path
        
        Returns:
            Path to generated PDF
        """
        
        if not os.path.exists(html_file):
            raise FileNotFoundError(f"HTML file not found: {html_file}")
        
        if pdf_file is None:
            pdf_file = str(Path(html_file).with_suffix('.pdf'))
        
        print(f"\n{'='*80}")
        print("HTML TO PDF CONVERSION (Playwright)")
        print(f"{'='*80}")
        print(f"Input:  {html_file}")
        print(f"Output: {pdf_file}")
        print(f"{'='*80}\n")
        
        html_path = Path(html_file).absolute()
        pdf_path = Path(pdf_file).absolute()
        
        print("Starting headless browser...")
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            print("Loading HTML...")
            page.goto(f"file://{html_path}")
            
            # Wait for page to fully load
            page.wait_for_load_state("networkidle")
            
            print("Generating PDF...")
            page.pdf(
                path=str(pdf_path),
                format='A4',
                margin={
                    'top': '1.5cm',
                    'right': '1.5cm',
                    'bottom': '1.5cm',
                    'left': '1.5cm'
                },
                print_background=True,
                display_header_footer=False
            )
            
            browser.close()
        
        file_size = os.path.getsize(pdf_path) / (1024 * 1024)
        print(f"\n✓ PDF generated successfully")
        print(f"Size: {file_size:.2f} MB")
        print(f"{'='*80}\n")
        
        return str(pdf_path)


def main():
    """Command-line interface"""
    
    if len(sys.argv) < 2:
        print("Usage: python playwright_pdf.py <html_file> [output.pdf]")
        print("\nInstall:")
        print("  pip install playwright")
        print("  playwright install chromium")
        sys.exit(1)
    
    html_file = sys.argv[1]
    pdf_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        converter = PlaywrightPDFConverter()
        result = converter.convert(html_file, pdf_file)
        print(f"✓ Success! PDF saved to: {result}")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
