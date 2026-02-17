"""
CLI runner for GitHub Actions
Usage: python run_analysis.py --date 2024-11-01 --event "Non-Farm Payrolls" --type pre --articles 30
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

# All analysis output goes into outputs/ relative to CWD
os.environ.setdefault("ANALYSIS_OUTPUT_DIR", str(Path.cwd() / "outputs"))

def parse_args():
    parser = argparse.ArgumentParser(description="Run Financial Analysis Pipeline")
    parser.add_argument("--date", required=True, help="Event date (YYYY-MM-DD)")
    parser.add_argument("--event", required=True, help="Event name")
    parser.add_argument("--type", required=True, choices=["pre", "post", "both"])
    parser.add_argument("--articles", type=int, default=30, help="Max news articles")
    parser.add_argument("--symbols", nargs="+", default=None, help="Symbols to analyze")
    return parser.parse_args()

async def run_pre(date, event, articles, symbols):
    from enhanced_pre_event_analyzer import PreEventAnalyzer
    from enhanced_report_generator import EnhancedReportGenerator

    print(f"\n🚀 Starting PRE-EVENT analysis: {event} ({date})")

    analyzer = PreEventAnalyzer(
        event_date=date,
        event_name=event,
        symbols=symbols,
        max_articles=articles
    )

    results = await analyzer.run_full_analysis()

    # Generate PDF into outputs/
    generator = EnhancedReportGenerator()
    pdf_path = generator.generate_pre_event_pdf(results)

    print(f"✓ PDF: {pdf_path}")
    print_summary(results)
    return results

async def run_post(date, event, articles, symbols):
    from enhanced_post_event_analyzer import PostEventAnalyzer
    from enhanced_report_generator import EnhancedReportGenerator

    print(f"\n🚀 Starting POST-EVENT analysis: {event} ({date})")

    analyzer = PostEventAnalyzer(
        event_date=date,
        event_name=event,
        symbols=symbols,
        max_articles=articles
    )

    results = await analyzer.run_full_analysis()

    generator = EnhancedReportGenerator()
    pdf_path = generator.generate_post_event_pdf(results)

    print(f"✓ PDF: {pdf_path}")
    print_summary(results)
    return results

def print_summary(results):
    sections = results.get("sections", {})
    exec_summary = sections.get("executive", {})

    print(f"\n{'='*60}")
    print(f"✅ Analysis Complete")
    print(f"   ID:        {results['analysis_id']}")
    print(f"   Sections:  {len(sections)}")
    if exec_summary:
        print(f"   Outlook:   {exec_summary.get('overall_outlook', 'N/A')}")
        print(f"   Confidence:{exec_summary.get('confidence_level', 0):.0f}%")
    print(f"{'='*60}\n")

async def main():
    args = parse_args()

    # Ensure outputs dir exists
    Path("outputs").mkdir(exist_ok=True)

    start = datetime.now()

    try:
        if args.type == "pre":
            await run_pre(args.date, args.event, args.articles, args.symbols)
        elif args.type == "post":
            await run_post(args.date, args.event, args.articles, args.symbols)
        elif args.type == "both":
            await run_pre(args.date, args.event, args.articles, args.symbols)
            await run_post(args.date, args.event, args.articles, args.symbols)

        elapsed = (datetime.now() - start).seconds
        print(f"⏱ Total time: {elapsed}s")
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
