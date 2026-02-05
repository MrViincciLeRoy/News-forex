#!/usr/bin/env python3
"""
Master Report Generator
Orchestrates the complete workflow:
1. Run enhanced comprehensive pipeline
2. Generate section JSON files
3. Create visualizations
4. Generate AI-powered HTML report with local LLM
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Master Report Generator - Complete Analysis Pipeline + HTML Report'
    )
    parser.add_argument(
        '--date',
        default='2024-11-01',
        help='Analysis date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--event',
        default='Non-Farm Payrolls',
        help='Event name (e.g., "Non-Farm Payrolls", "CPI", "FOMC")'
    )
    parser.add_argument(
        '--symbols',
        default=None,
        help='Comma-separated symbols (leave empty for auto-detect)'
    )
    parser.add_argument(
        '--output-dir',
        default='master_report_output',
        help='Output directory for all files'
    )
    parser.add_argument(
        '--max-articles',
        type=int,
        default=20,
        help='Maximum news articles to fetch'
    )
    parser.add_argument(
        '--skip-pipeline',
        action='store_true',
        help='Skip pipeline and use existing JSON (for testing)'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("MASTER REPORT GENERATOR")
    print("="*80)
    print(f"Date: {args.date}")
    print(f"Event: {args.event}")
    print(f"Output: {args.output_dir}")
    print(f"Max Articles: {args.max_articles}")
    print("="*80)
    
    # Parse symbols if provided
    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
    
    # Step 1: Run Enhanced Pipeline
    if not args.skip_pipeline:
        print("\n" + "="*80)
        print("STEP 1: RUNNING ENHANCED COMPREHENSIVE PIPELINE")
        print("="*80)
        
        try:
            from comprehensive_pipeline_enhanced import EnhancedComprehensivePipeline
            
            pipeline = EnhancedComprehensivePipeline(
                output_dir=args.output_dir,
                enable_hf=True,
                enable_viz=True,
                max_articles=args.max_articles
            )
            
            results = pipeline.analyze(
                date=args.date,
                event_name=args.event,
                symbols=symbols
            )
            
            main_json = results['report_file']
            
            print(f"\n✅ Pipeline Complete!")
            print(f"Main JSON: {main_json}")
            print(f"Section JSONs: {pipeline.sections_dir}")
            print(f"Visualizations: {pipeline.viz_dir}")
            
        except Exception as e:
            print(f"\n❌ Pipeline Error: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("\n⏭️  Skipping pipeline (using existing data)")
        
        # Find most recent JSON
        import glob
        json_files = glob.glob(f'{args.output_dir}/**/comprehensive_*.json', recursive=True)
        
        if not json_files:
            print(f"❌ No JSON files found in {args.output_dir}")
            sys.exit(1)
        
        main_json = max(json_files, key=os.path.getctime)
        print(f"Using: {main_json}")
    
    # Step 2: Generate AI-Powered HTML Report
    print("\n" + "="*80)
    print("STEP 2: GENERATING AI-POWERED HTML REPORT")
    print("="*80)
    
    try:
        from local_llm_report_generator import LocalLLMReportGenerator
        
        generator = LocalLLMReportGenerator()
        
        sections_dir = Path(args.output_dir) / 'sections'
        viz_dir = Path(args.output_dir) / 'visualizations'
        
        html_file = generator.generate_comprehensive_report(
            json_file=main_json,
            sections_dir=sections_dir,
            viz_dir=viz_dir
        )
        
        print(f"\n✅ HTML Report Complete!")
        print(f"Report: {html_file}")
        
    except Exception as e:
        print(f"\n❌ Report Generation Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Final Summary
    print("\n" + "="*80)
    print("✅ MASTER REPORT GENERATION COMPLETE")
    print("="*80)
    print(f"\n📂 Output Directory: {args.output_dir}")
    print(f"\n📄 Files Generated:")
    print(f"  • Main JSON: {main_json}")
    print(f"  • HTML Report: {html_file}")
    print(f"  • Section JSONs: {sections_dir}")
    print(f"  • Visualizations: {viz_dir}")
    
    print(f"\n🌐 Open Report:")
    print(f"  file://{os.path.abspath(html_file)}")
    
    print("\n" + "="*80)
    print("💡 Tip: The HTML report includes:")
    print("  ✓ AI-powered analysis of each section")
    print("  ✓ Interactive charts and visualizations")
    print("  ✓ Executive summary with key findings")
    print("  ✓ Complete raw data for reference")
    print("="*80)


if __name__ == "__main__":
    main()
