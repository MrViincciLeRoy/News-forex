"""
Test Script - Quick verification of the analysis pipeline
"""

import asyncio
from enhanced_pre_event_analyzer import PreEventAnalyzer
from enhanced_post_event_analyzer import PostEventAnalyzer
from enhanced_report_generator import EnhancedReportGenerator

async def test_pre_event():
    """Test pre-event analysis"""
    print("\n" + "="*80)
    print("TESTING PRE-EVENT ANALYSIS")
    print("="*80 + "\n")
    
    analyzer = PreEventAnalyzer(
        event_date="2024-11-01",
        event_name="Non-Farm Payrolls Test",
        max_articles=10
    )
    
    try:
        results = await analyzer.run_full_analysis()
        
        print(f"\n✓ Analysis completed successfully!")
        print(f"  Analysis ID: {results['analysis_id']}")
        print(f"  Sections generated: {len(results['sections'])}")
        
        # Generate PDF
        generator = EnhancedReportGenerator()
        pdf_file = generator.generate_pre_event_pdf(results)
        print(f"  PDF Report: {pdf_file}")
        
        return True
    except Exception as e:
        print(f"\n✗ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_post_event():
    """Test post-event analysis"""
    print("\n" + "="*80)
    print("TESTING POST-EVENT ANALYSIS")
    print("="*80 + "\n")
    
    analyzer = PostEventAnalyzer(
        event_date="2024-11-01",
        event_name="Non-Farm Payrolls Test",
        max_articles=10
    )
    
    try:
        results = await analyzer.run_full_analysis()
        
        print(f"\n✓ Analysis completed successfully!")
        print(f"  Analysis ID: {results['analysis_id']}")
        print(f"  Sections generated: {len(results['sections'])}")
        
        # Generate PDF
        generator = EnhancedReportGenerator()
        pdf_file = generator.generate_post_event_pdf(results)
        print(f"  PDF Report: {pdf_file}")
        
        return True
    except Exception as e:
        print(f"\n✗ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests"""
    print("\n🚀 Starting Financial Analysis Pipeline Tests\n")
    
    # Test pre-event
    pre_success = await test_pre_event()
    
    # Test post-event
    post_success = await test_post_event()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Pre-Event Analysis: {'✓ PASSED' if pre_success else '✗ FAILED'}")
    print(f"Post-Event Analysis: {'✓ PASSED' if post_success else '✗ FAILED'}")
    print("="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
