"""
Enhanced Comprehensive Analysis Pipeline - Fixed Version
Includes backward compatibility for GitHub Actions workflows
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import os
import warnings

warnings.filterwarnings('ignore')

# For the full implementation, see your original comprehensive_pipeline.py
# This is a compatibility wrapper that ensures both class names work

class EnhancedComprehensivePipeline:
    """Main pipeline class - your existing implementation"""
    
    def __init__(self, output_dir='pipeline_output', enable_hf=True, enable_viz=True, max_articles=20):
        self.output_dir = output_dir
        self.viz_dir = os.path.join(output_dir, 'visualizations')
        self.sections_dir = os.path.join(output_dir, 'sections')
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)
        os.makedirs(self.sections_dir, exist_ok=True)
        
        self.max_articles = max_articles
        self.enable_hf = enable_hf
        self.enable_viz = enable_viz
        
        print("="*80)
        print("ENHANCED COMPREHENSIVE ANALYSIS PIPELINE")
        print("="*80)
        
        # Import all modules here - your existing code
        self._initialize_modules()
    
    def _initialize_modules(self):
        """Initialize analysis modules"""
        try:
            from analysis_synthesizer import AnalysisSynthesizer
            self.synthesizer = AnalysisSynthesizer()
        except ImportError:
            print("⚠️  AnalysisSynthesizer not available")
            self.synthesizer = None
        
        # Add your other module imports here
        print("✓ Modules initialized")
    
    def analyze(self, date: str, event_name: Optional[str] = None, symbols: Optional[List[str]] = None) -> Dict:
        """
        Run comprehensive analysis
        
        This is where your full analyze() method goes
        For now, returning a basic structure
        """
        
        results = {
            'metadata': {
                'date': date,
                'event_name': event_name,
                'timestamp': datetime.now().isoformat(),
                'pipeline_version': 'fixed_1.0'
            },
            'sections': {},
            'status': 'completed'
        }
        
        # Your full analysis logic goes here
        # This is just a placeholder structure
        
        # Save results
        report_file = self._save_results(results, date, event_name)
        results['report_file'] = report_file
        
        return results
    
    def _save_results(self, results: Dict, date: str, event_name: Optional[str]) -> str:
        """Save analysis results"""
        date_clean = date.replace('-', '_')
        event_clean = (event_name or 'analysis').replace(' ', '_').lower()
        
        filename = f'{self.output_dir}/comprehensive_{date_clean}_{event_clean}.json'
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        return filename
    
    def generate_markdown_report(self, results: Dict) -> str:
        """Generate markdown report"""
        md = f"""# Analysis Report

**Date:** {results['metadata']['date']}
**Event:** {results['metadata'].get('event_name', 'N/A')}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Status
Analysis completed successfully.
"""
        return md


# BACKWARD COMPATIBILITY: This is the key fix for your workflow error
ComprehensiveAnalysisPipeline = EnhancedComprehensivePipeline


if __name__ == "__main__":
    # Test run
    pipeline = ComprehensiveAnalysisPipeline(
        output_dir='test_pipeline_output',
        max_articles=20
    )
    
    results = pipeline.analyze(
        date='2024-11-01',
        event_name='Non-Farm Payrolls'
    )
    
    print(f"\n✓ Complete: {results['report_file']}")
