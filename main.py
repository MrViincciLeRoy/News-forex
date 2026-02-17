"""
Financial Analysis Pipeline - Main API
Pre/Post Event Report Generation System
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import json
from pathlib import Path

from enhanced_pre_event_analyzer import PreEventAnalyzer
from enhanced_post_event_analyzer import PostEventAnalyzer
from enhanced_report_generator import EnhancedReportGenerator

app = FastAPI(
    title="Financial Analysis Pipeline",
    description="Generate comprehensive pre/post event financial reports",
    version="2.0"
)

class AnalysisRequest(BaseModel):
    date: str
    event_name: str
    symbols: Optional[List[str]] = None
    max_articles: int = 30
    report_type: str = "pre"  # "pre" or "post"

class AnalysisResponse(BaseModel):
    status: str
    analysis_id: str
    report_file: Optional[str] = None
    sections_generated: int
    message: str

@app.get("/")
def root():
    return {
        "service": "Financial Analysis Pipeline",
        "version": "2.0",
        "endpoints": {
            "pre_analysis": "/api/pre-event",
            "post_analysis": "/api/post-event",
            "status": "/api/status/{analysis_id}",
            "download": "/api/download/{filename}"
        }
    }

@app.post("/api/pre-event", response_model=AnalysisResponse)
async def generate_pre_event_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Generate comprehensive PRE-event analysis report"""
    
    try:
        analyzer = PreEventAnalyzer(
            event_date=request.date,
            event_name=request.event_name,
            symbols=request.symbols,
            max_articles=request.max_articles
        )
        
        # Run analysis
        results = await analyzer.run_full_analysis()
        
        # Generate report
        generator = EnhancedReportGenerator()
        report_file = generator.generate_pre_event_pdf(results)
        
        return AnalysisResponse(
            status="success",
            analysis_id=results['analysis_id'],
            report_file=report_file,
            sections_generated=len(results['sections']),
            message=f"Pre-event analysis complete for {request.event_name}"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/post-event", response_model=AnalysisResponse)
async def generate_post_event_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Generate comprehensive POST-event analysis report"""
    
    try:
        analyzer = PostEventAnalyzer(
            event_date=request.date,
            event_name=request.event_name,
            symbols=request.symbols,
            max_articles=request.max_articles
        )
        
        # Run analysis
        results = await analyzer.run_full_analysis()
        
        # Generate report
        generator = EnhancedReportGenerator()
        report_file = generator.generate_post_event_pdf(results)
        
        return AnalysisResponse(
            status="success",
            analysis_id=results['analysis_id'],
            report_file=report_file,
            sections_generated=len(results['sections']),
            message=f"Post-event analysis complete for {request.event_name}"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download/{filename}")
async def download_report(filename: str):
    """Download generated report"""
    
    file_path = Path("/mnt/user-data/outputs") / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type='application/pdf'
    )

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
