"""FastAPI main application for Initiation Pipeline."""

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
import asyncio
import csv
import io
import pandas as pd
from typing import List, Optional
from datetime import datetime

from ..services.ranking import FounderRankingService, FounderProfile
from ..services.company_discovery import ExaCompanyDiscovery
from ..services.profile_enrichment import LinkedInEnrichmentService
from .models import *
from .dependencies import get_ranking_service, get_discovery_service, get_enrichment_service

app = FastAPI(
    title="Initiation Pipeline API",
    description="AI-powered founder discovery and ranking system",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Dashboard endpoints
@app.get("/api/dashboard/stats", response_model=DashboardStats)
async def get_dashboard_stats():
    """Get dashboard statistics."""
    # TODO: Implement actual stats from database
    return DashboardStats(
        totalCompanies=0,
        totalFounders=0,
        rankedFounders=0,
        avgConfidenceScore=0.0,
        levelDistribution={},
        recentActivity=[]
    )

# Company discovery endpoints
@app.post("/api/companies/discover", response_model=DiscoveryJobResponse)
async def start_company_discovery(
    params: CompanyDiscoveryRequest,
    discovery_service: ExaCompanyDiscovery = Depends(get_discovery_service)
):
    """Start company discovery process."""
    try:
        # Start discovery in background
        task_id = f"discovery_{datetime.now().timestamp()}"
        
        # Run discovery using the new comprehensive method
        companies = await discovery_service.discover_companies(
            limit=params.limit,
            categories=params.categories,
            regions=params.regions,
            sources=params.sources
        )
        
        return DiscoveryJobResponse(
            jobId=task_id,
            status="completed",
            companiesFound=len(companies),
            message=f"Discovered {len(companies)} companies"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/companies", response_model=List[CompanyResponse])
async def get_companies():
    """Get all discovered companies."""
    # TODO: Implement database retrieval
    return []

@app.get("/api/companies/export")
async def export_companies():
    """Export companies to CSV."""
    # TODO: Implement actual export from database
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["name", "description", "website", "founded_year", "funding_total", "location", "ai_category"])
    
    def iter_csv():
        output.seek(0)
        yield output.read()
    
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=companies.csv"}
    )

# Founder ranking endpoints  
@app.post("/api/founders/rank", response_model=RankingJobResponse)
async def start_founder_ranking(
    file: UploadFile = File(...),
    minConfidence: float = 0.75,
    batchSize: int = 5,
    ranking_service: FounderRankingService = Depends(get_ranking_service)
):
    """Start founder ranking process."""
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Read CSV file
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        # Convert to FounderProfile objects
        profiles = []
        for _, row in df.iterrows():
            if pd.notna(row.get('person_name')) and pd.notna(row.get('company_name')):
                profile = FounderProfile.from_csv_row(row.to_dict())
                profiles.append(profile)
        
        # Start ranking in background
        task_id = f"ranking_{datetime.now().timestamp()}"
        
        # Run ranking
        rankings = await ranking_service.rank_founders_batch(
            profiles[:50],  # Limit for demo
            batch_size=batchSize
        )
        
        # Filter by confidence
        high_confidence = [r for r in rankings if r.classification.confidence_score >= minConfidence]
        
        return RankingJobResponse(
            jobId=task_id,
            status="completed",
            foundersRanked=len(rankings),
            highConfidenceCount=len(high_confidence),
            message=f"Ranked {len(rankings)} founders, {len(high_confidence)} high confidence"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/founders/rankings", response_model=List[FounderRankingResponse])
async def get_founder_rankings():
    """Get all founder rankings."""
    # TODO: Implement database retrieval
    return []

@app.get("/api/founders/rankings/export")
async def export_founder_rankings():
    """Export founder rankings to CSV."""
    # TODO: Implement actual export from database
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "name", "company", "level", "confidence_score", 
        "reasoning", "evidence", "verification_sources", "timestamp"
    ])
    
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=founder_rankings.csv"}
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
