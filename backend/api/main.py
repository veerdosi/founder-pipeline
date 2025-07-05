"""
FastAPI application to power the Initiation Pipeline web UI.
This version is updated to match the API endpoints expected by the frontend.
"""

import io
import csv
import logging
from typing import List, Dict, Any
from datetime import datetime
from urllib.parse import urlparse

import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .models import CompanyDiscoveryRequest, DashboardStats, PipelineJobResponse
from .dependencies import get_pipeline_service, get_ranking_service
from ..core.discovery import InitiationPipeline
from ..core.ranking import FounderRankingService
from ..core.ranking.models import FounderProfile
from ..models import EnrichedCompany

# --- Application Setup ---
app = FastAPI(
    title="Initiation Pipeline API",
    description="AI-powered founder discovery and ranking system.",
    version="1.1.0"
)

# --- CORS Middleware ---
# Allows the frontend (running on localhost:3000) to communicate with the backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- In-Memory Data Store ---
# A simple dictionary to store results. In a production app, use a database (e.g., PostgreSQL, Redis).
results_store: Dict[str, Any] = {
    "companies": [],
    "rankings": []
}

logger = logging.getLogger(__name__)

# --- API Endpoints ---

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# --- Dashboard Endpoints ---

@app.get("/api/dashboard/stats", response_model=DashboardStats)
async def get_dashboard_stats():
    """
    Get basic dashboard statistics from the current state.
    This is now connected to the in-memory store.
    """
    companies = results_store.get("companies", [])
    rankings = results_store.get("rankings", [])

    # Calculate level distribution
    level_dist = {}
    if rankings:
        for r in rankings:
            level = r.classification.level.value
            level_dist[level] = level_dist.get(level, 0) + 1

    return DashboardStats(
        totalCompanies=len(companies),
        totalFounders=sum(len(ec.profiles) for ec in companies),
        avgConfidenceScore=sum(r.classification.confidence_score for r in rankings) / len(rankings) if rankings else 0.0,
        levelDistribution=level_dist,
        recentActivity=[] # Placeholder for recent activity log
    )

# --- Company Discovery Endpoints ---

@app.post("/api/companies/discover", response_model=PipelineJobResponse)
async def discover_companies(
    params: CompanyDiscoveryRequest,
    pipeline_service: InitiationPipeline = Depends(get_pipeline_service),
):
    """
    Endpoint for the frontend to discover companies.
    This is a long-running task. Results are stored in memory.
    """
    try:
        enriched_companies: List[EnrichedCompany] = await pipeline_service.run_complete_pipeline_with_date_range(
            company_limit=params.limit,
            categories=params.categories,
            regions=params.regions,
            sources=params.sources,
            founded_after=params.founded_after,
            founded_before=params.founded_before
        )
        # Store results in our simple in-memory store
        results_store["companies"] = enriched_companies
        
        return PipelineJobResponse(
            jobId=f"job_{datetime.now().timestamp()}",
            status="completed",
            companiesFound=len(enriched_companies),
            foundersFound=sum(len(ec.profiles) for ec in enriched_companies),
            message="Discovery and enrichment complete."
        )
    except Exception as e:
        logger.error(f"Discovery failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/companies")
async def get_companies():
    """
    Endpoint to fetch the list of discovered companies for the UI table.
    Formats the data to match the frontend's expected structure.
    """
    enriched_companies = results_store.get("companies", [])
    formatted_companies = []
    for ec in enriched_companies:
        company = ec.company
        formatted_companies.append({
            "id": company.uuid,
            "name": company.name,
            "description": company.description or "No description available.",
            "website": str(company.website) if company.website else "",
            "aiCategory": company.ai_focus or "N/A",
            "fundingTotal": company.funding_total_usd,
            "location": f"{company.city}, {company.country}" if company.city else "N/A",
            "founders": [p.person_name for p in ec.profiles] if ec.profiles else [],
            "source": urlparse(company.source_url).hostname if company.source_url else "N/A"
        })
    return formatted_companies

@app.get("/api/companies/export")
async def export_companies():
    """
    Endpoint to export discovered companies to a CSV file.
    Uses the data stored in memory from the last discovery run.
    """
    enriched_companies = results_store.get("companies", [])
    if not enriched_companies:
        raise HTTPException(status_code=404, detail="No companies discovered yet. Run a discovery first.")

    output = io.StringIO()
    # Using pandas for robust CSV generation
    company_records = []
    for ec in enriched_companies:
        c = ec.company
        company_records.append({
            "company_name": c.name,
            "description": c.description,
            "website": str(c.website) if c.website else "",
            "founded_year": c.founded_year,
            "ai_focus": c.ai_focus,
            "sector": c.sector,
            "funding_total_usd": c.funding_total_usd,
            "funding_stage": c.funding_stage,
            "city": c.city,
            "country": c.country,
            "founders_count": len(ec.profiles)
        })
    df = pd.DataFrame(company_records)
    df.to_csv(output, index=False)
    
    output.seek(0)
    filename = f"companies_export_{datetime.now().strftime('%Y%m%d')}.csv"
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

# --- Founder Ranking Endpoints ---

@app.post("/api/founders/rank", response_model=PipelineJobResponse)
async def rank_founders_from_file(
    ranking_service: FounderRankingService = Depends(get_ranking_service),
    file: UploadFile = File(...)
):
    """
    Ranks founders from an uploaded CSV file.
    """
    try:
        # Read and parse the uploaded CSV file
        contents = await file.read()
        decoded_content = contents.decode('utf-8')
        csv_reader = csv.DictReader(io.StringIO(decoded_content))
        
        founder_profiles = [FounderProfile.from_csv_row(row) for row in csv_reader]
        
        if not founder_profiles:
            raise HTTPException(status_code=400, detail="CSV file is empty or in the wrong format.")

        # Rank the founders
        rankings = await ranking_service.rank_founders_batch(
            founder_profiles,
            batch_size=5,
            use_enhanced=True  # Use enhanced ranking with L-level validation
        )
        
        # Store results in memory
        results_store["rankings"] = rankings
        
        return PipelineJobResponse(
            jobId=f"rank_job_{datetime.now().timestamp()}",
            status="completed",
            companiesFound=0,
            foundersFound=len(rankings),
            message=f"Ranking complete for {len(rankings)} founders."
        )
    except Exception as e:
        logger.error(f"Ranking from file failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process file: {e}")

@app.get("/api/founders/rankings")
async def get_rankings():
    """
    Endpoint to fetch the list of ranked founders for the UI table.
    """
    rankings = results_store.get("rankings", [])
    formatted_rankings = []
    for r in rankings:
        formatted_rankings.append({
            "id": r.profile.linkedin_url or r.profile.name,
            "name": r.profile.name,
            "company": r.profile.company_name,
            "level": r.classification.level.value,
            "confidenceScore": r.classification.confidence_score,
            "reasoning": r.classification.reasoning,
            "evidence": r.classification.evidence,
            "verificationSources": r.classification.verification_sources,
            "timestamp": r.timestamp
        })
    return formatted_rankings

@app.get("/api/founders/rankings/export")
async def export_rankings():
    """
    Endpoint to export founder rankings to a CSV file.
    """
    rankings = results_store.get("rankings", [])
    if not rankings:
        raise HTTPException(status_code=404, detail="No rankings available to export. Run a ranking job first.")

    output = io.StringIO()
    ranking_records = []
    for r in rankings:
        ranking_records.append({
            "founder_name": r.profile.name,
            "company_name": r.profile.company_name,
            "linkedin_url": r.profile.linkedin_url,
            "l_level": r.classification.level.value,
            "confidence_score": r.classification.confidence_score,
            "reasoning": r.classification.reasoning,
            "evidence": " | ".join(r.classification.evidence),
        })
    df = pd.DataFrame(ranking_records)
    df.to_csv(output, index=False)
    
    output.seek(0)
    filename = f"founder_rankings_{datetime.now().strftime('%Y%m%d')}.csv"
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )