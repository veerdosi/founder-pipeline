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

from .models import CompanyDiscoveryRequest, DashboardStats, PipelineJobResponse, SimpleDateRangeRequest, YearBasedRequest
from .dependencies import get_pipeline_service, get_ranking_service
from ..core.discovery import InitiationPipeline
from ..core.ranking import FounderRankingService
from ..core.ranking.models import FounderProfile
from ..models import EnrichedCompany
from ..utils.checkpoint_manager import checkpointed_runner, checkpoint_manager

# --- Application Setup ---
app = FastAPI(
    title="Initiation Pipeline API",
    description="AI-powered founder discovery and ranking system.",
    version="1.1.0"
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Job-Based Data Store ---
# Track active jobs and their results with checkpointing
active_jobs: Dict[str, str] = {}  # {frontend_job_id: checkpoint_job_id}
latest_results: Dict[str, Any] = {
    "companies": [],
    "rankings": [],
    "last_job_id": None
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
    """Get basic dashboard statistics from checkpointed results."""
    companies = latest_results.get("companies", [])
    rankings = latest_results.get("rankings", [])

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
        recentActivity=[]
    )

# --- Company Discovery Endpoints ---

@app.post("/api/pipeline/run", response_model=PipelineJobResponse)
async def run_simple_pipeline(
    params: YearBasedRequest,
    pipeline_service: InitiationPipeline = Depends(get_pipeline_service),
):
    """Year-based endpoint that takes a single year and runs the full pipeline."""
    try:
        # Convert to standard discovery request
        discovery_params = params.to_discovery_request()
        
        # Convert to pipeline parameters
        pipeline_params = {
            'limit': discovery_params.limit,
            'categories': discovery_params.categories,
            'regions': discovery_params.regions,
            'sources': discovery_params.sources,
            'founded_after': discovery_params.founded_after,
            'founded_before': discovery_params.founded_before
        }
        
        # Run checkpointed pipeline
        result = await checkpointed_runner.run_checkpointed_pipeline(
            pipeline_service=pipeline_service,
            ranking_service=None,  # Not needed for discovery
            params=pipeline_params,
            force_restart=False
        )
        
        # Extract companies from result
        enriched_companies = result.get('companies', [])
        
        # Update latest results for API access
        latest_results["companies"] = enriched_companies
        latest_results["last_job_id"] = result['job_id']
        
        return PipelineJobResponse(
            jobId=result['job_id'],
            status="completed",
            companiesFound=len(enriched_companies),
            foundersFound=sum(len(ec.profiles) for ec in enriched_companies),
            message="Pipeline complete with checkpointing."
        )
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/companies/discover", response_model=PipelineJobResponse)
async def discover_companies(
    params: CompanyDiscoveryRequest,
    pipeline_service: InitiationPipeline = Depends(get_pipeline_service),
):
    """Discover companies using checkpointed pipeline for reliability."""
    try:
        # Convert request to pipeline parameters
        pipeline_params = {
            'limit': params.limit,
            'categories': params.categories,
            'regions': params.regions,
            'sources': params.sources,
            'founded_after': params.founded_after,
            'founded_before': params.founded_before
        }
        
        # Run checkpointed pipeline
        result = await checkpointed_runner.run_checkpointed_pipeline(
            pipeline_service=pipeline_service,
            ranking_service=None,  # Not needed for discovery
            params=pipeline_params,
            force_restart=False
        )
        
        # Extract companies from result
        enriched_companies = result.get('companies', [])
        
        # Update latest results for API access
        latest_results["companies"] = enriched_companies
        latest_results["last_job_id"] = result['job_id']
        
        return PipelineJobResponse(
            jobId=result['job_id'],
            status="completed",
            companiesFound=len(enriched_companies),
            foundersFound=sum(len(ec.profiles) for ec in enriched_companies),
            message="Discovery complete with checkpointing."
        )
    except Exception as e:
        logger.error(f"Discovery failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/companies")
async def get_companies():
    """Fetch discovered companies from latest checkpointed results."""
    enriched_companies = latest_results.get("companies", [])
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
    """Export companies from latest checkpointed results."""
    enriched_companies = latest_results.get("companies", [])
    if not enriched_companies:
        raise HTTPException(status_code=404, detail="No companies discovered yet. Run a discovery first.")

    output = io.StringIO()
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
    """Ranks founders from uploaded CSV with checkpointing."""
    try:
        contents = await file.read()
        decoded_content = contents.decode('utf-8')
        csv_reader = csv.DictReader(io.StringIO(decoded_content))
        
        founder_profiles = [FounderProfile.from_csv_row(row) for row in csv_reader]
        
        if not founder_profiles:
            raise HTTPException(status_code=400, detail="CSV file is empty or in the wrong format.")

        # Use checkpointed ranking service
        rankings = await ranking_service.rank_founders_batch(
            founder_profiles,
            batch_size=5,
            use_enhanced=True
        )
        
        # Store in latest results
        latest_results["rankings"] = rankings
        
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
    """Fetch ranked founders from latest checkpointed results."""
    rankings = latest_results.get("rankings", [])
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
    """Export founder rankings from latest checkpointed results."""
    rankings = latest_results.get("rankings", [])
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