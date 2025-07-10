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

# Ensure rankings is always a list, never None
def safe_get_rankings():
    rankings = latest_results.get("rankings", [])
    return rankings if rankings is not None else []

def safe_get_companies():
    companies = latest_results.get("companies", [])
    return companies if companies is not None else []

logger = logging.getLogger(__name__)

# --- API Endpoints ---

@app.get("/api/debug/data")
async def debug_data():
    """Debug endpoint to check what data is stored."""
    companies = safe_get_companies()
    rankings = safe_get_rankings()
    
    return {
        "companies_count": len(companies),
        "companies_sample": [ec.company.name for ec in companies[:5]] if companies else [],
        "rankings_count": len(rankings),
        "rankings_sample": [r.profile.name for r in rankings[:5]] if rankings else [],
        "last_job_id": latest_results.get("last_job_id"),
        "companies_structure": str(type(companies[0])) if companies else "No companies",
        "first_company_fields": list(vars(companies[0].company).keys()) if companies else "No companies"
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/checkpoints")
async def get_available_checkpoints():
    """Get list of available checkpoints."""
    try:
        jobs = checkpoint_manager.list_active_jobs()
        checkpoints = []
        
        for job_progress in jobs:
            job_id = job_progress['job_id']
            companies_stage = job_progress['stages'].get('companies', {})
            
            if companies_stage.get('completed'):
                checkpoints.append({
                    "id": job_id,
                    "created_at": companies_stage['timestamp'].isoformat() if hasattr(companies_stage['timestamp'], 'isoformat') else str(companies_stage['timestamp']),
                    "companies_count": companies_stage.get('data_count', 0),
                    "completion_percentage": job_progress['completion_percentage'],
                    "stages_completed": len([s for s in job_progress['stages'].values() if s.get('completed')])
                })
        
        # Sort by creation date, newest first
        checkpoints.sort(key=lambda x: x['created_at'], reverse=True)
        return checkpoints
        
    except Exception as e:
        logger.error(f"Error fetching checkpoints: {e}")
        return []

# --- Dashboard Endpoints ---

@app.get("/api/dashboard/stats", response_model=DashboardStats)
async def get_dashboard_stats():
    """Get basic dashboard statistics from checkpointed results."""
    companies = safe_get_companies()
    rankings = safe_get_rankings()

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

@app.post("/api/pipeline/resume/{checkpoint_id}", response_model=PipelineJobResponse)
async def resume_pipeline_from_checkpoint(
    checkpoint_id: str,
    pipeline_service: InitiationPipeline = Depends(get_pipeline_service),
    ranking_service: FounderRankingService = Depends(get_ranking_service),
):
    """Resume pipeline from a specific checkpoint."""
    try:
        # Try to resume from the specified checkpoint
        resume_data = checkpoint_manager.resume_pipeline(checkpoint_id)
        
        if resume_data is None:
            raise HTTPException(status_code=404, detail=f"Checkpoint {checkpoint_id} not found or empty")
        
        if resume_data.get('stage') == 'complete':
            # Pipeline already complete, just load results
            result = resume_data['data']
            enriched_companies = result.get('companies', [])
            rankings = result.get('rankings', [])
            
            # Update latest results for API access
            latest_results["companies"] = enriched_companies
            latest_results["rankings"] = rankings
            latest_results["last_job_id"] = checkpoint_id
            
            return PipelineJobResponse(
                jobId=checkpoint_id,
                status="completed",
                companiesFound=len(enriched_companies),
                foundersFound=len(rankings) if rankings else sum(len(ec.profiles) for ec in enriched_companies),
                message="Pipeline already complete. Results loaded from checkpoint."
            )
        
        # Resume from partial completion
        stage = resume_data['stage']
        logger.info(f"🔄 Resuming pipeline from checkpoint {checkpoint_id} at stage: {stage}")
        
        # Run remaining pipeline stages
        result = await checkpointed_runner.resume_checkpointed_pipeline(
            checkpoint_id=checkpoint_id,
            pipeline_service=pipeline_service,
            ranking_service=ranking_service,
            resume_data=resume_data
        )
        
        # Extract results
        enriched_companies = result.get('companies', [])
        rankings = result.get('rankings', [])
        
        # Update latest results for API access
        latest_results["companies"] = enriched_companies
        latest_results["rankings"] = rankings
        latest_results["last_job_id"] = checkpoint_id
        
        return PipelineJobResponse(
            jobId=checkpoint_id,
            status="completed",
            companiesFound=len(enriched_companies),
            foundersFound=len(rankings) if rankings else sum(len(ec.profiles) for ec in enriched_companies),
            message=f"Pipeline resumed from checkpoint and completed successfully."
        )
        
    except Exception as e:
        logger.error(f"Resume from checkpoint failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/pipeline/run", response_model=PipelineJobResponse)
async def run_complete_pipeline(
    params: YearBasedRequest,
    pipeline_service: InitiationPipeline = Depends(get_pipeline_service),
    ranking_service: FounderRankingService = Depends(get_ranking_service),
):
    """Year-based endpoint that runs the complete pipeline including ranking."""
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
        
        # Run complete checkpointed pipeline including ranking
        result = await checkpointed_runner.run_checkpointed_pipeline(
            pipeline_service=pipeline_service,
            ranking_service=ranking_service,
            params=pipeline_params,
            force_restart=False
        )
        
        # Extract results
        enriched_companies = result.get('companies', [])
        rankings = result.get('rankings', [])
        
        # Update latest results for API access
        latest_results["companies"] = enriched_companies
        latest_results["rankings"] = rankings
        latest_results["last_job_id"] = result['job_id']
        
        return PipelineJobResponse(
            jobId=result['job_id'],
            status="completed",
            companiesFound=len(enriched_companies),
            foundersFound=len(rankings) if rankings else sum(len(ec.profiles) for ec in enriched_companies),
            message="Complete pipeline with ranking finished successfully."
        )
    except Exception as e:
        logger.error(f"Complete pipeline failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/companies/discover", response_model=PipelineJobResponse)
async def discover_companies_only(
    params: CompanyDiscoveryRequest,
    pipeline_service: InitiationPipeline = Depends(get_pipeline_service),
):
    """Discover companies only (without ranking) using checkpointed pipeline."""
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
        
        # Run discovery-only pipeline
        result = await checkpointed_runner.run_checkpointed_pipeline(
            pipeline_service=pipeline_service,
            ranking_service=None,  # Skip ranking for discovery-only
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
            message="Company discovery complete (ranking skipped)."
        )
    except Exception as e:
        logger.error(f"Discovery failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/companies")
async def get_companies():
    """Fetch discovered companies from latest checkpointed results."""
    enriched_companies = safe_get_companies()
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
            "founders": getattr(company, 'founders', []) or [],
            "source": urlparse(company.source_url).hostname if company.source_url else "N/A"
        })
    return formatted_companies

@app.get("/api/companies/export")
async def export_companies():
    """Export companies to CSV file in outputs folder."""
    try:
        enriched_companies = safe_get_companies()
        logger.info(f"📊 Export request - Found {len(enriched_companies)} companies in latest_results")
        
        if not enriched_companies:
            logger.warning("No companies found for export")
            raise HTTPException(status_code=404, detail="No companies discovered yet. Run a discovery first.")

        company_records = []
        
        for i, ec in enumerate(enriched_companies):
            try:
                c = ec.company
                record = {
                    "company_name": c.name,
                    "description": c.description,
                    "short_description": getattr(c, 'short_description', None),
                    "website": str(c.website) if c.website else "",
                    "founded_year": c.founded_year,
                    "ai_focus": c.ai_focus,
                    "sector": c.sector,
                    "funding_total_usd": c.funding_total_usd,
                    "funding_stage": c.funding_stage.value if c.funding_stage else None,
                    "city": c.city,
                    "region": getattr(c, 'region', None),
                    "country": c.country,
                    "founders": "; ".join(getattr(c, 'founders', []) or []),
                    "founders_count": len(getattr(c, 'founders', []) or []),
                    "investors": "; ".join(getattr(c, 'investors', []) or []),
                    "categories": "; ".join(getattr(c, 'categories', []) or []),
                    "linkedin_url": getattr(c, 'linkedin_url', None),
                    "employee_count": getattr(c, 'employee_count', None),
                    "revenue_millions": getattr(c, 'revenue_millions', None),
                    "valuation_millions": getattr(c, 'valuation_millions', None),
                    "last_funding_date": getattr(c, 'last_funding_date', None),
                    "tech_stack": "; ".join(getattr(c, 'tech_stack', []) or []),
                    "competitors": "; ".join(getattr(c, 'competitors', []) or []),
                    "source_url": c.source_url,
                    "extraction_date": getattr(c, 'extraction_date', None),
                    "confidence_score": c.confidence_score,
                    "linkedin_profiles_found": len(ec.profiles)
                }
                company_records.append(record)
            except Exception as company_error:
                logger.error(f"Error processing company {i}: {company_error}")
                continue
        
        logger.info(f"📈 Created {len(company_records)} company records for CSV")
        
        # Save to outputs folder
        import os
        output_dir = "/Users/veerdosi/Documents/code/github/initiation-pipeline/output"
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"companies_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(output_dir, filename)
        
        df = pd.DataFrame(company_records)
        df.to_csv(filepath, index=False)
        
        logger.info(f"📄 Saved CSV to: {filepath}")
        
        return {
            "status": "success",
            "message": f"Companies exported successfully to {filename}",
            "filepath": filepath,
            "companies_count": len(company_records)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Companies export failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

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
    rankings = safe_get_rankings()
    
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
    """Export founder rankings to CSV file in outputs folder."""
    try:
        rankings = safe_get_rankings()
        logger.info(f"📊 Export request - Found {len(rankings)} rankings in latest_results")
        
        if not rankings:
            logger.warning("No rankings found for export")
            raise HTTPException(status_code=404, detail="No rankings available to export. Run a ranking job first.")

        ranking_records = []
        for i, r in enumerate(rankings):
            try:
                record = {
                    "founder_name": r.profile.name,
                    "company_name": r.profile.company_name,
                    "linkedin_url": r.profile.linkedin_url,
                    "l_level": r.classification.level.value,
                    "confidence_score": r.classification.confidence_score,
                    "reasoning": r.classification.reasoning,
                    "evidence": " | ".join(r.classification.evidence),
                    "verification_sources": " | ".join(r.classification.verification_sources),
                    "timestamp": r.timestamp
                }
                ranking_records.append(record)
            except Exception as ranking_error:
                logger.error(f"Error processing ranking {i}: {ranking_error}")
                continue
        
        logger.info(f"📈 Created {len(ranking_records)} ranking records for CSV")
        
        # Save to outputs folder
        import os
        output_dir = "/Users/veerdosi/Documents/code/github/initiation-pipeline/output"
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"founder_rankings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(output_dir, filename)
        
        df = pd.DataFrame(ranking_records)
        df.to_csv(filepath, index=False)
        
        logger.info(f"📄 Saved CSV to: {filepath}")
        
        return {
            "status": "success",
            "message": f"Founder rankings exported successfully to {filename}",
            "filepath": filepath,
            "rankings_count": len(ranking_records)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Rankings export failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")