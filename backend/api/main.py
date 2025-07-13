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

from .models import CompanyDiscoveryRequest, PipelineJobResponse, YearBasedRequest
from .dependencies import get_pipeline_service, get_ranking_service
from ..core.ranking import FounderRankingService
from ..models import LinkedInProfile
from ..core.analysis.market_analysis import PerplexityMarketAnalysis
from ..utils.checkpoint_manager import checkpointed_runner, checkpoint_manager
from ..core.config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(settings.log_file) if settings.log_file else logging.NullHandler()
    ]
)

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
    from backend.core.ranking.models import FounderRanking, LevelClassification, ExperienceLevel
    
    rankings = latest_results.get("rankings", [])
    
    # Only return rankings if pipeline is 100% complete
    if not rankings:
        try:
            jobs = checkpoint_manager.list_active_jobs()
            for job_progress in jobs:
                # Only load rankings if job is 100% complete
                if job_progress.get('completion_percentage', 0) == 100:
                    job_id = job_progress['job_id']
                    checkpoint_rankings = checkpoint_manager.load_checkpoint(job_id, 'rankings')
                    if checkpoint_rankings:
                        rankings = checkpoint_rankings
                        break
        except Exception as e:
            logger.warning(f"Failed to load rankings from checkpoint: {e}")
    
    # Convert EnrichedCompany objects to FounderRanking objects if needed
    if rankings and hasattr(rankings[0], 'profiles'):
        founder_rankings = []
        for enriched_company in rankings:
            for profile in enriched_company.profiles:
                # Check if profile has ranking data
                if hasattr(profile, 'l_level') and hasattr(profile, 'confidence_score'):
                    try:
                        level = ExperienceLevel(profile.l_level)
                        classification = LevelClassification(
                            level=level,
                            confidence_score=profile.confidence_score,
                            reasoning=getattr(profile, 'reasoning', ''),
                            evidence=getattr(profile, 'evidence', []),
                            verification_sources=getattr(profile, 'verification_sources', [])
                        )
                        founder_ranking = FounderRanking(
                            profile=profile,
                            classification=classification,
                            timestamp=getattr(profile, 'timestamp', ''),
                            processing_metadata={}
                        )
                        founder_rankings.append(founder_ranking)
                    except Exception as e:
                        logger.warning(f"Failed to convert profile to FounderRanking: {e}")
        rankings = founder_rankings
    
    return rankings if rankings is not None else []

def safe_get_companies():
    companies = latest_results.get("companies", [])
    
    # If no companies in memory, try to load from latest checkpoint
    if not companies:
        try:
            jobs = checkpoint_manager.list_active_jobs()
            for job_progress in jobs:
                job_id = job_progress['job_id']
                # Try to load companies from any completed stage (companies, enhanced_companies, profiles, or rankings)
                for stage in ['rankings', 'profiles', 'enhanced_companies', 'companies']:
                    if job_progress['stages'].get(stage, {}).get('completed'):
                        checkpoint_companies = checkpoint_manager.load_checkpoint(job_id, stage)
                        if checkpoint_companies:
                            companies = checkpoint_companies
                            break
                if companies:
                    break
        except Exception as e:
            logger.warning(f"Failed to load companies from checkpoint: {e}")
    
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
        "rankings_sample": [r.profile.person_name for r in rankings[:5]] if rankings else [],
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
            
            # Find all completed stages
            completed_stages = [(stage, data) for stage, data in job_progress['stages'].items() if data.get('completed')]
            
            if completed_stages:
                # Get foundation year (default to 2025 for current checkpoints)
                foundation_year = 2025  # Default for current data
                
                # Try to extract foundation year from checkpoint data if available
                try:
                    checkpoint_data = checkpoint_manager.load_checkpoint(job_id, 'companies')
                    if checkpoint_data and len(checkpoint_data) > 0:
                        # Find the most common foundation year from companies
                        years = [getattr(company, 'founded_year', None) for company in checkpoint_data if hasattr(company, 'founded_year') and getattr(company, 'founded_year')]
                        if years:
                            foundation_year = max(set(years), key=years.count)  # Most common year
                except:
                    pass  # Keep default 2025
                
                # Create a separate checkpoint entry for each completed stage
                stage_order = ['companies', 'enhanced_companies', 'profiles', 'rankings']
                for stage in stage_order:
                    if stage in job_progress['stages'] and job_progress['stages'][stage].get('completed'):
                        stage_data = job_progress['stages'][stage]
                        stage_index = stage_order.index(stage)
                        completion_percentage = ((stage_index + 1) / len(stage_order)) * 100
                        
                        checkpoints.append({
                            "id": f"{job_id}_{stage}",
                            "job_id": job_id,
                            "stage": stage,
                            "created_at": stage_data['timestamp'].isoformat() if hasattr(stage_data['timestamp'], 'isoformat') else str(stage_data['timestamp']),
                            "foundation_year": foundation_year,
                            "latest_stage": stage,
                            "completion_percentage": completion_percentage,
                            "stages_completed": stage_index + 1
                        })
        
        # Sort by creation date, newest first
        checkpoints.sort(key=lambda x: x['created_at'], reverse=True)
        return checkpoints
        
    except Exception as e:
        logger.error(f"Error fetching checkpoints: {e}")
        return []

# --- Company Discovery Endpoints ---

@app.post("/api/pipeline/resume/{checkpoint_id}", response_model=PipelineJobResponse)
async def resume_pipeline_from_checkpoint(
    checkpoint_id: str,
    ranking_service: FounderRankingService = Depends(get_ranking_service),
):
    """Resume pipeline from a specific checkpoint."""
    try:
        # Extract actual job_id from checkpoint_id (format: job_id_stage)
        if '_companies' in checkpoint_id or '_enhanced_companies' in checkpoint_id or '_profiles' in checkpoint_id or '_rankings' in checkpoint_id:
            # Remove the stage suffix to get the actual job_id
            for stage in ['_rankings', '_profiles', '_enhanced_companies', '_companies']:
                if checkpoint_id.endswith(stage):
                    actual_job_id = checkpoint_id[:-len(stage)]
                    break
        else:
            actual_job_id = checkpoint_id
        
        pipeline_service = get_pipeline_service(job_id=actual_job_id)
        result = await pipeline_service.run(force_restart=False)
        
        latest_results["companies"] = result
        latest_results["rankings"] = []
        latest_results["last_job_id"] = actual_job_id
        
        return PipelineJobResponse(
            jobId=checkpoint_id,
            status="completed",
            companiesFound=len(result),
            foundersFound=sum(len(ec.profiles) for ec in result),
            message="Pipeline resumed from checkpoint and completed successfully."
        )
        
    except Exception as e:
        logger.error(f"Resume from checkpoint failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/pipeline/run", response_model=PipelineJobResponse)
async def run_complete_pipeline(
    params: YearBasedRequest,
    ranking_service: FounderRankingService = Depends(get_ranking_service),
):
    """Year-based endpoint that runs the complete pipeline including ranking."""
    try:
        discovery_params = params.to_discovery_request()
        
        pipeline_params = {
            'limit': discovery_params.limit,
            'categories': discovery_params.categories,
            'regions': discovery_params.regions,
            'founded_after': discovery_params.founded_after,
            'founded_before': discovery_params.founded_before
        }
        
        job_id = checkpoint_manager.create_job_id(pipeline_params)
        pipeline_service = get_pipeline_service(job_id=job_id)
        
        result = await pipeline_service.run(
            limit=discovery_params.limit,
            categories=discovery_params.categories,
            regions=discovery_params.regions,
            founded_after=discovery_params.founded_after,
            founded_before=discovery_params.founded_before,
            force_restart=False
        )
        
        latest_results["companies"] = result
        latest_results["rankings"] = []
        latest_results["last_job_id"] = job_id
        
        return PipelineJobResponse(
            jobId=job_id,
            status="completed",
            companiesFound=len(result),
            foundersFound=sum(len(ec.profiles) for ec in result),
            message="Complete pipeline with ranking finished successfully."
        )
    except Exception as e:
        logger.error(f"Complete pipeline failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/companies/discover", response_model=PipelineJobResponse)
async def discover_companies_only(
    params: CompanyDiscoveryRequest,
):
    """Discover companies only (without ranking) using checkpointed pipeline."""
    try:
        pipeline_params = {
            'limit': params.limit,
            'categories': params.categories,
            'regions': params.regions,
            'founded_after': params.founded_after,
            'founded_before': params.founded_before
        }
        
        job_id = checkpoint_manager.create_job_id(pipeline_params)
        pipeline_service = get_pipeline_service(job_id=job_id)
        
        result = await pipeline_service.run(
            limit=params.limit,
            categories=params.categories,
            regions=params.regions,
            founded_after=params.founded_after,
            founded_before=params.founded_before,
            force_restart=False
        )
        
        latest_results["companies"] = result
        latest_results["last_job_id"] = job_id
        
        return PipelineJobResponse(
            jobId=job_id,
            status="completed",
            companiesFound=len(result),
            foundersFound=sum(len(ec.profiles) for ec in result),
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
        output_dir = "output"
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
        
        founder_profiles = [LinkedInProfile.from_csv_row(row) for row in csv_reader]
        
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
            "id": r.profile.linkedin_url or r.profile.person_name,
            "name": r.profile.person_name,
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
                    "founder_name": r.profile.person_name,
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
        output_dir = "output"
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


@app.get("/api/companies/list")
async def get_companies_list():
    """Get list of companies from latest results for market analysis dropdown."""
    try:
        companies = safe_get_companies()
        
        if not companies:
            return []
        
        # Extract company names and info for dropdown
        company_list = []
        for ec in companies:
            company_info = {
                "id": ec.company.name,
                "name": ec.company.name,
                "sector": ec.company.sector or "Unknown",
                "founded_year": ec.company.founded_year or "Unknown",
                "ai_focus": ec.company.ai_focus or "AI"
            }
            company_list.append(company_info)
        
        return company_list
        
    except Exception as e:
        logger.error(f"❌ Failed to get companies list: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get companies: {str(e)}")


@app.post("/api/companies/{company_name}/market-analysis")
async def generate_market_analysis(company_name: str):
    """Generate market analysis for a specific company."""
    try:
        # Find the company in the latest results
        companies = safe_get_companies()
        target_company = None
        
        for ec in companies:
            if ec.company.name == company_name:
                target_company = ec.company
                break
        
        if not target_company:
            raise HTTPException(status_code=404, detail=f"Company '{company_name}' not found in latest results")
        
        # Initialize market analysis service
        market_analysis = PerplexityMarketAnalysis()
        
        # Get company sector and founded year
        sector = target_company.ai_focus or target_company.sector or "artificial intelligence"
        founded_year = target_company.founded_year or 2023
        
        # Generate market analysis
        async with market_analysis:
            metrics = await market_analysis.analyze_market(
                sector=sector,
                year=founded_year,
                region="United States"
            )
        
        # Format response
        analysis_data = {
            "company_name": company_name,
            "sector": sector,
            "founded_year": founded_year,
            "market_size_billion": metrics.market_size_billion,
            "cagr_percent": metrics.cagr_percent,
            "timing_score": metrics.timing_score,
            "us_sentiment": metrics.us_sentiment,
            "sea_sentiment": metrics.sea_sentiment,
            "competitor_count": metrics.competitor_count,
            "total_funding_billion": metrics.total_funding_billion,
            "momentum_score": metrics.momentum_score,
            "market_stage": metrics.market_stage.value if metrics.market_stage else "unknown",
            "confidence_score": metrics.confidence_score,
            "analysis_date": metrics.analysis_date.isoformat() if metrics.analysis_date else datetime.now().isoformat(),
            "execution_time": metrics.execution_time
        }
        
        return {
            "status": "success",
            "data": analysis_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Market analysis failed for {company_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Market analysis failed: {str(e)}")