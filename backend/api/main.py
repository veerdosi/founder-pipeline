"""Simplified FastAPI main application for date-range based discovery with CSV output."""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
import asyncio
import csv
import io
from typing import List, Optional
from datetime import datetime, date

from ..services.ranking import FounderRankingService, FounderProfile
from ..services.company_discovery import ExaCompanyDiscovery
from ..services.profile_enrichment import LinkedInEnrichmentService
from ..services.pipeline import InitiationPipeline
from .models import *
from .dependencies import get_ranking_service, get_discovery_service, get_enrichment_service, get_pipeline_service
from ..utils.checkpoint_manager import checkpoint_manager, checkpointed_runner

app = FastAPI(
    title="Initiation Pipeline API",
    description="AI-powered founder discovery and ranking system with date range filtering",
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

@app.get("/api/dashboard/stats", response_model=DashboardStats)
async def get_dashboard_stats():
    """Get basic dashboard statistics."""
    return DashboardStats(
        totalCompanies=0,
        totalFounders=0,
        avgConfidenceScore=0.0,
        levelDistribution={},
        recentActivity=[]
    )

@app.post("/api/discover-and-process", response_model=PipelineJobResponse)
async def discover_and_process_companies(
    params: CompanyDiscoveryRequest,
    pipeline_service: InitiationPipeline = Depends(get_pipeline_service),
    ranking_service: FounderRankingService = Depends(get_ranking_service)
):
    """
    Main endpoint: Discover companies in date range, find founders, and prepare CSV data.
    Uses checkpointing for reliability and resume capability.
    """
    try:
        # Convert request params to dict for job ID creation
        params_dict = {
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
            ranking_service=ranking_service,
            params=params_dict,
            force_restart=False
        )
        
        return PipelineJobResponse(
            jobId=result['job_id'],
            status="completed",
            companiesFound=result['stats']['total_companies'],
            foundersFound=result['stats']['total_founders'],
            message=f"Processed {result['stats']['total_companies']} companies and {result['stats']['total_founders']} founders (checkpointed)"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/export-companies-csv")
async def export_companies_csv(
    params: CompanyDiscoveryRequest,
    pipeline_service: InitiationPipeline = Depends(get_pipeline_service)
):
    """Export companies dataset to CSV based on date range."""
    try:
        # Run discovery and enrichment
        enriched_companies = await pipeline_service.run_complete_pipeline_with_date_range(
            company_limit=params.limit,
            categories=params.categories,
            regions=params.regions,
            sources=params.sources,
            founded_after=params.founded_after,
            founded_before=params.founded_before
        )
        
        # Generate company CSV
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            "company_name", "description", "website", "founded_year", 
            "funding_total_usd", "funding_stage", "city", "country", 
            "ai_focus", "sector", "founders_count", "source_url"
        ])
        
        # Write data
        for ec in enriched_companies:
            company = ec.company
            writer.writerow([
                company.name,
                company.description or "",
                str(company.website) if company.website else "",
                company.founded_year,
                company.funding_total_usd,
                company.funding_stage,
                company.city or "",
                company.country or "",
                company.ai_focus or "",
                company.sector or "",
                len(ec.profiles),
                company.source_url or ""
            ])
        
        output.seek(0)
        filename = f"companies_{params.founded_after}_{params.founded_before}.csv"
        
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/export-founders-csv")
async def export_founders_csv(
    params: CompanyDiscoveryRequest,
    pipeline_service: InitiationPipeline = Depends(get_pipeline_service)
):
    """Export founders dataset to CSV based on date range."""
    try:
        # Run discovery and enrichment
        enriched_companies = await pipeline_service.run_complete_pipeline_with_date_range(
            company_limit=params.limit,
            categories=params.categories,
            regions=params.regions,
            sources=params.sources,
            founded_after=params.founded_after,
            founded_before=params.founded_before
        )
        
        # Generate founders CSV
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            "company_name", "person_name", "title", "linkedin_url", 
            "location", "about", "estimated_age",
            "experience_1_title", "experience_1_company",
            "experience_2_title", "experience_2_company", 
            "experience_3_title", "experience_3_company",
            "education_1_school", "education_1_degree",
            "education_2_school", "education_2_degree",
            "skill_1", "skill_2", "skill_3",
            "l_level", "confidence_score", "reasoning"
        ])
        
        # Write data
        for ec in enriched_companies:
            company_name = ec.company.name
            
            for profile in ec.profiles:
                # Get L-level classification if available
                l_level = getattr(profile, 'l_level', '')
                confidence_score = getattr(profile, 'confidence_score', '')
                reasoning = getattr(profile, 'reasoning', '')
                
                writer.writerow([
                    company_name,
                    profile.person_name,
                    profile.title or "",
                    str(profile.linkedin_url),
                    profile.location or "",
                    profile.about or "",
                    profile.estimated_age,
                    profile.experience_1_title or "",
                    profile.experience_1_company or "",
                    profile.experience_2_title or "",
                    profile.experience_2_company or "",
                    profile.experience_3_title or "",
                    profile.experience_3_company or "",
                    profile.education_1_school or "",
                    profile.education_1_degree or "",
                    profile.education_2_school or "",
                    profile.education_2_degree or "",
                    profile.skill_1 or "",
                    profile.skill_2 or "",
                    profile.skill_3 or "",
                    l_level,
                    confidence_score,
                    reasoning
                ])
        
        output.seek(0)
        filename = f"founders_{params.founded_after}_{params.founded_before}.csv"
        
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/discover-rank-export")
async def discover_rank_and_export(
    params: CompanyDiscoveryRequest,
    pipeline_service: InitiationPipeline = Depends(get_pipeline_service),
    ranking_service: FounderRankingService = Depends(get_ranking_service)
):
    """Complete pipeline: discover companies, find founders, rank them, export both CSVs."""
    try:
        task_id = f"complete_{datetime.now().timestamp()}"
        
        # Step 1: Run complete pipeline
        enriched_companies = await pipeline_service.run_complete_pipeline_with_date_range(
            company_limit=params.limit,
            categories=params.categories,
            regions=params.regions,
            sources=params.sources,
            founded_after=params.founded_after,
            founded_before=params.founded_before
        )
        
        # Step 2: Rank founders using L1-L10 framework
        all_profiles = []
        for ec in enriched_companies:
            for profile in ec.profiles:
                founder_profile = FounderProfile(
                    name=profile.person_name,
                    company_name=ec.company.name,
                    title=profile.title or "",
                    linkedin_url=str(profile.linkedin_url),
                    location=profile.location,
                    about=profile.about,
                    estimated_age=profile.estimated_age,
                    experience_1_title=profile.experience_1_title,
                    experience_1_company=profile.experience_1_company,
                    experience_2_title=profile.experience_2_title,
                    experience_2_company=profile.experience_2_company,
                    experience_3_title=profile.experience_3_title,
                    experience_3_company=profile.experience_3_company,
                    education_1_school=profile.education_1_school,
                    education_1_degree=profile.education_1_degree,
                    education_2_school=profile.education_2_school,
                    education_2_degree=profile.education_2_degree,
                    skill_1=profile.skill_1,
                    skill_2=profile.skill_2,
                    skill_3=profile.skill_3
                )
                all_profiles.append((founder_profile, profile))
        
        # Rank founders
        rankings = await ranking_service.rank_founders_batch(
            [fp for fp, _ in all_profiles], 
            batch_size=5
        )
        
        # Apply rankings back to profiles
        for i, ranking in enumerate(rankings):
            if i < len(all_profiles):
                _, original_profile = all_profiles[i]
                # Add ranking data to profile
                original_profile.l_level = ranking.classification.level.value
                original_profile.confidence_score = ranking.classification.confidence_score
                original_profile.reasoning = ranking.classification.reasoning
        
        total_founders = len(all_profiles)
        high_confidence = len([r for r in rankings if r.classification.confidence_score >= 0.75])
        
        return PipelineJobResponse(
            jobId=task_id,
            status="completed",
            companiesFound=len(enriched_companies),
            foundersFound=total_founders,
            message=f"Complete pipeline: {len(enriched_companies)} companies, {total_founders} founders ranked ({high_confidence} high confidence)"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
