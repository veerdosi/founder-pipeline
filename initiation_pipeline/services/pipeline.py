"""Main pipeline orchestrator."""

import asyncio
import time
from pathlib import Path
from typing import List, Optional

from ..core import (
    get_logger,
    console,
    Timer,
    save_checkpoint,
    load_checkpoint,
    create_progress_bar,
    settings
)
from ..models import (
    Company,
    LinkedInProfile,
    MarketMetrics,
    EnrichedCompany,
    PipelineResult
)
from .company_discovery import ExaCompanyDiscovery
from .profile_enrichment import LinkedInEnrichmentService
from .market_analysis import MarketAnalysisProvider


logger = get_logger(__name__)


class InitiationPipeline:
    """Main pipeline for AI company discovery and analysis."""
    
    def __init__(self):
        self.company_discovery = ExaCompanyDiscovery()
        self.profile_enrichment = LinkedInEnrichmentService()
        self.market_analysis = MarketAnalysisProvider()
    
    async def run_complete_pipeline(
        self,
        company_limit: int = 50,
        include_profiles: bool = True,
        include_market_analysis: bool = True,
        checkpoint_prefix: str = "pipeline"
    ) -> PipelineResult:
        """Run the complete pipeline with all steps."""
        logger.info("🚀 Starting Complete AI Company Pipeline")
        console.print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Step 1: Company Discovery
            companies = await self._run_company_discovery(
                company_limit, checkpoint_prefix
            )
            
            if not companies:
                console.print("❌ No companies found, aborting pipeline")
                return PipelineResult(companies=[], stats={}, execution_time=0)
            
            # Step 2: Profile Enrichment (optional)
            enriched_companies = []
            if include_profiles:
                enriched_companies = await self._run_profile_enrichment(
                    companies, checkpoint_prefix
                )
            else:
                enriched_companies = [
                    EnrichedCompany(company=company, profiles=[])
                    for company in companies
                ]
            
            # Step 3: Market Analysis (optional)
            if include_market_analysis:
                enriched_companies = await self._run_market_analysis(
                    enriched_companies, checkpoint_prefix
                )
            
            execution_time = time.time() - start_time
            
            # Generate stats
            stats = self._generate_pipeline_stats(enriched_companies, execution_time)
            
            console.print(f"\n🎉 Pipeline Complete!")
            console.print(f"⏱️  Total execution time: {execution_time:.1f} seconds")
            console.print(f"🏢 Companies processed: {len(enriched_companies)}")
            
            return PipelineResult(
                companies=enriched_companies,
                stats=stats,
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"❌ Pipeline error: {e}")
            raise
    
    async def run_company_discovery_only(
        self,
        limit: int = 50,
        categories: Optional[List[str]] = None,
        regions: Optional[List[str]] = None
    ) -> List[Company]:
        """Run only company discovery."""
        logger.info("🔍 Running company discovery only")
        
        with Timer("Company Discovery"):
            companies = await self.company_discovery.find_companies(
                limit=limit,
                categories=categories,
                regions=regions
            )
        
        return companies
    
    async def run_profile_enrichment_only(
        self,
        companies: List[Company]
    ) -> List[EnrichedCompany]:
        """Run only profile enrichment."""
        logger.info("👤 Running profile enrichment only")
        
        enriched_companies = []
        
        with create_progress_bar() as progress:
            task = progress.add_task(
                "Finding LinkedIn profiles...", 
                total=len(companies)
            )
            
            for company in companies:
                try:
                    profiles = await self.profile_enrichment.find_profiles(company)
                    
                    # Enrich profiles with full data
                    enriched_profiles = []
                    for profile in profiles[:3]:  # Limit to 3 profiles per company
                        enriched_profile = await self.profile_enrichment.enrich_profile(profile)
                        enriched_profiles.append(enriched_profile)
                    
                    enriched_company = EnrichedCompany(
                        company=company,
                        profiles=enriched_profiles
                    )
                    enriched_companies.append(enriched_company)
                    
                except Exception as e:
                    logger.error(f"Error processing {company.name}: {e}")
                    # Add company without profiles
                    enriched_companies.append(
                        EnrichedCompany(company=company, profiles=[])
                    )
                
                progress.update(task, advance=1)
        
        return enriched_companies
    
    async def run_market_analysis_only(
        self,
        companies: List[Company]
    ) -> List[EnrichedCompany]:
        """Run only market analysis."""
        logger.info("📊 Running market analysis only")
        
        enriched_companies = []
        
        async with self.market_analysis as analyzer:
            with create_progress_bar() as progress:
                task = progress.add_task(
                    "Analyzing markets...", 
                    total=len(companies)
                )
                
                for company in companies:
                    try:
                        # Determine sector for analysis
                        sector = company.ai_focus or company.sector or "Artificial Intelligence"
                        year = company.founded_year or 2020
                        
                        market_metrics = await analyzer.analyze_market(
                            sector=sector,
                            year=year
                        )
                        
                        enriched_company = EnrichedCompany(
                            company=company,
                            profiles=[],
                            market_metrics=market_metrics
                        )
                        enriched_companies.append(enriched_company)
                        
                    except Exception as e:
                        logger.error(f"Error analyzing {company.name}: {e}")
                        # Add company without market analysis
                        enriched_companies.append(
                            EnrichedCompany(company=company, profiles=[])
                        )
                    
                    progress.update(task, advance=1)
        
        return enriched_companies
    
    async def _run_company_discovery(
        self, 
        limit: int, 
        checkpoint_prefix: str
    ) -> List[Company]:
        """Run company discovery with checkpointing."""
        checkpoint_name = f"{checkpoint_prefix}_companies"
        
        # Try to load from checkpoint
        companies = await load_checkpoint(checkpoint_name)
        if companies:
            console.print(f"📂 Loaded {len(companies)} companies from checkpoint")
            return companies
        
        console.print("🔍 Finding AI companies...")
        with Timer("Company Discovery"):
            companies = await self.company_discovery.find_companies(limit=limit)
        
        # Save checkpoint
        await save_checkpoint(companies, checkpoint_name)
        
        return companies
    
    async def _run_profile_enrichment(
        self, 
        companies: List[Company], 
        checkpoint_prefix: str
    ) -> List[EnrichedCompany]:
        """Run profile enrichment with checkpointing."""
        checkpoint_name = f"{checkpoint_prefix}_profiles"
        
        # Try to load from checkpoint
        enriched_companies = await load_checkpoint(checkpoint_name)
        if enriched_companies:
            console.print(f"📂 Loaded {len(enriched_companies)} enriched companies from checkpoint")
            return enriched_companies
        
        console.print("👤 Finding LinkedIn profiles...")
        enriched_companies = []
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(settings.concurrent_requests)
        
        async def process_company(company: Company, progress_callback) -> EnrichedCompany:
            async with semaphore:
                try:
                    profiles = await self.profile_enrichment.find_profiles(company)
                    
                    # Enrich profiles with full data (limit to 3 per company)
                    enriched_profiles = []
                    for profile in profiles[:3]:
                        enriched_profile = await self.profile_enrichment.enrich_profile(profile)
                        enriched_profiles.append(enriched_profile)
                    
                    result = EnrichedCompany(
                        company=company,
                        profiles=enriched_profiles
                    )
                    
                    # Update progress
                    progress_callback()
                    return result
                    
                except Exception as e:
                    logger.error(f"Error processing {company.name}: {e}")
                    progress_callback()
                    return EnrichedCompany(company=company, profiles=[])
        
        # Process companies with better progress tracking
        with create_progress_bar() as progress:
            task = progress.add_task(
                "Finding LinkedIn profiles...", 
                total=len(companies)
            )
            
            # Process companies in smaller batches to avoid overwhelming APIs
            batch_size = min(5, settings.concurrent_requests)
            
            for i in range(0, len(companies), batch_size):
                batch = companies[i:i+batch_size]
                
                # Process current batch
                tasks = [process_company(company, lambda: None) for company in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Handle results and update progress
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Error processing {batch[j].name}: {result}")
                        enriched_companies.append(EnrichedCompany(company=batch[j], profiles=[]))
                    else:
                        enriched_companies.append(result)
                    
                    # Update progress for each completed company
                    progress.update(task, advance=1)
        
        # Save checkpoint
        await save_checkpoint(enriched_companies, checkpoint_name)
        
        return enriched_companies
    
    async def _run_market_analysis(
        self, 
        enriched_companies: List[EnrichedCompany], 
        checkpoint_prefix: str
    ) -> List[EnrichedCompany]:
        """Run market analysis with checkpointing."""
        checkpoint_name = f"{checkpoint_prefix}_analysis"
        
        # Try to load from checkpoint
        analyzed_companies = await load_checkpoint(checkpoint_name)
        if analyzed_companies:
            console.print(f"📂 Loaded {len(analyzed_companies)} analyzed companies from checkpoint")
            return analyzed_companies
        
        console.print("📊 Analyzing markets...")
        
        # Group companies by sector for efficient analysis
        sector_groups = self._group_companies_by_sector(enriched_companies)
        
        # Analyze each sector
        sector_cache = {}
        
        async with self.market_analysis as analyzer:
            with create_progress_bar() as progress:
                task = progress.add_task(
                    "Analyzing markets...", 
                    total=len(enriched_companies)
                )
                
                for enriched_company in enriched_companies:
                    company = enriched_company.company
                    sector = company.ai_focus or company.sector or "Artificial Intelligence"
                    year = company.founded_year or 2020
                    
                    # Use cached analysis if available
                    cache_key = f"{sector}_{year}"
                    if cache_key in sector_cache:
                        market_metrics = sector_cache[cache_key]
                    else:
                        try:
                            market_metrics = await analyzer.analyze_market(
                                sector=sector,
                                year=year
                            )
                            sector_cache[cache_key] = market_metrics
                        except Exception as e:
                            logger.error(f"Error analyzing sector {sector}: {e}")
                            market_metrics = None
                    
                    # Update enriched company with market metrics
                    enriched_company.market_metrics = market_metrics
                    progress.update(task, advance=1)
        
        # Save checkpoint
        await save_checkpoint(enriched_companies, checkpoint_name)
        
        return enriched_companies
    
    def _group_companies_by_sector(
        self, 
        enriched_companies: List[EnrichedCompany]
    ) -> dict:
        """Group companies by AI sector."""
        sector_groups = {}
        
        for enriched_company in enriched_companies:
            company = enriched_company.company
            sector = company.ai_focus or company.sector or "Artificial Intelligence"
            
            if sector not in sector_groups:
                sector_groups[sector] = []
            sector_groups[sector].append(enriched_company)
        
        return sector_groups
    
    def _generate_pipeline_stats(
        self, 
        enriched_companies: List[EnrichedCompany], 
        execution_time: float
    ) -> dict:
        """Generate pipeline execution statistics."""
        total_companies = len(enriched_companies)
        companies_with_profiles = sum(
            1 for ec in enriched_companies if ec.profiles
        )
        companies_with_analysis = sum(
            1 for ec in enriched_companies if ec.market_metrics
        )
        total_profiles = sum(
            len(ec.profiles) for ec in enriched_companies
        )
        
        # Sector distribution
        sectors = {}
        for ec in enriched_companies:
            sector = ec.company.ai_focus or ec.company.sector or "Unknown"
            sectors[sector] = sectors.get(sector, 0) + 1
        
        # Funding stage distribution
        funding_stages = {}
        for ec in enriched_companies:
            stage = ec.company.funding_stage or "Unknown"
            funding_stages[str(stage)] = funding_stages.get(str(stage), 0) + 1
        
        return {
            "total_companies": total_companies,
            "companies_with_profiles": companies_with_profiles,
            "companies_with_analysis": companies_with_analysis,
            "total_profiles": total_profiles,
            "execution_time": execution_time,
            "sector_distribution": sectors,
            "funding_stage_distribution": funding_stages,
            "avg_profiles_per_company": total_profiles / max(total_companies, 1),
            "profile_success_rate": companies_with_profiles / max(total_companies, 1),
            "analysis_success_rate": companies_with_analysis / max(total_companies, 1)
        }
