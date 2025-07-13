"""Main pipeline orchestrator with multi-source data fusion and advanced analytics."""

import asyncio
import time
from typing import List, Optional
from datetime import date

from .config import settings
from ..models import Company, EnrichedCompany
from .data.company_discovery import ExaCompanyDiscovery
from .data.profile_enrichment import LinkedInEnrichmentService
from .analysis.market_analysis import PerplexityMarketAnalysis
from .data.data_fusion import DataFusionService, FusedCompanyData
from .ranking.ranking_service import FounderRankingService

import logging
from rich.console import Console
from ..utils.checkpoint_manager import checkpoint_manager, CheckpointedPipelineRunner

logger = logging.getLogger(__name__)
console = Console()


class InitiationPipeline:
    
    def __init__(self, job_id: str):
        self.company_discovery = ExaCompanyDiscovery()
        self.profile_enrichment = LinkedInEnrichmentService()
        self.market_analysis = PerplexityMarketAnalysis()
        self.data_fusion = DataFusionService()
        self.ranking_service = FounderRankingService()
        self.job_id = job_id
        self.runner = CheckpointedPipelineRunner(checkpoint_manager)

    async def run(
        self,
        limit: int = 50,
        categories: Optional[List[str]] = None,
        regions: Optional[List[str]] = None,
        founded_after: Optional[date] = None,
        founded_before: Optional[date] = None,
        force_restart: bool = False,
    ) -> List[EnrichedCompany]:
        logger.info(f"🚀 Starting pipeline for job_id: {self.job_id}")
        console.print("=" * 70)
        start_time = time.time()

        try:
            companies = await self._discover_companies_checkpointed(
                limit, categories, regions, founded_after, founded_before, force_restart
            )
            if not companies:
                console.print("❌ No companies found, aborting pipeline")
                return []

            enhanced_companies = await self._enhance_companies_checkpointed(companies, force_restart)

            enriched_companies = await self._enrich_profiles_checkpointed(enhanced_companies, force_restart)

            ranked_companies = await self._rank_founders_checkpointed(enriched_companies, force_restart)

            execution_time = time.time() - start_time
            stats = self._generate_stats_from_enriched(ranked_companies, execution_time)
            self._print_summary(stats)
            
            logger.info(f"🎉 Pipeline complete! Processed {len(ranked_companies)} companies in {execution_time:.1f}s")
            return ranked_companies

        except Exception as e:
            logger.error(f"❌ Pipeline error for job_id {self.job_id}: {e}")
            raise

    async def _discover_companies_checkpointed(
        self, limit, categories, regions, founded_after, founded_before, force_restart
    ):
        stage_name = "companies"
        if not force_restart:
            cached_data = checkpoint_manager.load_checkpoint(self.job_id, stage_name)
            if cached_data:
                return cached_data

        console.print("🔍 Discovering companies...")
        companies = await self.company_discovery.find_companies(
            limit=limit,
            categories=categories,
            regions=regions,
            founded_year=founded_after.year if founded_after else None,
        )
        
        if founded_after or founded_before:
            companies = self._filter_by_date(companies, founded_after, founded_before)

        checkpoint_manager.save_checkpoint(self.job_id, stage_name, companies)
        return companies

    def _filter_by_date(self, companies, founded_after, founded_before):
        filtered_companies = []
        for company in companies:
            if company.founded_year:
                company_date = date(company.founded_year, 1, 1)
                if founded_after and company_date < founded_after:
                    continue
                if founded_before and company_date > founded_before:
                    continue
            filtered_companies.append(company)
        
        console.print(f"📅 Filtered to {len(filtered_companies)} companies in date range")
        return filtered_companies

    async def _enhance_companies_checkpointed(self, companies, force_restart):
        stage_name = "enhanced_companies"
        if not force_restart:
            cached_data = checkpoint_manager.load_checkpoint(self.job_id, stage_name)
            if cached_data:
                return cached_data

        console.print("🔄 Enhancing companies with Crunchbase data fusion...")
        enhanced_companies = await self.data_fusion.batch_fuse_companies(companies, batch_size=min(3, settings.concurrent_requests))
        checkpoint_manager.save_checkpoint(self.job_id, stage_name, enhanced_companies)
        return enhanced_companies

    async def _enrich_profiles_checkpointed(self, companies, force_restart):
        stage_name = "profiles"
        if not force_restart:
            cached_data = checkpoint_manager.load_checkpoint(self.job_id, stage_name)
            if cached_data:
                return cached_data

        console.print("👤 Finding and enriching LinkedIn profiles...")
        enriched_companies = []
        for i, company in enumerate(companies):
            console.print(f"   👤 [{i+1}/{len(companies)}] Processing {company.name}...")
            try:
                # Step 1: Find LinkedIn URLs
                profiles = await self.profile_enrichment.find_profiles(company)
                
                # Step 2: Enrich profiles with full LinkedIn data
                if profiles:
                    console.print(f"   🔍 Found {len(profiles)} profiles, enriching with full LinkedIn data...")
                    enriched_profiles = await self.profile_enrichment.enrich_profiles_batch(profiles)
                    enriched_companies.append(EnrichedCompany(company=company, profiles=enriched_profiles))
                else:
                    enriched_companies.append(EnrichedCompany(company=company, profiles=[]))
                    
            except Exception as e:
                logger.error(f"Error enriching {company.name}: {e}")
                enriched_companies.append(EnrichedCompany(company=company, profiles=[]))
        
        checkpoint_manager.save_checkpoint(self.job_id, stage_name, enriched_companies)
        return enriched_companies

    async def _rank_founders_checkpointed(self, enriched_companies, force_restart):
        stage_name = "rankings"
        if not force_restart:
            cached_data = checkpoint_manager.load_checkpoint(self.job_id, stage_name)
            if cached_data:
                return cached_data

        console.print("🏆 Ranking founders...")
        for i, enriched in enumerate(enriched_companies):
            if enriched.profiles:
                console.print(f"   🏆 [{i+1}/{len(enriched_companies)}] Ranking founders for {enriched.company.name}")
                try:
                    ranked_profiles = await self.ranking_service.rank_founders_batch(enriched.profiles)
                    enriched.profiles = ranked_profiles
                except Exception as e:
                    logger.error(f"Ranking failed for {enriched.company.name}: {e}")

        checkpoint_manager.save_checkpoint(self.job_id, stage_name, enriched_companies)
        return enriched_companies


    def _generate_stats_from_enriched(
        self, 
        enriched_companies: List[EnrichedCompany], 
        execution_time: float
    ) -> dict:
        """Generate pipeline statistics from EnrichedCompany objects."""
        total_companies = len(enriched_companies)
        profiles_found = sum(len(ec.profiles) for ec in enriched_companies)
        companies_with_profiles = sum(1 for ec in enriched_companies if ec.profiles)
        
        return {
            'total_companies': total_companies,
            'profiles_found': profiles_found,
            'companies_with_profiles': companies_with_profiles,
            'execution_time': execution_time,
        }

    def _print_summary(self, stats: dict):
        """Print pipeline summary."""
        console.print("\n" + "=" * 70)
        console.print("🎉 [bold green]Pipeline Summary[/bold green]")
        console.print("=" * 70)
        
        console.print(f"📊 Total Companies Processed: [bold cyan]{stats['total_companies']}[/bold cyan]")
        console.print(f"⏱️  Execution Time: [bold yellow]{stats['execution_time']:.1f}s[/bold yellow]")
        console.print(f"👤 LinkedIn Profiles Found: [bold green]{stats['profiles_found']}[/bold green]")
        console.print(f"🏢 Companies with Profiles: [bold blue]{stats['companies_with_profiles']}[/bold blue]")
        
        console.print("=" * 70)
