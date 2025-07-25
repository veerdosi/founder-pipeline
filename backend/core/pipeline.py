"""Main pipeline orchestrator with 3-stage company enrichment flow."""

import asyncio
import time
from typing import List, Optional
from datetime import date

from .config import settings
from ..models import Company, EnrichedCompany
from .data.company_enrichment import company_enrichment_service
from .data.profile_enrichment import LinkedInEnrichmentService
from .ranking.ranking_service import FounderRankingService

import logging
from rich.console import Console
from ..utils.checkpoint_manager import checkpoint_manager, CheckpointedPipelineRunner

logger = logging.getLogger(__name__)
console = Console()


class InitiationPipeline:
    
    def __init__(self, job_id: str):
        self.company_enrichment = company_enrichment_service
        self.profile_enrichment = LinkedInEnrichmentService()
        self.ranking_service = FounderRankingService()
        self.job_id = job_id
        self.runner = CheckpointedPipelineRunner(checkpoint_manager)

    async def run(
        self,
        year: int,
        limit: Optional[int] = None,
        force_restart: bool = False,
    ) -> List[EnrichedCompany]:
        logger.info(f"🚀 Starting 3-stage pipeline for job_id: {self.job_id}, year: {year}")
        console.print("=" * 70)
        start_time = time.time()

        try:
            # Automatically determine CSV file path based on year
            csv_file_path = self._get_csv_file_path(year)
            console.print(f"📂 Using input file: {csv_file_path}")

            # Stage 1: Company Enrichment (replaces discovery + data fusion)
            companies = await self._enrich_companies_checkpointed(csv_file_path, limit, force_restart)
            if not companies:
                console.print("❌ No companies found, aborting pipeline")
                return []

            # Stage 2: Profile Enrichment (unchanged)
            enriched_companies = await self._enrich_profiles_checkpointed(companies, force_restart)

            # Stage 3: Founder Ranking (unchanged)
            ranked_companies = await self._rank_founders_checkpointed(enriched_companies, force_restart)

            execution_time = time.time() - start_time
            stats = self._generate_stats_from_enriched(ranked_companies, execution_time)
            self._print_summary(stats)
            
            logger.info(f"🎉 Pipeline complete! Processed {len(ranked_companies)} companies in {execution_time:.1f}s")
            return ranked_companies

        except Exception as e:
            logger.error(f"❌ Pipeline error for job_id {self.job_id}: {e}")
            raise

    def _get_csv_file_path(self, year: int) -> str:
        """Get the CSV file path for a given year."""
        from pathlib import Path
        
        input_dir = Path("./input")
        
        # First try the exact year file
        year_file = input_dir / f"{year}companies.csv"
        if year_file.exists():
            return str(year_file)
        
        # For 2023, try the split files
        if year == 2023:
            file1 = input_dir / "2023companies-1.csv"
            file2 = input_dir / "2023companies-2.csv"
            if file1.exists():
                logger.warning(f"Found split 2023 files, using {file1}")
                return str(file1)
        
        # If no specific file found, raise error
        available_files = list(input_dir.glob("*companies*.csv"))
        available_years = []
        for f in available_files:
            try:
                if "companies" in f.name:
                    year_str = f.name.replace("companies", "").replace(".csv", "").replace("-1", "").replace("-2", "")
                    available_years.append(year_str)
            except:
                pass
        
        raise FileNotFoundError(
            f"No CSV file found for year {year}. "
            f"Available files: {[f.name for f in available_files]}"
        )

    async def _enrich_companies_checkpointed(self, csv_file_path: str, limit: Optional[int], force_restart: bool):
        stage_name = "enriched_companies"
        if not force_restart:
            cached_data = checkpoint_manager.load_checkpoint(self.job_id, stage_name)
            if cached_data:
                # Export companies CSV when loading from checkpoint
                try:
                    runner = CheckpointedPipelineRunner(checkpoint_manager)
                    await runner._export_companies_csv(cached_data, self.job_id)
                    logger.info("📊 Companies CSV export completed from checkpoint")
                except Exception as e:
                    logger.error(f"Failed to export companies CSV from checkpoint: {e}")
                return cached_data

        console.print("🔄 Enriching companies from Crunchbase CSV...")
        companies = await self.company_enrichment.process_crunchbase_csv(csv_file_path, limit)
        checkpoint_manager.save_checkpoint(self.job_id, stage_name, companies)
        
        # Export companies CSV after enrichment
        try:
            runner = CheckpointedPipelineRunner(checkpoint_manager)
            await runner._export_companies_csv(companies, self.job_id)
            logger.info("📊 Companies CSV export completed")
        except Exception as e:
            logger.error(f"Failed to export companies CSV: {e}")
        
        return companies

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
        logger.info(f"💾 Saved {len(enriched_companies)} profiles to checkpoint")
        return enriched_companies

    async def _rank_founders_checkpointed(self, enriched_companies, force_restart):
        stage_name = "rankings"
        
        # Check for complete stage checkpoint first
        if not force_restart:
            cached_data = checkpoint_manager.load_checkpoint(self.job_id, stage_name)
            if cached_data:
                # Export founders CSV when loading from checkpoint (100% complete)
                try:
                    runner = CheckpointedPipelineRunner(checkpoint_manager)
                    await runner._export_founders_csv(cached_data, self.job_id)
                    logger.info("📊 Founders CSV export completed from checkpoint")
                except Exception as e:
                    logger.error(f"Failed to export founders CSV from checkpoint: {e}")
                return cached_data
        
        # Check for incremental checkpoint and resume if possible
        incremental_data = None
        completed_companies = set()
        
        if not force_restart:
            incremental_data = checkpoint_manager.load_incremental_checkpoint(self.job_id, stage_name)
            if incremental_data:
                # Extract which companies are already completed
                completed_companies = set(incremental_data['progress_info'].get('completed_companies', []))
                logger.info(f"🔄 Resuming ranking from incremental checkpoint: {len(completed_companies)} companies already ranked")
                console.print(f"🔄 Resuming ranking: {len(completed_companies)}/{len(enriched_companies)} companies already completed")

        console.print("🏆 Ranking founders...")
        failure_count = 0
        max_consecutive_failures = 3
        
        # Start with incremental data if available, otherwise use original data
        if incremental_data and incremental_data['data']:
            working_companies = incremental_data['data']
        else:
            working_companies = enriched_companies
        
        for i, enriched in enumerate(working_companies):
            company_id = f"{enriched.company.name}_{enriched.company.website or 'no_website'}"
            
            # Skip if company already completed
            if company_id in completed_companies:
                console.print(f"   ✅ [{i+1}/{len(working_companies)}] Skipping already ranked {enriched.company.name}")
                continue
            
            if enriched.profiles:
                console.print(f"   🏆 [{i+1}/{len(working_companies)}] Ranking founders for {enriched.company.name}")
                try:
                    ranked_profiles = await self.ranking_service.rank_founders_batch(enriched.profiles)
                    enriched.profiles = ranked_profiles
                    
                    # Mark this company as completed
                    completed_companies.add(company_id)
                    failure_count = 0  # Reset failure count on success
                    
                    # Save incremental checkpoint after each successful company
                    progress_info = {
                        'completed_companies': list(completed_companies),
                        'total_companies': len(working_companies),
                        'progress_percentage': round((len(completed_companies) / len(working_companies)) * 100, 1)
                    }
                    
                    checkpoint_success = checkpoint_manager.save_incremental_checkpoint(
                        self.job_id, 
                        stage_name, 
                        working_companies, 
                        progress_info
                    )
                    
                    if checkpoint_success:
                        logger.info(f"💾 Incremental checkpoint saved: {len(completed_companies)}/{len(working_companies)} companies ranked")
                    
                except Exception as e:
                    failure_count += 1
                    logger.error(f"Ranking failed for {enriched.company.name}: {e}")
                    
                    # Check if we've hit the consecutive failure limit
                    if failure_count >= max_consecutive_failures:
                        logger.error(f"❌ {max_consecutive_failures} consecutive ranking failures - stopping ranking process")
                        console.print(f"❌ Stopping due to {max_consecutive_failures} consecutive API failures")
                        
                        # Save current progress before stopping
                        progress_info = {
                            'completed_companies': list(completed_companies),
                            'total_companies': len(working_companies),
                            'progress_percentage': round((len(completed_companies) / len(working_companies)) * 100, 1),
                            'stopped_due_to_failures': True,
                            'last_error': str(e)
                        }
                        
                        checkpoint_manager.save_incremental_checkpoint(
                            self.job_id, 
                            stage_name, 
                            working_companies, 
                            progress_info
                        )
                        
                        logger.info(f"💾 Progress saved before stopping: {len(completed_companies)}/{len(working_companies)} companies completed")
                        console.print(f"💾 Progress saved: {len(completed_companies)}/{len(working_companies)} companies ranked")
                        console.print("🔄 You can resume ranking later by running the pipeline again")
                        
                        # Don't save the final checkpoint or export CSV yet
                        return working_companies

        # All companies completed successfully - save final checkpoint
        checkpoint_manager.save_checkpoint(self.job_id, stage_name, working_companies)
        logger.info(f"✅ All founders ranked successfully - final checkpoint saved")
        
        # Clean up incremental checkpoint since we're done
        incremental_file = checkpoint_manager.checkpoint_dir / f"{self.job_id}_{stage_name}_incremental.pkl"
        if incremental_file.exists():
            incremental_file.unlink()
            logger.info("🗑️ Cleaned up incremental checkpoint file")
        
        # Export founders CSV with ranking data (100% complete)
        try:
            runner = CheckpointedPipelineRunner(checkpoint_manager)
            await runner._export_founders_csv(working_companies, self.job_id)
            logger.info("📊 Founders CSV export completed")
        except Exception as e:
            logger.error(f"Failed to export founders CSV: {e}")
            import traceback
            traceback.print_exc()
        
        return working_companies


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
