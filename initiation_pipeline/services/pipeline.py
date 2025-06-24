"""Main pipeline orchestrator with multi-source data fusion and advanced analytics."""

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
from ..models import Company, EnrichedCompany, PipelineResult
from .company_discovery import ExaCompanyDiscovery
from .profile_enrichment import LinkedInEnrichmentService
from .market_analysis import MarketAnalysisProvider
from .data_fusion import DataFusionService, FusedCompanyData


logger = get_logger(__name__)


class InitiationPipeline:
    """Main pipeline with multi-source data fusion and advanced analytics."""
    
    def __init__(self):
        self.company_discovery = ExaCompanyDiscovery()
        self.profile_enrichment = LinkedInEnrichmentService()
        self.market_analysis = MarketAnalysisProvider()
        self.data_fusion = DataFusionService()
    
    async def run_complete_pipeline(
        self,
        company_limit: int = 50,
        include_profiles: bool = True,
        include_market_analysis: bool = True,
        enable_data_fusion: bool = True,
        checkpoint_prefix: str = "pipeline"
    ) -> List[FusedCompanyData]:
        """Run the complete pipeline with data fusion and advanced analytics."""
        logger.info("🚀 Starting AI Company Pipeline with Data Fusion")
        console.print("=" * 70)
        
        start_time = time.time()
        
        try:
            # Step 1: Multi-source Company Discovery
            companies = await self._run_company_discovery(
                company_limit, checkpoint_prefix
            )
            
            if not companies:
                console.print("❌ No companies found, aborting pipeline")
                return []
            
            # Step 2: Data Fusion (includes all analytics)
            if enable_data_fusion:
                fused_companies = await self._run_data_fusion(
                    companies, checkpoint_prefix
                )
            else:
                # Fallback to basic processing
                fused_companies = await self._run_basic_processing(
                    companies, include_profiles, include_market_analysis
                )
            
            execution_time = time.time() - start_time
            
            # Generate stats and summary
            stats = self._generate_stats(fused_companies, execution_time)
            self._print_summary(stats)
            
            logger.info(f"🎉 Pipeline complete! Processed {len(fused_companies)} companies in {execution_time:.1f}s")
            return fused_companies
            
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
                        
                        # Use current year for market analysis (what matters for investment decisions)
                        from datetime import datetime
                        current_year = datetime.now().year
                        
                        market_metrics = await analyzer.analyze_market(
                            sector=sector,
                            year=current_year
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
        
        console.print("🔍 Multi-source AI company discovery...")
        
        with Timer("Company Discovery"):
            companies = await self.company_discovery.find_companies(limit=limit)
        
        # Save checkpoint
        await save_checkpoint(companies, checkpoint_name)
        
        return companies
    
    async def _run_data_fusion(
        self, 
        companies: List[Company], 
        checkpoint_prefix: str
    ) -> List[FusedCompanyData]:
        """Run comprehensive data fusion with all enhancement services."""
        checkpoint_name = f"{checkpoint_prefix}_fused"
        
        # Try to load from checkpoint
        fused_companies = await load_checkpoint(checkpoint_name)
        if fused_companies:
            console.print(f"📂 Loaded {len(fused_companies)} fused companies from checkpoint")
            return fused_companies
        
        console.print("🔄 Multi-source data fusion and enhancement...")
        
        with create_progress_bar() as progress:
            task = progress.add_task(
                "Fusing company data...", 
                total=len(companies)
            )
            
            # Process companies in batches for efficiency
            batch_size = min(3, settings.concurrent_requests)  # Smaller batches for complex processing
            fused_companies = []
            
            for i in range(0, len(companies), batch_size):
                batch = companies[i:i + batch_size]
                
                # Process batch with data fusion
                batch_results = await self.data_fusion.batch_fuse_companies(batch, batch_size)
                fused_companies.extend(batch_results)
                
                # Update progress
                progress.update(task, advance=len(batch))
        
        # Save checkpoint
        await save_checkpoint(fused_companies, checkpoint_name)
        
        return fused_companies
    
    async def _run_basic_processing(
        self,
        companies: List[Company],
        include_profiles: bool,
        include_market_analysis: bool
    ) -> List[FusedCompanyData]:
        """Fallback to basic processing without data fusion."""
        console.print("⚠️  Running basic processing (data fusion disabled)")
        
        # Convert to FusedCompanyData format for consistency
        fused_companies = []
        
        for company in companies:
            # Create basic fused data structure
            fused_data = self.data_fusion._create_fallback_data(company)
            fused_companies.append(fused_data)
        
        return fused_companies
    
    def _generate_stats(
        self, 
        fused_companies: List[FusedCompanyData], 
        execution_time: float
    ) -> dict:
        """Generate pipeline statistics."""
        total_companies = len(fused_companies)
        
        # Data source analysis
        data_source_counts = {}
        for company in fused_companies:
            for source in company.data_sources:
                data_source_counts[source] = data_source_counts.get(source, 0) + 1
        
        # Data quality analysis
        quality_scores = [c.data_quality_score for c in fused_companies]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        # Confidence analysis
        confidence_scores = [c.confidence_score for c in fused_companies]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        # Sector distribution
        sector_counts = {}
        for company in fused_companies:
            sector = company.primary_sector
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        
        # Funding analysis
        companies_with_funding = sum(1 for c in fused_companies if c.total_funding_usd)
        total_funding = sum(c.total_funding_usd for c in fused_companies if c.total_funding_usd)
        
        return {
            "total_companies": total_companies,
            "execution_time": execution_time,
            "avg_data_quality": avg_quality,
            "avg_confidence": avg_confidence,
            "data_source_usage": data_source_counts,
            "sector_distribution": sector_counts,
            "companies_with_funding": companies_with_funding,
            "total_funding_usd": total_funding,
            "funding_success_rate": companies_with_funding / total_companies if total_companies > 0 else 0
        }
    
    def _print_summary(self, stats: dict):
        """Print pipeline summary."""
        console.print("\n" + "=" * 70)
        console.print("🎉 [bold green]Pipeline Summary[/bold green]")
        console.print("=" * 70)
        
        # Basic stats
        console.print(f"📊 Total Companies Processed: [bold cyan]{stats['total_companies']}[/bold cyan]")
        console.print(f"⏱️  Execution Time: [bold yellow]{stats['execution_time']:.1f}s[/bold yellow]")
        console.print(f"🏆 Average Data Quality: [bold green]{stats['avg_data_quality']:.2f}[/bold green]")
        console.print(f"🎯 Average Confidence: [bold blue]{stats['avg_confidence']:.2f}[/bold blue]")
        
        # Data sources
        console.print(f"\n📡 [bold]Data Sources Used:[/bold]")
        for source, count in stats['data_source_usage'].items():
            percentage = (count / stats['total_companies']) * 100
            console.print(f"   • {source.title()}: {count} companies ({percentage:.1f}%)")
        
        # Top sectors
        console.print(f"\n🔬 [bold]Top AI Sectors:[/bold]")
        top_sectors = sorted(stats['sector_distribution'].items(), key=lambda x: x[1], reverse=True)[:5]
        for sector, count in top_sectors:
            console.print(f"   • {sector.replace('_', ' ').title()}: {count} companies")
        
        # Funding insights
        if stats['companies_with_funding'] > 0:
            console.print(f"\n💰 [bold]Funding Insights:[/bold]")
            console.print(f"   • Companies with funding data: {stats['companies_with_funding']}")
            console.print(f"   • Success rate: {stats['funding_success_rate']:.1%}")
            if stats['total_funding_usd']:
                console.print(f"   • Total funding tracked: ${stats['total_funding_usd']:,.0f}")
        
        console.print("=" * 70)
    
    def export_results(
        self, 
        fused_companies: List[FusedCompanyData], 
        output_path: Path,
        format: str = "csv"
    ):
        """Export results to various formats."""
        if format.lower() == "csv":
            self._export_to_csv(fused_companies, output_path)
        elif format.lower() == "json":
            self._export_to_json(fused_companies, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_to_csv(self, fused_companies: List[FusedCompanyData], output_path: Path):
        """Export to CSV with comprehensive data."""
        import csv
        
        if not fused_companies:
            return
        
        # Define CSV columns
        columns = [
            'name', 'description', 'website', 'founded_year',
            'primary_sector', 'ai_focus', 'business_model', 'target_market',
            'total_funding_usd', 'latest_funding_usd', 'funding_stage',
            'current_valuation_usd', 'annual_revenue_usd',
            'employee_count', 'customer_count',
            'headquarters_location', 'linkedin_url', 'crunchbase_url',
            'data_quality_score', 'confidence_score',
            'founders', 'key_investors', 'technology_stack', 'sub_sectors',
            'data_sources', 'last_updated'
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            
            for company in fused_companies:
                row = {
                    'name': company.name,
                    'description': company.description,
                    'website': company.website,
                    'founded_year': company.founded_year,
                    'primary_sector': company.primary_sector,
                    'ai_focus': company.ai_focus,
                    'business_model': company.business_model,
                    'target_market': company.target_market,
                    'total_funding_usd': company.total_funding_usd,
                    'latest_funding_usd': company.latest_funding_usd,
                    'funding_stage': company.funding_stage,
                    'current_valuation_usd': company.current_valuation_usd,
                    'annual_revenue_usd': company.annual_revenue_usd,
                    'employee_count': company.employee_count,
                    'customer_count': company.customer_count,
                    'headquarters_location': company.headquarters_location,
                    'linkedin_url': company.linkedin_url,
                    'crunchbase_url': company.crunchbase_url,
                    'data_quality_score': company.data_quality_score,
                    'confidence_score': company.confidence_score,
                    'founders': '|'.join(company.founders) if company.founders else '',
                    'key_investors': '|'.join(company.key_investors) if company.key_investors else '',
                    'technology_stack': '|'.join(company.technology_stack) if company.technology_stack else '',
                    'sub_sectors': '|'.join(company.sub_sectors) if company.sub_sectors else '',
                    'data_sources': '|'.join(company.data_sources) if company.data_sources else '',
                    'last_updated': company.last_updated
                }
                writer.writerow(row)
    
    def _export_to_json(self, fused_companies: List[FusedCompanyData], output_path: Path):
        """Export to JSON with full data structure."""
        import json
        
        data = {
            'companies': [self.data_fusion.to_dict(company) for company in fused_companies],
            'metadata': {
                'total_companies': len(fused_companies),
                'export_timestamp': time.time(),
                'pipeline_version': 'enhanced_v1.0'
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(data, jsonfile, indent=2, ensure_ascii=False)
