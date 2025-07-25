"""Company enrichment service that processes Crunchbase CSV files and enriches them with additional data."""

import asyncio
import csv
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from ...models import Company
from ..analysis.sector_classification import sector_description_service
from ..analysis.market_analysis import PerplexityMarketAnalysis
from ..analysis.funding_stage_detection import funding_stage_detection_service

logger = logging.getLogger(__name__)


class CompanyEnrichmentService:
    """Service for processing Crunchbase CSV files and enriching company data."""
    
    def __init__(self):
        self.input_dir = Path("./input")
        self.market_analyzer = PerplexityMarketAnalysis()
    
    async def process_crunchbase_csv(
        self, 
        csv_file_path: str,
        max_companies: Optional[int] = None
    ) -> List[Company]:
        """
        Process a Crunchbase CSV file and return enriched Company objects.
        
        Args:
            csv_file_path: Path to the Crunchbase CSV file
            max_companies: Maximum number of companies to process (for testing)
        
        Returns:
            List of enriched Company objects
        """
        try:
            logger.info(f"🔄 Processing Crunchbase CSV: {csv_file_path}")
            
            # Load and filter companies from CSV
            companies = await self._load_companies_from_csv(csv_file_path, max_companies)
            
            if not companies:
                logger.warning("No companies found in CSV file")
                return []
            
            # Enrich companies with additional data
            enriched_companies = await self._enrich_companies_batch(companies)
            
            logger.info(f"✅ Enrichment complete: {len(enriched_companies)} companies processed")
            return enriched_companies
            
        except Exception as e:
            logger.error(f"Error processing Crunchbase CSV: {e}")
            raise
    
    async def _load_companies_from_csv(
        self, 
        csv_file_path: str, 
        max_companies: Optional[int] = None
    ) -> List[Company]:
        """Load companies from Crunchbase CSV and filter out closed companies."""
        companies = []
        file_path = Path(csv_file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                
                for row_num, row in enumerate(reader, 1):
                    # Parse company data from CSV (no longer filtering out closed companies)
                    company = self._parse_company_from_row(row)
                    if company:
                        companies.append(company)
                    
                    # Limit for testing purposes
                    if max_companies and len(companies) >= max_companies:
                        break
                        
        except Exception as e:
            logger.error(f"Error reading CSV file {csv_file_path}: {e}")
            raise
        
        logger.info(f"📂 Loaded {len(companies)} companies from {file_path.name}")
        return companies
    
    def _parse_company_from_row(self, row: Dict[str, str]) -> Optional[Company]:
        """Parse a Company object from a CSV row."""
        try:
            # Extract basic company information
            name = row.get('Organization Name', '').strip()
            if not name:
                return None
            
            description = row.get('Description', '').strip()
            full_description = row.get('Full Description', '').strip()
            # Use full description if available, otherwise use short description
            final_description = full_description if full_description else description
            
            # Parse founded year
            founded_date = row.get('Founded Date', '').strip()
            founded_year = None
            if founded_date:
                try:
                    # Handle different date formats (YYYY-MM-DD, YYYY, etc.)
                    if '-' in founded_date:
                        founded_year = int(founded_date.split('-')[0])
                    else:
                        founded_year = int(founded_date)
                except (ValueError, IndexError):
                    pass
            
            # Parse funding information
            total_funding_usd = self._parse_float(row.get('Total Funding Amount (in USD)', ''))
            last_funding_amount_usd = self._parse_float(row.get('Last Funding Amount (in USD)', ''))
            number_of_funding_rounds = self._parse_int(row.get('Number of Funding Rounds', ''))
            
            # Parse last funding date
            last_funding_date = row.get('Last Funding Date', '').strip()
            
            # Parse location
            headquarters = row.get('Headquarters Location', '').strip()
            city, region, country = self._parse_location(headquarters)
            
            # Parse founders and investors
            founders_str = row.get('Founders', '').strip()
            founders = [f.strip() for f in founders_str.split(',') if f.strip()] if founders_str else []
            
            investors_str = row.get('Top 5 Investors', '').strip()
            investors = [i.strip() for i in investors_str.split(',') if i.strip()] if investors_str else []
            
            # Parse employee count
            employee_count = self._parse_employee_count(row.get('Number of Employees', ''))
            
            # Parse URLs
            website = row.get('Website', '').strip()
            linkedin_url = row.get('LinkedIn', '').strip()
            crunchbase_url = row.get('Organization Name URL', '').strip()
            
            # Parse industries/categories
            industries = row.get('Industries', '').strip()
            categories = [cat.strip() for cat in industries.split(',') if cat.strip()] if industries else []
            
            # Parse operating status
            operating_status = row.get('Operating Status', '').strip()
            
            return Company(
                name=name,
                description=final_description,
                short_description=description,
                website=website if website else None,
                founded_year=founded_year,
                funding_total_usd=total_funding_usd,
                funding_stage=None,  # Will be enriched later
                founders=founders,
                investors=investors,
                categories=categories,
                city=city,
                region=region,
                country=country,
                sector=None,  # Will be enriched later
                ai_focus=None,
                linkedin_url=linkedin_url if linkedin_url else None,
                crunchbase_url=crunchbase_url if crunchbase_url else None,
                source_url=crunchbase_url if crunchbase_url else None,
                employee_count=employee_count,
                extraction_date=datetime.now().isoformat(),
                # Additional fields for enrichment
                last_funding_amount_usd=last_funding_amount_usd,
                number_of_funding_rounds=number_of_funding_rounds,
                last_funding_date=last_funding_date,
                operating_status=operating_status,
                market_metrics=None  # Will be enriched later
            )
            
        except Exception as e:
            logger.warning(f"Error parsing company row: {e}")
            return None
    
    def _parse_float(self, value: str) -> Optional[float]:
        """Parse float value from string, handling empty/invalid values."""
        if not value or not value.strip():
            return None
        try:
            return float(value.replace(',', ''))
        except (ValueError, TypeError):
            return None
    
    def _parse_int(self, value: str) -> Optional[int]:
        """Parse integer value from string, handling empty/invalid values."""
        if not value or not value.strip():
            return None
        try:
            return int(value.replace(',', ''))
        except (ValueError, TypeError):
            return None
    
    def _parse_employee_count(self, value: str) -> Optional[int]:
        """Parse employee count, handling ranges like '11-50'."""
        if not value or not value.strip():
            return None
        
        value = value.strip()
        
        # Handle ranges like '11-50', '1-10', etc.
        if '-' in value:
            try:
                parts = value.split('-')
                if len(parts) == 2:
                    # Take the midpoint of the range
                    min_val = int(parts[0])
                    max_val = int(parts[1])
                    return (min_val + max_val) // 2
            except (ValueError, IndexError):
                pass
        
        # Handle exact numbers
        try:
            return int(value.replace(',', '').replace('+', ''))
        except (ValueError, TypeError):
            return None
    
    def _parse_location(self, headquarters: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Parse location string into city, region, country."""
        if not headquarters:
            return None, None, None
        
        # Split by comma and clean up
        parts = [part.strip() for part in headquarters.split(',')]
        
        if len(parts) >= 3:
            return parts[0], parts[1], parts[2]
        elif len(parts) == 2:
            return parts[0], None, parts[1]
        elif len(parts) == 1:
            return parts[0], None, None
        else:
            return None, None, None
    
    async def _enrich_companies_batch(
        self, 
        companies: List[Company], 
        batch_size: int = 3
    ) -> List[Company]:
        """Enrich companies with sector classification, market analysis, and funding stage."""
        enriched_companies = []
        
        for i in range(0, len(companies), batch_size):
            batch = companies[i:i + batch_size]
            batch_num = i//batch_size + 1
            total_batches = (len(companies) + batch_size - 1) // batch_size
            logger.info(f"🔄 Enriching batch {batch_num}/{total_batches} ({len(batch)} companies) - adding sector classification, market analysis, and funding stages")
            
            # Process batch concurrently
            batch_tasks = [self._enrich_single_company(company) for company in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to enrich {batch[j].name}: {result}")
                    # Keep original company if enrichment fails
                    enriched_companies.append(batch[j])
                else:
                    enriched_companies.append(result)
        
        return enriched_companies
    
    async def _enrich_single_company(self, company: Company) -> Company:
        """Enrich a single company with all additional data."""
        try:
            # Step 1: Get sector classification first
            sector = await self._get_sector_classification(company)
            if sector:
                company.sector = sector
            
            # Step 2: Run market analysis and funding stage concurrently, using the classified sector
            tasks = [
                self._get_market_analysis(company),  # Now uses company.sector from step 1
                self._get_funding_stage(company)
            ]
            
            market_metrics, funding_stage = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Update company with enriched data
            if not isinstance(market_metrics, Exception) and market_metrics:
                company.market_metrics = market_metrics
            
            if not isinstance(funding_stage, Exception) and funding_stage:
                company.funding_stage = funding_stage
            
            return company
            
        except Exception as e:
            logger.error(f"Error enriching company {company.name}: {e}")
            return company
    
    async def _get_sector_classification(self, company: Company) -> Optional[str]:
        """Get sector classification for the company."""
        try:
            sector = await sector_description_service.get_sector_description(
                company_name=company.name,
                company_description=company.description or "",
                website_content="",
                additional_context=f"Categories: {', '.join(company.categories)}"
            )
            return sector
        except Exception as e:
            logger.warning(f"Sector classification failed for {company.name}: {e}")
            return None
    
    async def _get_market_analysis(self, company: Company):
        """Get market analysis for the company."""
        try:
            async with PerplexityMarketAnalysis() as market_analyzer:
                market_metrics = await market_analyzer.analyze_market(
                    sector=company.sector or "AI Software",
                    year=company.founded_year or 2024,
                    company_name=company.name,
                    include_text_analysis=False  # Only get numerical metrics for CSV
                )
                return market_metrics
        except Exception as e:
            logger.warning(f"Market analysis failed for {company.name}: {e}")
            return None
    
    async def _get_funding_stage(self, company: Company) -> Optional[str]:
        """Get funding stage classification for the company."""
        try:
            funding_stage = await funding_stage_detection_service.detect_funding_stage(
                company_name=company.name,
                description=company.description or "",
                total_funding_usd=company.funding_total_usd,
                last_funding_amount_usd=getattr(company, 'last_funding_amount_usd', None),
                number_of_funding_rounds=getattr(company, 'number_of_funding_rounds', None),
                founded_year=company.founded_year
            )
            return funding_stage
        except Exception as e:
            logger.warning(f"Funding stage detection failed for {company.name}: {e}")
            return None


# Global instance for easy import
company_enrichment_service = CompanyEnrichmentService()