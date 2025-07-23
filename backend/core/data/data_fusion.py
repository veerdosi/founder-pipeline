"""Multi-source data fusion service for comprehensive company intelligence."""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import statistics

from ...models import Company, EnrichedCompany, MarketMetrics

import logging
logger = logging.getLogger(__name__)
from ..analysis.metrics_extraction import MetricsExtractor
from ..analysis.sector_classification import sector_description_service
from ..analysis.market_analysis import PerplexityMarketAnalysis

logger = logging.getLogger(__name__)


@dataclass
class FusedCompanyData:
    """Comprehensive company data fused from multiple sources."""
    # Core company info
    name: str
    description: str
    website: Optional[str]
    founded_year: Optional[int]
    
    # Enhanced sector classification
    primary_sector: str
    sub_sectors: List[str]
    ai_focus: str
    technology_stack: List[str]
    business_model: str
    target_market: str
    
    # Financial metrics (fused from multiple sources)
    total_funding_usd: Optional[float]
    latest_funding_usd: Optional[float]
    funding_stage: Optional[str]
    current_valuation_usd: Optional[float]
    annual_revenue_usd: Optional[float]
    
    # Operational metrics
    employee_count: Optional[int]
    customer_count: Optional[int]
    
    # Founder and investor data
    founders: List[str]
    key_investors: List[str]
    
    # Location data
    headquarters_location: Optional[str]
    
    # Social and web presence
    linkedin_url: Optional[str]
    crunchbase_url: Optional[str]
    
    # Market analysis data
    market_metrics: Optional[MarketMetrics]
    
    # Data quality and confidence
    data_sources: List[str]
    data_quality_score: float
    confidence_score: float
    
    # Metadata
    last_updated: str
    fusion_timestamp: str


class DataFusionService:
    """Service for fusing data from multiple sources into comprehensive company profiles."""
    
    def __init__(self):
        self.metrics_extractor = MetricsExtractor()
        
        # Confidence weights for different data sources
        self.source_weights = {
            'crunchbase': 0.9,
            'exa': 0.7,
            'website_extraction': 0.6,
            'ai_classification': 0.8,
            'regex_extraction': 0.5
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - ensure all sessions are closed."""
        pass  # No persistent sessions to close since we use fresh CrunchbaseService instances
    
    async def fuse_company_data(
        self, 
        base_company: Company,
        website_content: str = "",
        additional_sources: Optional[Dict[str, Any]] = None,
        target_year: Optional[int] = None
    ) -> FusedCompanyData:
        """Fuse data from multiple sources to create comprehensive company profile (without Crunchbase)."""
        enhanced_metrics = {}
        sector_description = None
        market_metrics = None
        
        try:
            logger.info(f"🔄 Fusing data for {base_company.name}")
                        
            # Get enhanced metrics
            try:
                if website_content:
                    enhanced_metrics = await asyncio.wait_for(
                        self.metrics_extractor.extract_comprehensive_metrics(
                            website_content, base_company.name
                        ),
                        timeout=30
                    )
            except Exception as e:
                logger.warning(f"Enhanced metrics failed for {base_company.name}: {e}")
                enhanced_metrics = {}
            
            # Get sector description
            try:
                sector_description = await asyncio.wait_for(
                    sector_description_service.get_sector_description(
                        company_name=base_company.name,
                        company_description=base_company.description or "",
                        website_content=website_content,
                        additional_context=f"AI Focus: {base_company.ai_focus or 'N/A'}"
                    ),
                    timeout=30
                )
            except Exception as e:
                logger.warning(f"Sector description failed for {base_company.name}: {e}")
                sector_description = None
            
            # Get market analysis metrics
            try:
                async with PerplexityMarketAnalysis() as market_analyzer:
                    market_metrics = await asyncio.wait_for(
                        market_analyzer.analyze_market(
                            sector=base_company.sector or sector_description or "AI Software",
                            year=base_company.founded_year or target_year or 2024,
                            company_name=base_company.name
                        ),
                        timeout=60  # 60 second timeout for market analysis
                    )
                    logger.info(f"📊 Market analysis complete for {base_company.name}")
            except asyncio.TimeoutError:
                logger.warning(f"Market analysis timeout for {base_company.name}")
                market_metrics = None
            except Exception as e:
                logger.warning(f"Market analysis failed for {base_company.name}: {e}")
                market_metrics = None
            
            # Fuse all data sources (without Crunchbase)
            fused_data = self._fuse_data_sources(
                base_company=base_company,
                enhanced_metrics=enhanced_metrics,
                sector_description=sector_description,
                market_metrics=market_metrics,
                website_content=website_content,
                additional_sources=additional_sources or {},
                target_year=target_year
            )
            
            logger.info(f"✅ Data fusion complete for {base_company.name} (quality: {fused_data.data_quality_score:.2f})")
            return fused_data
            
        except Exception as e:
            logger.error(f"Error in data fusion for {base_company.name}: {e}")
            raise e
    
    async def batch_fuse_companies(
        self, 
        companies, 
        batch_size: int = 3,
        target_year: Optional[int] = None
    ) -> List[Company]:
        """Batch process companies through data fusion with target year fallback."""
        enhanced_companies = []
        
        for i in range(0, len(companies), batch_size):
            batch = companies[i:i + batch_size]
            batch_tasks = []
            
            for company_item in batch:
                # Handle both Company and EnrichedCompany objects
                if hasattr(company_item, 'company'):
                    # This is an EnrichedCompany object
                    base_company = company_item.company
                else:
                    # This is a Company object
                    base_company = company_item
                
                task = self.fuse_company_data(
                    base_company=base_company,
                    website_content="",
                    additional_sources=None,
                    target_year=target_year
                )
                batch_tasks.append(task)
            
            # Process batch concurrently
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    # Get the company name safely for logging
                    company_item = batch[j]
                    if hasattr(company_item, 'company'):
                        company_name = company_item.company.name
                        base_company = company_item.company
                    else:
                        company_name = company_item.name
                        base_company = company_item
                    
                    logger.error(f"Failed to fuse data for {company_name}: {result}")
                    # Keep original company if fusion fails
                    enhanced_companies.append(base_company)
                else:
                    # Get the original company for conversion
                    company_item = batch[j]
                    if hasattr(company_item, 'company'):
                        original_company = company_item.company
                    else:
                        original_company = company_item
                    
                    # Convert FusedCompanyData back to Company model
                    enhanced_company = self._convert_fused_to_company(result, original_company)
                    enhanced_companies.append(enhanced_company)
        
        return enhanced_companies
    
    def _convert_fused_to_company(self, fused_data: FusedCompanyData, original_company: Company) -> Company:
        """Convert FusedCompanyData back to Company model with enhanced data."""
        # Update the original company with fused data
        original_company.name = fused_data.name
        original_company.description = fused_data.description
        original_company.website = fused_data.website
        original_company.founded_year = fused_data.founded_year
        original_company.ai_focus = fused_data.ai_focus
        original_company.sector = fused_data.primary_sector
        original_company.funding_total_usd = fused_data.total_funding_usd
        original_company.funding_stage = fused_data.funding_stage
        original_company.founders = fused_data.founders
        original_company.investors = fused_data.key_investors
        original_company.linkedin_url = fused_data.linkedin_url
        original_company.crunchbase_url = fused_data.crunchbase_url
        original_company.employee_count = fused_data.employee_count
        original_company.confidence_score = fused_data.confidence_score
        original_company.data_quality_score = fused_data.data_quality_score
        original_company.market_metrics = fused_data.market_metrics
        
        # Update founding year with the researched year from market analysis
        if fused_data.market_metrics and hasattr(fused_data.market_metrics, 'researched_founding_year') and fused_data.market_metrics.researched_founding_year:
            original_company.founded_year = fused_data.market_metrics.researched_founding_year
        
        return original_company
    
    
    def _fuse_data_sources(
        self,
        base_company: Company,
        enhanced_metrics: Dict[str, Any],
        sector_description: Optional[str],
        market_metrics: Optional[MarketMetrics],
        website_content: str,
        additional_sources: Dict[str, Any],
        target_year: Optional[int] = None
    ) -> FusedCompanyData:
        """Fuse data from multiple sources using intelligent conflict resolution."""
        
        data_sources = ['exa_websets']  # Base company from enriched websets
        
        if enhanced_metrics:
            data_sources.append('website_extraction')
        if sector_description:
            data_sources.append('ai_classification')
        if market_metrics:
            data_sources.append('market_analysis')
        
        # Core company information (simplified without Crunchbase conflict resolution)
        name = base_company.name
        description = base_company.description
        website = str(base_company.website) if base_company.website else None
        
        # Founded year with target year fallback
        founded_year = base_company.founded_year
        
        # Use target_year as fallback if no founded_year found from data sources
        if founded_year is None and target_year is not None:
            founded_year = target_year
        
        # Sector classification (use centralized sector description)
        if sector_description:
            primary_sector = sector_description
            sub_sectors = []
            ai_focus = base_company.ai_focus or "Artificial Intelligence"
            technology_stack = []
            business_model = "b2b_saas"
            target_market = "Enterprise"
        else:
            primary_sector = base_company.sector or "AI Software Solutions"
            sub_sectors = []
            ai_focus = base_company.ai_focus or "Artificial Intelligence"
            technology_stack = []
            business_model = "b2b_saas"
            target_market = "Enterprise"
        
        # Financial metrics (with intelligent fusion)
        total_funding = self._resolve_funding_amount([
            base_company.funding_total_usd,
            crunchbase_data.funding_total if crunchbase_data else None,
            enhanced_metrics.get('total_funding_usd')
        ])
        
        latest_funding = self._resolve_funding_amount([
            crunchbase_data.last_funding_amount if crunchbase_data else None,
            enhanced_metrics.get('latest_funding_usd')
        ])
        
        funding_stage = self._resolve_field(
            [base_company.funding_stage, 
             crunchbase_data.funding_stage if crunchbase_data else None,
             enhanced_metrics.get('funding_stage')],
            ['exa', 'crunchbase', 'website_extraction']
        )
        
        # Valuation and revenue
        current_valuation = enhanced_metrics.get('current_valuation_usd')
        annual_revenue = enhanced_metrics.get('annual_revenue_usd')
        
        # Operational metrics
        employee_count = self._resolve_numeric_field([
            crunchbase_data.employee_count if crunchbase_data else None,
            enhanced_metrics.get('employee_count')
        ])
        
        customer_count = enhanced_metrics.get('customer_count')
        
        # People data
        founders = self._merge_lists([
            base_company.founders or [],
            crunchbase_data.founder_names if crunchbase_data else []
        ])
        
        key_investors = crunchbase_data.key_investors if crunchbase_data else []
        
        # Location data
        headquarters = self._resolve_field([
            f"{base_company.city}, {base_company.country}" if base_company.city and base_company.country else None,
            crunchbase_data.headquarters_location if crunchbase_data else None
        ])
        
        # Social presence
        linkedin_url = self._resolve_field([
            base_company.linkedin_url,
            crunchbase_data.linkedin_url if crunchbase_data else None
        ])
        
        crunchbase_url = self._resolve_field([
            base_company.crunchbase_url,
            crunchbase_data.crunchbase_url if crunchbase_data else None
        ])
        
        # Calculate data quality and confidence scores
        data_quality_score = self._calculate_overall_data_quality(
            enhanced_metrics.get('data_quality_score', 0.0),
            crunchbase_data.data_quality_score if crunchbase_data else 0.0,
            0.8 if sector_description else 0.0  # Fixed confidence score for sector description
        )
        
        confidence_score = self._calculate_confidence_score(data_sources, data_quality_score)
        
        return FusedCompanyData(
            name=name,
            description=description,
            website=website,
            founded_year=founded_year,
            primary_sector=primary_sector,
            sub_sectors=sub_sectors,
            ai_focus=ai_focus,
            technology_stack=technology_stack,
            business_model=business_model,
            target_market=target_market,
            total_funding_usd=total_funding,
            latest_funding_usd=latest_funding,
            funding_stage=funding_stage,
            current_valuation_usd=current_valuation,
            annual_revenue_usd=annual_revenue,
            employee_count=employee_count,
            customer_count=customer_count,
            founders=founders,
            key_investors=key_investors,
            headquarters_location=headquarters,
            linkedin_url=linkedin_url,
            crunchbase_url=crunchbase_url,
            market_metrics=market_metrics,
            data_sources=data_sources,
            data_quality_score=data_quality_score,
            confidence_score=confidence_score,
            last_updated=datetime.now().isoformat(),
            fusion_timestamp=datetime.now().isoformat()
        )
    
    def _resolve_field(self, values: List[Optional[Any]], sources: List[str] = None) -> Optional[Any]:
        """Resolve conflicts between field values from different sources."""
        if not values:
            return None
        
        # Remove None values and get weights
        valid_pairs = []
        sources = sources or ['unknown'] * len(values)
        
        for value, source in zip(values, sources):
            if value is not None and str(value).strip():
                weight = self.source_weights.get(source, 0.5)
                valid_pairs.append((value, weight))
        
        if not valid_pairs:
            return None
        
        # Return highest weighted value
        return max(valid_pairs, key=lambda x: x[1])[0]
    
    def _resolve_numeric_field(self, values: List[Optional[float]], sources: List[str] = None) -> Optional[float]:
        """Resolve numeric field conflicts with statistical methods."""
        valid_values = [v for v in values if v is not None and isinstance(v, (int, float))]
        
        if not valid_values:
            return None
        
        if len(valid_values) == 1:
            return valid_values[0]
        
        # Use median for funding amounts to avoid outliers
        return statistics.median(valid_values)
    
    def _resolve_funding_amount(self, amounts: List[Optional[float]]) -> Optional[float]:
        """Resolve funding amounts with validation and outlier detection."""
        valid_amounts = []
        
        for amount in amounts:
            if amount is not None and isinstance(amount, (int, float)) and amount > 0:
                # Basic validation - reasonable funding range
                if 1000 <= amount <= 100_000_000_000:  # $1K to $100B
                    valid_amounts.append(amount)
        
        if not valid_amounts:
            return None
        
        if len(valid_amounts) == 1:
            return valid_amounts[0]
        
        # If values are close (within 50%), take the average
        min_val, max_val = min(valid_amounts), max(valid_amounts)
        if max_val / min_val <= 1.5:
            return statistics.mean(valid_amounts)
        
        # Otherwise, take the median to avoid outliers
        return statistics.median(valid_amounts)
    
    def _merge_lists(self, lists: List[List[str]]) -> List[str]:
        """Merge multiple lists while removing duplicates."""
        merged = []
        seen = set()
        
        for lst in lists:
            if lst:
                for item in lst:
                    item_clean = str(item).strip().lower()
                    if item_clean and item_clean not in seen:
                        seen.add(item_clean)
                        merged.append(str(item).strip())
        
        return merged
    
    def _calculate_overall_data_quality(self, *quality_scores) -> float:
        """Calculate overall data quality from multiple sources."""
        valid_scores = [score for score in quality_scores if score is not None and 0 <= score <= 1]
        
        if not valid_scores:
            return 0.2  # Low default quality
        
        return statistics.mean(valid_scores)
    
    def _calculate_confidence_score(self, data_sources: List[str], data_quality: float) -> float:
        """Calculate overall confidence score based on sources and quality."""
        if not data_sources:
            return 0.1
        
        # Base score from data sources
        source_score = sum(self.source_weights.get(source, 0.3) for source in data_sources) / len(data_sources)
        
        # Boost for multiple sources
        multi_source_boost = min(len(data_sources) * 0.1, 0.3)
        
        # Combine with data quality
        confidence = (source_score * 0.6) + (data_quality * 0.4) + multi_source_boost
        
        return min(confidence, 1.0)
    
    
    
    
