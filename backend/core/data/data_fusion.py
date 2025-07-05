"""Multi-source data fusion service for comprehensive company intelligence."""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import statistics

from ...models import Company, EnrichedCompany

import logging
logger = logging.getLogger(__name__)
from ..analysis.metrics_extraction import MetricsExtractor
from ..analysis.sector_classification import SectorClassifier, SectorClassification
from .crunchbase_integration import CrunchbaseService, CrunchbaseCompany


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
        self.sector_classifier = SectorClassifier()
        self.crunchbase = CrunchbaseService()
        
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
        try:
            # Force cleanup any remaining sessions
            if hasattr(self.crunchbase, 'session') and self.crunchbase.session:
                await self.crunchbase.session.close()
        except Exception as e:
            logger.warning(f"Error closing sessions: {e}")
    
    async def fuse_company_data(
        self, 
        base_company: Company,
        website_content: str = "",
        additional_sources: Optional[Dict[str, Any]] = None
    ) -> FusedCompanyData:
        """Fuse data from multiple sources to create comprehensive company profile."""
        crunchbase_data = None
        enhanced_metrics = {}
        sector_classification = None
        
        try:
            logger.info(f"🔄 Fusing data for {base_company.name}")
            
            # Get crunchbase data with proper session management
            try:
                async with self.crunchbase as cb_service:
                    crunchbase_data = await asyncio.wait_for(
                        cb_service.enrich_existing_company(
                            base_company.name, 
                            str(base_company.website) if base_company.website else None
                        ),
                        timeout=30  # 30 second timeout for crunchbase
                    )
            except asyncio.TimeoutError:
                logger.warning(f"Crunchbase lookup timeout for {base_company.name}")
                crunchbase_data = None
            except Exception as e:
                logger.warning(f"Crunchbase lookup failed for {base_company.name}: {e}")
                crunchbase_data = None
            
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
            
            # Get sector classification
            try:
                sector_classification = await asyncio.wait_for(
                    self.sector_classifier.classify_company(
                        company_name=base_company.name,
                        description=base_company.description or "",
                        website_content=website_content,
                        additional_context=f"AI Focus: {base_company.ai_focus or 'N/A'}"
                    ),
                    timeout=30
                )
            except Exception as e:
                logger.warning(f"Sector classification failed for {base_company.name}: {e}")
                sector_classification = None
            
            # Fuse all data sources
            fused_data = self._fuse_data_sources(
                base_company=base_company,
                crunchbase_data=crunchbase_data,
                enhanced_metrics=enhanced_metrics,
                sector_classification=sector_classification,
                website_content=website_content,
                additional_sources=additional_sources or {}
            )
            
            logger.info(f"✅ Data fusion complete for {base_company.name} (quality: {fused_data.data_quality_score:.2f})")
            return fused_data
            
        except Exception as e:
            logger.error(f"Error in data fusion for {base_company.name}: {e}")
            return self._create_fallback_data(base_company)
    
    async def batch_fuse_companies(
        self, 
        companies: List[Company],
        batch_size: int = 3  # Reduced batch size for better reliability
    ) -> List[FusedCompanyData]:
        """Fuse data for multiple companies in smaller, more reliable batches."""
        fused_companies = []
        
        for i in range(0, len(companies), batch_size):
            batch = companies[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(companies) + batch_size - 1)//batch_size}")
            
            # Process companies sequentially within batch to avoid overwhelming APIs
            for company in batch:
                try:
                    result = await asyncio.wait_for(
                        self.fuse_company_data(company),
                        timeout=90  # 90 second timeout per company
                    )
                    
                    if result is not None:
                        fused_companies.append(result)
                        
                    # Small delay to be nice to APIs
                    await asyncio.sleep(0.5)
                        
                except asyncio.TimeoutError:
                    logger.error(f"Timeout processing {company.name}")
                    # Add fallback data instead of skipping
                    fallback = self._create_fallback_data(company)
                    fused_companies.append(fallback)
                except Exception as e:
                    logger.error(f"Error processing {company.name}: {e}")
                    # Add fallback data instead of skipping
                    fallback = self._create_fallback_data(company)
                    fused_companies.append(fallback)
        
        return fused_companies
    
    def _fuse_data_sources(
        self,
        base_company: Company,
        crunchbase_data: Optional[CrunchbaseCompany],
        enhanced_metrics: Dict[str, Any],
        sector_classification: Optional[SectorClassification],
        website_content: str,
        additional_sources: Dict[str, Any]
    ) -> FusedCompanyData:
        """Fuse data from multiple sources using intelligent conflict resolution."""
        
        # Track data sources used
        data_sources = ['exa']  # Base company always from Exa
        
        if crunchbase_data:
            data_sources.append('crunchbase')
        if enhanced_metrics:
            data_sources.append('website_extraction')
        if sector_classification:
            data_sources.append('ai_classification')
        
        # Core company information (with conflict resolution)
        name = self._resolve_field(
            [base_company.name, crunchbase_data.name if crunchbase_data else None],
            ['exa', 'crunchbase']
        )
        
        description = self._resolve_field(
            [base_company.description, crunchbase_data.description if crunchbase_data else None],
            ['exa', 'crunchbase']
        )
        
        website = self._resolve_field(
            [str(base_company.website) if base_company.website else None, 
             crunchbase_data.website if crunchbase_data else None],
            ['exa', 'crunchbase']
        )
        
        # Founded year with validation
        founded_year = self._resolve_numeric_field(
            [base_company.founded_year, 
             int(crunchbase_data.founded_date.split('-')[0]) if crunchbase_data and crunchbase_data.founded_date else None],
            ['exa', 'crunchbase']
        )
        
        # Sector classification (prioritize AI classification)
        if sector_classification:
            primary_sector = sector_classification.primary_sector
            sub_sectors = sector_classification.sub_sectors
            ai_focus = sector_classification.ai_focus
            technology_stack = sector_classification.technology_stack
            business_model = sector_classification.business_model
            target_market = sector_classification.target_market
        else:
            primary_sector = base_company.ai_focus or "machine_learning"
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
            crunchbase_data.linkedin_url if crunchbase_data else None
        ])
        
        crunchbase_url = crunchbase_data.crunchbase_url if crunchbase_data else None
        
        # Calculate data quality and confidence scores
        data_quality_score = self._calculate_overall_data_quality(
            enhanced_metrics.get('data_quality_score', 0.0),
            crunchbase_data.data_quality_score if crunchbase_data else 0.0,
            sector_classification.confidence_score if sector_classification else 0.0
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
    
    def _create_fallback_data(self, base_company: Company) -> FusedCompanyData:
        """Create fallback data when fusion fails."""
        return FusedCompanyData(
            name=base_company.name,
            description=base_company.description or "",
            website=str(base_company.website) if base_company.website else None,
            founded_year=base_company.founded_year,
            primary_sector=base_company.ai_focus or "machine_learning",
            sub_sectors=[],
            ai_focus=base_company.ai_focus or "Artificial Intelligence",
            technology_stack=[],
            business_model="b2b_saas",
            target_market="Enterprise",
            total_funding_usd=base_company.funding_total_usd,
            latest_funding_usd=None,
            funding_stage=base_company.funding_stage,
            current_valuation_usd=None,
            annual_revenue_usd=None,
            employee_count=None,
            customer_count=None,
            founders=base_company.founders or [],
            key_investors=[],
            headquarters_location=f"{base_company.city}, {base_company.country}" if base_company.city and base_company.country else None,
            linkedin_url=None,
            crunchbase_url=None,
            data_sources=['exa'],
            data_quality_score=0.3,
            confidence_score=0.2,
            last_updated=datetime.now().isoformat(),
            fusion_timestamp=datetime.now().isoformat()
        )
    
    def to_dict(self, fused_data: FusedCompanyData) -> Dict[str, Any]:
        """Convert fused data to dictionary for export."""
        return asdict(fused_data)
