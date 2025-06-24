"""Core data models for the initiation pipeline."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum

from pydantic import BaseModel, Field, HttpUrl


class FundingStage(str, Enum):
    """Funding stages for companies."""
    SEED = "seed"
    SERIES_A = "series-a"
    SERIES_B = "series-b"
    SERIES_C = "series-c"
    SERIES_D = "series-d"
    IPO = "ipo"
    ACQUIRED = "acquired"
    UNKNOWN = "unknown"


class MarketStage(str, Enum):
    """Market maturity stages."""
    EMERGING = "emerging"
    GROWTH = "growth"
    MATURE = "mature"
    UNKNOWN = "unknown"


class Company(BaseModel):
    """Company data model."""
    uuid: Optional[str] = None
    name: str
    description: Optional[str] = None
    short_description: Optional[str] = None
    website: Optional[HttpUrl] = None
    linkedin_url: Optional[HttpUrl] = None
    founded_year: Optional[int] = None
    funding_total_usd: Optional[float] = None
    funding_stage: Optional[FundingStage] = None
    founders: List[str] = Field(default_factory=list)
    investors: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    
    # Location
    city: Optional[str] = None
    region: Optional[str] = None
    country: Optional[str] = None
    
    # AI specific
    ai_focus: Optional[str] = None
    sector: Optional[str] = None
    
    # Metadata
    source_url: Optional[str] = None
    extraction_date: Optional[datetime] = None
    
    class Config:
        use_enum_values = True


class LinkedInProfile(BaseModel):
    """LinkedIn profile data model."""
    person_name: str
    linkedin_url: HttpUrl
    title: Optional[str] = None
    role: Optional[str] = None
    company_name: Optional[str] = None
    
    # Profile details
    headline: Optional[str] = None
    location: Optional[str] = None
    about: Optional[str] = None
    estimated_age: Optional[int] = None
    
    # Experience (up to 5)
    experience_1_title: Optional[str] = None
    experience_1_company: Optional[str] = None
    experience_2_title: Optional[str] = None
    experience_2_company: Optional[str] = None
    experience_3_title: Optional[str] = None
    experience_3_company: Optional[str] = None
    experience_4_title: Optional[str] = None
    experience_4_company: Optional[str] = None
    experience_5_title: Optional[str] = None
    experience_5_company: Optional[str] = None
    
    # Education (up to 3)
    education_1_school: Optional[str] = None
    education_1_degree: Optional[str] = None
    education_2_school: Optional[str] = None
    education_2_degree: Optional[str] = None
    education_3_school: Optional[str] = None
    education_3_degree: Optional[str] = None
    
    # Skills (up to 5)
    skill_1: Optional[str] = None
    skill_2: Optional[str] = None
    skill_3: Optional[str] = None
    skill_4: Optional[str] = None
    skill_5: Optional[str] = None


class MarketMetrics(BaseModel):
    """Market analysis metrics."""
    market_size_billion: Optional[float] = None
    cagr_percent: Optional[float] = None
    timing_score: Optional[float] = Field(None, ge=1.0, le=5.0)
    us_sentiment: Optional[float] = Field(None, ge=1.0, le=5.0)
    sea_sentiment: Optional[float] = Field(None, ge=1.0, le=5.0)
    competitor_count: Optional[int] = None
    total_funding_billion: Optional[float] = None
    momentum_score: Optional[float] = Field(None, ge=1.0, le=5.0)
    market_stage: Optional[MarketStage] = None
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Analysis metadata
    analysis_date: Optional[datetime] = None
    execution_time: Optional[float] = None


class EnrichedCompany(BaseModel):
    """Complete company data with profiles and market analysis."""
    # Company data
    company: Company
    
    # LinkedIn profiles
    profiles: List[LinkedInProfile] = Field(default_factory=list)
    
    # Market analysis
    market_metrics: Optional[MarketMetrics] = None
    
    # Processing metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None


class PipelineResult(BaseModel):
    """Complete pipeline execution result."""
    companies: List[EnrichedCompany]
    stats: Dict[str, Any]
    execution_time: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    def to_separate_csv_records(self) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Convert to separate company and founder records for CSV export."""
        company_records = []
        founder_records = []
        
        for enriched in self.companies:
            company = enriched.company
            metrics = enriched.market_metrics
            
            # Company record (one per company)
            company_record = {
                "company_uuid": company.uuid,
                "company_name": company.name,
                "description": company.description,
                "short_description": company.short_description,
                "website": str(company.website) if company.website else None,
                "company_linkedin": str(company.linkedin_url) if company.linkedin_url else None,
                "founded_year": company.founded_year,
                "funding_total_usd": company.funding_total_usd,
                "funding_stage": company.funding_stage,
                "founders": "|".join(company.founders) if company.founders else None,
                "investors": "|".join(company.investors) if company.investors else None,
                "categories": "|".join(company.categories) if company.categories else None,
                "city": company.city,
                "region": company.region,
                "country": company.country,
                "ai_focus": company.ai_focus,
                "sector": company.sector,
                
                # Market metrics
                "market_size_billion": metrics.market_size_billion if metrics else None,
                "cagr_percent": metrics.cagr_percent if metrics else None,
                "timing_score": metrics.timing_score if metrics else None,
                "us_sentiment": metrics.us_sentiment if metrics else None,
                "sea_sentiment": metrics.sea_sentiment if metrics else None,
                "competitor_count": metrics.competitor_count if metrics else None,
                "total_funding_billion": metrics.total_funding_billion if metrics else None,
                "momentum_score": metrics.momentum_score if metrics else None,
                "market_stage": metrics.market_stage if metrics else None,
                "confidence_score": metrics.confidence_score if metrics else None,
            }
            company_records.append(company_record)
            
            # Founder records (one per profile)
            if enriched.profiles:
                for profile in enriched.profiles:
                    founder_record = {
                        "company_name": company.name,
                        "person_name": profile.person_name,
                        "linkedin_url": str(profile.linkedin_url),
                        "title": profile.title,
                        "role": profile.role,
                        "headline": profile.headline,
                        "location": profile.location,
                        "about": profile.about,
                        "estimated_age": profile.estimated_age,
                        
                        # Experience
                        "experience_1_title": profile.experience_1_title,
                        "experience_1_company": profile.experience_1_company,
                        "experience_2_title": profile.experience_2_title,
                        "experience_2_company": profile.experience_2_company,
                        "experience_3_title": profile.experience_3_title,
                        "experience_3_company": profile.experience_3_company,
                        
                        # Education
                        "education_1_school": profile.education_1_school,
                        "education_1_degree": profile.education_1_degree,
                        "education_2_school": profile.education_2_school,
                        "education_2_degree": profile.education_2_degree,
                        
                        # Skills
                        "skill_1": profile.skill_1,
                        "skill_2": profile.skill_2,
                        "skill_3": profile.skill_3,
                    }
                    founder_records.append(founder_record)
        
        return company_records, founder_records
