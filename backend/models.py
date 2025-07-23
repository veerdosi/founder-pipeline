"""Core data models for the Initiation Pipeline."""

from pydantic import BaseModel, HttpUrl, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import date, datetime
from uuid import uuid4


class FundingStage(str, Enum):
    """Funding stage enumeration."""
    PRE_SEED = "pre-seed"
    SEED = "seed"
    SERIES_A = "series-a"
    SERIES_B = "series-b"
    SERIES_C = "series-c"
    GROWTH = "growth"
    IPO = "ipo"
    ACQUIRED = "acquired"
    UNKNOWN = "unknown"


class MarketStage(str, Enum):
    """Market stage enumeration."""
    EARLY = "early"
    GROWTH = "growth"
    MATURE = "mature"
    DECLINING = "declining"
    UNKNOWN = "unknown"


class Company(BaseModel):
    """Company data model."""
    uuid: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: Optional[str] = None
    short_description: Optional[str] = None
    website: Optional[HttpUrl] = None
    founded_year: Optional[int] = None
    ai_focus: Optional[str] = None
    sector: Optional[str] = None
    funding_total_usd: Optional[float] = None
    funding_stage: Optional[FundingStage] = None
    city: Optional[str] = None
    region: Optional[str] = None
    country: Optional[str] = None
    founders: Optional[List[str]] = Field(default_factory=list)
    investors: Optional[List[str]] = Field(default_factory=list)
    categories: Optional[List[str]] = Field(default_factory=list)
    linkedin_url: Optional[str] = None
    crunchbase_url: Optional[str] = None
    employee_count: Optional[int] = None
    revenue_millions: Optional[float] = None
    valuation_millions: Optional[float] = None
    last_funding_date: Optional[date] = None
    tech_stack: Optional[List[str]] = Field(default_factory=list)
    competitors: Optional[List[str]] = Field(default_factory=list)
    source_url: Optional[str] = None
    extraction_date: Optional[datetime] = None
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    data_quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    market_metrics: Optional["MarketMetrics"] = None
    
    class Config:
        json_encoders = {
            HttpUrl: str
        }


class MediaCoverageData(BaseModel):
    """Media coverage and public presence data model."""
    media_mentions_count: Optional[int] = None
    awards_and_recognitions: Optional[List[str]] = Field(default_factory=list)
    speaking_engagements: Optional[List[str]] = Field(default_factory=list)
    social_media_followers: Optional[int] = None
    thought_leadership_score: Optional[float] = None
    overall_sentiment: Optional[str] = None


class FinancialProfileData(BaseModel):
    """Financial profile and investment data model."""
    companies_founded: Optional[List[str]] = Field(default_factory=list)
    investment_activities: Optional[List[str]] = Field(default_factory=list)
    board_positions: Optional[List[str]] = Field(default_factory=list)
    notable_achievements: Optional[List[str]] = Field(default_factory=list)
    estimated_net_worth: Optional[str] = None
    confidence_level: Optional[str] = None


class LinkedInProfile(BaseModel):
    """LinkedIn profile data model."""
    person_name: str  # Field expected by ranking system
    company_name: Optional[str] = None
    linkedin_url: Optional[str] = None
    title: Optional[str] = None  # Simplified from current_position
    role: Optional[str] = None  # Role extracted from title (CEO, Founder, etc.)
    about: Optional[str] = None  # Simplified from summary
    location: Optional[str] = None
    
    # Founder identity mapping
    founder_name: Optional[str] = None  # Original founder name from Company.founders that this profile matches
    
    # Structured data that ranking system expects
    experience: Optional[List[Dict[str, str]]] = Field(default_factory=list)  # [{'title': '', 'company': ''}]
    education: Optional[List[Dict[str, str]]] = Field(default_factory=list)   # [{'school': '', 'degree': ''}]
    skills: Optional[List[str]] = Field(default_factory=list)
    
    # Optional metadata
    estimated_age: Optional[int] = None
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    
    @classmethod
    def from_csv_row(cls, row: dict) -> "LinkedInProfile":
        """Create LinkedInProfile from CSV row data."""
        return cls(
            person_name=row.get("founder_name", ""),
            company_name=row.get("company_name"),
            linkedin_url=row.get("linkedin_url"),
            title=row.get("title"),
            role=row.get("role"),
            about=row.get("about"),
            location=row.get("location"),
            founder_name=row.get("founder_name"),
            experience=[],  # These would need to be parsed from complex CSV format
            education=[],   # These would need to be parsed from complex CSV format
            skills=[],      # These would need to be parsed from complex CSV format
            estimated_age=int(row["estimated_age"]) if row.get("estimated_age") else None,
            confidence_score=float(row.get("confidence_score", 0.0))
        )
    
    # Enhanced data from Perplexity
    media_coverage: Optional[MediaCoverageData] = None
    financial_profile: Optional[FinancialProfileData] = None
    
    # Individual experience fields for ranking system compatibility
    experience_1_title: Optional[str] = None
    experience_1_company: Optional[str] = None
    experience_2_title: Optional[str] = None
    experience_2_company: Optional[str] = None
    experience_3_title: Optional[str] = None
    experience_3_company: Optional[str] = None
    
    # Individual education fields for ranking system compatibility
    education_1_school: Optional[str] = None
    education_1_degree: Optional[str] = None
    education_2_school: Optional[str] = None
    education_2_degree: Optional[str] = None
    
    # Individual skill fields for ranking system compatibility
    skill_1: Optional[str] = None
    skill_2: Optional[str] = None
    skill_3: Optional[str] = None
    skill_4: Optional[str] = None
    skill_5: Optional[str] = None
    
    # Fields for ranking system compatibility
    l_level: Optional[str] = None  # L1-L10 ranking result
    reasoning: Optional[str] = None  # Ranking reasoning


class MarketMetrics(BaseModel):
    """Market analysis metrics."""
    # Store researched founding year temporarily to update company data
    researched_founding_year: Optional[int] = None
    
    # Numerical metrics
    market_size_usd: Optional[float] = None
    market_size_billion: Optional[float] = None
    growth_rate: Optional[float] = None
    cagr_percent: Optional[float] = None
    timing_score: Optional[float] = None
    us_sentiment: Optional[float] = None
    sea_sentiment: Optional[float] = None
    competitor_count: Optional[int] = None
    total_funding_billion: Optional[float] = None
    momentum_score: Optional[float] = None
    competition_level: Optional[str] = None
    market_stage: Optional[MarketStage] = None
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    analysis_date: Optional[datetime] = None
    execution_time: Optional[float] = None
    
    # Comprehensive text analysis
    market_overview: Optional[str] = None
    market_size_analysis: Optional[str] = None
    growth_drivers: Optional[str] = None
    timing_analysis: Optional[str] = None
    regional_analysis: Optional[str] = None
    competitive_landscape: Optional[str] = None
    investment_climate: Optional[str] = None
    regulatory_environment: Optional[str] = None
    technology_trends: Optional[str] = None
    consumer_adoption: Optional[str] = None
    supply_chain_analysis: Optional[str] = None
    risk_assessment: Optional[str] = None
    strategic_recommendations: Optional[str] = None
    
    # Structured lists (enhanced)
    key_trends: Optional[List[str]] = Field(default_factory=list)
    major_players: Optional[List[str]] = Field(default_factory=list)
    barriers_to_entry: Optional[List[str]] = Field(default_factory=list)
    opportunities: Optional[List[str]] = Field(default_factory=list)
    threats: Optional[List[str]] = Field(default_factory=list)
    regulatory_changes: Optional[List[str]] = Field(default_factory=list)
    emerging_technologies: Optional[List[str]] = Field(default_factory=list)


class EnrichedCompany(BaseModel):
    """Enriched company with profiles and market data."""
    company: Company
    profiles: List[LinkedInProfile] = Field(default_factory=list)
    market_metrics: Optional[MarketMetrics] = None
    funding_history: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    competitors: Optional[List[str]] = Field(default_factory=list)
    news_mentions: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    patent_count: Optional[int] = None
    employee_count: Optional[int] = None
    tech_stack: Optional[List[str]] = Field(default_factory=list)
    social_media: Optional[Dict[str, str]] = Field(default_factory=dict)
    enrichment_timestamp: datetime = Field(default_factory=datetime.now)
    data_sources: List[str] = Field(default_factory=list)


class PipelineResult(BaseModel):
    """Pipeline execution result."""
    companies: List[EnrichedCompany] = Field(default_factory=list)
    total_discovered: int = 0
    total_enriched: int = 0
    execution_time_seconds: float = 0.0
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    checkpoint_data: Optional[Dict[str, Any]] = None
    config_used: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class SearchFilter(BaseModel):
    """Search filter configuration."""
    categories: Optional[List[str]] = None
    regions: Optional[List[str]] = None
    funding_stages: Optional[List[FundingStage]] = None
    founded_after: Optional[date] = None
    founded_before: Optional[date] = None
    min_funding: Optional[float] = None
    max_funding: Optional[float] = None
    min_employees: Optional[int] = None
    max_employees: Optional[int] = None


class DataSource(BaseModel):
    """Data source configuration."""
    name: str
    enabled: bool = True
    priority: int = 1
    rate_limit: Optional[int] = None
    api_key_required: bool = False
    config: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ProcessingStatus(BaseModel):
    """Processing status for async operations."""
    job_id: str
    status: str  # "pending", "running", "completed", "failed"
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    message: Optional[str] = None
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    result_count: int = 0
