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
    
    class Config:
        json_encoders = {
            HttpUrl: str
        }


class LinkedInProfile(BaseModel):
    """LinkedIn profile data model."""
    person_name: str
    company_name: Optional[str] = None
    linkedin_url: Optional[str] = None
    current_position: Optional[str] = None
    experience_years: Optional[int] = None
    education: Optional[List[str]] = Field(default_factory=list)
    skills: Optional[List[str]] = Field(default_factory=list)
    previous_companies: Optional[List[str]] = Field(default_factory=list)
    summary: Optional[str] = None
    location: Optional[str] = None
    connections_count: Optional[int] = None
    followers_count: Optional[int] = None
    profile_picture_url: Optional[str] = None
    verified: bool = False
    premium: bool = False
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)


class MarketMetrics(BaseModel):
    """Market analysis metrics."""
    market_size_usd: Optional[float] = None
    growth_rate: Optional[float] = None
    competition_level: Optional[str] = None
    market_stage: Optional[MarketStage] = None
    key_trends: Optional[List[str]] = Field(default_factory=list)
    major_players: Optional[List[str]] = Field(default_factory=list)
    barriers_to_entry: Optional[List[str]] = Field(default_factory=list)
    opportunities: Optional[List[str]] = Field(default_factory=list)
    threats: Optional[List[str]] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)


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
