"""Pydantic models for API requests and responses."""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime


class CompanyDiscoveryRequest(BaseModel):
    limit: int = 50
    categories: List[str] = []
    regions: List[str] = []
    sources: List[str] = ["techcrunch", "crunchbase", "ycombinator"]


class DiscoveryJobResponse(BaseModel):
    jobId: str
    status: str
    companiesFound: int
    message: str


class CompanyResponse(BaseModel):
    id: str
    name: str
    description: str
    website: Optional[str] = None
    foundedYear: Optional[int] = None
    fundingTotal: Optional[float] = None
    fundingStage: Optional[str] = None
    founders: List[str] = []
    location: str
    aiCategory: str
    source: str


class RankingJobResponse(BaseModel):
    jobId: str
    status: str
    foundersRanked: int
    highConfidenceCount: int
    message: str


class FounderRankingResponse(BaseModel):
    id: str
    name: str
    company: str
    level: str
    confidenceScore: float
    reasoning: str
    evidence: List[str]
    verificationSources: List[str]
    timestamp: str


class DashboardStats(BaseModel):
    totalCompanies: int
    totalFounders: int
    rankedFounders: int
    avgConfidenceScore: float
    levelDistribution: Dict[str, int]
    recentActivity: List[Dict[str, Any]]
