"""Pydantic models for API requests and responses."""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, date


class CompanyDiscoveryRequest(BaseModel):
    limit: int = 50
    categories: List[str] = []
    regions: List[str] = []
    sources: List[str] = ["techcrunch", "crunchbase", "ycombinator"]
    # Date range for company founding
    founded_after: Optional[date] = None
    founded_before: Optional[date] = None


class PipelineJobResponse(BaseModel):
    jobId: str
    status: str
    companiesFound: int
    foundersFound: int
    message: str


class DashboardStats(BaseModel):
    totalCompanies: int
    totalFounders: int
    avgConfidenceScore: float
    levelDistribution: Dict[str, int]
    recentActivity: List[Dict[str, Any]]
