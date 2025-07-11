"""Pydantic models for API requests and responses."""

from pydantic import BaseModel, validator
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

class YearBasedRequest(BaseModel):
    """Request model for year-based company discovery."""
    year: int
    limit: int = 100
    
    @validator('year')
    def validate_year(cls, v):
        current_year = datetime.now().year
        if v < 1990 or v > current_year:
            raise ValueError(f'Year must be between 1990 and {current_year}')
        return v
    
    def to_discovery_request(self) -> CompanyDiscoveryRequest:
        """Convert to standard discovery request format."""
        start_date = date(self.year, 1, 1)
        end_date = date(self.year, 12, 31)
        
        return CompanyDiscoveryRequest(
            limit=self.limit,
            categories=[],
            regions=[],
            sources=["techcrunch", "crunchbase", "ycombinator"],
            founded_after=start_date,
            founded_before=end_date
        )
class PipelineJobResponse(BaseModel):
    jobId: str
    status: str
    companiesFound: int
    foundersFound: int
    message: str