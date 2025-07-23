"""Pydantic models for API requests and responses."""

from pydantic import BaseModel, validator
from typing import List, Optional, Dict, Any
from datetime import datetime, date
from ..core.config import settings

class CompanyDiscoveryRequest(BaseModel):
    limit: int = settings.default_company_limit
    categories: List[str] = []
    regions: List[str] = []
    sources: List[str] = ["techcrunch", "crunchbase", "ycombinator"]
    # Date range for company founding
    founded_after: Optional[date] = None
    founded_before: Optional[date] = None

class AcceleratorDiscoveryRequest(BaseModel):
    """Request model for accelerator-based company discovery."""
    limit: int = settings.default_company_limit
    accelerators: List[str] = ["yc", "techstars", "500co"]  # Which accelerators to search
    
    @validator('accelerators')
    def validate_accelerators(cls, v):
        valid_accelerators = {"yc", "ycombinator", "techstars", "500co", "500global"}
        for accelerator in v:
            if accelerator.lower() not in valid_accelerators:
                raise ValueError(f'Invalid accelerator: {accelerator}. Valid options: {valid_accelerators}')
        return [acc.lower() for acc in v]

class YearBasedRequest(BaseModel):
    """Request model for year-based company discovery."""
    year: int
    limit: int = settings.default_company_limit
    
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