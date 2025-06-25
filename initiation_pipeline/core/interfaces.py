"""Base service interfaces and protocols."""

from abc import ABC, abstractmethod
from typing import List, Optional, Protocol

from ..models import Company, LinkedInProfile, MarketMetrics


class CompanyDiscoveryService(ABC):
    """Abstract base class for company discovery services."""
    
    @abstractmethod
    async def find_companies(
        self, 
        limit: int = 50,
        categories: Optional[List[str]] = None,
        regions: Optional[List[str]] = None
    ) -> List[Company]:
        """Find AI companies based on criteria."""
        pass


class ProfileEnrichmentService(ABC):
    """Abstract base class for LinkedIn profile enrichment."""
    
    @abstractmethod
    async def find_profiles(self, company: Company) -> List[LinkedInProfile]:
        """Find LinkedIn profiles for company executives."""
        pass
    
    @abstractmethod
    async def enrich_profile(self, profile: LinkedInProfile) -> LinkedInProfile:
        """Enrich profile with additional data."""
        pass
    
    async def enrich_profiles_batch(self, profiles: List[LinkedInProfile]) -> List[LinkedInProfile]:
        """Enrich multiple profiles efficiently. Default implementation calls enrich_profile individually."""
        return [await self.enrich_profile(profile) for profile in profiles]


class MarketAnalysisService(ABC):
    """Abstract base class for market analysis."""
    
    @abstractmethod
    async def analyze_market(
        self, 
        sector: str, 
        year: int,
        region: Optional[str] = None
    ) -> MarketMetrics:
        """Analyze market metrics for a sector."""
        pass


class SearchProvider(Protocol):
    """Protocol for search providers."""
    
    async def search(
        self, 
        query: str, 
        limit: int = 10,
        **kwargs
    ) -> List[dict]:
        """Execute a search query."""
        ...


class LLMProvider(Protocol):
    """Protocol for LLM providers."""
    
    async def generate(
        self, 
        prompt: str, 
        model: str = "gpt-4o-mini",
        **kwargs
    ) -> str:
        """Generate text using LLM."""
        ...
    
    async def extract_structured(
        self, 
        prompt: str, 
        response_model,
        **kwargs
    ):
        """Extract structured data using LLM."""
        ...
