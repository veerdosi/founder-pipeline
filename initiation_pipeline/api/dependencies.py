"""FastAPI dependencies for service injection."""

from ..services.ranking import FounderRankingService
from ..services.company_discovery import ExaCompanyDiscovery  
from ..services.profile_enrichment import LinkedInEnrichmentService


def get_ranking_service() -> FounderRankingService:
    """Get founder ranking service instance."""
    return FounderRankingService()


def get_discovery_service() -> ExaCompanyDiscovery:
    """Get company discovery service instance."""
    return ExaCompanyDiscovery()


def get_enrichment_service() -> LinkedInEnrichmentService:
    """Get founder enrichment service instance.""" 
    return LinkedInEnrichmentService()
