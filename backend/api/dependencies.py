"""FastAPI dependencies for service injection."""

from ..core.ranking import FounderRankingService
from ..core.discovery import ExaCompanyDiscovery
from ..core.pipeline import InitiationPipeline
from ..core.data import LinkedInEnrichmentService
from ..core.data import FounderDataPipeline


def get_ranking_service() -> FounderRankingService:
    """Get founder ranking service instance."""
    return FounderRankingService()


def get_discovery_service() -> ExaCompanyDiscovery:
    """Get company discovery service instance."""
    return ExaCompanyDiscovery()


def get_enrichment_service() -> LinkedInEnrichmentService:
    """Get founder enrichment service instance.""" 
    return LinkedInEnrichmentService()

def get_founder_pipeline_service() -> FounderDataPipeline:
    return FounderDataPipeline()

def get_pipeline_service() -> InitiationPipeline:
    """Get complete pipeline service instance."""
    return InitiationPipeline()
