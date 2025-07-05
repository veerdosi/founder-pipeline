"""FastAPI dependencies for service injection."""

from ..core.ranking import FounderRankingService
from ..core.discovery import ExaCompanyDiscovery, InitiationPipeline  
from ..core.data import LinkedInEnrichmentService
from ..core.analysis import PerplexityVerificationService


def get_ranking_service() -> FounderRankingService:
    """Get founder ranking service instance."""
    return FounderRankingService()


def get_discovery_service() -> ExaCompanyDiscovery:
    """Get company discovery service instance."""
    return ExaCompanyDiscovery()


def get_enrichment_service() -> LinkedInEnrichmentService:
    """Get founder enrichment service instance.""" 
    return LinkedInEnrichmentService()


def get_verification_service() -> PerplexityVerificationService:
    """Get Perplexity verification service instance."""
    return PerplexityVerificationService()


def get_pipeline_service() -> InitiationPipeline:
    """Get complete pipeline service instance."""
    return InitiationPipeline()
