"""FastAPI dependencies for service injection."""

from ..services.ranking import FounderRankingService
from ..services.company_discovery import ExaCompanyDiscovery  
from ..services.profile_enrichment import LinkedInEnrichmentService
from ..services.perplexity_verification import PerplexityVerificationService
from ..services.pipeline import InitiationPipeline


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
