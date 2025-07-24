"""FastAPI dependencies for service injection."""

from ..core.ranking import FounderRankingService
from ..core.pipeline import InitiationPipeline
from ..core.data import LinkedInEnrichmentService


def get_ranking_service() -> FounderRankingService:
    """Get founder ranking service instance."""
    return FounderRankingService()


def get_enrichment_service() -> LinkedInEnrichmentService:
    """Get founder enrichment service instance.""" 
    return LinkedInEnrichmentService()

def get_pipeline_service(job_id: str = None) -> InitiationPipeline:
    """Get complete 3-stage pipeline service instance."""
    return InitiationPipeline(job_id=job_id)
