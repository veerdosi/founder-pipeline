"""Initiation Pipeline - AI Company Discovery and Analysis."""

from .core import settings, console
from .models import (
    Company,
    LinkedInProfile,
    MarketMetrics,
    EnrichedCompany,
    PipelineResult,
    FundingStage,
    MarketStage
)
from .services import (
    ExaCompanyDiscovery,
    LinkedInEnrichmentService,
    MarketAnalysisProvider,
    InitiationPipeline
)

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    "settings",
    "console",
    "Company",
    "LinkedInProfile", 
    "MarketMetrics",
    "EnrichedCompany",
    "PipelineResult",
    "FundingStage",
    "MarketStage",
    "ExaCompanyDiscovery",
    "LinkedInEnrichmentService",
    "MarketAnalysisProvider", 
    "InitiationPipeline"
]
