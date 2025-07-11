"""Core data module for data fusion and enrichment.

This module includes comprehensive founder intelligence gathering:
- Financial data (exits, investments, board positions)
- Media coverage and thought leadership 
- Web intelligence via Perplexity and search
- Orchestrated data collection pipeline
- Multi-source ranking system
- Comprehensive export functionality
"""

from .data_fusion import DataFusionService, FusedCompanyData
from .profile_enrichment import LinkedInEnrichmentService
from .company_discovery import ExaCompanyDiscovery
from .financial_collector import FinancialDataCollector
from .media_collector import MediaCollector
from .intel_service import PerplexitySearchService
from .founder_pipeline import FounderDataPipeline

__all__ = [
    "DataFusionService",
    "FusedCompanyData", 
    "LinkedInEnrichmentService",
    "FinancialDataCollector",
    "MediaCollector",
    "PerplexitySearchService", 
    "FounderDataPipeline",
    "ExaCompanyDiscovery"
]
