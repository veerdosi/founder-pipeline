"""AI system for founder ranking using Claude Sonnet 4."""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from ...core import settings
from ..ranking.ranking_service import FounderRankingService
from ...models import LinkedInProfile

import logging
logger = logging.getLogger(__name__)

@dataclass 
class FounderAnalysisResult:
    """founder analysis result."""
    experience_level: str  # L1-L10
    confidence_score: float
    reasoning: str
    evidence: List[str]
    verification_sources: List[str]

class AIAnalysisService:
    """AI analysis service for founder ranking."""
    
    def __init__(self):
        self.ranking_service = FounderRankingService()
        
    async def rank_founders_batch(
        self, 
        founders_data: List[Dict[str, Any]],
        batch_size: int = 3,
        use_verification: bool = True
    ) -> List[FounderAnalysisResult]:
        """Rank multiple founders in batches."""
        
        founder_profiles = [LinkedInProfile.from_csv_row(data) for data in founders_data]

        results = await self.ranking_service.rank_founders_batch(
            founder_profiles=founder_profiles,
            batch_size=batch_size,
            use_enhanced=use_verification
        )
        
        analysis_results = []
        for result in results:
            classification = result['classification']
            analysis_results.append(
                FounderAnalysisResult(
                    experience_level=classification.level.value,
                    confidence_score=classification.confidence_score,
                    reasoning=classification.reasoning,
                    evidence=classification.evidence,
                    verification_sources=classification.verification_sources
                )
            )
            
        return analysis_results
