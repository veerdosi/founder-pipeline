"""Simplified founder ranking service using Claude Sonnet 4 and Perplexity verification."""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

from ...core import get_logger, settings
from .models import FounderProfile, FounderRanking, LevelClassification, ExperienceLevel
from ..analysis.ai_analysis import ClaudeSonnet4RankingService, FounderAnalysisResult


logger = get_logger(__name__)


class FounderRankingService:
    """Simplified service for ranking founders using Claude Sonnet 4 + Perplexity verification."""
    
    def __init__(self):
        self.claude_ranking_service = ClaudeSonnet4RankingService()
    
    async def rank_founder(
        self, 
        profile: FounderProfile,
        use_enhanced: bool = True
    ) -> FounderRanking:
        """Rank a single founder using Claude Sonnet 4."""
        logger.info(f"🎯 Ranking founder: {profile.name}")
        
        try:
            # Convert profile to dict for AI analysis
            founder_data = self._profile_to_dict(profile)
            
            # Analyze with Claude Sonnet 4 + Perplexity verification
            analysis_result = await self.claude_ranking_service.rank_founder(
                founder_data=founder_data,
                use_verification=use_enhanced
            )
            
            # Convert to L-level classification
            classification = self._convert_to_classification(analysis_result)
            
            # Create ranking result
            ranking = FounderRanking(
                profile=profile,
                classification=classification,
                timestamp=datetime.now().isoformat(),
                processing_metadata={
                    "enhanced_verification": use_enhanced,
                    "ai_model": "claude-sonnet-4",
                    "verification_sources": len(analysis_result.verification_sources),
                    "processing_time": datetime.now().isoformat()
                }
            )
            
            logger.info(f"✅ Ranked {profile.name} as {classification.level.value} (confidence: {classification.confidence_score:.2f})")
            return ranking
            
        except Exception as e:
            logger.error(f"Error ranking founder {profile.name}: {e}")
            return self._create_fallback_ranking(profile, str(e))
    
    async def rank_founders_batch(
        self, 
        profiles: List[FounderProfile],
        batch_size: int = 3,
        use_enhanced: bool = True
    ) -> List[FounderRanking]:
        """Rank multiple founders in batches."""
        rankings = []
        
        # Convert profiles to founder data
        founders_data = [self._profile_to_dict(profile) for profile in profiles]
        
        # Use batch ranking from AI service
        analysis_results = await self.claude_ranking_service.rank_founders_batch(
            founders_data=founders_data,
            batch_size=batch_size,
            use_verification=use_enhanced
        )
        
        # Convert results to rankings
        for profile, analysis_result in zip(profiles, analysis_results):
            try:
                classification = self._convert_to_classification(analysis_result)
                
                ranking = FounderRanking(
                    profile=profile,
                    classification=classification,
                    timestamp=datetime.now().isoformat(),
                    processing_metadata={
                        "enhanced_verification": use_enhanced,
                        "ai_model": "claude-sonnet-4",
                        "verification_sources": len(analysis_result.verification_sources),
                        "processing_time": datetime.now().isoformat()
                    }
                )
                rankings.append(ranking)
                
            except Exception as e:
                logger.error(f"Error converting analysis for {profile.name}: {e}")
                rankings.append(self._create_fallback_ranking(profile, str(e)))
        
        return rankings
    
    def _profile_to_dict(self, profile: FounderProfile) -> Dict[str, Any]:
        """Convert FounderProfile to dict for AI analysis."""
        return {
            "name": profile.name,
            "company_name": profile.company_name,
            "title": profile.title,
            "about": profile.about,
            "location": profile.location,
            "estimated_age": profile.estimated_age,
            "experience_1_title": profile.experience_1_title,
            "experience_1_company": profile.experience_1_company,
            "experience_2_title": profile.experience_2_title,
            "experience_2_company": profile.experience_2_company,
            "experience_3_title": profile.experience_3_title,
            "experience_3_company": profile.experience_3_company,
            "education_1_school": profile.education_1_school,
            "education_1_degree": profile.education_1_degree,
            "education_2_school": profile.education_2_school,
            "education_2_degree": profile.education_2_degree,
            "skill_1": profile.skill_1,
            "skill_2": profile.skill_2,
            "skill_3": profile.skill_3,
            "linkedin_url": profile.linkedin_url
        }
    
    def _convert_to_classification(self, analysis_result: FounderAnalysisResult) -> LevelClassification:
        """Convert FounderAnalysisResult to LevelClassification."""
        
        # Map experience level string to enum
        try:
            level_enum = ExperienceLevel(analysis_result.experience_level)
        except ValueError:
            # Fallback to INSUFFICIENT_DATA if level not recognized
            level_enum = ExperienceLevel.INSUFFICIENT_DATA
        
        return LevelClassification(
            level=level_enum,
            confidence_score=analysis_result.confidence_score,
            reasoning=analysis_result.reasoning,
            evidence=analysis_result.evidence,
            verification_sources=analysis_result.verification_sources
        )
    
    def _create_fallback_ranking(self, profile: FounderProfile, error_message: str) -> FounderRanking:
        """Create fallback ranking when analysis fails."""
        
        classification = LevelClassification(
            level=ExperienceLevel.INSUFFICIENT_DATA,
            confidence_score=0.1,
            reasoning=f"Analysis failed: {error_message}",
            evidence=[],
            verification_sources=[]
        )
        
        return FounderRanking(
            profile=profile,
            classification=classification,
            timestamp=datetime.now().isoformat(),
            processing_metadata={
                "ai_model": "claude-sonnet-4",
                "error": error_message,
                "processing_time": datetime.now().isoformat()
            }
        )
    
    def get_ranking_summary(self, rankings: List[FounderRanking]) -> Dict[str, Any]:
        """Generate summary statistics for rankings."""
        
        if not rankings:
            return {"total": 0, "distribution": {}, "average_confidence": 0.0}
        
        # Count by level
        level_counts = {}
        total_confidence = 0.0
        successful_rankings = 0
        
        for ranking in rankings:
            if ranking.classification.confidence_score > 0.1:  # Successful ranking
                level = ranking.classification.level.value
                level_counts[level] = level_counts.get(level, 0) + 1
                total_confidence += ranking.classification.confidence_score
                successful_rankings += 1
        
        return {
            "total": len(rankings),
            "successful": successful_rankings,
            "failed": len(rankings) - successful_rankings,
            "distribution": level_counts,
            "average_confidence": total_confidence / successful_rankings if successful_rankings > 0 else 0.0
        }
