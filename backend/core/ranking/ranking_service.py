"""Pure ranking service that ONLY adds L-level classifications to founder datasets."""

import asyncio
from typing import List, Dict, Any
from datetime import datetime
import pandas as pd

from .models import FounderProfile, LevelClassification, ExperienceLevel
from ..analysis.ai_analysis import ClaudeSonnet4RankingService, FounderAnalysisResult

import logging
logger = logging.getLogger(__name__)


class FounderRankingService:
    """Pure ranking service - assumes data is already enhanced elsewhere."""
    
    def __init__(self):
        self.claude_ranking_service = ClaudeSonnet4RankingService()
    
    async def rank_founders_batch(
        self,
        founder_profiles: List[FounderProfile],
        batch_size: int = 5,
        use_enhanced: bool = True
    ) -> List[Dict[str, Any]]:
        """Simple ranking method without verification complexity."""
        logger.info(f"🎯 Ranking {len(founder_profiles)} founders (simplified)")
        
        # Convert profiles to founder data for AI analysis
        founders_data = [
            self._profile_to_ai_dict(profile) 
            for profile in founder_profiles
        ]
        
        # Batch AI analysis for L-level classification
        try:
            analysis_results = await self.claude_ranking_service.rank_founders_batch(
                founders_data=founders_data,
                batch_size=batch_size,
                use_verification=False
            )
        except Exception as e:
            logger.error(f"Claude ranking service failed: {e}")
            # Create fallback results for all founders
            analysis_results = []
            for _ in founder_profiles:
                from ..analysis.ai_analysis import FounderAnalysisResult
                fallback_result = FounderAnalysisResult(
                    experience_level="INSUFFICIENT_DATA",
                    confidence_score=0.1,
                    reasoning=f"Ranking service unavailable: {str(e)}",
                    evidence=[],
                    verification_sources=[]
                )
                analysis_results.append(fallback_result)
        
        if len(analysis_results) != len(founder_profiles):
            logger.warning(f"Mismatch in analysis results: {len(analysis_results)} vs {len(founder_profiles)} profiles")
            # Pad with fallback results if needed
            while len(analysis_results) < len(founder_profiles):
                from ..analysis.ai_analysis import FounderAnalysisResult
                fallback_result = FounderAnalysisResult(
                    experience_level="INSUFFICIENT_DATA",
                    confidence_score=0.1,
                    reasoning="Analysis incomplete - missing result",
                    evidence=[],
                    verification_sources=[]
                )
                analysis_results.append(fallback_result)
        
        # Create rankings dataset
        rankings = []
        for profile, analysis_result in zip(founder_profiles, analysis_results):
            try:
                # Convert AI result to L-level classification
                classification = self._convert_to_classification(analysis_result, profile)
                
                # Create simple ranking result
                ranking = {
                    'profile': profile,
                    'classification': classification,
                    'timestamp': datetime.now().isoformat()
                }
                rankings.append(ranking)
                
            except Exception as e:
                logger.error(f"Error ranking {profile.name}: {e}")
                # Add fallback ranking
                fallback_classification = LevelClassification(
                    level=ExperienceLevel.INSUFFICIENT_DATA,
                    confidence_score=0.1,
                    reasoning=f"Ranking failed: {str(e)}",
                    evidence=[],
                    verification_sources=[]
                )
                ranking = {
                    'profile': profile,
                    'classification': fallback_classification,
                    'timestamp': datetime.now().isoformat()
                }
                rankings.append(ranking)
        
        logger.info(f"✅ Ranked {len(rankings)} founders")
        return rankings
    
    async def add_rankings_to_dataset(
        self, 
        enhanced_profiles: List[FounderProfile],
        batch_size: int = 5
    ) -> List[Dict[str, Any]]:
        """Add L-level rankings to enhanced founder dataset."""
        logger.info(f"🎯 Adding L-level rankings to {len(enhanced_profiles)} founder profiles")
        
        # Convert profiles to founder data for AI analysis
        founders_data = [
            self._profile_to_ai_dict(profile) 
            for profile in enhanced_profiles
        ]
        
        # Simplified batch AI analysis without verification
        analysis_results = await self.claude_ranking_service.rank_founders_batch(
            founders_data=founders_data,
            batch_size=batch_size,
            use_verification=False
        )
        
        # Create final dataset with rankings added
        ranked_dataset = []
        for profile, analysis_result in zip(enhanced_profiles, analysis_results):
            try:
                # Convert AI result to L-level classification
                classification = self._convert_to_classification(analysis_result, profile)
                
                # Create dataset row with ranking added
                dataset_row = self._create_dataset_row(profile, classification)
                ranked_dataset.append(dataset_row)
                
            except Exception as e:
                logger.error(f"Error ranking {profile.name}: {e}")
                # Add fallback ranking
                fallback_classification = LevelClassification(
                    level=ExperienceLevel.INSUFFICIENT_DATA,
                    confidence_score=0.1,
                    reasoning=f"Ranking failed: {str(e)}",
                    evidence=[],
                    verification_sources=[]
                )
                dataset_row = self._create_dataset_row(profile, fallback_classification)
                ranked_dataset.append(dataset_row)
        
        logger.info(f"✅ Added L-level rankings to {len(ranked_dataset)} founders")
        return ranked_dataset
    
    async def export_ranked_csv(
        self,
        enhanced_profiles: List[FounderProfile],
        output_path: str
    ) -> str:
        """Export ranked dataset to CSV."""
        logger.info(f"📊 Exporting ranked dataset to {output_path}")
        
        # Get ranked dataset
        ranked_dataset = await self.add_rankings_to_dataset(enhanced_profiles)
        
        # Convert to DataFrame and export
        df = pd.DataFrame(ranked_dataset)
        
        # Reorder columns: ranking first, then core info, then enhanced data
        ranking_cols = ['l_level', 'confidence_score', 'reasoning']
        core_cols = ['name', 'company_name', 'title', 'location']
        enhanced_cols = [col for col in df.columns if col.startswith(('total_', 'phd_', 'accelerator_', 'sec_', 'unicorn_', 'highest_', 'top_tier'))]
        other_cols = [col for col in df.columns if col not in ranking_cols + core_cols + enhanced_cols]
        
        column_order = ranking_cols + core_cols + enhanced_cols + other_cols
        df = df[column_order]
        
        # Export
        df.to_csv(output_path, index=False)
        
        logger.info(f"✅ Exported {len(df)} ranked founders to {output_path}")
        return output_path
    
    def _profile_to_ai_dict(self, profile: FounderProfile) -> Dict[str, Any]:
        """Convert profile to dict for AI ranking (includes enhanced data if available)."""
        # Basic profile data
        data = {
            "name": profile.name,
            "company_name": profile.company_name,
            "title": profile.title,
            "about": profile.about,
            "location": profile.location,
            "estimated_age": profile.estimated_age,
            "experience_1_title": profile.experience_1_title,
            "experience_1_company": profile.experience_1_company,
            "education_1_school": profile.education_1_school,
            "education_1_degree": profile.education_1_degree,
            "linkedin_url": profile.linkedin_url
        }
        
        # Add enhanced data if available (for better AI ranking)
        if profile.has_enhanced_data():
            if profile.financial_profile:
                financial_metrics = profile.financial_profile.get("metrics", {})
                data.update({
                    "total_exits": financial_metrics.get("total_exits", 0),
                    "total_exit_value_usd": financial_metrics.get("total_exit_value_usd", 0),
                    "unicorn_companies_count": financial_metrics.get("unicorn_companies_count", 0),
                    "years_entrepreneurship": financial_metrics.get("years_entrepreneurship", 0)
                })
            
            if profile.education_profile:
                data.update({
                    "phd_degrees_count": len(profile.education_profile.get("phd_degrees", [])),
                    "technical_background": profile.education_profile.get("technical_field_background", False),
                    "top_tier_institution": profile.education_profile.get("top_tier_institution", False)
                })
            
            if profile.accelerator_profile:
                data.update({
                    "accelerator_programs": profile.accelerator_profile.get("total_programs", 0),
                    "top_accelerator": profile.accelerator_profile.get("has_top_accelerator", False)
                })
            
            if profile.sec_profile:
                data.update({
                    "sec_verified_exits": profile.sec_profile.get("exit_count", 0),
                    "sec_highest_exit": profile.sec_profile.get("highest_exit_value", 0)
                })
        
        return data
    
    def _convert_to_classification(
        self, 
        analysis_result: FounderAnalysisResult,
        profile: FounderProfile
    ) -> LevelClassification:
        """Convert AI analysis to L-level classification."""
        
        # Map to enum
        try:
            level_enum = ExperienceLevel(analysis_result.experience_level)
        except ValueError:
            level_enum = ExperienceLevel.INSUFFICIENT_DATA
        
        # Boost confidence if enhanced data is available
        confidence_score = analysis_result.confidence_score
        if profile.has_enhanced_data():
            confidence_score = min(confidence_score + 0.1, 1.0)
        
        return LevelClassification(
            level=level_enum,
            confidence_score=confidence_score,
            reasoning=analysis_result.reasoning,
            evidence=analysis_result.evidence,
            verification_sources=analysis_result.verification_sources
        )
    
    def _create_dataset_row(
        self, 
        profile: FounderProfile, 
        classification: LevelClassification
    ) -> Dict[str, Any]:
        """Create final dataset row with ranking + all enhanced data."""
        
        # Start with ranking results
        row = {
            # L-level ranking (primary output)
            "l_level": classification.level.value,
            "confidence_score": round(classification.confidence_score, 3),
            "reasoning": classification.reasoning,
            
            # Core founder info
            "name": profile.name,
            "company_name": profile.company_name,
            "title": profile.title,
            "location": profile.location,
            "about": profile.about,
            "linkedin_url": profile.linkedin_url,
            
            # Basic profile data
            "estimated_age": profile.estimated_age,
            "experience_1_title": profile.experience_1_title,
            "experience_1_company": profile.experience_1_company,
            "education_1_school": profile.education_1_school,
            "education_1_degree": profile.education_1_degree,
            "skill_1": profile.skill_1,
            
            # Enhanced data availability
            "enhanced_data_available": profile.has_enhanced_data(),
            "ranking_timestamp": datetime.now().isoformat()
        }
        
        # Add all enhanced data if available
        if profile.has_enhanced_data():
            # Financial intelligence
            if profile.financial_profile:
                financial_metrics = profile.financial_profile.get("metrics", {})
                row.update({
                    "total_exits": financial_metrics.get("total_exits", 0),
                    "total_exit_value_usd": financial_metrics.get("total_exit_value_usd", 0),
                    "total_value_created_usd": financial_metrics.get("total_value_created_usd", 0),
                    "unicorn_companies_count": financial_metrics.get("unicorn_companies_count", 0),
                    "major_exits_count": financial_metrics.get("companies_with_major_exits_count", 0),
                    "highest_exit_value_usd": financial_metrics.get("highest_exit_value_usd", 0),
                    "years_entrepreneurship": financial_metrics.get("years_entrepreneurship", 0),
                    "companies_founded_count": len(profile.financial_profile.get("companies_founded", []))
                })
            
            # Education verification
            if profile.education_profile:
                row.update({
                    "phd_degrees_count": len(profile.education_profile.get("phd_degrees", [])),
                    "highest_degree_verified": profile.education_profile.get("highest_degree"),
                    "technical_background_verified": profile.education_profile.get("technical_field_background", False),
                    "top_tier_institution": profile.education_profile.get("top_tier_institution", False),
                    "academic_publications_count": len(profile.education_profile.get("academic_publications", []))
                })
            
            # Accelerator verification
            if profile.accelerator_profile:
                row.update({
                    "accelerator_programs_count": profile.accelerator_profile.get("total_programs", 0),
                    "top_tier_accelerator_verified": profile.accelerator_profile.get("has_top_accelerator", False),
                    "total_accelerator_funding": profile.accelerator_profile.get("total_accelerator_funding", 0),
                    "accelerator_network_strength": profile.accelerator_profile.get("accelerator_network_strength", 0)
                })
            
            # SEC verification  
            if profile.sec_profile:
                row.update({
                    "sec_verified_exits_count": profile.sec_profile.get("exit_count", 0),
                    "sec_highest_exit_value": profile.sec_profile.get("highest_exit_value", 0),
                    "sec_total_exit_value": profile.sec_profile.get("total_verified_exit_value", 0)
                })
        
        return row
