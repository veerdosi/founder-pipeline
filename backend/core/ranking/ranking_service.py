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
        
        # Enhanced data columns
        financial_cols = [col for col in df.columns if col.startswith(('total_exits', 'total_exit_value', 'companies_founded', 'total_funding', 'financial_'))]
        media_cols = [col for col in df.columns if col.startswith(('media_mentions', 'awards_', 'thought_leader', 'media_confidence'))]
        web_cols = [col for col in df.columns if col.startswith(('verified_facts', 'web_data', 'searches_'))]
        confidence_cols = [col for col in df.columns if col.endswith('_confidence') or col == 'overall_confidence']
        
        # Legacy enhanced columns
        legacy_cols = [col for col in df.columns if col.startswith(('phd_', 'accelerator_', 'sec_', 'unicorn_', 'highest_', 'top_tier'))]
        
        # Other columns
        other_cols = [col for col in df.columns 
                     if col not in ranking_cols + core_cols + financial_cols + media_cols + web_cols + confidence_cols + legacy_cols]
        
        column_order = ranking_cols + core_cols + financial_cols + media_cols + web_cols + confidence_cols + legacy_cols + other_cols
        
        # Only include columns that exist in the dataframe
        existing_columns = [col for col in column_order if col in df.columns]
        df = df[existing_columns]
        
        # Export
        df.to_csv(output_path, index=False)
        
        logger.info(f"✅ Exported {len(df)} ranked founders to {output_path}")
        return output_path
    
    def _profile_to_ai_dict(self, profile: FounderProfile) -> Dict[str, Any]:
        """Convert FounderProfile to dict for AI ranking (includes enhanced data if available)."""
        # Ensure we only accept FounderProfile objects
        if not isinstance(profile, FounderProfile):
            raise TypeError(f"Expected FounderProfile, got {type(profile).__name__}. "
                          f"LinkedInProfile objects should be converted to FounderProfile "
                          f"in the founder intelligence stage before ranking.")
        
        # FounderProfile handling
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
        
        # Add enhanced financial data if available
        if profile.financial_profile and hasattr(profile.financial_profile, 'company_exits'):
            financial_data = {
                "total_exits": len(profile.financial_profile.company_exits),
                "total_exit_value_usd": profile.financial_profile.total_exit_value_usd,
                "companies_founded_count": len(profile.financial_profile.companies_founded),
                "total_funding_raised_usd": profile.financial_profile.total_funding_raised_usd,
                "number_of_investments": profile.financial_profile.number_of_investments,
                "board_positions_count": len(profile.financial_profile.board_positions)
            }
            
            # Add details about largest exit
            if profile.financial_profile.company_exits:
                largest_exit = max(profile.financial_profile.company_exits, 
                                 key=lambda x: x.exit_value_usd or 0)
                financial_data.update({
                    "largest_exit_company": largest_exit.company_name,
                    "largest_exit_value_usd": largest_exit.exit_value_usd,
                    "largest_exit_type": largest_exit.exit_type.value
                })
            
            # Add company founding details
            if profile.financial_profile.companies_founded:
                companies_summary = []
                for company in profile.financial_profile.companies_founded[:3]:  # Top 3
                    companies_summary.append({
                        "name": company.company_name,
                        "founding_date": company.founding_date.isoformat() if company.founding_date else None,
                        "current_valuation_usd": company.current_valuation_usd,
                        "is_current": company.is_current_company
                    })
                financial_data["companies_founded_details"] = companies_summary
            
            data["financial_profile"] = financial_data
        
        # Add enhanced media data if available
        if profile.media_profile and hasattr(profile.media_profile, 'media_mentions'):
            media_data = {
                "media_mentions_count": len(profile.media_profile.media_mentions),
                "awards_count": len(profile.media_profile.awards),
                "thought_leadership_count": len(profile.media_profile.thought_leadership),
                "thought_leader_score": profile.media_profile.thought_leader_score,
                "public_profile_score": profile.media_profile.public_profile_score,
                "positive_sentiment_ratio": profile.media_profile.positive_sentiment_ratio,
                "twitter_followers": profile.media_profile.twitter_followers,
                "linkedin_connections": profile.media_profile.linkedin_connections
            }
            
            # Add top awards
            if profile.media_profile.awards:
                top_awards = []
                for award in profile.media_profile.awards[:3]:  # Top 3 awards
                    top_awards.append({
                        "name": award.award_name,
                        "organization": award.awarding_organization,
                        "date": award.award_date.isoformat() if award.award_date else None
                    })
                media_data["top_awards"] = top_awards
            
            # Add high-impact media mentions
            high_impact_mentions = [
                mention for mention in profile.media_profile.media_mentions
                if mention.importance_score > 0.7
            ]
            media_data["high_impact_mentions_count"] = len(high_impact_mentions)
            
            data["media_profile"] = media_data
        
        # Add web intelligence summary
        if profile.web_search_data and hasattr(profile.web_search_data, 'verified_facts'):
            web_data = {
                "verified_facts_count": len(profile.web_search_data.verified_facts),
                "data_quality_score": profile.web_search_data.overall_data_quality,
                "searches_performed": profile.web_search_data.total_searches_performed,
                "data_gaps_count": len(profile.web_search_data.data_gaps)
            }
            
            # Add key verified facts
            if profile.web_search_data.verified_facts:
                web_data["key_verified_facts"] = profile.web_search_data.verified_facts[:5]
            
            data["web_intelligence"] = web_data
        
        # Add overall intelligence data indicators
        data["has_intelligence_data"] = profile.has_intelligence_data()
        data["has_financial_data"] = profile.has_financial_data()
        data["has_media_presence"] = profile.has_media_presence()
        data["overall_confidence"] = profile.calculate_overall_confidence()
        
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
        
        # Boost confidence if intelligence data is available
        confidence_score = analysis_result.confidence_score
        if profile.has_intelligence_data():
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
            "experience_2_title": profile.experience_2_title,
            "experience_2_company": profile.experience_2_company,
            "education_1_school": profile.education_1_school,
            "education_1_degree": profile.education_1_degree,
            "education_2_school": profile.education_2_school,
            "education_2_degree": profile.education_2_degree,
            "skill_1": profile.skill_1,
            "skill_2": profile.skill_2,
            "skill_3": profile.skill_3,
            
            "ranking_timestamp": datetime.now().isoformat()
        }
        
        # Add intelligence data using the profile's dict method
        if hasattr(profile, 'to_dict'):
            data_dict = profile.to_dict()
            
            # Add financial data
            row.update({
                "total_exits": data_dict.get("total_exits", 0),
                "total_exit_value_usd": data_dict.get("total_exit_value_usd"),
                "companies_founded_count": data_dict.get("companies_founded_count", 0),
                "total_funding_raised_usd": data_dict.get("total_funding_raised_usd"),
                "financial_confidence": data_dict.get("financial_confidence", 0.0)
            })
            
            # Add media data
            row.update({
                "media_mentions_count": data_dict.get("media_mentions_count", 0),
                "awards_count": data_dict.get("awards_count", 0),
                "thought_leader_score": data_dict.get("thought_leader_score", 0.0),
                "media_confidence": data_dict.get("media_confidence", 0.0)
            })
            
            # Add web intelligence data
            row.update({
                "verified_facts_count": data_dict.get("verified_facts_count", 0),
                "web_data_quality": data_dict.get("web_data_quality", 0.0),
                "searches_performed": data_dict.get("searches_performed", 0)
            })
            
            # Add overall metrics
            row.update({
                "data_collected": data_dict.get("data_collected", False),
                "overall_confidence": data_dict.get("overall_confidence", 0.0)
            })
        else:
            # Fallback for profiles without dict method
            row.update({
                "total_exits": 0,
                "total_exit_value_usd": None,
                "companies_founded_count": 0,
                "total_funding_raised_usd": None,
                "financial_confidence": 0.0,
                "media_mentions_count": 0,
                "awards_count": 0,
                "thought_leader_score": 0.0,
                "media_confidence": 0.0,
                "verified_facts_count": 0,
                "web_data_quality": 0.0,
                "searches_performed": 0,
                "data_collected": profile.data_collected,
                "overall_confidence": 0.1
            })
        
        return row
