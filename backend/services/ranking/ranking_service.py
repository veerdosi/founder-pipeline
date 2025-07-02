"""Enhanced founder ranking service with L-level validation and real-time verification."""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
import aiohttp
import os

from .models import FounderProfile, FounderRanking, LevelClassification, ExperienceLevel
from .prompts import RankingPrompts
from .level_thresholds import LevelValidator, LevelThresholds, DataPoint, SourceType
from .verification_service import RealTimeVerificationOrchestrator

logger = logging.getLogger(__name__)


class ClaudeSonnet4Provider:
    """Claude Sonnet 4 API provider for founder analysis."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.model = "claude-3-5-sonnet-20241022"  # Claude Sonnet 4 model
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
    
    async def analyze_founder(self, prompt: str, max_tokens: int = 1500) -> str:
        """Send founder analysis request to Claude Sonnet 4."""
        
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": RankingPrompts.SYSTEM_PROMPT,
            "messages": [
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "temperature": 0.1  # Low temperature for consistent analysis
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    self.base_url, 
                    headers=self.headers, 
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["content"][0]["text"]
                    else:
                        error_text = await response.text()
                        logger.error(f"Claude API error {response.status}: {error_text}")
                        raise Exception(f"Claude API error: {response.status} - {error_text}")
                        
            except asyncio.TimeoutError:
                logger.error("Claude API request timed out")
                raise Exception("Claude API request timed out")
            except Exception as e:
                logger.error(f"Claude API request failed: {e}")
                raise


class FounderRankingService:
    """Enhanced founder ranking service with L1-L10 framework and real-time verification."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.claude_provider = ClaudeSonnet4Provider(api_key)
        self.prompts = RankingPrompts()
        self.level_validator = LevelValidator()
        self.realtime_verifier = RealTimeVerificationOrchestrator()
        self.level_definitions = LevelThresholds.get_level_definitions()
    
    async def rank_founder(self, profile: FounderProfile) -> FounderRanking:
        """Basic founder ranking using Claude analysis only."""
        
        logger.info(f"Basic ranking for {profile.name} at {profile.company_name}")
        
        # Generate analysis prompt
        analysis_prompt = self.prompts.create_founder_analysis_prompt(profile)
        
        try:
            # Get Claude's analysis
            response = await self.claude_provider.analyze_founder(analysis_prompt)
            
            # Parse the JSON response
            classification_data = self._parse_claude_response(response)
            
            # Create classification object
            classification = LevelClassification(
                level=ExperienceLevel(classification_data["level"]),
                confidence_score=classification_data["confidence_score"],
                reasoning=classification_data["reasoning"],
                evidence=classification_data["evidence"],
                verification_sources=classification_data["verification_sources"]
            )
            
            # Create final ranking result
            ranking = FounderRanking(
                profile=profile,
                classification=classification,
                timestamp=datetime.now().isoformat(),
                processing_metadata={
                    "model_used": self.claude_provider.model,
                    "prompt_version": "v1.0_basic",
                    "processing_time": datetime.now().isoformat()
                }
            )
            
            logger.info(f"Classified {profile.name} as {classification.level.value} (confidence: {classification.confidence_score:.2f})")
            return ranking
            
        except Exception as e:
            logger.error(f"Failed to rank founder {profile.name}: {e}")
            raise
    
    async def rank_founder_enhanced(
        self, 
        profile: FounderProfile,
        last_update: Optional[datetime] = None,
        require_realtime_verification: bool = True,
        minimum_confidence: float = 0.75
    ) -> FounderRanking:
        """Enhanced ranking with L-level validation and real-time verification for stale data."""
        
        logger.info(f"Enhanced ranking for {profile.name} at {profile.company_name}")
        
        try:
            # Step 1: Get initial Claude analysis
            initial_ranking = await self.rank_founder(profile)
            proposed_level = initial_ranking.classification.level.value
            
            logger.info(f"Claude proposed {proposed_level} for {profile.name}")
            
            # Step 2: Real-time verification (if data is stale)
            verification_result = None
            if require_realtime_verification:
                verification_result = await self.realtime_verifier.update_stale_founder_data(
                    profile.name,
                    profile.company_name,
                    last_update
                )
                
                if verification_result['updated']:
                    logger.info(f"Real-time verification updated data for {profile.name}")
                else:
                    logger.info(f"No real-time updates needed for {profile.name}")
            
            # Step 3: L-level threshold validation
            # Convert verification data to DataPoints for validation
            data_points = []
            if verification_result and verification_result.get('verification_data'):
                data_points = verification_result['verification_data']
            
            validation_result = self.level_validator.validate_level_assignment(
                proposed_level,
                data_points
            )
            
            logger.info(f"L-level validation: {validation_result['valid']} (recommended: {validation_result['recommendation']})")
            
            # Step 4: Determine final classification
            final_classification = self._determine_final_classification(
                initial_ranking.classification,
                verification_result,
                validation_result,
                minimum_confidence
            )
            
            # Step 5: Create enhanced ranking result
            enhanced_ranking = FounderRanking(
                profile=profile,
                classification=final_classification,
                timestamp=datetime.now().isoformat(),
                processing_metadata={
                    "model_used": self.claude_provider.model,
                    "prompt_version": "v2.0_enhanced",
                    "initial_claude_level": proposed_level,
                    "realtime_verification_used": require_realtime_verification,
                    "data_updated": verification_result['updated'] if verification_result else False,
                    "threshold_validation_passed": validation_result['valid'],
                    "confidence_score": verification_result.get('update_confidence', 0.0) if verification_result else 0.0,
                    "processing_time": datetime.now().isoformat(),
                    "verification_summary": verification_result if verification_result else None,
                    "validation_details": validation_result
                }
            )
            
            logger.info(f"Final classification: {final_classification.level.value} (confidence: {final_classification.confidence_score:.2f})")
            return enhanced_ranking
            
        except Exception as e:
            logger.error(f"Enhanced ranking failed for {profile.name}: {e}")
            # Fallback to basic ranking
            return await self.rank_founder(profile)
    
    async def rank_founders_batch(
        self, 
        profiles: List[FounderProfile], 
        batch_size: int = 5,
        delay_between_requests: float = 1.0,
        use_enhanced: bool = False
    ) -> List[FounderRanking]:
        """Rank multiple founders in batches."""
        
        logger.info(f"Starting batch ranking for {len(profiles)} founders (enhanced: {use_enhanced})")
        rankings = []
        
        # Process in batches to avoid rate limits
        for i in range(0, len(profiles), batch_size):
            batch = profiles[i:i + batch_size]
            
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(profiles)-1)//batch_size + 1}")
            
            # Choose ranking method
            if use_enhanced:
                batch_tasks = [self.rank_founder_enhanced(profile) for profile in batch]
            else:
                batch_tasks = [self.rank_founder(profile) for profile in batch]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Handle results and exceptions
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to rank {batch[j].name}: {result}")
                    # Create failed ranking with error info
                    failed_ranking = self._create_failed_ranking(batch[j], str(result))
                    rankings.append(failed_ranking)
                else:
                    rankings.append(result)
            
            # Rate limiting delay between batches
            if i + batch_size < len(profiles):
                await asyncio.sleep(delay_between_requests)
        
        successful = len([r for r in rankings if r.classification.confidence_score > 0])
        failed = len(rankings) - successful
        
        logger.info(f"Batch ranking complete: {successful} successful, {failed} failed")
        return rankings
    
    def _determine_final_classification(
        self,
        claude_classification: LevelClassification,
        verification_result: Optional[Dict[str, Any]],
        validation_result: Dict[str, Any],
        minimum_confidence: float
    ) -> LevelClassification:
        """Determine final classification using real-time verification and L-level validation."""
        
        claude_level = claude_classification.level.value
        recommended_level = validation_result['recommendation']
        threshold_validation_passed = validation_result['valid']
        
        # Real-time verification data available?
        has_fresh_data = verification_result and verification_result.get('updated', False)
        verification_confidence = verification_result.get('update_confidence', 0.0) if verification_result else 0.0
        
        # Decision logic
        if threshold_validation_passed and (not has_fresh_data or verification_confidence >= minimum_confidence):
            # Threshold validation passed and no conflicting fresh data
            final_level = claude_level
            confidence = min(0.95, claude_classification.confidence_score + (0.1 if has_fresh_data else 0.0))
            reasoning = f"VALIDATED: {claude_classification.reasoning}"
            if has_fresh_data:
                reasoning += " Real-time verification supports this classification."
            
        elif has_fresh_data and verification_confidence >= minimum_confidence:
            # Fresh data available with high confidence - may override threshold validation
            final_level = claude_level  # Trust Claude + fresh data
            confidence = min(0.9, (claude_classification.confidence_score + verification_confidence) / 2)
            reasoning = f"REAL-TIME VERIFIED: {claude_classification.reasoning}. Updated with recent data."
            
        elif not threshold_validation_passed:
            # Threshold validation failed - use recommended level
            final_level = recommended_level
            confidence = max(0.3, claude_classification.confidence_score * 0.7)
            reasoning = f"ADJUSTED: Initially {claude_level}, adjusted to {recommended_level} based on L-level criteria."
            
        else:
            # Use Claude's classification but with lower confidence
            final_level = claude_level
            confidence = claude_classification.confidence_score * 0.8
            reasoning = f"STANDARD: {claude_classification.reasoning}. Limited verification data available."
        
        # Compile evidence
        evidence = claude_classification.evidence.copy()
        
        # Add real-time verification evidence
        if verification_result and verification_result.get('key_updates'):
            for update in verification_result['key_updates'][:3]:  # Limit to 3 key updates
                evidence.append(f"Recent update: {update['type']} - {update.get('source_text', 'verified')}")
        
        # Add validation evidence
        evidence.extend(validation_result['verified_criteria'])
        if validation_result['missing_criteria']:
            evidence.append(f"Missing criteria: {', '.join(validation_result['missing_criteria'][:2])}")
        
        # Verification sources
        verification_sources = claude_classification.verification_sources.copy()
        if verification_result and verification_result.get('updated'):
            verification_sources.append("Perplexity real-time search")
        
        return LevelClassification(
            level=ExperienceLevel(final_level),
            confidence_score=confidence,
            reasoning=reasoning,
            evidence=evidence,
            verification_sources=list(set(verification_sources))
        )
    
    def _parse_claude_response(self, response: str) -> Dict[str, Any]:
        """Parse Claude's JSON response and validate structure."""
        
        try:
            # Try to extract JSON from response
            response_clean = response.strip()
            
            # Remove any markdown formatting
            if response_clean.startswith("```json"):
                response_clean = response_clean[7:]
            if response_clean.endswith("```"):
                response_clean = response_clean[:-3]
            
            data = json.loads(response_clean)
            
            # Validate required fields
            required_fields = ["level", "confidence_score", "reasoning", "evidence", "verification_sources"]
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Validate level is valid
            if data["level"] not in [level.value for level in ExperienceLevel]:
                raise ValueError(f"Invalid level: {data['level']}")
            
            # Validate confidence score
            if not 0.0 <= data["confidence_score"] <= 1.0:
                raise ValueError(f"Invalid confidence score: {data['confidence_score']}")
            
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Claude response as JSON: {e}")
            logger.error(f"Response was: {response}")
            raise ValueError(f"Invalid JSON response from Claude: {e}")
        except Exception as e:
            logger.error(f"Failed to validate Claude response: {e}")
            raise
    
    def _create_failed_ranking(self, profile: FounderProfile, error_message: str) -> FounderRanking:
        """Create a ranking object for failed analysis."""
        
        classification = LevelClassification(
            level=ExperienceLevel.L1,  # Default to L1 for failed cases
            confidence_score=0.0,
            reasoning=f"Analysis failed: {error_message}",
            evidence=[],
            verification_sources=[]
        )
        
        return FounderRanking(
            profile=profile,
            classification=classification,
            timestamp=datetime.now().isoformat(),
            processing_metadata={
                "model_used": self.claude_provider.model,
                "error": error_message,
                "processing_time": datetime.now().isoformat()
            }
        )
    
    def get_ranking_summary(self, rankings: List[FounderRanking]) -> Dict[str, Any]:
        """Generate summary statistics for a set of rankings."""
        
        if not rankings:
            return {"total": 0, "distribution": {}, "average_confidence": 0.0}
        
        # Count by level
        level_counts = {}
        total_confidence = 0.0
        successful_rankings = 0
        enhanced_used = 0
        
        for ranking in rankings:
            if ranking.classification.confidence_score > 0:  # Successful ranking
                level = ranking.classification.level.value
                level_counts[level] = level_counts.get(level, 0) + 1
                total_confidence += ranking.classification.confidence_score
                successful_rankings += 1
                
                # Check if enhanced ranking was used
                if ranking.processing_metadata.get('prompt_version', '').startswith('v2.0'):
                    enhanced_used += 1
        
        return {
            "total": len(rankings),
            "successful": successful_rankings,
            "failed": len(rankings) - successful_rankings,
            "enhanced_used": enhanced_used,
            "distribution": level_counts,
            "average_confidence": total_confidence / successful_rankings if successful_rankings > 0 else 0.0,
            "level_descriptions": self.prompts.get_level_descriptions()
        }
    
    def get_verification_requirements(self, level: str) -> Dict[str, Any]:
        """Get verification requirements for a specific L-level."""
        
        if level not in self.level_definitions:
            return {"error": f"Invalid level: {level}"}
        
        threshold = self.level_definitions[level]
        search_strategies = LevelThresholds.get_search_strategies()
        
        return {
            "level": level,
            "description": threshold.description,
            "primary_criteria": threshold.primary_criteria,
            "financial_thresholds": threshold.financial_thresholds,
            "experience_requirements": threshold.experience_requirements,
            "verification_sources": [src.value for src in threshold.verification_requirements],
            "minimum_sources": threshold.minimum_sources,
            "search_strategies": search_strategies.get(level, []),
            "credibility_hierarchy": {
                src.value: src.credibility_score 
                for src in threshold.verification_requirements
            }
        }
