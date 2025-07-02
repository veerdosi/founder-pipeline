"""Founder ranking service using Claude Sonnet 4 API."""

import json
import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import aiohttp
import os
from dataclasses import asdict

from .models import FounderProfile, FounderRanking, LevelClassification, ExperienceLevel
from .prompts import RankingPrompts


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
    """Main service for ranking founders using the L1-L10 framework."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.claude_provider = ClaudeSonnet4Provider(api_key)
        self.prompts = RankingPrompts()
    
    async def rank_founder(self, profile: FounderProfile) -> FounderRanking:
        """Rank a single founder using L1-L10 classification."""
        
        logger.info(f"Analyzing founder: {profile.name} at {profile.company_name}")
        
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
                    "prompt_version": "v1.0",
                    "processing_time": datetime.now().isoformat()
                }
            )
            
            logger.info(f"Classified {profile.name} as {classification.level.value} (confidence: {classification.confidence_score:.2f})")
            return ranking
            
        except Exception as e:
            logger.error(f"Failed to rank founder {profile.name}: {e}")
            raise
    
    async def rank_founders_batch(
        self, 
        profiles: List[FounderProfile], 
        batch_size: int = 5,
        delay_between_requests: float = 1.0
    ) -> List[FounderRanking]:
        """Rank multiple founders in batches with rate limiting."""
        
        logger.info(f"Starting batch ranking for {len(profiles)} founders")
        rankings = []
        
        # Process in batches to avoid rate limits
        for i in range(0, len(profiles), batch_size):
            batch = profiles[i:i + batch_size]
            
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(profiles)-1)//batch_size + 1}")
            
            # Process batch concurrently
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
        
        logger.info(f"Completed batch ranking. {len([r for r in rankings if r.classification.level])} successful, {len([r for r in rankings if not r.classification.level])} failed")
        return rankings
    
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
    
    async def verify_ranking(self, ranking: FounderRanking) -> FounderRanking:
        """Verify and potentially refine an existing ranking."""
        
        verification_prompt = self.prompts.create_verification_prompt(
            ranking.profile, 
            ranking.classification.level.value
        )
        
        try:
            response = await self.claude_provider.analyze_founder(verification_prompt)
            verification_data = self._parse_claude_response(response)
            
            # Update classification with verification results
            ranking.classification = LevelClassification(
                level=ExperienceLevel(verification_data["level"]),
                confidence_score=verification_data["confidence_score"],
                reasoning=f"VERIFIED: {verification_data['reasoning']}",
                evidence=verification_data["evidence"],
                verification_sources=verification_data["verification_sources"]
            )
            
            ranking.processing_metadata["verified"] = True
            ranking.processing_metadata["verification_time"] = datetime.now().isoformat()
            
            return ranking
            
        except Exception as e:
            logger.error(f"Failed to verify ranking for {ranking.profile.name}: {e}")
            return ranking  # Return original ranking if verification fails
    
    def get_ranking_summary(self, rankings: List[FounderRanking]) -> Dict[str, Any]:
        """Generate summary statistics for a set of rankings."""
        
        if not rankings:
            return {"total": 0, "distribution": {}, "average_confidence": 0.0}
        
        # Count by level
        level_counts = {}
        total_confidence = 0.0
        successful_rankings = 0
        
        for ranking in rankings:
            if ranking.classification.confidence_score > 0:  # Successful ranking
                level = ranking.classification.level.value
                level_counts[level] = level_counts.get(level, 0) + 1
                total_confidence += ranking.classification.confidence_score
                successful_rankings += 1
        
        return {
            "total": len(rankings),
            "successful": successful_rankings,
            "failed": len(rankings) - successful_rankings,
            "distribution": level_counts,
            "average_confidence": total_confidence / successful_rankings if successful_rankings > 0 else 0.0,
            "level_descriptions": self.prompts.get_level_descriptions()
        }
