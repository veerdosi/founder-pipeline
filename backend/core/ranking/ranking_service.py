"""Pure ranking service that ONLY adds L-level classifications to founder datasets."""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import pandas as pd
import asyncio
import json
import anthropic


from .models import LevelClassification, ExperienceLevel
from .. import config
from .prompts import RankingPrompts


import logging
logger = logging.getLogger(__name__)


class ClaudeSonnet4RankingService:
    """Claude Sonnet 4 ranking service for L1-L10 founder classification."""
    
    def __init__(self):
        self.anthropic_client = anthropic.AsyncAnthropic(api_key=config.settings.anthropic_api_key)
        self.perplexity_verifier = None  # Lazy load to avoid circular import
        
    async def rank_founder(
        self, 
        founder_data: Dict[str, Any],
        company_data: Dict[str, Any] = None,
        use_verification: bool = True
    ) -> LevelClassification:
        """Rank a single founder using Claude Sonnet 4."""
        
        founder_name = founder_data.get('name', 'Unknown')
        company_name = company_data.get('name', '') if company_data else founder_data.get('company_name', '')
        
        logger.info(f"Ranking founder: {founder_name}")
        
        # Step 1: Initial ranking with Claude Sonnet 4
        initial_ranking = await self._analyze_with_claude(founder_data, company_data)
        
        if not initial_ranking:
            logger.warning(f"No initial ranking returned for {founder_name}")
            return self._create_fallback_result(founder_name)
        
        logger.debug(f"Initial ranking for {founder_name}: {initial_ranking}")
        
        # Step 2: Verification with Perplexity (if enabled)
        verification_data = {}
        level_key = initial_ranking.get('level') or initial_ranking.get('experience_level')
        if use_verification and level_key:
            try:
                verifier = self._get_perplexity_verifier()
                verification_data = await verifier.verify_founder_data(
                    founder_name=founder_name,
                    company_name=company_name,
                    claimed_level=level_key,
                    evidence=initial_ranking.get('evidence', [])
                )
            except Exception as e:
                logger.warning(f"Perplexity verification failed for {founder_name}: {e}")
                verification_data = {}
        
        # Step 3: Final ranking with verification data
        final_ranking = await self._finalize_ranking_with_verification(
            initial_ranking, verification_data
        )
        
        # Add safety checks for required fields
        if not final_ranking or not isinstance(final_ranking, dict):
            logger.error(f"Invalid final_ranking: {final_ranking}")
            return self._create_fallback_result(founder_name)
        
        # Check for required fields and provide defaults
        level = final_ranking.get('experience_level') or final_ranking.get('level', 'INSUFFICIENT_DATA')
        confidence_score = final_ranking.get('confidence_score', 0.1)
        reasoning = final_ranking.get('reasoning', 'Unable to analyze due to parsing error')
        evidence = final_ranking.get('evidence', [])
        
        logger.debug(f"Final ranking for {founder_name}: level={level}, confidence={confidence_score}")
        
        return LevelClassification(
            level=level,
            confidence_score=confidence_score,
            reasoning=reasoning,
            evidence=evidence,
            verification_sources=verification_data.get('additional_sources', [])
        )
    
    async def rank_founders_batch(
        self, 
        founders_data: List[Dict[str, Any]],
        batch_size: int = 3,
        use_verification: bool = True
    ) -> List[LevelClassification]:
        """Rank multiple founders in batches."""
        
        results = []
        
        for i in range(0, len(founders_data), batch_size):
            batch = founders_data[i:i + batch_size]
            logger.info(f"Processing founder batch {i//batch_size + 1}/{(len(founders_data) + batch_size - 1)//batch_size}")
            
            # Process batch sequentially to avoid rate limits
            for founder_data in batch:
                try:
                    result = await self.rank_founder(founder_data, use_verification=use_verification)
                    results.append(result)
                    
                    # Rate limiting between founders
                    await asyncio.sleep(1.0)
                    
                except Exception as e:
                    logger.error(f"Error ranking founder {founder_data.get('name', 'Unknown')}: {e}")
                    results.append(self._create_fallback_result(founder_data.get('name', 'Unknown')))
        
        return results
    
    async def _analyze_with_claude(
        self, 
        founder_data: Dict[str, Any], 
        company_data: Dict[str, Any] = None
    ) -> Optional[Dict[str, Any]]:
        """Analyze founder using Claude Sonnet 4."""
        
        system_prompt = RankingPrompts.SYSTEM_PROMPT

        # Prepare founder context
        founder_context = RankingPrompts.create_founder_analysis_prompt(founder_data, company_data)
        
        try:
            response = await self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                temperature=0.1,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": f"Analyze this founder and classify their experience level:\n\n{founder_context}"
                    }
                ]
            )
            
            # Parse JSON response
            content = response.content[0].text.strip()
            
            # Clean and extract JSON from response
            parsed_json = self._extract_json_from_response(content)
            if parsed_json:
                return parsed_json
            
            logger.error(f"Failed to parse JSON from Claude response: {content[:200]}...")
            return None
            
        except Exception as e:
            logger.error(f"Claude analysis failed: {e}")
            return None
    
    def _extract_json_from_response(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON from Claude response with robust error handling."""
        import re
        
        # Strategy 1: Extract from markdown code blocks
        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            if json_end != -1:
                json_content = content[json_start:json_end].strip()
                try:
                    return json.loads(json_content)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode error in markdown block: {e}")
        
        # Strategy 2: Extract from generic code blocks
        if "```" in content:
            json_start = content.find("```") + 3
            json_end = content.find("```", json_start)
            if json_end != -1:
                json_content = content[json_start:json_end].strip()
                try:
                    return json.loads(json_content)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode error in code block: {e}")
        
        # Strategy 3: Look for JSON object patterns
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, content, re.DOTALL)
        
        for match in matches:
            try:
                # Clean the JSON content
                clean_json = self._clean_json_string(match)
                return json.loads(clean_json)
            except json.JSONDecodeError:
                continue
        
        # Strategy 4: Try to parse the entire content as JSON
        try:
            clean_content = self._clean_json_string(content)
            return json.loads(clean_content)
        except json.JSONDecodeError:
            pass
        
        # Strategy 5: Look for key-value pairs and construct JSON
        try:
            return self._construct_json_from_text(content)
        except Exception as e:
            logger.warning(f"Failed to construct JSON from text: {e}")
        
        return None
    
    def _clean_json_string(self, json_str: str) -> str:
        """Clean JSON string by removing control characters and fixing common issues."""
        import re
        
        # Remove control characters except newlines and tabs
        json_str = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', json_str)
        
        # Fix common JSON issues
        json_str = json_str.replace('```json', '').replace('```', '')
        json_str = json_str.strip()
        
        # Remove trailing commas before closing brackets
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        return json_str
    
    def _construct_json_from_text(self, text: str) -> Dict[str, Any]:
        """Attempt to construct JSON from text by extracting key information."""
        import re
        
        # Look for experience level
        experience_match = re.search(r'experience[_\s]*level[:\s]*["\\]?([Ll]\d+|INSUFFICIENT_DATA)["\\]?', text, re.IGNORECASE)
        experience_level = experience_match.group(1) if experience_match else "INSUFFICIENT_DATA"
        
        # Look for confidence score
        confidence_match = re.search(r'confidence[_\s]*score[:\s]*([0-9.]+)', text, re.IGNORECASE)
        confidence_score = float(confidence_match.group(1)) if confidence_match else 0.1
        
        # Look for reasoning
        reasoning_match = re.search(r'reasoning[:\s]*["\\]([^"]+)["\\]', text, re.IGNORECASE | re.DOTALL)
        reasoning = reasoning_match.group(1) if reasoning_match else "Unable to parse reasoning from response"
        
        # Look for evidence
        evidence_match = re.search(r'evidence[:\s]*\[(.*?)\]', text, re.IGNORECASE | re.DOTALL)
        evidence = []
        if evidence_match:
            evidence_text = evidence_match.group(1)
            evidence = [item.strip().strip('"\'') for item in evidence_text.split(',') if item.strip()]
        
        return {
            "experience_level": experience_level,
            "confidence_score": min(1.0, max(0.0, confidence_score)),
            "reasoning": reasoning[:500],  # Limit reasoning length
            "evidence": evidence[:5]  # Limit evidence items
        }
    
    async def _finalize_ranking_with_verification(
        self, 
        initial_ranking: Dict[str, Any], 
        verification_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Finalize ranking incorporating verification data."""
        
        if not verification_data.get('verified_facts'):
            # No verification data - return initial ranking with lower confidence
            initial_ranking['confidence_score'] *= 0.8
            return initial_ranking
        
        # Check verification status
        verification_status = verification_data.get('verification_status', 'unverified')
        verification_confidence = verification_data.get('confidence_score', 0.0)
        
        # Adjust confidence based on verification
        if verification_status == 'verified' and verification_confidence > 0.7:
            # Verification supports initial ranking
            initial_ranking['confidence_score'] = min(1.0, initial_ranking.get('confidence_score', 0) + 0.2)
            initial_ranking['reasoning'] += f"\n\nVERIFIED: {verification_data.get('verified_facts', [])[:2]}"
        elif verification_status == 'unverified':
            # Verification contradicts initial ranking
            initial_ranking['confidence_score'] *= 0.6
            initial_ranking['reasoning'] += f"\n\nVERIFICATION CONCERNS: {verification_data.get('contradictions', [])[:2]}"
        
        return initial_ranking
    
    def _create_fallback_result(self, founder_name: str) -> LevelClassification:
        """Create fallback result when analysis fails."""
        return LevelClassification(
            level=ExperienceLevel.INSUFFICIENT_DATA,
            confidence_score=0.1,
            reasoning=f"Unable to analyze {founder_name} due to insufficient data or API errors",
            evidence=[],
            verification_sources=[]
        )


class FounderRankingService:
    """Pure ranking service - assumes data is already enhanced elsewhere."""
    
    def __init__(self):
        self.claude_ranking_service = ClaudeSonnet4RankingService()
    
    async def rank_founders_batch(
        self,
        founder_profiles: List[Any],  # LinkedInProfile objects
        batch_size: int = 5,
        use_enhanced: bool = True
    ) -> List[Any]:
        logger.info(f"🎯 Ranking {len(founder_profiles)} founders")
        
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
            for profile in founder_profiles:
                fallback_result = self.claude_ranking_service._create_fallback_result(getattr(profile, 'person_name', 'Unknown Founder'))
                analysis_results.append(fallback_result)
        
        if len(analysis_results) != len(founder_profiles):
            logger.warning(f"Mismatch in analysis results: {len(analysis_results)} vs {len(founder_profiles)} profiles")
            # Pad with fallback results if needed
            while len(analysis_results) < len(founder_profiles):
                fallback_result = self.claude_ranking_service._create_fallback_result("Unknown Founder")
                analysis_results.append(fallback_result)
        
        # Attach ranking data directly to profiles
        for profile, analysis_result in zip(founder_profiles, analysis_results):
            try:
                # Convert AI result to L-level classification
                classification = self._convert_to_classification(analysis_result, profile)
                
                # Attach ranking data to profile object
                profile.l_level = classification.level.value
                profile.confidence_score = classification.confidence_score
                profile.reasoning = classification.reasoning
                
            except Exception as e:
                logger.error(f"Error ranking {getattr(profile, 'person_name', 'Unknown Founder')}: {e}")
                # Add fallback ranking data to profile
                profile.l_level = ExperienceLevel.INSUFFICIENT_DATA.value
                profile.confidence_score = 0.1
                profile.reasoning = f"Ranking failed: {str(e)}"
        
        logger.info(f"✅ Ranked {len(founder_profiles)} founders")
        return founder_profiles
    
    def _profile_to_ai_dict(self, profile: Any) -> Dict[str, Any]:
        """Convert LinkedInProfile to dict for AI ranking."""
        # Handle None profile
        if profile is None:
            return {
                "name": "Unknown Founder",
                "company_name": "",
                "title": "Founder",
                "about": "",
                "location": "",
                "linkedin_url": "",
                "experience_1_title": "",
                "experience_1_company": "",
                "education_1_school": "",
                "education_1_degree": "",
                "overall_confidence": 0.1
            }
        
        # Try to extract from structured data first, then fall back to individual fields
        experience_1_title = ""
        experience_1_company = ""
        
        # Check for structured experience data
        experience = getattr(profile, 'experience', [])
        if experience and len(experience) > 0 and isinstance(experience[0], dict):
            experience_1_title = experience[0].get('title', '') or ""
            experience_1_company = experience[0].get('company', '') or ""
        
        # Fall back to individual fields if structured data not available
        if not experience_1_title:
            experience_1_title = getattr(profile, 'experience_1_title', '') or ""
        if not experience_1_company:
            experience_1_company = getattr(profile, 'experience_1_company', '') or ""
        
        # Same approach for education
        education_1_school = ""
        education_1_degree = ""
        
        # Check for structured education data
        education = getattr(profile, 'education', [])
        if education and len(education) > 0 and isinstance(education[0], dict):
            education_1_school = education[0].get('school', '') or ""
            education_1_degree = education[0].get('degree', '') or ""
        
        # Fall back to individual fields if structured data not available
        if not education_1_school:
            education_1_school = getattr(profile, 'education_1_school', '') or ""
        if not education_1_degree:
            education_1_degree = getattr(profile, 'education_1_degree', '') or ""
        
        # Convert LinkedInProfile to dict format with better null handling
        profile_name = getattr(profile, 'person_name', '') or "Unknown Founder"
        
        data = {
            "name": profile_name,
            "company_name": getattr(profile, 'company_name', '') or "",
            "title": getattr(profile, 'title', '') or "Founder",
            "about": getattr(profile, 'about', '') or "",
            "location": getattr(profile, 'location', '') or "",
            "linkedin_url": getattr(profile, 'linkedin_url', '') or "",
            "experience_1_title": experience_1_title,
            "experience_1_company": experience_1_company,
            "education_1_school": education_1_school,
            "education_1_degree": education_1_degree,
            "overall_confidence": 0.3 if getattr(profile, 'linkedin_url', '') else 0.1
        }
        return data
    
    def _convert_to_classification(
        self, 
        analysis_result: LevelClassification,
        profile: Any
    ) -> LevelClassification:
        """Convert AI analysis to L-level classification."""
        
        # Map to enum
        try:
            level_enum = ExperienceLevel(analysis_result.level)
        except ValueError:
            level_enum = ExperienceLevel.INSUFFICIENT_DATA
        
        # Use confidence score as-is for basic LinkedIn data
        confidence_score = analysis_result.confidence_score
        
        return LevelClassification(
            level=level_enum,
            confidence_score=confidence_score,
            reasoning=analysis_result.reasoning,
            evidence=analysis_result.evidence,
            verification_sources=analysis_result.verification_sources
        )
    
    def _create_dataset_row(
        self, 
        profile: Any, 
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
            "name": getattr(profile, 'person_name', ''),
            "company_name": getattr(profile, 'company_name', ''),
            "title": getattr(profile, 'title', ''),
            "location": getattr(profile, 'location', ''),
            "about": getattr(profile, 'about', ''),
            "linkedin_url": getattr(profile, 'linkedin_url', ''),
            
            # Basic profile data from LinkedInProfile
            "experience_1_title": getattr(profile, 'experience', [{}])[0].get('title', '') if getattr(profile, 'experience', []) else '',
            "experience_1_company": getattr(profile, 'experience', [{}])[0].get('company', '') if getattr(profile, 'experience', []) else '',
            "experience_2_title": getattr(profile, 'experience', [{}])[1].get('title', '') if len(getattr(profile, 'experience', [])) > 1 else '',
            "experience_2_company": getattr(profile, 'experience', [{}])[1].get('company', '') if len(getattr(profile, 'experience', [])) > 1 else '',
            "education_1_school": getattr(profile, 'education', [{}])[0].get('school', '') if getattr(profile, 'education', []) else '',
            "education_1_degree": getattr(profile, 'education', [{}])[0].get('degree', '') if getattr(profile, 'education', []) else '',
            "education_2_school": getattr(profile, 'education', [{}])[1].get('school', '') if len(getattr(profile, 'education', [])) > 1 else '',
            "education_2_degree": getattr(profile, 'education', [{}])[1].get('degree', '') if len(getattr(profile, 'education', [])) > 1 else '',
            "skills": '|'.join(getattr(profile, 'skills', [])),
            
            "ranking_timestamp": datetime.now().isoformat()
        }
        
        # Add basic metadata
        row.update({
            "data_collected": True,
            "overall_confidence": 0.3 if getattr(profile, 'linkedin_url', '') else 0.1
        })
        
        return row
