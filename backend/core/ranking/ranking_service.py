"""Pure ranking service that ONLY adds L-level classifications to founder datasets."""

from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
import asyncio
import json
import anthropic


from .models import FounderProfile, LevelClassification, ExperienceLevel
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
            return self._create_fallback_result(founder_name)
        
        # Step 2: Verification with Perplexity (if enabled)
        verification_data = {}
        if use_verification and initial_ranking.get('experience_level'):
            try:
                verifier = self._get_perplexity_verifier()
                verification_data = await verifier.verify_founder_data(
                    founder_name=founder_name,
                    company_name=company_name,
                    claimed_level=initial_ranking['experience_level'],
                    evidence=initial_ranking.get('evidence', [])
                )
            except Exception as e:
                logger.warning(f"Perplexity verification failed for {founder_name}: {e}")
                verification_data = {}
        
        # Step 3: Final ranking with verification data
        final_ranking = await self._finalize_ranking_with_verification(
            initial_ranking, verification_data
        )
        
        return LevelClassification(
            level=final_ranking['experience_level'],
            confidence_score=final_ranking['confidence_score'],
            reasoning=final_ranking['reasoning'],
            evidence=final_ranking['evidence'],
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
            for profile in founder_profiles:
                fallback_result = self.claude_ranking_service._create_fallback_result(profile.name)
                analysis_results.append(fallback_result)
        
        if len(analysis_results) != len(founder_profiles):
            logger.warning(f"Mismatch in analysis results: {len(analysis_results)} vs {len(founder_profiles)} profiles")
            # Pad with fallback results if needed
            while len(analysis_results) < len(founder_profiles):
                fallback_result = self.claude_ranking_service._create_fallback_result("Unknown Founder")
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
                
        # Other columns
        other_cols = [col for col in df.columns 
                     if col not in ranking_cols + core_cols + financial_cols + media_cols + web_cols + confidence_cols]
        
        column_order = ranking_cols + core_cols + financial_cols + media_cols + web_cols + confidence_cols + other_cols
        
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
        
        # Add basic confidence score
        data["overall_confidence"] = profile.calculate_overall_confidence()
        
        return data
    
    def _convert_to_classification(
        self, 
        analysis_result: LevelClassification,
        profile: FounderProfile
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
        
        # Add basic metadata
        row.update({
            "data_collected": profile.data_collected,
            "overall_confidence": profile.calculate_overall_confidence()
        })
        
        return row
