"""AI system for founder ranking using Claude Sonnet 4."""

import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json

import anthropic

from ...core import settings

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


class ClaudeSonnet4RankingService:
    """Claude Sonnet 4 ranking service for L1-L10 founder classification."""
    
    def __init__(self):
        self.anthropic_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self.perplexity_verifier = None  # Lazy load to avoid circular import
        
    def _get_perplexity_verifier(self):
        """Lazy load perplexity verifier to avoid circular import."""
        if self.perplexity_verifier is None:
            from ..ranking.verification_service import RealTimeFounderVerifier
            self.perplexity_verifier = RealTimeFounderVerifier()
        return self.perplexity_verifier
        
    async def rank_founder(
        self, 
        founder_data: Dict[str, Any],
        company_data: Dict[str, Any] = None,
        use_verification: bool = True
    ) -> FounderAnalysisResult:
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
        
        return FounderAnalysisResult(
            experience_level=final_ranking['experience_level'],
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
    ) -> List[FounderAnalysisResult]:
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
        
        system_prompt = """You are an expert venture capital analyst specializing in founder evaluation.

Analyze the founder's background and classify them into one of the L1-L10 experience levels:

L10 - Legendary Entrepreneurs: Multiple IPOs >$1B, industry pioneers
L9 - Transformational Leaders: 1 IPO >$1B, building second company  
L8 - Proven Unicorn Builders: Built 1+ companies to $1B+ valuation
L7 - Elite Serial Entrepreneurs: 2+ exits >$100M OR 2+ unicorn companies
L6 - Market Innovators: Groundbreaking innovation, patents, awards, thought leadership
L5 - Growth-Stage Entrepreneurs: Companies with >$50M funding, IPO preparation
L4 - Proven Operators: $10M-$100M exits OR C-level roles at notable tech companies  
L3 - Technical Veterans: 10+ years experience, PhD, senior technical roles
L2 - Early-Stage Entrepreneurs: Accelerator graduates, 2-5 years experience, seed funding
L1 - Nascent Founders: <2 years experience, first-time founders, recent graduates

Return JSON with:
- experience_level: L1-L10
- confidence_score: 0.0-1.0
- reasoning: detailed explanation
- evidence: list of key supporting facts

Be conservative - require strong evidence for higher levels."""

        # Prepare founder context
        founder_context = self._prepare_founder_context(founder_data, company_data)
        
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
            
            # Extract JSON from response (handle markdown code blocks)
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()
            
            return json.loads(content)
            
        except Exception as e:
            logger.error(f"Claude analysis failed: {e}")
            return None
    
    def _prepare_founder_context(
        self, 
        founder_data: Dict[str, Any], 
        company_data: Dict[str, Any] = None
    ) -> str:
        """Prepare founder context for Claude analysis."""
        
        context_parts = []
        
        # Basic info
        context_parts.append(f"Name: {founder_data.get('name', 'Unknown')}")
        context_parts.append(f"Title: {founder_data.get('title', 'Founder')}")
        
        if company_data:
            context_parts.append(f"Company: {company_data.get('name', 'Unknown')}")
            context_parts.append(f"Company Description: {company_data.get('description', 'N/A')}")
            if company_data.get('funding_total_usd'):
                context_parts.append(f"Total Funding: ${company_data['funding_total_usd']:,.0f}")
        
        # Professional background
        if founder_data.get('about'):
            context_parts.append(f"Background: {founder_data['about']}")
        
        # Experience
        for i in range(1, 4):
            exp_title = founder_data.get(f'experience_{i}_title')
            exp_company = founder_data.get(f'experience_{i}_company')
            if exp_title and exp_company:
                context_parts.append(f"Experience {i}: {exp_title} at {exp_company}")
        
        # Education
        for i in range(1, 3):
            edu_school = founder_data.get(f'education_{i}_school')
            edu_degree = founder_data.get(f'education_{i}_degree')
            if edu_school:
                degree_text = f" - {edu_degree}" if edu_degree else ""
                context_parts.append(f"Education {i}: {edu_school}{degree_text}")
        
        # Skills
        skills = []
        for i in range(1, 4):
            skill = founder_data.get(f'skill_{i}')
            if skill:
                skills.append(skill)
        
        if skills:
            context_parts.append(f"Skills: {', '.join(skills)}")
        
        # Additional context
        if founder_data.get('location'):
            context_parts.append(f"Location: {founder_data['location']}")
        
        if founder_data.get('estimated_age'):
            context_parts.append(f"Estimated Age: {founder_data['estimated_age']}")
        
        return "\n".join(context_parts)
    
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
    
    def _create_fallback_result(self, founder_name: str) -> FounderAnalysisResult:
        """Create fallback result when analysis fails."""
        return FounderAnalysisResult(
            experience_level="INSUFFICIENT_DATA",
            confidence_score=0.1,
            reasoning=f"Unable to analyze {founder_name} due to insufficient data or API errors",
            evidence=[],
            verification_sources=[]
        )
