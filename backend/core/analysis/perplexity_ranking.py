"""Perplexity Sonar fallback ranking service for founder classification."""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
import httpx

from ...core.config import settings
from .ai_analysis import FounderAnalysisResult

logger = logging.getLogger(__name__)


class PerplexityRankingService:
    """Perplexity Sonar service for L1-L10 founder classification as Claude fallback."""
    
    def __init__(self):
        self.api_key = settings.perplexity_api_key
        self.base_url = "https://api.perplexity.ai/chat/completions"
        
    async def rank_founder(
        self, 
        founder_data: Dict[str, Any],
        company_data: Dict[str, Any] = None
    ) -> FounderAnalysisResult:
        """Rank a single founder using Perplexity Sonar."""
        
        founder_name = founder_data.get('name', 'Unknown')
        logger.info(f"Ranking founder with Perplexity fallback: {founder_name}")
        
        try:
            analysis = await self._analyze_with_perplexity(founder_data, company_data)
            
            if not analysis:
                return self._create_fallback_result(founder_name)
            
            return FounderAnalysisResult(
                experience_level=analysis['experience_level'],
                confidence_score=analysis['confidence_score'],
                reasoning=analysis['reasoning'],
                evidence=analysis['evidence'],
                verification_sources=[]
            )
            
        except Exception as e:
            logger.error(f"Perplexity analysis failed for {founder_name}: {e}")
            return self._create_fallback_result(founder_name)
    
    async def rank_founders_batch(
        self, 
        founders_data: List[Dict[str, Any]],
        batch_size: int = 3
    ) -> List[FounderAnalysisResult]:
        """Rank multiple founders in batches using Perplexity."""
        
        results = []
        
        for i in range(0, len(founders_data), batch_size):
            batch = founders_data[i:i + batch_size]
            logger.info(f"Processing founder batch {i//batch_size + 1}/{(len(founders_data) + batch_size - 1)//batch_size} with Perplexity")
            
            for founder_data in batch:
                try:
                    result = await self.rank_founder(founder_data)
                    results.append(result)
                    
                    # Rate limiting
                    await asyncio.sleep(2.0)
                    
                except Exception as e:
                    logger.error(f"Error ranking founder {founder_data.get('name', 'Unknown')}: {e}")
                    results.append(self._create_fallback_result(founder_data.get('name', 'Unknown')))
        
        return results
    
    async def _analyze_with_perplexity(
        self, 
        founder_data: Dict[str, Any], 
        company_data: Dict[str, Any] = None
    ) -> Optional[Dict[str, Any]]:
        """Analyze founder using Perplexity Sonar."""
        
        system_prompt = """You are an expert venture capital analyst specializing in founder evaluation.

Analyze the founder's background and classify them into one of the L1-L10 experience levels:

L10 - Legendary Entrepreneurs: Multiple IPOs >$1B, industry pioneers (e.g., Jeff Bezos, Elon Musk)
L9 - Transformational Leaders: 1 IPO >$1B, building second company (e.g., Jack Dorsey, Reid Hoffman)
L8 - Proven Unicorn Builders: Built 1+ companies to $1B+ valuation (e.g., Brian Chesky, Daniel Ek)
L7 - Elite Serial Entrepreneurs: 2+ exits >$100M OR 2+ unicorn companies
L6 - Market Innovators: Groundbreaking innovation, patents, awards, thought leadership
L5 - Growth-Stage Entrepreneurs: Companies with >$50M funding, IPO preparation
L4 - Proven Operators: $10M-$100M exits OR C-level roles at notable tech companies
L3 - Technical Veterans: 10+ years experience, PhD, senior technical roles
L2 - Early-Stage Entrepreneurs: Accelerator graduates, 2-5 years experience, seed funding
L1 - Nascent Founders: <2 years experience, first-time founders, recent graduates

Return ONLY valid JSON with:
- experience_level: L1-L10
- confidence_score: 0.0-1.0
- reasoning: detailed explanation
- evidence: list of key supporting facts

Be conservative - require strong evidence for higher levels."""

        founder_context = self._prepare_founder_context(founder_data, company_data)
        
        payload = {
            "model": "sonar-reasoning-pro",
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"Analyze this founder and classify their experience level:\n\n{founder_context}"
                }
            ],
            "max_tokens": 2000,
            "temperature": 0.1,
            "top_p": 0.9,
            "return_citations": True,
            "search_domain_filter": ["perplexity.ai"],
            "return_images": False,
            "return_related_questions": False
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json=payload,
                    timeout=30.0
                )
                
                response.raise_for_status()
                data = response.json()
                
                content = data["choices"][0]["message"]["content"].strip()
                
                # Extract JSON from response
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
            logger.error(f"Perplexity API call failed: {e}")
            return None
    
    def _prepare_founder_context(
        self, 
        founder_data: Dict[str, Any], 
        company_data: Dict[str, Any] = None
    ) -> str:
        """Prepare founder context for Perplexity analysis."""
        
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
    
    def _create_fallback_result(self, founder_name: str) -> FounderAnalysisResult:
        """Create fallback result when analysis fails."""
        return FounderAnalysisResult(
            experience_level="INSUFFICIENT_DATA",
            confidence_score=0.1,
            reasoning=f"Unable to analyze {founder_name} due to insufficient data or API errors",
            evidence=[],
            verification_sources=[]
        )
