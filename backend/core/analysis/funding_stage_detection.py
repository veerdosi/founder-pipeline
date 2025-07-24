"""Funding stage detection service using ChatGPT for accurate stage classification."""

import logging
from typing import Optional
from openai import AsyncOpenAI
from ..config import settings
from ...utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class FundingStageDetectionService:
    """Service for detecting accurate funding stages using AI analysis."""
    
    def __init__(self):
        self.openai = AsyncOpenAI(api_key=settings.openai_api_key)
        self.rate_limiter = RateLimiter(max_requests=60, time_window=60)
    
    async def detect_funding_stage(
        self, 
        company_name: str,
        description: str,
        total_funding_usd: Optional[float] = None,
        last_funding_amount_usd: Optional[float] = None,
        number_of_funding_rounds: Optional[int] = None,
        founded_year: Optional[int] = None
    ) -> Optional[str]:
        """
        Detect the most likely funding stage for a company using AI analysis.
        
        Returns standardized funding stage names:
        - Pre-Seed
        - Seed  
        - Series A
        - Series B
        - Series C
        - Series D+
        - Growth/Late Stage
        - Unknown
        """
        try:
            context = self._prepare_context(
                company_name, description, total_funding_usd, 
                last_funding_amount_usd, number_of_funding_rounds, founded_year
            )
            
            funding_stage = await self._analyze_funding_stage(company_name, context)
            
            return funding_stage or "Unknown"
            
        except Exception as e:
            logger.error(f"Error detecting funding stage for {company_name}: {e}")
            return "Unknown"
    
    def _prepare_context(
        self, 
        company_name: str, 
        description: str, 
        total_funding_usd: Optional[float],
        last_funding_amount_usd: Optional[float],
        number_of_funding_rounds: Optional[int],
        founded_year: Optional[int]
    ) -> str:
        """Prepare combined context for funding stage detection."""
        contexts = [
            f"Company: {company_name}",
            f"Description: {description}",
        ]
        
        if total_funding_usd is not None:
            contexts.append(f"Total Funding: ${total_funding_usd:,.0f} USD")
        
        if last_funding_amount_usd is not None:
            contexts.append(f"Last Funding Amount: ${last_funding_amount_usd:,.0f} USD")
            
        if number_of_funding_rounds is not None:
            contexts.append(f"Number of Funding Rounds: {number_of_funding_rounds}")
            
        if founded_year is not None:
            contexts.append(f"Founded Year: {founded_year}")
        
        return "\n".join(contexts)
    
    async def _analyze_funding_stage(self, company_name: str, context: str) -> Optional[str]:
        """Analyze funding stage using ChatGPT with specific prompt requirements."""
        
        prompt = f"""
        Based on the company information provided, determine the most likely current funding stage. 

        Company Information:
        {context}

        Funding Stage Guidelines:
        - Pre-Seed: Usually < $500K, very early stage, often bootstrapped or friends/family
        - Seed: $500K - $3M, early product development, small team
        - Series A: $3M - $15M, product-market fit, scaling team and operations
        - Series B: $15M - $50M, established revenue, expanding market reach
        - Series C: $50M - $100M, profitable or near-profitable, market expansion
        - Series D+: $100M+, multiple later rounds, preparing for exit
        - Growth/Late Stage: Very large rounds, mature company, pre-IPO
        - Unknown: Insufficient information to determine

        Instructions:
        1. Consider the total funding amount as the primary indicator
        2. Factor in the number of rounds - more rounds typically means later stage
        3. Consider company age and description maturity
        4. Use the last funding amount to understand current stage
        5. Return ONLY one of these exact values: Pre-Seed, Seed, Series A, Series B, Series C, Series D+, Growth/Late Stage, Unknown

        Answer with just the funding stage name, nothing else.
        """
        
        try:
            await self.rate_limiter.acquire()
            
            response = await self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=50
            )
            
            content = response.choices[0].message.content
            if content:
                stage = content.strip()
                
                # Validate that the response is one of our expected stages
                valid_stages = [
                    "Pre-Seed", "Seed", "Series A", "Series B", 
                    "Series C", "Series D+", "Growth/Late Stage", "Unknown"
                ]
                
                if stage in valid_stages:
                    return stage
                else:
                    logger.warning(f"Invalid funding stage returned: {stage}, defaulting to Unknown")
                    return "Unknown"
            
            return None
            
        except Exception as e:
            logger.warning(f"AI funding stage detection failed for {company_name}: {e}")
            return None


# Global instance for easy import
funding_stage_detection_service = FundingStageDetectionService()