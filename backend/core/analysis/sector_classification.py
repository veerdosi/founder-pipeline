"""Centralized sector description service for consistent company categorization."""

import json
from typing import Optional
from openai import AsyncOpenAI
from ...core import settings
from ...utils.rate_limiter import RateLimiter
import logging

logger = logging.getLogger(__name__)


class SectorDescriptionService:
    """Service for generating detailed, searchable sector descriptions for companies."""
    
    def __init__(self):
        self.openai = AsyncOpenAI(api_key=settings.openai_api_key)
        self.rate_limiter = RateLimiter(max_requests=60, time_window=60)
    
    async def get_sector_description(
        self, 
        company_name: str,
        company_description: str,
        website_content: str = "",
        additional_context: str = ""
    ) -> Optional[str]:
        """
        Generate a detailed, searchable sector description for a company.
        
        The description will be:
        - Detailed and accurate for market research
        - Searchable online and in research reports
        - Less than 10 words
        - Suitable for running market analysis
        """
        try:
            combined_context = self._prepare_context(
                company_name, company_description, website_content, additional_context
            )
            
            sector_description = await self._generate_sector_description(company_name, combined_context)
            
            return sector_description
            
        except Exception as e:
            logger.error(f"Error generating sector description for {company_name}: {e}")
            return "AI Software Solutions"  # Default fallback
    
    def _prepare_context(
        self, 
        company_name: str, 
        description: str, 
        website_content: str, 
        additional_context: str
    ) -> str:
        """Prepare combined context for sector description generation."""
        contexts = [
            f"Company: {company_name}",
            f"Description: {description}",
            f"Website Content: {website_content[:1000]}" if website_content else "",
            f"Additional Context: {additional_context[:300]}" if additional_context else ""
        ]
        return "\n".join(filter(None, contexts))
    
    async def _generate_sector_description(self, company_name: str, context: str) -> Optional[str]:
        """Generate AI-powered sector description using the specified prompt requirements."""
        
        prompt = f"""
        I need a startup sector description that I can run a market research on so the description needs to be detailed and accurate and searchable online and in research reports. Recommend the best sector description for me. Make sure the output is less than 10 words.

        Company Information:
        {context}

        Requirements:
        - Must be less than 10 words
        - Must be detailed and accurate
        - Must be searchable online and in research reports
        - Must be suitable for market research
        - Should capture the specific niche/vertical the company operates in

        Examples of good sector descriptions:
        - "AI-Powered Healthcare Diagnostics Platform"
        - "Enterprise Data Analytics Software"
        - "Autonomous Vehicle Navigation Technology"
        - "Fintech Payment Processing Solutions"
        - "B2B Customer Support Automation"

        Return only the sector description, nothing else.
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
                sector_description = content.strip()
                
                # Validate word count
                word_count = len(sector_description.split())
                if word_count <= 10:
                    return sector_description
                else:
                    logger.warning(f"Generated sector description exceeds 10 words ({word_count} words): {sector_description}")
                    # Try to truncate while keeping meaning
                    words = sector_description.split()[:10]
                    return ' '.join(words)
            
            return None
            
        except Exception as e:
            logger.warning(f"AI sector description generation failed for {company_name}: {e}")
            return None


# Global instance for easy import
sector_description_service = SectorDescriptionService()


