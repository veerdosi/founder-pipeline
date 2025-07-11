"""AI-powered sector classification service for accurate company categorization."""

import json
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from openai import AsyncOpenAI

from ...core import settings
from ...utils.rate_limiter import RateLimiter

import logging
logger = logging.getLogger(__name__)


@dataclass
class SectorClassification:
    """Structured sector classification result."""
    primary_sector: str = "unknown"
    sub_sectors: List[str] = field(default_factory=list)
    ai_focus: str = "unknown"
    technology_stack: List[str] = field(default_factory=list)
    business_model: str = "unknown"
    target_market: str = "unknown"
    confidence_score: float = 0.0
    reasoning: str = "No classification performed"
    classification_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class SectorClassifier:
    """AI-powered classification of companies into detailed sectors and categories."""
    
    def __init__(self):
        self.openai = AsyncOpenAI(api_key=settings.openai_api_key)
        self.rate_limiter = RateLimiter(max_requests=60, time_window=60)
    
    async def classify_company(
        self, 
        company_name: str,
        description: str,
        website_content: str = "",
        additional_context: str = ""
    ) -> SectorClassification:
        """Classify a company into detailed sectors using AI analysis."""
        try:
            combined_text = self._prepare_classification_text(
                company_name, description, website_content, additional_context
            )
            
            ai_result = await self._get_ai_classification(company_name, combined_text)
            
            if not ai_result:
                return self._get_default_classification("AI classification returned no result.")

            return SectorClassification(
                primary_sector=ai_result.get("primary_sector", "unknown"),
                sub_sectors=ai_result.get("sub_sectors", []),
                ai_focus=ai_result.get("ai_focus", "unknown"),
                technology_stack=ai_result.get("technology_stack", []),
                business_model=ai_result.get("business_model", "unknown"),
                target_market=ai_result.get("target_market", "unknown"),
                confidence_score=ai_result.get("confidence_score", 0.0),
                reasoning=ai_result.get("reasoning", "No reasoning provided."),
                classification_timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error classifying company {company_name}: {e}")
            return self._get_default_classification(str(e))
    
    def _prepare_classification_text(
        self, 
        company_name: str, 
        description: str, 
        website_content: str, 
        additional_context: str
    ) -> str:
        """Prepare combined text for classification."""
        texts = [
            f"Company: {company_name}",
            f"Description: {description}",
            f"Website Content: {website_content[:1500]}" if website_content else "",
            f"Additional Context: {additional_context[:500]}" if additional_context else ""
        ]
        return "\n".join(filter(None, texts))
    
    async def _get_ai_classification(self, company_name: str, text: str) -> Optional[Dict]:
        """Get AI-powered sector classification."""
        prompt = f"""
        Analyze the following information about '{company_name}' and classify it.
        
        Information:
        {text}
        
        Provide a detailed classification in JSON format with the following fields:
        - primary_sector: The main industry sector (e.g., 'ai_fintech', 'ai_healthcare').
        - sub_sectors: A list of specific sub-categories.
        - ai_focus: A detailed description of their AI application.
        - technology_stack: A list of key technologies, languages, and platforms.
        - business_model: The company's business model (e.g., 'b2b_saas', 'api_platform').
        - target_market: The primary customer segment (e.g., 'enterprise', 'developers').
        - confidence_score: Your confidence in this classification (0.0 to 1.0).
        - reasoning: A brief explanation for your classification.
        
        Return only the JSON object.
        """
        
        try:
            await self.rate_limiter.acquire()
            
            response = await self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            if content:
                return json.loads(content)
            return None
            
        except Exception as e:
            logger.warning(f"AI classification request failed for {company_name}: {e}")
            return None
    
    def _get_default_classification(self, reason: str) -> SectorClassification:
        """Return default classification for error cases."""
        return SectorClassification(
            reasoning=f"Default classification due to analysis error: {reason}",
            confidence_score=0.1
        )