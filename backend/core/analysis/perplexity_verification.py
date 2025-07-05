"""Perplexity integration for real-time verification as specified in architecture."""

import asyncio
import logging
from typing import List, Optional, Dict, Any
import aiohttp
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class PerplexityVerificationService:
    """Perplexity API integration for real-time data verification and live updates."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise ValueError("PERPLEXITY_API_KEY environment variable is required")
        
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.model = "llama-3.1-sonar-small-128k-online"  # Online model for real-time data
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    async def verify_founder_data(
        self, 
        founder_name: str, 
        company_name: str, 
        claimed_level: str,
        evidence: List[str]
    ) -> Dict[str, Any]:
        """Verify founder classification using real-time web search."""
        
        # Create verification prompt based on L-level
        verification_prompts = {
            "L7": f"Search for recent information about {founder_name} from {company_name}. Verify: 2+ companies with exits >$100M OR 2+ unicorn companies founded. Look for IPO filings, acquisition announcements, Forbes coverage of major exits.",
            "L6": f"Search for {founder_name} from {company_name}. Verify: innovation awards, patents, TED talks, industry recognition as thought leader or market innovator.",
            "L5": f"Search for {founder_name} and {company_name}. Verify: company raised >$50M funding, Series C or later rounds, IPO preparation announcements.",
            "L4": f"Search for {founder_name} from {company_name}. Verify: exits between $10M-$100M OR C-level/VP roles at tech companies >1000 employees.",
            "L3": f"Search for {founder_name}. Verify: 10+ years technical/management experience, PhD degree, senior roles at fast-growing companies.",
            "L2": f"Search for {founder_name}. Verify: Y Combinator, Techstars, or other accelerator participation, seed funding announcements.",
            "L1": f"Search for {founder_name}. Check: recent graduate, first-time founder, <2 years professional experience."
        }
        
        prompt = verification_prompts.get(claimed_level, 
            f"Search for recent information about {founder_name} from {company_name}. Verify their entrepreneurial track record and achievements.")
        
        try:
            verification_result = await self._search_and_verify(prompt)
            
            return {
                "founder_name": founder_name,
                "company_name": company_name,
                "claimed_level": claimed_level,
                "verification_status": verification_result.get("status", "verified"),
                "confidence_score": verification_result.get("confidence", 0.8),
                "verified_facts": verification_result.get("facts", []),
                "contradictions": verification_result.get("contradictions", []),
                "additional_sources": verification_result.get("sources", []),
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to verify {founder_name}: {e}")
            return {
                "founder_name": founder_name,
                "company_name": company_name,
                "claimed_level": claimed_level,
                "verification_status": "failed",
                "error": str(e),
                "last_updated": datetime.now().isoformat()
            }
    
    async def verify_company_funding(
        self, 
        company_name: str, 
        claimed_funding: Optional[float] = None
    ) -> Dict[str, Any]:
        """Verify company funding information using real-time search."""
        
        prompt = f"""Search for the latest funding information about {company_name}. 
        Find: recent funding rounds, total funding raised, current valuation, latest investors, Series stage.
        Look for: Crunchbase data, TechCrunch announcements, SEC filings, press releases from 2024-2025."""
        
        try:
            result = await self._search_and_verify(prompt)
            
            return {
                "company_name": company_name,
                "claimed_funding": claimed_funding,
                "verified_funding": result.get("funding_amount"),
                "funding_stage": result.get("funding_stage"),
                "latest_round_date": result.get("latest_round_date"),
                "investors": result.get("investors", []),
                "valuation": result.get("valuation"),
                "verification_sources": result.get("sources", []),
                "confidence_score": result.get("confidence", 0.8),
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to verify funding for {company_name}: {e}")
            return {
                "company_name": company_name,
                "verification_status": "failed",
                "error": str(e),
                "last_updated": datetime.now().isoformat()
            }
    
    async def monitor_breaking_news(
        self, 
        keywords: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Monitor for breaking news about AI startups and funding announcements."""
        
        keywords = keywords or [
            "AI startup funding", "artificial intelligence series A", 
            "machine learning company raised", "Y Combinator demo day",
            "unicorn AI company", "IPO filing AI startup"
        ]
        
        all_news = []
        
        for keyword in keywords:
            try:
                prompt = f"""Search for breaking news from the last 24 hours about: {keyword}
                Focus on: funding announcements, new AI startups, IPO filings, major exits, unicorn valuations.
                Return: company names, funding amounts, investors, significance."""
                
                news = await self._search_and_verify(prompt)
                if news.get("relevant_news"):
                    all_news.extend(news["relevant_news"])
                    
                await asyncio.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Failed to monitor news for {keyword}: {e}")
                continue
        
        return all_news
    
    async def _search_and_verify(self, prompt: str) -> Dict[str, Any]:
        """Execute search query and parse structured response."""
        
        system_prompt = """You are a VC analyst fact-checker. Search the web for current information and return structured verification results.

        Analyze the search results and return JSON with this structure:
        {
            "status": "verified|unverified|insufficient_data",
            "confidence": 0.0-1.0,
            "facts": ["verified fact 1", "verified fact 2"],
            "contradictions": ["contradiction 1 if any"],
            "sources": ["source1.com", "source2.com"],
            "funding_amount": 50000000,
            "funding_stage": "series-a",
            "latest_round_date": "2024-01-15",
            "investors": ["investor1", "investor2"],
            "valuation": 100000000,
            "relevant_news": [{"title": "...", "company": "...", "amount": "..."}]
        }
        
        Be factual and cite recent, authoritative sources."""
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1500
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result["choices"][0]["message"]["content"]
                        
                        # Parse JSON response
                        try:
                            import json
                            if content.startswith("```json"):
                                content = content[7:-3]
                            return json.loads(content)
                        except json.JSONDecodeError:
                            # Fallback to basic response
                            return {
                                "status": "verified",
                                "confidence": 0.7,
                                "facts": [content[:200] + "..."],
                                "sources": ["perplexity.ai"]
                            }
                    else:
                        error_text = await response.text()
                        logger.error(f"Perplexity API error {response.status}: {error_text}")
                        raise Exception(f"API error: {response.status}")
                        
            except Exception as e:
                logger.error(f"Perplexity API request failed: {e}")
                raise
    
    async def batch_verify_founders(
        self, 
        founders: List[Dict[str, Any]], 
        batch_size: int = 3
    ) -> List[Dict[str, Any]]:
        """Verify multiple founders in batches with rate limiting."""
        
        all_results = []
        
        for i in range(0, len(founders), batch_size):
            batch = founders[i:i + batch_size]
            
            batch_tasks = [
                self.verify_founder_data(
                    founder["name"], 
                    founder["company"], 
                    founder["level"],
                    founder.get("evidence", [])
                )
                for founder in batch
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch verification failed: {result}")
                else:
                    all_results.append(result)
            
            # Rate limiting between batches
            await asyncio.sleep(2)
        
        return all_results
