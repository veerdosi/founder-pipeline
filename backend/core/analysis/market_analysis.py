"""Market analysis service using Perplexity AI for enhanced reasoning."""

import asyncio
import re
import time
from datetime import datetime
from typing import Dict, List, Optional

import aiohttp
from openai import AsyncOpenAI

from ..interfaces import MarketAnalysisService
from ..config import settings
from ...utils.data_processing import clean_text
from ...utils.rate_limiter import RateLimiter
from ...models import MarketMetrics, MarketStage

import logging
logger = logging.getLogger(__name__)


class PerplexityMarketAnalysis(MarketAnalysisService):
    """Market analysis using Perplexity AI for comprehensive research."""
    
    def __init__(self):
        self.perplexity_client = AsyncOpenAI(
            api_key=settings.perplexity_api_key,
            base_url="https://api.perplexity.ai"
        )
        self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.rate_limiter = RateLimiter(
            max_requests=settings.requests_per_minute,
            time_window=60
        )
        self.search_rate_limiter = RateLimiter(max_requests=20, time_window=60)
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    def _safe_json_parse(self, content: str) -> dict:
        """Safely parse JSON from a string, returning a dictionary."""
        if not content or not content.strip():
            logger.warning("Empty content provided for JSON parsing.")
            return {}
            
        try:
            # Clean up common JSON formatting issues
            content = content.strip()
            
            # Remove markdown code blocks if present
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            # Try to parse JSON
            import json
            return json.loads(content)
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}. Content: '{content[:200]}...'")
            return {}
        except Exception as e:
            logger.error(f"An unexpected error occurred during JSON parsing: {e}")
            return {}

    async def analyze_market(
        self, 
        sector: str, 
        year: int,
        region: Optional[str] = None
    ) -> MarketMetrics:
        """Analyze market metrics for a sector using Perplexity."""
        logger.info(f"📊 Analyzing market for {sector} ({year}) using Perplexity")
        
        start_time = time.time()
        
        try:
            # Run analysis tasks with Perplexity
            tasks = [
                self._get_market_size_and_cagr_perplexity(sector, year),
                self._get_timing_analysis_perplexity(sector, year),
                self._get_regional_sentiment_perplexity(sector, year, "United States"),
                self._get_regional_sentiment_perplexity(sector, year, "Asia"),
                self._get_competitor_analysis_perplexity(sector, year)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Safely extract results
            market_data = results[0] if isinstance(results[0], dict) else {}
            timing_data = results[1] if isinstance(results[1], dict) else {}
            us_sentiment = results[2] if isinstance(results[2], (int, float)) else 0.0
            asia_sentiment = results[3] if isinstance(results[3], (int, float)) else 0.0
            competitor_data = results[4] if isinstance(results[4], dict) else {}

            # Safely access and convert values, providing 0 as a default
            market_size = float(market_data.get("market_size", 0) or 0)
            cagr = float(market_data.get("cagr", 0) or 0)
            timing_score = float(timing_data.get("timing_score", 0) or 0)
            competitor_count = int(competitor_data.get("competitor_count", 0) or 0)
            total_funding = float(competitor_data.get("total_funding", 0) or 0)
            momentum_score = float(competitor_data.get("momentum_score", 0) or 0)

            # Determine market stage
            market_stage = self._determine_market_stage(
                cagr,
                competitor_count,
                timing_score
            )
            
            # Calculate confidence score
            confidence = self._calculate_confidence_score(
                market_data, competitor_data, timing_score
            )
            
            execution_time = time.time() - start_time
            
            metrics = MarketMetrics(
                market_size_billion=market_size,
                market_size_usd=market_size * 1_000_000_000,
                cagr_percent=cagr,
                growth_rate=cagr,
                timing_score=timing_score,
                us_sentiment=us_sentiment,
                sea_sentiment=asia_sentiment,
                competitor_count=competitor_count,
                total_funding_billion=total_funding,
                momentum_score=momentum_score,
                market_stage=market_stage,
                confidence_score=confidence,
                analysis_date=datetime.utcnow(),
                execution_time=execution_time
            )
            
            logger.info(f"✅ Perplexity market analysis complete for {sector}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing market for {sector}: {e}")
            # Return default metrics on failure
            return MarketMetrics(
                market_size_billion=0,
                market_size_usd=0,
                cagr_percent=0,
                growth_rate=0,
                timing_score=0.0,
                us_sentiment=0.0,
                sea_sentiment=0.0,
                competitor_count=0,
                total_funding_billion=0.0,
                momentum_score=0.0,
                market_stage=MarketStage.UNKNOWN,
                confidence_score=0.1,
                analysis_date=datetime.utcnow(),
                execution_time=time.time() - start_time
            )
    
    async def _get_market_size_and_cagr_perplexity(self, sector: str, year: int) -> Optional[Dict]:
        """Get market size and CAGR using Perplexity, returning structured JSON."""
        prompt = f"""
        Research the {sector} market globally for {year}. Provide the following data in a clear JSON format:
        
        - Market size in billions USD
        - Expected CAGR (Compound Annual Growth Rate) as a percentage
        
        Return only a valid JSON object with the keys "market_size" and "cagr".
        Example: {{"market_size": 15.5, "cagr": 12.5}}
        """
        
        try:
            await self.rate_limiter.acquire()
            
            response = await self.perplexity_client.chat.completions.create(
                model="sonar-pro",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            
            content = response.choices[0].message.content
            return self._safe_json_parse(content)
            
        except Exception as e:
            logger.error(f"Error getting market size/CAGR from Perplexity: {e}")
            return None
    
    async def _get_timing_analysis_perplexity(self, sector: str, year: int) -> Optional[Dict]:
        """Get market timing analysis using Perplexity, returning structured JSON."""
        prompt = f"""
        Analyze the market timing for early-stage {sector} companies in {year}.
        Rate the market timing on a scale of 1-5 (5 being excellent).
        
        Return only a valid JSON object with the key "timing_score".
        Example: {{"timing_score": 4.0}}
        """
        
        try:
            await self.rate_limiter.acquire()
            
            response = await self.perplexity_client.chat.completions.create(
                model="sonar-pro",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            
            content = response.choices[0].message.content
            return self._safe_json_parse(content)
            
        except Exception as e:
            logger.error(f"Error getting timing analysis from Perplexity: {e}")
            return None
    
    async def _get_regional_sentiment_perplexity(self, sector: str, year: int, region: str) -> float:
        """Get regional market sentiment using Perplexity, returning a score."""
        prompt = f"""
        Analyze the {sector} market sentiment in {region} for {year}.
        Rate the sentiment for early-stage companies on a scale of 1-5 (5 being very positive).
        
        Return only a valid JSON object with the key "sentiment_score".
        Example: {{"sentiment_score": 4.5}}
        """
        
        try:
            await self.rate_limiter.acquire()
            
            response = await self.perplexity_client.chat.completions.create(
                model="sonar-pro",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            
            content = response.choices[0].message.content
            result = self._safe_json_parse(content)
            return float(result.get("sentiment_score", 0.0) or 0.0)

        except Exception as e:
            logger.error(f"Error getting regional sentiment from Perplexity for {region}: {e}")
            return 0.0
    
    async def _get_competitor_analysis_perplexity(self, sector: str, year: int) -> Optional[Dict]:
        """Get competitor analysis using Perplexity, returning structured JSON."""
        prompt = f"""
        Analyze the competitive landscape for early-stage {sector} companies in {year}.
        Provide the following data in a clear JSON format:
        
        - Number of major competitors as an integer
        - Total venture funding in this sector in billions USD
        - Market momentum score on a 1-5 scale (5 being high momentum)
        
        Return only a valid JSON object with keys "competitor_count", "total_funding", and "momentum_score".
        Example: {{"competitor_count": 25, "total_funding": 2.1, "momentum_score": 4.0}}
        """
        
        try:
            await self.rate_limiter.acquire()
            
            response = await self.perplexity_client.chat.completions.create(
                model="sonar-pro",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            
            content = response.choices[0].message.content
            return self._safe_json_parse(content)
            
        except Exception as e:
            logger.error(f"Error getting competitor analysis from Perplexity: {e}")
            return None
    
    def _determine_market_stage(
        self, 
        cagr: float, 
        competitor_count: int, 
        timing_score: float
    ) -> MarketStage:
        """Determine market maturity stage."""
        if cagr > 30 and competitor_count < 20 and timing_score >= 4.0:
            return MarketStage.EARLY  # Changed from EMERGING to EARLY
        elif cagr > 15 and competitor_count < 50 and timing_score >= 3.0:
            return MarketStage.GROWTH
        elif cagr < 15 and competitor_count >= 30:
            return MarketStage.MATURE
        else:
            return MarketStage.UNKNOWN
    
    def _calculate_confidence_score(
        self, 
        market_data: Dict, 
        competitor_data: Dict, 
        timing_score: float
    ) -> float:
        """Calculate confidence score for the analysis."""
        scores = []
        
        # Market size confidence
        if market_data.get("market_size", 0) > 0:
            scores.append(0.9)  # Higher confidence with Perplexity
        else:
            scores.append(0.3)
        
        # CAGR confidence
        if market_data.get("cagr", 0) > 0:
            scores.append(0.9)
        else:
            scores.append(0.4)
        
        # Competitor data confidence
        if competitor_data.get("competitor_count", 0) > 0:
            scores.append(0.8)
        else:
            scores.append(0.4)
        
        # Timing confidence
        if timing_score > 1.0:
            scores.append(0.8)
        else:
            scores.append(0.3)
        
        return round(sum(scores) / len(scores), 2)


# Keep the old class name for compatibility
MarketAnalysisProvider = PerplexityMarketAnalysis
