"""Market analysis service using Perplexity AI for enhanced reasoning."""

import asyncio
import re
import statistics
import time
from datetime import datetime
from typing import Dict, List, Optional

import aiohttp
from openai import AsyncOpenAI

from ..core import (
    MarketAnalysisService,
    get_logger,
    RateLimiter,
    clean_text,
    settings
)
from ..models import MarketMetrics, MarketStage


logger = get_logger(__name__)


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
    
    def _safe_json_parse(self, content: str, fallback_type: str = "generic") -> dict:
        """Safely parse JSON with fallback handling."""
        if not content or not content.strip():
            logger.warning("Empty content for JSON parsing")
            return self._get_fallback_data(fallback_type)
            
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
            
            # Try to extract numbers with regex as fallback
            import re
            numbers = re.findall(r'\d+\.?\d*', content)
            if numbers:
                logger.info(f"Extracted numbers as fallback: {numbers}")
                return self._create_fallback_from_numbers(numbers, fallback_type)
            
            return self._get_fallback_data(fallback_type)
        except Exception as e:
            logger.error(f"Unexpected error in JSON parsing: {e}")
            return self._get_fallback_data(fallback_type)
    
    def _get_fallback_data(self, fallback_type: str) -> dict:
        """Get fallback data based on type."""
        fallbacks = {
            "market_size": {"market_size": 0, "cagr": 0},
            "timing": {"timing_score": 3},
            "sentiment": {"us_sentiment": 3, "sea_sentiment": 3},
            "competitor": {"competitor_count": 10, "total_funding": 1.0, "momentum_score": 3.0},
            "generic": {}
        }
        return fallbacks.get(fallback_type, {})
    
    def _create_fallback_from_numbers(self, numbers: list, fallback_type: str) -> dict:
        """Create fallback data from extracted numbers."""
        if fallback_type == "market_size":
            return {
                "market_size": float(numbers[0]) if numbers else 0,
                "cagr": float(numbers[1]) if len(numbers) > 1 else 0
            }
        elif fallback_type == "timing":
            return {"timing_score": min(5, max(1, float(numbers[0]))) if numbers else 3}
        elif fallback_type == "sentiment":
            return {
                "us_sentiment": min(5, max(1, float(numbers[0]))) if numbers else 3,
                "sea_sentiment": min(5, max(1, float(numbers[1]))) if len(numbers) > 1 else 3
            }
        elif fallback_type == "competitor":
            return {
                "competitor_count": int(float(numbers[0])) if numbers else 10,
                "total_funding": float(numbers[1]) if len(numbers) > 1 else 1.0,
                "momentum_score": min(5, max(1, float(numbers[2]))) if len(numbers) > 2 else 3.0
            }
        return {}
    
    def _safe_float(self, value) -> float:
        """Safely convert value to float."""
        try:
            if value is None:
                return 0.0
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                # Extract first number from string
                import re
                numbers = re.findall(r'\d+\.?\d*', value)
                return float(numbers[0]) if numbers else 0.0
            if isinstance(value, dict):
                # If it's a dict, try to get a numeric value from it
                for key in ['value', 'number', 'amount', 'size', 'cagr', 'score']:
                    if key in value:
                        return self._safe_float(value[key])
                return 0.0
            return 0.0
        except (ValueError, TypeError, AttributeError):
            return 0.0
    
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
            
            # Extract results safely
            market_data = results[0] if not isinstance(results[0], Exception) else {"market_size": 0, "cagr": 0}
            timing_data = results[1] if not isinstance(results[1], Exception) else {"timing_score": 3.0}
            us_sentiment = results[2] if not isinstance(results[2], Exception) else 3.0
            asia_sentiment = results[3] if not isinstance(results[3], Exception) else 3.0
            competitor_data = results[4] if not isinstance(results[4], Exception) else {"competitor_count": 10, "total_funding": 1.0, "momentum_score": 3.0}
            
            # Determine market stage
            market_stage = self._determine_market_stage(
                market_data.get("cagr", 0),
                competitor_data.get("competitor_count", 10),
                timing_data.get("timing_score", 3.0)
            )
            
            # Calculate confidence score
            confidence = self._calculate_confidence_score(
                market_data, competitor_data, timing_data.get("timing_score", 3.0)
            )
            
            execution_time = time.time() - start_time
            
            metrics = MarketMetrics(
                market_size_billion=market_data.get("market_size", 0),
                cagr_percent=market_data.get("cagr", 0),
                timing_score=timing_data.get("timing_score", 3.0),
                us_sentiment=us_sentiment,
                sea_sentiment=asia_sentiment,  # Using Asia as proxy for SEA
                competitor_count=competitor_data.get("competitor_count", 10),
                total_funding_billion=competitor_data.get("total_funding", 1.0),
                momentum_score=competitor_data.get("momentum_score", 3.0),
                market_stage=market_stage,
                confidence_score=confidence,
                analysis_date=datetime.utcnow(),
                execution_time=execution_time
            )
            
            logger.info(f"✅ Perplexity market analysis complete for {sector}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing market for {sector}: {e}")
            # Return default metrics
            return MarketMetrics(
                market_size_billion=0,
                cagr_percent=0,
                timing_score=3.0,
                us_sentiment=3.0,
                sea_sentiment=3.0,
                competitor_count=10,
                total_funding_billion=1.0,
                momentum_score=3.0,
                market_stage=MarketStage.UNKNOWN,
                confidence_score=0.1,
                analysis_date=datetime.utcnow(),
                execution_time=time.time() - start_time
            )
    
    async def _get_market_size_and_cagr_perplexity(self, sector: str, year: int) -> Dict:
        """Get market size and CAGR using Perplexity's research capabilities."""
        prompt = f"""
        Research the {sector} market globally for {year}. Provide specific data on:
        
        1. Current market size in billions USD
        2. Expected CAGR (Compound Annual Growth Rate) percentage
        3. Key growth drivers
        4. Market forecasts for the next 3-5 years
        
        Focus on early-stage companies and emerging opportunities in this sector.
        Please provide specific numerical values where available.
        """
        
        try:
            await self.rate_limiter.acquire()
            
            response = await self.perplexity_client.chat.completions.create(
                model="sonar",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            
            # Extract metrics using OpenAI for structured parsing
            extraction_prompt = f"""
            Extract market metrics from this research content. Return valid JSON only:
            
            {content}
            
            {{
                "market_size": "market size in billions USD as number or null",
                "cagr": "CAGR percentage as number or null"
            }}
            """
            
            extraction_response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": extraction_prompt}],
                temperature=0.1
            )
            
            result = self._safe_json_parse(extraction_response.choices[0].message.content, "market_size")
            
            return {
                "market_size": self._safe_float(result.get("market_size", 0)),
                "cagr": self._safe_float(result.get("cagr", 0))
            }
            
        except Exception as e:
            logger.error(f"Error getting market size/CAGR from Perplexity: {e}")
            return {"market_size": 0, "cagr": 0}
    
    async def _get_timing_analysis_perplexity(self, sector: str, year: int) -> Dict:
        """Get market timing analysis using Perplexity."""
        prompt = f"""
        Analyze the market timing for early-stage {sector} companies in {year}. Consider:
        
        1. Technology maturity and adoption readiness
        2. Market barriers and entry opportunities
        3. Competitive landscape for new entrants
        4. Investment climate and funding availability
        5. Regulatory environment
        
        Rate the market timing on a scale of 1-5 (5 being excellent timing for new companies).
        Explain your reasoning.
        """
        
        try:
            await self.rate_limiter.acquire()
            
            response = await self.perplexity_client.chat.completions.create(
                model="sonar",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            
            # Extract timing score
            extraction_prompt = f"""
            Extract the timing score from this analysis. Return valid JSON only:
            
            {content}
            
            {{
                "timing_score": "timing score from 1-5 as number"
            }}
            """
            
            extraction_response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": extraction_prompt}],
                temperature=0.1
            )
            
            result = self._safe_json_parse(extraction_response.choices[0].message.content, "timing")
            
            return {
                "timing_score": self._safe_float(result.get("timing_score", 3.0))
            }
            
        except Exception as e:
            logger.error(f"Error getting timing analysis from Perplexity: {e}")
            return {"timing_score": 3.0}
    
    async def _get_regional_sentiment_perplexity(self, sector: str, year: int, region: str) -> float:
        """Get regional market sentiment using Perplexity."""
        prompt = f"""
        Analyze the {sector} market sentiment and opportunities in {region} for {year}. Consider:
        
        1. Government support and policies
        2. Investment climate and VC activity
        3. Talent availability and ecosystem maturity
        4. Market adoption rates
        5. Regulatory environment
        6. Success stories and market traction
        
        Rate the overall sentiment for early-stage companies on a scale of 1-5 
        (5 being very positive/favorable conditions).
        """
        
        try:
            await self.rate_limiter.acquire()
            
            response = await self.perplexity_client.chat.completions.create(
                model="sonar",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            
            # Extract sentiment score
            extraction_prompt = f"""
            Extract the sentiment score from this analysis. Return valid JSON only:
            
            {content}
            
            {{
                "sentiment_score": "sentiment score from 1-5 as number"
            }}
            """
            
            extraction_response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": extraction_prompt}],
                temperature=0.1
            )
            
            result = self._safe_json_parse(extraction_response.choices[0].message.content, "sentiment")
            
            return self._safe_float(result.get("sentiment_score", 3.0))
            
        except Exception as e:
            logger.error(f"Error getting regional sentiment from Perplexity: {e}")
            return 3.0
    
    async def _get_competitor_analysis_perplexity(self, sector: str, year: int) -> Dict:
        """Get competitor analysis using Perplexity."""
        prompt = f"""
        Analyze the competitive landscape for early-stage {sector} companies in {year}:
        
        1. Number of major competitors/players in the market
        2. Total venture funding raised in this sector
        3. Key market leaders and their positions
        4. Barriers to entry for new companies
        5. Market consolidation trends
        6. Opportunities for disruption
        
        Provide specific numbers where possible (competitor count, funding amounts).
        Rate market momentum on 1-5 scale (5 being very high momentum/activity).
        """
        
        try:
            await self.rate_limiter.acquire()
            
            response = await self.perplexity_client.chat.completions.create(
                model="sonar",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            
            # Extract competitor metrics
            extraction_prompt = f"""
            Extract competitor metrics from this analysis. Return valid JSON only:
            
            {content}
            
            {{
                "competitor_count": "number of major competitors as integer",
                "total_funding": "total funding in billions USD as number",
                "momentum_score": "momentum score from 1-5 as number"
            }}
            """
            
            extraction_response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": extraction_prompt}],
                temperature=0.1
            )
            
            result = self._safe_json_parse(extraction_response.choices[0].message.content, "competitor")
            
            return {
                "competitor_count": int(self._safe_float(result.get("competitor_count", 10))),
                "total_funding": self._safe_float(result.get("total_funding", 1.0)),
                "momentum_score": self._safe_float(result.get("momentum_score", 3.0))
            }
            
        except Exception as e:
            logger.error(f"Error getting competitor analysis from Perplexity: {e}")
            return {"competitor_count": 10, "total_funding": 1.0, "momentum_score": 3.0}
    
    def _determine_market_stage(
        self, 
        cagr: float, 
        competitor_count: int, 
        timing_score: float
    ) -> MarketStage:
        """Determine market maturity stage."""
        if cagr > 30 and competitor_count < 20 and timing_score >= 4.0:
            return MarketStage.EMERGING
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
