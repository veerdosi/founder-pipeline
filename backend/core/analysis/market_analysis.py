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
            
            # Handle truncated JSON by finding the last complete object
            if not content.endswith('}') and not content.endswith(']'):
                # Find the last complete object
                brace_count = 0
                last_valid_pos = -1
                for i, char in enumerate(content):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            last_valid_pos = i + 1
                
                if last_valid_pos > 0:
                    content = content[:last_valid_pos]
                    logger.warning(f"Truncated JSON detected, using content up to position {last_valid_pos}")
            
            # Try to parse JSON
            import json
            return json.loads(content)
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}. Content: '{content[:200]}...'")
            # Try to extract key-value pairs manually if JSON parsing fails
            return self._extract_key_values_fallback(content)
        except Exception as e:
            logger.error(f"An unexpected error occurred during JSON parsing: {e}")
            return {}
    
    def _extract_key_values_fallback(self, content: str) -> dict:
        """Fallback method to extract key-value pairs from malformed JSON."""
        import re
        result = {}
        
        # Extract string values
        string_pattern = r'"([^"]+)":\s*"([^"]*)"'
        for match in re.finditer(string_pattern, content):
            key, value = match.groups()
            result[key] = value
        
        # Extract numeric values  
        number_pattern = r'"([^"]+)":\s*([0-9.]+)'
        for match in re.finditer(number_pattern, content):
            key, value = match.groups()
            try:
                result[key] = float(value) if '.' in value else int(value)
            except ValueError:
                pass
        
        # Extract array values (simplified)
        array_pattern = r'"([^"]+)":\s*\[([^\]]*)\]'
        for match in re.finditer(array_pattern, content):
            key, array_content = match.groups()
            # Simple array parsing for strings
            items = []
            for item in array_content.split(','):
                item = item.strip().strip('"')
                if item:
                    items.append(item)
            result[key] = items
        
        logger.warning(f"Used fallback parsing, extracted {len(result)} key-value pairs")
        return result
    
    def _ensure_string(self, value) -> str:
        """Ensure a value is converted to a string, handling lists and dicts."""
        if value is None:
            return ""
        elif isinstance(value, str):
            return value
        elif isinstance(value, list):
            # Join list items with bullet points for readability
            return "• " + "\n• ".join(str(item) for item in value if item)
        elif isinstance(value, dict):
            # Convert dict to readable text
            parts = []
            for k, v in value.items():
                if v:
                    parts.append(f"{k.replace('_', ' ').title()}: {str(v)}")
            return "\n".join(parts)
        else:
            return str(value)
    
    def _ensure_list(self, value) -> list:
        """Ensure a value is converted to a list."""
        if value is None:
            return []
        elif isinstance(value, list):
            return value
        elif isinstance(value, str):
            # Try to split on common delimiters
            if '•' in value:
                return [item.strip() for item in value.split('•') if item.strip()]
            elif '\n' in value:
                return [item.strip() for item in value.split('\n') if item.strip()]
            else:
                return [value]
        else:
            return [str(value)]

    async def analyze_market(
        self, 
        sector: str, 
        year: int,
        region: Optional[str] = None,
        company_name: Optional[str] = None
    ) -> MarketMetrics:
        """Analyze market metrics for a sector using Perplexity."""
        logger.info(f"📊 Analyzing market for {sector} ({year}) using Perplexity")
        
        start_time = time.time()
        
        try:
            # Run comprehensive analysis tasks with Perplexity
            tasks = [
                self._get_market_size_and_cagr_perplexity(sector, year, company_name),
                self._get_timing_analysis_perplexity(sector, year, company_name),
                self._get_regional_sentiment_perplexity(sector, year, "United States", company_name),
                self._get_regional_sentiment_perplexity(sector, year, "Asia", company_name),
                self._get_competitor_analysis_perplexity(sector, year, company_name),
                self._get_comprehensive_market_overview(sector, year, company_name),
                self._get_investment_and_regulatory_analysis(sector, year, company_name),
                self._get_technology_and_trends_analysis(sector, year, company_name),
                self._get_risks_and_recommendations(sector, year, company_name)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Safely extract numerical results
            market_data = results[0] if isinstance(results[0], dict) else {}
            timing_data = results[1] if isinstance(results[1], dict) else {}
            us_sentiment = results[2] if isinstance(results[2], (int, float)) else 0.0
            asia_sentiment = results[3] if isinstance(results[3], (int, float)) else 0.0
            competitor_data = results[4] if isinstance(results[4], dict) else {}
            
            # Extract comprehensive text analysis
            overview_data = results[5] if isinstance(results[5], dict) else {}
            investment_data = results[6] if isinstance(results[6], dict) else {}
            tech_data = results[7] if isinstance(results[7], dict) else {}
            risk_data = results[8] if isinstance(results[8], dict) else {}

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
                # Numerical metrics
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
                execution_time=execution_time,
                
                # Comprehensive text analysis (ensure strings)
                market_overview=self._ensure_string(overview_data.get("market_overview")),
                market_size_analysis=self._ensure_string(overview_data.get("market_size_analysis")),
                growth_drivers=self._ensure_string(overview_data.get("growth_drivers")),
                timing_analysis=self._ensure_string(timing_data.get("timing_analysis")),
                regional_analysis=self._ensure_string(overview_data.get("regional_analysis")),
                competitive_landscape=self._ensure_string(competitor_data.get("competitive_landscape")),
                investment_climate=self._ensure_string(investment_data.get("investment_climate")),
                regulatory_environment=self._ensure_string(investment_data.get("regulatory_environment")),
                technology_trends=self._ensure_string(tech_data.get("technology_trends")),
                consumer_adoption=self._ensure_string(tech_data.get("consumer_adoption")),
                supply_chain_analysis=self._ensure_string(tech_data.get("supply_chain_analysis")),
                risk_assessment=self._ensure_string(risk_data.get("risk_assessment")),
                strategic_recommendations=self._ensure_string(risk_data.get("strategic_recommendations")),
                
                # Enhanced structured lists (ensure lists)
                key_trends=self._ensure_list(tech_data.get("key_trends")),
                major_players=self._ensure_list(competitor_data.get("major_players")),
                barriers_to_entry=self._ensure_list(competitor_data.get("barriers_to_entry")),
                opportunities=self._ensure_list(risk_data.get("opportunities")),
                threats=self._ensure_list(risk_data.get("threats")),
                regulatory_changes=self._ensure_list(investment_data.get("regulatory_changes")),
                emerging_technologies=self._ensure_list(tech_data.get("emerging_technologies"))
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
                execution_time=time.time() - start_time,
                market_overview=f"Analysis failed for {sector} market.",
                market_size_analysis="Unable to determine market size due to analysis error.",
                growth_drivers="Growth drivers could not be analyzed.",
                timing_analysis="Market timing analysis unavailable.",
                regional_analysis="Regional analysis unavailable.",
                competitive_landscape="Competitive analysis unavailable.",
                investment_climate="Investment climate analysis unavailable.",
                regulatory_environment="Regulatory analysis unavailable.",
                technology_trends="Technology trend analysis unavailable.",
                consumer_adoption="Consumer adoption analysis unavailable.",
                supply_chain_analysis="Supply chain analysis unavailable.",
                risk_assessment="Risk assessment unavailable.",
                strategic_recommendations="Strategic recommendations unavailable due to analysis failure."
            )
    
    async def _get_market_size_and_cagr_perplexity(self, sector: str, year: int, company_name: Optional[str] = None) -> Optional[Dict]:
        """Get market size and CAGR using Perplexity, returning structured JSON."""
        company_context = f" for companies like {company_name}" if company_name else ""
        prompt = f"""
        Research the {sector} market globally for {year}, focusing on the market opportunity{company_context}. Provide the following data in a clear JSON format:
        
        - Market size in billions USD (total addressable market)
        - Expected CAGR (Compound Annual Growth Rate) as a percentage
        
        Return only a valid JSON object with the keys "market_size" and "cagr".
        Example: {{"market_size": 15.5, "cagr": 12.5}}
        """
        
        try:
            await self.rate_limiter.acquire()
            
            response = await self.perplexity_client.chat.completions.create(
                model="sonar",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            
            content = response.choices[0].message.content
            return self._safe_json_parse(content)
            
        except Exception as e:
            logger.error(f"Error getting market size/CAGR from Perplexity: {e}")
            return None
    
    async def _get_timing_analysis_perplexity(self, sector: str, year: int, company_name: Optional[str] = None) -> Optional[Dict]:
        """Get market timing analysis using Perplexity, returning structured JSON."""
        company_context = f" specifically for {company_name} and similar companies" if company_name else " for early-stage companies"
        prompt = f"""
        Analyze the market timing for entering the {sector} market in {year}{company_context}.
        
        You MUST respond with ONLY a valid JSON object in this exact format:
        {{
            "timing_score": 4.2,
            "timing_analysis": "Detailed explanation of why this is a good/bad time to enter the market, considering market conditions, competition, and opportunity"
        }}
        
        Do not include any text before or after the JSON. Timing_score must be a number between 1-5. Timing_analysis must be a string.
        """
        
        try:
            await self.rate_limiter.acquire()
            
            response = await self.perplexity_client.chat.completions.create(
                model="sonar",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            
            content = response.choices[0].message.content
            return self._safe_json_parse(content)
            
        except Exception as e:
            logger.error(f"Error getting timing analysis from Perplexity: {e}")
            return None
    
    async def _get_regional_sentiment_perplexity(self, sector: str, year: int, region: str, company_name: Optional[str] = None) -> float:
        """Get regional market sentiment using Perplexity, returning a score."""
        company_context = f" for companies like {company_name}" if company_name else " for early-stage companies"
        prompt = f"""
        Analyze the {sector} market sentiment in {region} for {year}{company_context}.
        Rate the sentiment on a scale of 1-5 (5 being very positive).
        
        Return only a valid JSON object with the key "sentiment_score".
        Example: {{"sentiment_score": 4.5}}
        """
        
        try:
            await self.rate_limiter.acquire()
            
            response = await self.perplexity_client.chat.completions.create(
                model="sonar",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            
            content = response.choices[0].message.content
            result = self._safe_json_parse(content)
            return float(result.get("sentiment_score", 0.0) or 0.0)

        except Exception as e:
            logger.error(f"Error getting regional sentiment from Perplexity for {region}: {e}")
            return 0.0
    
    async def _get_competitor_analysis_perplexity(self, sector: str, year: int, company_name: Optional[str] = None) -> Optional[Dict]:
        """Get competitor analysis using Perplexity, returning structured JSON."""
        company_context = f" relevant to {company_name}" if company_name else ""
        prompt = f"""
        Analyze the competitive landscape in the {sector} market for {year}{company_context}.
        
        You MUST respond with ONLY a valid JSON object in this exact format:
        {{
            "competitor_count": 25,
            "total_funding": 2.1,
            "momentum_score": 4.0,
            "competitive_landscape": "Detailed analysis of the competitive environment and how companies compete in this space",
            "major_players": ["Company 1", "Company 2", "Company 3", "Company 4", "Company 5"],
            "barriers_to_entry": ["Barrier 1", "Barrier 2", "Barrier 3", "Barrier 4"]
        }}
        
        Do not include any text before or after the JSON. Competitor_count must be an integer, total_funding and momentum_score must be numbers, competitive_landscape must be a string, major_players and barriers_to_entry must be arrays of strings.
        """
        
        try:
            await self.rate_limiter.acquire()
            
            response = await self.perplexity_client.chat.completions.create(
                model="sonar",
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
    
    async def _get_comprehensive_market_overview(self, sector: str, year: int, company_name: Optional[str] = None) -> Optional[Dict]:
        """Get comprehensive market overview and size analysis."""
        company_context = f" with particular focus on opportunities for {company_name}" if company_name else ""
        prompt = f"""
        Provide a comprehensive market analysis for the {sector} industry in {year}{company_context}.
        
        You MUST respond with ONLY a valid JSON object in this exact format:
        {{
            "market_overview": "2-3 paragraph detailed overview of the current state of the {sector} market and opportunity landscape",
            "market_size_analysis": "Detailed explanation of market size, segments, and how companies can capture value",
            "growth_drivers": "Key factors driving market growth and what this means for new entrants", 
            "regional_analysis": "Regional market dynamics and geographic opportunities for companies"
        }}
        
        Do not include any text before or after the JSON. Each value must be a single string.
        """
        
        try:
            await self.rate_limiter.acquire()
            
            response = await self.perplexity_client.chat.completions.create(
                model="sonar",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            
            content = response.choices[0].message.content
            return self._safe_json_parse(content)
            
        except Exception as e:
            logger.error(f"Error getting market overview from Perplexity: {e}")
            return None

    async def _get_investment_and_regulatory_analysis(self, sector: str, year: int, company_name: Optional[str] = None) -> Optional[Dict]:
        """Get investment climate and regulatory environment analysis."""
        company_context = f" particularly for companies like {company_name}" if company_name else ""
        prompt = f"""
        Analyze the investment and regulatory landscape for {sector} companies in {year}{company_context}.
        
        You MUST respond with ONLY a valid JSON object in this exact format:
        {{
            "investment_climate": "Current funding trends, investor sentiment, and what this means for companies seeking capital",
            "regulatory_environment": "Key regulations, compliance requirements, and how companies should navigate regulatory challenges",
            "regulatory_changes": ["Change 1", "Change 2", "Change 3"]
        }}
        
        Do not include any text before or after the JSON. Investment_climate and regulatory_environment must be strings. Regulatory_changes must be an array of strings.
        """
        
        try:
            await self.rate_limiter.acquire()
            
            response = await self.perplexity_client.chat.completions.create(
                model="sonar",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            
            content = response.choices[0].message.content
            return self._safe_json_parse(content)
            
        except Exception as e:
            logger.error(f"Error getting investment/regulatory analysis from Perplexity: {e}")
            return None

    async def _get_technology_and_trends_analysis(self, sector: str, year: int, company_name: Optional[str] = None) -> Optional[Dict]:
        """Get technology trends and consumer adoption analysis."""
        company_context = f" with implications for companies like {company_name}" if company_name else ""
        prompt = f"""
        Analyze technology trends and market dynamics for the {sector} industry in {year}{company_context}.
        
        You MUST respond with ONLY a valid JSON object in this exact format:
        {{
            "technology_trends": "Current and emerging technology trends and how companies can leverage them",
            "consumer_adoption": "Consumer behavior, adoption patterns, and what this means for companies",
            "supply_chain_analysis": "Supply chain dynamics, dependencies, and how companies should manage risks",
            "key_trends": ["Trend 1", "Trend 2", "Trend 3", "Trend 4", "Trend 5"],
            "emerging_technologies": ["Tech 1", "Tech 2", "Tech 3", "Tech 4"]
        }}
        
        Do not include any text before or after the JSON. Technology_trends, consumer_adoption, and supply_chain_analysis must be strings. Key_trends and emerging_technologies must be arrays of strings.
        """
        
        try:
            await self.rate_limiter.acquire()
            
            response = await self.perplexity_client.chat.completions.create(
                model="sonar",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            
            content = response.choices[0].message.content
            return self._safe_json_parse(content)
            
        except Exception as e:
            logger.error(f"Error getting technology/trends analysis from Perplexity: {e}")
            return None

    async def _get_risks_and_recommendations(self, sector: str, year: int, company_name: Optional[str] = None) -> Optional[Dict]:
        """Get risk assessment and strategic recommendations."""
        company_context = f" particularly for {company_name}" if company_name else ""
        prompt = f"""
        Provide risk assessment and strategic recommendations for {sector} companies in {year}{company_context}.
        
        You MUST respond with ONLY a valid JSON object in this exact format:
        {{
            "risk_assessment": "Comprehensive analysis of market risks, challenges, and how companies should mitigate them",
            "strategic_recommendations": "Specific actionable recommendations for companies entering or scaling in this market",
            "opportunities": ["Opportunity 1", "Opportunity 2", "Opportunity 3", "Opportunity 4"],
            "threats": ["Threat 1", "Threat 2", "Threat 3", "Threat 4"]
        }}
        
        Do not include any text before or after the JSON. Risk_assessment and strategic_recommendations must be strings. Opportunities and threats must be arrays of strings.
        """
        
        try:
            await self.rate_limiter.acquire()
            
            response = await self.perplexity_client.chat.completions.create(
                model="sonar",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            
            content = response.choices[0].message.content
            return self._safe_json_parse(content)
            
        except Exception as e:
            logger.error(f"Error getting risks/recommendations from Perplexity: {e}")
            return None

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
