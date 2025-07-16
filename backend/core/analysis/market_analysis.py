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
            
            # Remove any leading/trailing text that isn't JSON
            # Find the first '{' and last '}'
            start_idx = content.find('{')
            end_idx = content.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                content = content[start_idx:end_idx + 1]
            
            # Handle truncated JSON by finding the last complete object
            if not content.endswith('}') and not content.endswith(']'):
                # Find the last complete object by properly tracking braces
                brace_count = 0
                last_valid_pos = -1
                in_string = False
                escape_next = False
                
                for i, char in enumerate(content):
                    if escape_next:
                        escape_next = False
                        continue
                    
                    if char == '\\':
                        escape_next = True
                        continue
                    
                    if char == '"' and not escape_next:
                        in_string = not in_string
                        continue
                    
                    if not in_string:
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
        
        # Extract string values (handling escaped quotes and multiline strings)
        string_pattern = r'"([^"]+)":\s*"((?:[^"\\]|\\.)*)"'
        for match in re.finditer(string_pattern, content, re.DOTALL):
            key, value = match.groups()
            # Unescape common JSON escape sequences
            value = value.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
            result[key] = value
        
        # Extract numeric values  
        number_pattern = r'"([^"]+)":\s*([0-9.]+)(?:\s*[,}])'
        for match in re.finditer(number_pattern, content):
            key, value = match.groups()
            try:
                result[key] = float(value) if '.' in value else int(value)
            except ValueError:
                pass
        
        # Extract array values (improved handling)
        array_pattern = r'"([^"]+)":\s*\[((?:[^\]\\]|\\.)*)\]'
        for match in re.finditer(array_pattern, content, re.DOTALL):
            key, array_content = match.groups()
            # Parse array items more carefully
            items = []
            # Split on commas but handle quoted strings
            current_item = ""
            in_quotes = False
            for char in array_content:
                if char == '"' and (not current_item or current_item[-1] != '\\'):
                    in_quotes = not in_quotes
                elif char == ',' and not in_quotes:
                    item = current_item.strip().strip('"')
                    if item:
                        items.append(item)
                    current_item = ""
                    continue
                current_item += char
            
            # Don't forget the last item
            if current_item.strip():
                item = current_item.strip().strip('"')
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
        foundation_year_instruction = f"""
        IMPORTANT: First, find the exact founding year of {company_name}. Use that founding year as the base year for your market analysis instead of {year}.
        
        Step 1: Research when {company_name} was founded
        Step 2: Use that founding year for all market metrics and analysis
        """ if company_name else ""
        
        prompt = f"""
        {foundation_year_instruction}
        
        Research the {sector} market globally, focusing on the market opportunity{company_context}. Provide the following data in a clear JSON format:
        
        - Market size in billions USD (total addressable market) for the founding year
        - Expected CAGR (Compound Annual Growth Rate) as a percentage from the founding year onwards
        
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
        foundation_year_instruction = f"""
        IMPORTANT: First, find the exact founding year of {company_name}. Use that founding year as the base year for your market timing analysis instead of {year}.
        
        Step 1: Research when {company_name} was founded
        Step 2: Analyze the market timing for entering the {sector} market in that founding year
        """ if company_name else ""
        
        prompt = f"""
        {foundation_year_instruction}
        
        Analyze the market timing for entering the {sector} market{company_context}.
        
        You MUST respond with ONLY a valid JSON object in this exact format:
        {{
            "timing_score": 4.2,
            "timing_analysis": "Detailed explanation of why this was a good/bad time to enter the market in the founding year, considering market conditions, competition, and opportunity at that time"
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
        foundation_year_instruction = f"""
        IMPORTANT: First, find the exact founding year of {company_name}. Use that founding year for your regional sentiment analysis instead of {year}.
        
        Step 1: Research when {company_name} was founded
        Step 2: Analyze the {sector} market sentiment in {region} for that founding year
        """ if company_name else ""
        
        prompt = f"""
        {foundation_year_instruction}
        
        Analyze the {sector} market sentiment in {region}{company_context}.
        Rate the sentiment on a scale of 1-5 (5 being very positive) based on the market conditions in the founding year.
        
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
        foundation_year_instruction = f"""
        IMPORTANT: First, find the exact founding year of {company_name}. Use that founding year for your competitive analysis instead of {year}.
        
        Step 1: Research when {company_name} was founded
        Step 2: Analyze the competitive landscape in the {sector} market for that founding year
        """ if company_name else ""
        
        prompt = f"""
        {foundation_year_instruction}
        
        Analyze the competitive landscape in the {sector} market{company_context}.
        
        You MUST respond with ONLY a valid JSON object in this exact format:
        {{
            "competitor_count": 25,
            "total_funding": 2.1,
            "momentum_score": 4.0,
            "competitive_landscape": "Detailed analysis of the competitive environment and how companies competed in this space during the founding year",
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
        foundation_year_instruction = f"""
        IMPORTANT: First, find the exact founding year of {company_name}. Use that founding year for your market overview analysis instead of {year}.
        
        Step 1: Research when {company_name} was founded
        Step 2: Provide a comprehensive market analysis for the {sector} industry in that founding year
        """ if company_name else ""
        
        prompt = f"""
        {foundation_year_instruction}
        
        Provide a comprehensive market analysis for the {sector} industry{company_context}.
        
        You MUST respond with ONLY a valid JSON object in this exact format:
        {{
            "market_overview": "2-3 paragraph detailed overview of the state of the {sector} market and opportunity landscape during the founding year",
            "market_size_analysis": "Detailed explanation of market size, segments, and how companies could capture value during the founding year",
            "growth_drivers": "Key factors driving market growth during the founding year and what this meant for new entrants", 
            "regional_analysis": "Regional market dynamics and geographic opportunities for companies during the founding year"
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
        foundation_year_instruction = f"""
        IMPORTANT: First, find the exact founding year of {company_name}. Use that founding year for your investment and regulatory analysis instead of {year}.
        
        Step 1: Research when {company_name} was founded
        Step 2: Analyze the investment and regulatory landscape for {sector} companies in that founding year
        """ if company_name else ""
        
        prompt = f"""
        {foundation_year_instruction}
        
        Analyze the investment and regulatory landscape for {sector} companies{company_context}.
        
        You MUST respond with ONLY a valid JSON object in this exact format:
        {{
            "investment_climate": "Funding trends, investor sentiment, and what this meant for companies seeking capital during the founding year",
            "regulatory_environment": "Key regulations, compliance requirements, and how companies had to navigate regulatory challenges during the founding year",
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
        foundation_year_instruction = f"""
        IMPORTANT: First, find the exact founding year of {company_name}. Use that founding year for your technology and trends analysis instead of {year}.
        
        Step 1: Research when {company_name} was founded
        Step 2: Analyze technology trends and market dynamics for the {sector} industry in that founding year
        """ if company_name else ""
        
        prompt = f"""
        {foundation_year_instruction}
        
        Analyze technology trends and market dynamics for the {sector} industry{company_context}.
        
        You MUST respond with ONLY a valid JSON object in this exact format:
        {{
            "technology_trends": "Technology trends during the founding year and how companies could leverage them",
            "consumer_adoption": "Consumer behavior, adoption patterns, and what this meant for companies during the founding year",
            "supply_chain_analysis": "Supply chain dynamics, dependencies, and how companies had to manage risks during the founding year",
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
        foundation_year_instruction = f"""
        IMPORTANT: First, find the exact founding year of {company_name}. Use that founding year for your risk assessment and recommendations instead of {year}.
        
        Step 1: Research when {company_name} was founded
        Step 2: Provide risk assessment and strategic recommendations for {sector} companies in that founding year
        """ if company_name else ""
        
        prompt = f"""
        {foundation_year_instruction}
        
        Provide risk assessment and strategic recommendations for {sector} companies{company_context}.
        
        You MUST respond with ONLY a valid JSON object in this exact format:
        {{
            "risk_assessment": "Comprehensive analysis of market risks, challenges, and how companies had to mitigate them during the founding year",
            "strategic_recommendations": "Specific actionable recommendations for companies entering or scaling in this market during the founding year",
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
