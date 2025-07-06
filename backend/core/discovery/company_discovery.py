"""Comprehensive company discovery service implementing 20+ source monitoring."""

import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import json
import aiohttp
import re
from dataclasses import dataclass

from exa_py import Exa
from openai import AsyncOpenAI

from .. import (
    CompanyDiscoveryService, 
    settings
)
from ...models import Company, FundingStage
from ...utils.data_processing import clean_text

import logging
import asyncio
import time

logger = logging.getLogger(__name__)

class RateLimiter:
    """Simple rate limiter for API requests."""
    def __init__(self, max_requests: int, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
    
    async def acquire(self):
        """Acquire permission to make a request."""
        now = time.time()
        self.requests = [req_time for req_time in self.requests 
                        if now - req_time < self.time_window]
        
        if len(self.requests) >= self.max_requests:
            oldest_request = min(self.requests)
            wait_time = self.time_window - (now - oldest_request)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                return await self.acquire()
        
        self.requests.append(now)


logger = logging.getLogger(__name__)


class ExaCompanyDiscovery(CompanyDiscoveryService):
    """Enhanced company discovery using comprehensive 20+ source monitoring."""
    
    def __init__(self):
        self.exa = Exa(settings.exa_api_key)
        self.openai = AsyncOpenAI(api_key=settings.openai_api_key)
        self.rate_limiter = RateLimiter(
            max_requests=settings.requests_per_minute,
            time_window=60
        )
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def discover_companies(
        self,
        limit: int = 50,
        categories: Optional[List[str]] = None,
        regions: Optional[List[str]] = None,
        sources: Optional[List[str]] = None
    ) -> List[Company]:
        """Discover companies using comprehensive 20+ source monitoring."""
        logger.info(f"🔍 Starting comprehensive discovery from 20+ sources...")
        
        companies = await self.find_companies(
            limit=limit,
            categories=categories,
            regions=regions
        )
        
        logger.info(f"✅ Comprehensive discovery complete: {len(companies)} companies")
        return companies
    
    async def find_companies(
        self, 
        limit: int = 50,
        categories: Optional[List[str]] = None,
        regions: Optional[List[str]] = None,
        founded_year: Optional[int] = None
    ) -> List[Company]:
        """Find AI companies using Exa search."""
        year_context = f" (founded in {founded_year})" if founded_year else ""
        logger.info(f"🔍 Finding {limit} AI companies with Exa{year_context}...")
        
        categories = categories or settings.ai_categories
        
        # Generate search queries with optional year filtering
        queries = self._generate_search_queries(categories, regions, founded_year)
        
        all_companies = []
        # Increase results per query to account for deduplication
        results_per_query = max(8, (limit * 2) // len(queries))  # 2x multiplier for dedup buffer
        
        # Execute searches
        for i, query in enumerate(queries):
            logger.info(f"  Query {i+1}/{len(queries)}: {query[:50]}...")
            
            try:
                await self.rate_limiter.acquire()
                
                # Perform search with error handling
                try:
                    result = self.exa.search_and_contents(
                        query,
                        type="neural",
                        use_autoprompt=True,
                        num_results=results_per_query,
                        text={"max_characters": 2000},
                        include_domains=[
                            "techcrunch.com", 
                            "venturebeat.com", 
                            "crunchbase.com",
                            "pitchbook.com", 
                            "bloomberg.com", 
                            "reuters.com",
                            "forbes.com",
                            "wired.com",
                            "eu-startups.com",
                            "tech.eu",
                            "techinasia.com",
                            "startupindia.gov.in",
                            "dealstreetasia.com",
                            "theinformation.com",
                            "sifted.eu",
                            "axios.com",
                            "technode.com",
                            "e27.co",
                            "inc.com",
                            "fastcompany.com",
                            "businessinsider.com",
                            "theblock.co",
                            "venturebeat.com",
                            "producthunt.com"
                        ]
                    )
                except Exception as search_error:
                    logger.error(f"Exa search failed for query '{query}': {search_error}")
                    continue
                
                if not result or not result.results:
                    logger.warning(f"No results for query: {query}")
                    continue
                
                # Extract company data from results with individual error handling
                for item in result.results:
                    try:
                        company = await self._extract_company_data(
                            item.text,
                            item.url,
                            item.title,
                            founded_year
                        )
                        if company:
                            all_companies.append(company)
                    except Exception as extract_error:
                        logger.warning(f"Failed to extract data from {item.url}: {extract_error}")
                        continue
                        
            except Exception as e:
                logger.error(f"Unexpected error with query '{query}': {e}")
                continue
        
        # Deduplicate companies
        unique_companies = self._deduplicate_companies(all_companies)
        logger.info(f"✅ Found {len(unique_companies)} unique AI companies")
        
        return unique_companies[:limit]
    
    def _generate_search_queries(
        self, 
        categories: List[str], 
        regions: Optional[List[str]] = None,
        founded_year: Optional[int] = None
    ) -> List[str]:
        """Generate targeted search queries for AI companies, optionally filtered by founding year."""
        
        if founded_year:
            # Year-specific queries - focus heavily on that specific year
            logger.info(f"🎯 Generating year-specific queries for companies founded in {founded_year}")
            queries = [
                f"AI companies founded in {founded_year} startups",
                f"AI startups launched {founded_year} artificial intelligence",
                f"new AI companies established {founded_year} machine learning",
                f"AI startups founded {founded_year} funding seed",
                f"artificial intelligence companies started {founded_year}",
                f"AI company founded {founded_year} venture capital",
                f"machine learning startups founded {founded_year}",
                f"AI companies that began in {founded_year}",
                f"startups founded {founded_year} artificial intelligence technology",
                f"AI companies established {founded_year} deep learning"
            ]
            
            # Add category-specific year queries
            for category in categories[:3]:
                queries.extend([
                    f"{category} AI companies founded {founded_year}",
                    f"{category} startups established {founded_year}"
                ])
            
            # Add region-specific year queries
            regions = regions or settings.target_regions
            for region in regions[:3]:
                queries.append(f"AI companies founded {founded_year} {region}")
            
            return queries[:12]  # Focus on fewer, more targeted queries
        else:
            # Generate dynamic year range for recent companies (last 5-6 years)
            current_year = datetime.now().year
            recent_years = [str(year) for year in range(current_year - 5, current_year + 1)]
            year_range = " ".join(recent_years)
            
            # Early-stage focused queries
            queries = [
                f"AI startup announced funding seed series A {current_year}",
                f"new AI company launched artificial intelligence {current_year}",
                f"AI startup raised funding machine learning {current_year}",
                f"recently founded AI startups {current_year} {current_year - 1} seed or pre-seed investment",
                f"emerging AI companies venture capital {current_year}",
                "latest AI startups from Y Combinator or Techstars announcing seed funding",
                f"new artificial intelligence companies {current_year} raising a seed round",
                "AI startup news for pre-seed funding announcements",
                "pre-seed AI companies machine learning computer vision",
                f"startup AI companies founded in {year_range} announcing seed funding",
                f"European AI startups seed series A funding {current_year} {current_year - 1}",
                f"Asian AI startups Singapore Israel India machine learning {current_year} {current_year - 1}"
            ]
            
            # Add category-specific early-stage queries
            for category in categories[:4]:  # Increase to 4 categories
                queries.extend([
                    f"early stage {category} startups seed funding {current_year}",
                    f"{category} startup companies pre-seed series A investment"
                ])
            
            # Add region-specific early-stage queries
            regions = regions or settings.target_regions
            for region in regions[:6]:  # More regions for global coverage
                queries.append(f"early stage AI startup {region} seed funding artificial intelligence")
            
            return queries[:15]  # Increase total queries for better coverage
    
    async def _extract_company_data(
        self, 
        content: str, 
        url: str, 
        title: str,
        target_year: Optional[int] = None
    ) -> Optional[Company]:
        """Extract structured company data using GPT-4."""
        year_instruction = ""
        if target_year:
            year_instruction = f"""
IMPORTANT: Pay special attention to the founding year. Only extract companies that were founded in {target_year}. 
If the company was not founded in {target_year}, return null for the founded_year field.
"""
        
        prompt = f"""
Extract company information from this content. Return valid JSON only:

Content: {content[:1500]}
Title: {title}
{year_instruction}
{{
    "name": "exact company name",
    "description": "what the company does in 1-2 sentences",
    "short_description": "brief 1-sentence description",
    "founded_year": "YYYY as integer (be precise about the founding year) or null if not {target_year if target_year else 'found'}",
    "funding_amount_millions": "funding amount in millions USD as number or null",
    "funding_stage": "seed/series-a/series-b/series-c/ipo or null",
    "founders": ["founder names if mentioned"],
    "investors": ["investor names if mentioned"],
    "categories": ["industry categories/tags"],
    "city": "city name only",
    "region": "state/province/region",
    "country": "country name",
    "ai_focus": "specific AI area like NLP, computer vision, robotics, etc",
    "sector": "business sector like fintech, healthcare, retail, etc",
    "website": "company website if mentioned",
    "linkedin_url": "company LinkedIn URL if mentioned (format: https://linkedin.com/company/...)"
}}
"""
        
        try:
            await self.rate_limiter.acquire()
            
            # OpenAI API call with timeout and retry
            try:
                response = await self.openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    timeout=30.0  # 30 second timeout
                )
            except Exception as api_error:
                logger.error(f"OpenAI API call failed for {url}: {api_error}")
                return None
            
            if not response or not response.choices:
                logger.warning(f"Empty response from OpenAI for {url}")
                return None
            
            raw_content = response.choices[0].message.content
            if not raw_content:
                logger.warning(f"Empty content from OpenAI for {url}")
                return None
                
            raw_content = raw_content.strip()
            
            # Remove markdown code blocks if present
            if raw_content.startswith("```json"):
                raw_content = raw_content[7:]
            if raw_content.startswith("```"):
                raw_content = raw_content[3:]
            if raw_content.endswith("```"):
                raw_content = raw_content[:-3]
            
            raw_content = raw_content.strip()
            
            # Parse JSON with error handling
            try:
                result = json.loads(raw_content)
            except json.JSONDecodeError as json_error:
                logger.error(f"JSON parsing failed for {url}: {json_error}")
                logger.debug(f"Raw content: {raw_content[:500]}...")
                return None
            
            # Validate required fields
            if not result.get("name"):
                logger.warning(f"No company name found for {url}")
                return None
            
            # Map funding stage with error handling
            funding_stage = None
            if result.get("funding_stage"):
                try:
                    funding_stage = FundingStage(result["funding_stage"])
                except ValueError as stage_error:
                    logger.warning(f"Invalid funding stage '{result['funding_stage']}' for {result['name']}: {stage_error}")
                    funding_stage = FundingStage.UNKNOWN
            
            # Preprocess website URL
            website = result.get("website")
            if website and not website.startswith(('http://', 'https://')):
                website = f'https://{website}'
            
            # Process LinkedIn URL
            linkedin_url = result.get("linkedin_url")
            if linkedin_url and not linkedin_url.startswith(('http://', 'https://')):
                linkedin_url = f'https://{linkedin_url}'
            
            # Convert funding from millions to USD
            funding_millions = result.get("funding_amount_millions")
            funding_total_usd = None
            if funding_millions and isinstance(funding_millions, (int, float)):
                funding_total_usd = funding_millions * 1_000_000
            
            # Create Company object with error handling
            try:
                company = Company(
                    uuid=f"comp_{hash(result.get('name', ''))}", # Generate simple UUID
                    name=clean_text(result.get("name", "")),
                    description=clean_text(result.get("description", "")),
                    short_description=clean_text(result.get("short_description", "")),
                    founded_year=result.get("founded_year"),
                    funding_total_usd=funding_total_usd,
                    funding_stage=funding_stage,
                    founders=result.get("founders", []),
                    investors=result.get("investors", []),
                    categories=result.get("categories", []),
                    city=clean_text(result.get("city", "")),
                    region=clean_text(result.get("region", "")),
                    country=clean_text(result.get("country", "")),
                    ai_focus=clean_text(result.get("ai_focus", "")),
                    sector=clean_text(result.get("sector", "")),
                    website=website,
                    linkedin_url=linkedin_url,
                    source_url=url,
                    extraction_date=datetime.utcnow()
                )
                
                # Final validation
                if not company.name or len(company.name.strip()) < 2:
                    logger.warning(f"Company name too short or empty for {url}")
                    return None
                
                return company
                
            except Exception as company_error:
                logger.error(f"Failed to create Company object for {result.get('name', 'unknown')}: {company_error}")
                return None
            
        except Exception as e:
            logger.error(f"Error extracting company data: {e}")
            return None
    
    def _deduplicate_companies(self, companies: List[Company]) -> List[Company]:
        """Remove duplicate companies based on name similarity."""
        unique_companies = []
        seen_names = set()
        
        for company in companies:
            name_lower = company.name.lower().strip()
            if name_lower and len(name_lower) > 2 and name_lower not in seen_names:
                seen_names.add(name_lower)
                unique_companies.append(company)
        
        return unique_companies