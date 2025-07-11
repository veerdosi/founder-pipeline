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
            max_requests=settings.requests_per_minute * 2,  # Double the rate limit for better throughput
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
        logger.info(f"🚀 Starting discovery: {limit} companies from 20+ sources...")
        
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
        start_time = time.time()
        year_context = f" (founded in {founded_year})" if founded_year else ""
        logger.info(f"🔍 Finding {limit} AI companies with Exa{year_context}...")
        
        categories = categories or settings.ai_categories
        
        # Generate search queries with optional year filtering
        queries = self._generate_search_queries(categories, regions, founded_year)
        logger.info(f"📝 Running {len(queries)} search queries...")
        
        all_companies = []
        processed_urls = set()  # Track processed URLs to avoid duplicates
        # Increase results per query significantly for better yield
        results_per_query = max(12, (limit * 3) // len(queries))  # 3x multiplier for better coverage
        
        # Execute searches with progress tracking
        for i, query in enumerate(queries):
            logger.info(f"🔍 [{i+1}/{len(queries)}] {query[:50]}...")
            print(f"🔍 [{i+1}/{len(queries)}] Searching: {query[:50]}...")
            
            try:
                await self.rate_limiter.acquire()
                
                # Perform search with error handling
                try:
                    print(f"   📡 Executing search query...")
                    result = self.exa.search_and_contents(
                        query,
                        type="neural",
                        use_autoprompt=True,
                        num_results=results_per_query,
                        text={"max_characters": 2000},
                        include_domains=[
                            # Tech news and startup coverage
                            "techcrunch.com", 
                            "venturebeat.com", 
                            "theverge.com",
                            "wired.com",
                            "arstechnica.com",
                            "thenextweb.com",  
                            "mashable.com",
                            "engadget.com",
                            
                            # Business and funding news
                            "bloomberg.com", 
                            "reuters.com",  
                            "forbes.com",
                            "businessinsider.com",
                            "inc.com",
                            "fastcompany.com",
                            "axios.com",
                            "fortune.com",
                            "wsj.com",
                            
                            # Startup databases and platforms
                            "crunchbase.com",
                            "pitchbook.com", 
                            "producthunt.com",
                            "angel.co", 
                            "startupgrind.com",
                            "founder.org",
                            
                            # Industry publications
                            "theinformation.com",
                            "protocol.com",  
                            "stratechery.com",
                            "recode.net",  
                            "readwrite.com",
                            
                            # Accelerator and VC sites
                            "500.co",
                            "angellist.com",
                            "antler.co", 
                            "plugandplaytechcenter.com",
                            "masschallenge.org"
                        ]
                    )
                    if result and result.results:
                        print(f"   📊 Found {len(result.results)} articles")
                        logger.info(f"  Found {len(result.results)} articles")
                    else:
                        print(f"   ⚠️  No results for query: {query}")
                        logger.debug(f"  No results for query: {query}")
                        continue
                except Exception as search_error:
                    logger.error(f"  Search failed: {search_error}")
                    continue
                
                if not result or not result.results:
                    continue
                
                # Extract company data from results with individual error handling
                companies_found_this_query = 0
                print(f"   🔍 Processing {len(result.results)} articles for company data...")
                
                for j, item in enumerate(result.results):
                    try:
                        # Skip if we've already processed this URL
                        if item.url in processed_urls:
                            print(f"     ⏭️  [{j+1}/{len(result.results)}] Skipping duplicate: {item.title[:50]}...")
                            continue
                        
                        processed_urls.add(item.url)
                        print(f"     📄 [{j+1}/{len(result.results)}] Processing: {item.title[:50]}...")
                        company = await self._extract_company_data(
                            item.text,
                            item.url,
                            item.title,
                            founded_year
                        )
                        if company:
                            all_companies.append(company)
                            companies_found_this_query += 1
                            print(f"     ✅ {company.name}")
                            logger.info(f"  ✅ {company.name}")
                    except Exception as extract_error:
                        logger.debug(f"  Failed to extract from {item.url}: {extract_error}")
                        continue
                
                if companies_found_this_query > 0:
                    print(f"   📊 {companies_found_this_query} companies found ({len(all_companies)} total)")
                    logger.info(f"  📊 {companies_found_this_query} companies found ({len(all_companies)} total)")
                else:
                    print(f"   ❌ No companies found in this query")
                        
            except Exception as e:
                logger.error(f"Unexpected error with query '{query}': {e}")
                continue
        
        # Deduplicate companies
        print(f"🔄 Deduplicating {len(all_companies)} companies...")
        unique_companies = self._deduplicate_companies(all_companies)
        
        elapsed_time = time.time() - start_time
        print(f"✅ Discovery complete: {len(unique_companies)} unique companies ({elapsed_time:.1f}s)")
        logger.info(f"✅ Discovery complete: {len(unique_companies)} unique companies ({elapsed_time:.1f}s)")
        
        return unique_companies[:limit]
    
    def _generate_search_queries(
        self, 
        categories: List[str], 
        regions: Optional[List[str]] = None,
        founded_year: Optional[int] = None
    ) -> List[str]:
        """Generate US-focused search queries for AI companies."""
        
        if founded_year:
            # Year-specific queries - US market focused
            logger.info(f"🇺🇸 Generating US-focused queries for companies founded in {founded_year}")
            queries = [
                # US funding announcements
                f"US AI startups founded {founded_year} announced seed funding",
                f"American artificial intelligence companies {founded_year} pre-seed",
                f"Silicon Valley AI startups {founded_year} Series A funding",
                f"US machine learning companies {founded_year} venture capital",
                f"American AI companies {founded_year} raised seed round",
                f"US startup funding {founded_year} artificial intelligence",
                f"American generative AI companies {founded_year} investment",
                f"US computer vision startups {founded_year} funding",
                
                # US-specific accelerators and VCs
                f"Y Combinator AI startups {founded_year} batch",
                f"Techstars US AI companies {founded_year} demo day",
                f"500 Startups American AI batch {founded_year}",
                f"Andreessen Horowitz AI investments {founded_year} a16z",
                f"Sequoia Capital US AI startups {founded_year}",
                f"Google Ventures American AI companies {founded_year}",
                f"Microsoft Ventures US AI startups {founded_year}",
                f"Intel Capital American AI investments {founded_year}",
                f"NVIDIA Inception US AI startups {founded_year}",
                f"Amazon Alexa Fund American AI companies {founded_year}",
                f"Salesforce Ventures US AI startups {founded_year}",
                
                # US geographic hubs
                f"Silicon Valley AI startups {founded_year} Palo Alto",
                f"San Francisco AI companies {founded_year} Bay Area",
                f"Seattle AI startups {founded_year} Washington",
                f"New York AI companies {founded_year} NYC",
                f"Boston AI startups {founded_year} Cambridge",
                f"Austin AI companies {founded_year} Texas",
                f"Los Angeles AI startups {founded_year} California",
                f"Chicago AI companies {founded_year} Illinois",
                f"Denver AI startups {founded_year} Colorado",
                f"Atlanta AI companies {founded_year} Georgia",
                
                # US industry-specific
                f"US healthcare AI startups {founded_year} medical",
                f"American fintech AI companies {founded_year} financial",
                f"US enterprise AI startups {founded_year} B2B",
                f"American consumer AI apps {founded_year} B2C",
                f"US AI infrastructure companies {founded_year} cloud",
                f"American AI developer tools {founded_year} programming",
                f"US cybersecurity AI startups {founded_year} security",
                f"American autonomous vehicle AI {founded_year} self-driving",
                f"US AI robotics companies {founded_year} automation",
                f"American AI drug discovery {founded_year} biotech",
                
                # US universities and research
                f"Stanford AI startups {founded_year} university spinoff",
                f"MIT AI companies {founded_year} research spin-out",
                f"Carnegie Mellon AI startups {founded_year} CMU",
                f"UC Berkeley AI companies {founded_year} California",
                f"Harvard AI startups {founded_year} Cambridge",
                f"Caltech AI companies {founded_year} Pasadena"
            ]
            return queries[:35]
        else:
            # Current year searches - US market focused
            current_year = datetime.now().year
            previous_year = current_year - 1
            
            queries = [
                # US funding announcements
                f"US AI startups announced pre-seed funding {current_year}",
                f"American artificial intelligence companies raised seed {current_year}",
                f"Silicon Valley AI startups Series A funding {current_year}",
                f"US machine learning companies venture capital {current_year}",
                f"American generative AI startups funding {current_year}",
                f"US computer vision companies investment {current_year}",
                f"American NLP startups funding {current_year}",
                f"US AI robotics companies venture capital {current_year}",
                
                # US accelerators and VCs
                f"Y Combinator AI startups demo day {current_year}",
                f"Techstars US AI companies {current_year}",
                f"500 Startups American AI batch {current_year}",
                f"Andreessen Horowitz US AI investments {current_year}",
                f"Sequoia Capital American AI startups {current_year}",
                f"Google Ventures US AI companies {current_year}",
                f"Microsoft Ventures American AI startups {current_year}",
                f"Intel Capital US AI investments {current_year}",
                f"NVIDIA Inception American AI startups {current_year}",
                f"Amazon Alexa Fund US AI companies {current_year}",
                f"Salesforce Ventures American AI startups {current_year}",
                f"Founders Fund US AI companies {current_year}",
                f"Kleiner Perkins American AI investments {current_year}",
                f"Bessemer Venture Partners US AI {current_year}",
                
                # US geographic hubs
                f"Silicon Valley AI startups seed funding {current_year}",
                f"San Francisco AI companies Bay Area {current_year}",
                f"Seattle AI startups Washington state {current_year}",
                f"New York AI companies NYC funding {current_year}",
                f"Boston AI startups Cambridge {current_year}",
                f"Austin AI companies Texas {current_year}",
                f"Los Angeles AI startups California {current_year}",
                f"Chicago AI companies Illinois {current_year}",
                f"Denver AI startups Colorado {current_year}",
                f"Atlanta AI companies Georgia {current_year}",
                f"Miami AI startups Florida {current_year}",
                f"Phoenix AI companies Arizona {current_year}",
                
                # US industry verticals
                f"US healthcare AI startups medical devices {current_year}",
                f"American fintech AI companies financial {current_year}",
                f"US enterprise AI startups B2B software {current_year}",
                f"American consumer AI apps B2C {current_year}",
                f"US AI infrastructure companies cloud {current_year}",
                f"American AI developer tools programming {current_year}",
                f"US cybersecurity AI startups security {current_year}",
                f"American autonomous vehicle AI {current_year}",
                f"US AI robotics companies automation {current_year}",
                f"American AI drug discovery biotech {current_year}",
                f"US logistics AI companies supply chain {current_year}",
                f"American legal AI startups lawtech {current_year}",
                
                # US universities and research
                f"Stanford AI startups university spinoff {current_year}",
                f"MIT AI companies research commercialization {current_year}",
                f"Carnegie Mellon AI startups CMU {current_year}",
                f"UC Berkeley AI companies California {current_year}",
                f"Harvard AI startups Cambridge {current_year}",
                f"Caltech AI companies Pasadena {current_year}",
                f"Georgia Tech AI startups Atlanta {current_year}",
                f"University of Washington AI companies Seattle {current_year}",
                
                # US platform-specific
                f"Product Hunt AI tools launched US {current_year}",
                f"AngelList AI startups fundraising America {current_year}",
                f"TechCrunch US AI companies {current_year}",
                f"VentureBeat American AI startups {current_year}",
                f"Crunchbase US AI funding {current_year}",
                f"PitchBook American AI companies {current_year}"
            ]
            return queries[:40]
    
    async def _is_company_alive(self, company_name: str, website: str = None) -> bool:
        """Use Perplexity to verify if a company is still active and operating."""
        try:
            # Import here to avoid circular imports
            from openai import AsyncOpenAI
            
            # Use OpenAI client configured for Perplexity
            perplexity_client = AsyncOpenAI(
                api_key=settings.perplexity_api_key,
                base_url="https://api.perplexity.ai"
            )
            
            query = f"Is {company_name} still active and operating in 2024? Are they still in business?"
            if website:
                query += f" Website: {website}"
            
            response = await perplexity_client.chat.completions.create(
                model="sonar",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a business research assistant. Answer with 'ACTIVE' if the company is still operating and in business, 'INACTIVE' if they have shut down or ceased operations, or 'UNKNOWN' if you cannot determine their status."
                    },
                    {
                        "role": "user", 
                        "content": query
                    }
                ],
                max_tokens=50,
                temperature=0.1
            )
            
            if response and response.choices:
                answer = response.choices[0].message.content.strip().upper()
                if "ACTIVE" in answer:
                    return True
                elif "INACTIVE" in answer:
                    logger.info(f"Company {company_name} appears to be inactive - filtering out")
                    return False
                else:
                    # Default to including if uncertain
                    return True
                    
        except Exception as e:
            logger.debug(f"Error checking if {company_name} is alive: {e}")
            # Default to including company if verification fails
            return True
        
        return True
    
    async def _is_early_stage_startup(self, company_data: dict, content: str, target_year: Optional[int] = None) -> bool:
        """Use Perplexity to determine if a company is an early-stage startup suitable for VC investment."""
        
        # Determine the context year for assessment
        current_year = datetime.now().year
        assessment_year = target_year if target_year else current_year
        
        # Calculate time-based context for what constitutes "early stage" at that time
        if target_year:
            # For historical searches, assess based on the target year context
            founded_year = company_data.get('founded_year')
            if founded_year and isinstance(founded_year, int):
                years_since_founding = max(0, assessment_year - founded_year)
                time_context = f"This analysis is for companies in {target_year}. A company founded in {founded_year} would be {years_since_founding} years old in {target_year}."
            else:
                time_context = f"This analysis is for companies in {target_year}. Company founding year is unknown."
        else:
            # For current searches, use present-day context
            time_context = f"This analysis is for present-day ({current_year}) investment opportunities."
        
        prompt = f"""
Analyze this company information and determine if it was an early-stage startup suitable for VC investment at the time of analysis.

Company: {company_data.get('name', 'Unknown')}
Description: {company_data.get('description', 'No description')}
Founded Year: {company_data.get('founded_year', 'Unknown')}
Funding Stage: {company_data.get('funding_stage', 'Unknown')}
Funding Amount: {company_data.get('funding_amount_millions', 'Unknown')}

{time_context}

Context from article: {content[:800]}

CLASSIFICATION CRITERIA:
- EARLY-STAGE STARTUP: Pre-seed, seed, Series A, Series B, Series C (if recent), private company with growth potential, seeking venture capital, founded after 2015
- MATURE/ESTABLISHED: Public companies, mega-unicorns (>$10B valuation), Big Tech (Google, Meta, Apple, Microsoft, Amazon, etc.), well-established enterprises with >10 years and >$100M revenue

IMPORTANT: Be INCLUSIVE rather than exclusive. If unsure, classify as early-stage. Focus on excluding only obviously mature companies.

Answer with ONLY "YES" if this is an early-stage startup suitable for VC investment, or "NO" if it's clearly a mature/established company.
"""
        
        try:
            # Import here to avoid circular imports
            from openai import AsyncOpenAI
            
            # Use OpenAI client configured for Perplexity
            perplexity_client = AsyncOpenAI(
                api_key=settings.perplexity_api_key,
                base_url="https://api.perplexity.ai"
            )
            
            await self.rate_limiter.acquire()
            
            response = await perplexity_client.chat.completions.create(
                model="sonar",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10,
                timeout=15.0
            )
            
            if response and response.choices:
                answer = response.choices[0].message.content.strip().upper()
                return answer == "YES"
            
        except Exception as e:
            logger.debug(f"Error in maturity assessment for {company_data.get('name', 'Unknown')}: {e}")
            # Default to including the company if AI assessment fails
            return True
        
        return True
    
    async def _get_crunchbase_url(self, company_name: str, website: str = None) -> Optional[str]:
        """Use Perplexity to find the crunchbase URL for a company."""
        try:
            # Import here to avoid circular imports
            from openai import AsyncOpenAI
            
            # Use OpenAI client configured for Perplexity
            perplexity_client = AsyncOpenAI(
                api_key=settings.perplexity_api_key,
                base_url="https://api.perplexity.ai"
            )
            
            query = f"What is the Crunchbase URL for {company_name}?"
            if website:
                query += f" Their website is {website}"
            query += " Please provide the exact Crunchbase URL."
            
            response = await perplexity_client.chat.completions.create(
                model="sonar",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a business research assistant. Find the exact Crunchbase URL for companies. Return only the URL if found, or 'NOT_FOUND' if you cannot find it."
                    },
                    {
                        "role": "user", 
                        "content": query
                    }
                ],
                max_tokens=100,
                temperature=0.1
            )
            
            if response and response.choices:
                answer = response.choices[0].message.content.strip()
                
                # Extract URL from the response
                import re
                url_pattern = r'https?://(?:www\.)?crunchbase\.com/[^\s<>"\']*'
                urls = re.findall(url_pattern, answer)
                
                if urls:
                    return urls[0]
                elif "NOT_FOUND" in answer.upper():
                    return None
                else:
                    return None
                    
        except Exception as e:
            logger.debug(f"Error getting crunchbase URL for {company_name}: {e}")
            return None
        
        return None
    
    async def _extract_company_data(
        self, 
        content: str, 
        url: str, 
        title: str,
        target_year: Optional[int] = None
    ) -> Optional[Company]:
        """Extract structured company data using GPT-4o-mini with improved extraction."""
        year_instruction = ""
        if target_year:
            year_instruction = f"""
IMPORTANT: Pay special attention to the founding year. Only extract companies that were founded in {target_year}. 
If the company was not founded in {target_year}, return null for the founded_year field.
"""
        
        # Enhanced prompt for better extraction
        prompt = f"""
Extract company information from this content. Focus on finding ANY startup mentioned, not just the main subject.

EXTRACTION RULES:
1. Extract ALL companies mentioned in the article, even if briefly mentioned
2. For each company, include: name, description, founding year, funding info, founders, location
3. If multiple companies are mentioned, extract the MOST RELEVANT early-stage startup
4. IGNORE: Google, Meta, Apple, Microsoft, Amazon, OpenAI, Anthropic, public companies, unicorns >$1B
5. FOCUS ON: Recently founded companies, seed/Series A/B stage, private startups

IMPORTANT: Return ONLY valid JSON. No explanations.

Content: {content[:2000]}  
Title: {title}
{year_instruction}

REQUIRED JSON FORMAT (return exactly this structure):
{{
    "name": "exact company name",
    "description": "detailed description of what the company does",
    "short_description": "1-sentence summary",
    "founded_year": "YYYY as integer or null",
    "funding_amount_millions": "amount in USD millions as number or null",
    "funding_stage": "pre-seed/seed/series-a/series-b/series-c or null",
    "founders": ["founder names"],
    "investors": ["investor/VC names"],
    "categories": ["AI/ML category tags"],
    "city": "city name",
    "region": "state/province",
    "country": "country name",
    "ai_focus": "specific AI area (NLP, computer vision, robotics, etc.)",
    "sector": "industry sector (fintech, healthcare, etc.)",
    "website": "company website URL",
    "linkedin_url": "LinkedIn company URL"
}}

If NO suitable early-stage startup found, return: null
"""
        
        result = None  # Initialize result to avoid NoneType errors
        try:
            await self.rate_limiter.acquire()
            
            # OpenAI API call with timeout and retry
            try:
                response = await self.openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    timeout=45.0  # Increased timeout for better reliability
                )
            except Exception as api_error:
                logger.debug(f"OpenAI API call failed for {url}: {api_error}")
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
            
            # Handle common AI response patterns
            if raw_content.lower().startswith("based on"):
                # AI sometimes adds explanatory text before JSON
                lines = raw_content.split('\n')
                for i, line in enumerate(lines):
                    if line.strip().startswith('{') or line.strip().lower() == 'null':
                        raw_content = '\n'.join(lines[i:])
                        break
            
            # Clean up any remaining text around JSON
            raw_content = raw_content.strip()
            if raw_content.startswith("Here's the JSON:") or raw_content.startswith("Here is the JSON:"):
                raw_content = raw_content.split(":", 1)[1].strip()
            
            raw_content = raw_content.strip()
            
            # Parse JSON with error handling
            try:
                result = json.loads(raw_content)
            except json.JSONDecodeError as json_error:
                logger.error(f"JSON parsing failed for {url}: {json_error}")
                logger.debug(f"Raw content: {raw_content[:500]}...")
                return None
            
            # Handle null response (no company found)
            if result is None:
                logger.debug(f"No company found in content for {url}")
                return None
            
            # Handle different JSON structures
            if not isinstance(result, dict):
                logger.warning(f"Invalid JSON structure for {url} - Expected dict or null, got {type(result)}")
                logger.debug(f"Actual result: {result}")
                
                # Try to handle common non-dict structures
                if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
                    logger.info(f"Converting list to dict for {url}")
                    result = result[0]
                elif isinstance(result, str):
                    # Check if it's a string that contains "null" or similar
                    if result.lower().strip() in ['null', 'none', 'no company']:
                        logger.debug(f"AI returned no company indication for {url}")
                        return None
                    logger.warning(f"AI returned string instead of JSON for {url}: {result[:200]}...")
                    return None
                else:
                    return None
            
            # Validate required fields
            if not result.get("name") or not isinstance(result.get("name"), str):
                logger.warning(f"No valid company name found for {url}")
                return None
            
            # Use AI to assess if this is a mature company vs early-stage startup
            if not await self._is_early_stage_startup(result, content, target_year):
                print(f"     🚫 Filtered out mature company: {result.get('name')}")
                logger.debug(f"Filtered out mature/established company: {result.get('name')} for {url}")
                return None
            
            # Check if company is still alive and operating
            company_name = result.get('name')
            website = result.get('website')
            if not await self._is_company_alive(company_name, website):
                print(f"     💀 Filtered out inactive company: {company_name}")
                logger.debug(f"Filtered out inactive company: {company_name} for {url}")
                return None
            
            # Helper function to safely get values from result
            def safe_get(key, default=None):
                try:
                    return result.get(key, default) if result and isinstance(result, dict) else default
                except Exception:
                    return default
            
            # Map funding stage with error handling
            funding_stage = None
            funding_stage_raw = safe_get("funding_stage")
            if funding_stage_raw and isinstance(funding_stage_raw, str):
                try:
                    # Normalize funding stage string
                    raw_stage = funding_stage_raw.lower().strip()
                    
                    # Common funding stage mappings
                    stage_mappings = {
                        "early stage vc": FundingStage.SEED,
                        "early stage": FundingStage.SEED,
                        "early-stage": FundingStage.SEED,
                        "pre-seed": FundingStage.PRE_SEED,
                        "pre seed": FundingStage.PRE_SEED,
                        "preseed": FundingStage.PRE_SEED,
                        "seed": FundingStage.SEED,
                        "series a": FundingStage.SERIES_A,
                        "series-a": FundingStage.SERIES_A,
                        "series_a": FundingStage.SERIES_A,
                        "series b": FundingStage.SERIES_B,
                        "series-b": FundingStage.SERIES_B,
                        "series_b": FundingStage.SERIES_B,
                        "series c": FundingStage.SERIES_C,
                        "series-c": FundingStage.SERIES_C,
                        "series_c": FundingStage.SERIES_C,
                        "growth": FundingStage.GROWTH,
                        "growth stage": FundingStage.GROWTH,
                        "ipo": FundingStage.IPO,
                        "public": FundingStage.IPO,
                        "acquired": FundingStage.ACQUIRED,
                        "acquisition": FundingStage.ACQUIRED,
                        "unknown": FundingStage.UNKNOWN
                    }
                    
                    if raw_stage in stage_mappings:
                        funding_stage = stage_mappings[raw_stage]
                    else:
                        # Try direct enum lookup
                        funding_stage = FundingStage(funding_stage_raw)
                except ValueError as stage_error:
                    logger.warning(f"Invalid funding stage '{funding_stage_raw}' for {safe_get('name', 'unknown')}: {stage_error}")
                    funding_stage = FundingStage.UNKNOWN
            
            # Preprocess website URL
            website = safe_get("website")
            if website and isinstance(website, str) and not website.startswith(('http://', 'https://')):
                website = f'https://{website}'
            
            # Process LinkedIn URL
            linkedin_url = safe_get("linkedin_url")
            if linkedin_url and isinstance(linkedin_url, str) and not linkedin_url.startswith(('http://', 'https://')):
                linkedin_url = f'https://{linkedin_url}'
            
            # Convert funding from millions to USD
            funding_millions = safe_get("funding_amount_millions")
            funding_total_usd = None
            if funding_millions and isinstance(funding_millions, (int, float)):
                funding_total_usd = funding_millions * 1_000_000
            
            # Create Company object with error handling
            try:
                company_name = safe_get("name", "")
                
                # Get crunchbase URL using Perplexity
                crunchbase_url = await self._get_crunchbase_url(company_name, website)
                
                company = Company(
                    uuid=f"comp_{hash(company_name)}", # Generate simple UUID
                    name=clean_text(company_name),
                    description=clean_text(safe_get("description", "")),
                    short_description=clean_text(safe_get("short_description", "")),
                    founded_year=safe_get("founded_year"),
                    funding_total_usd=funding_total_usd,
                    funding_stage=funding_stage,
                    founders=safe_get("founders", []),
                    investors=safe_get("investors", []),
                    categories=safe_get("categories", []),
                    city=clean_text(safe_get("city", "")),
                    region=clean_text(safe_get("region", "")),
                    country=clean_text(safe_get("country", "")),
                    ai_focus=clean_text(safe_get("ai_focus", "")),
                    sector=clean_text(safe_get("sector", "")),
                    website=website,
                    linkedin_url=linkedin_url,
                    crunchbase_url=crunchbase_url,
                    source_url=url,
                    extraction_date=datetime.utcnow()
                )
                
                # Final validation
                if not company.name or len(company.name.strip()) < 2:
                    logger.warning(f"Company name too short or empty for {url}")
                    return None
                
                return company
                
            except Exception as company_error:
                company_name = safe_get('name', 'unknown')
                logger.error(f"Failed to create Company object for {company_name}: {company_error}")
                return None
            
        except Exception as e:
            company_name = "unknown"
            try:
                if result is not None and isinstance(result, dict):
                    company_name = result.get('name', 'unknown')
            except Exception:
                pass  # Keep default company_name
            logger.error(f"Error extracting company data from {url} (company: {company_name}): {e}")
            return None
    
    def _deduplicate_companies(self, companies: List[Company]) -> List[Company]:
        """Remove duplicate companies based on name similarity and other factors."""
        import difflib
        
        unique_companies = []
        seen_names = set()
        seen_websites = set()
        
        for company in companies:
            # Skip if no valid name
            if not company.name or len(company.name.strip()) < 2:
                continue
                
            name_lower = company.name.lower().strip()
            
            # Skip obvious duplicates by exact name match
            if name_lower in seen_names:
                continue
            
            # Check for similar names (fuzzy matching)
            is_duplicate = False
            for existing_name in seen_names:
                # Use difflib for similarity matching
                similarity = difflib.SequenceMatcher(None, name_lower, existing_name).ratio()
                if similarity > 0.85:  # 85% similarity threshold
                    logger.debug(f"Filtering similar company name: {company.name} (similar to existing: {existing_name})")
                    is_duplicate = True
                    break
            
            if is_duplicate:
                continue
            
            # Check for duplicate websites
            if company.website:
                website_str = str(company.website).lower()
                if website_str in seen_websites:
                    logger.debug(f"Filtering duplicate website: {company.name} ({website_str})")
                    continue
                seen_websites.add(website_str)
            
            # Add to unique companies
            seen_names.add(name_lower)
            unique_companies.append(company)
        
        logger.info(f"Deduplication: {len(companies)} → {len(unique_companies)} companies")
        return unique_companies