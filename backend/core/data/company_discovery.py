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
from ..analysis.sector_classification import sector_description_service

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
    """Enhanced company discovery using comprehensive 20+ source monitoring with better uniqueness."""
    
    def __init__(self):
        self.exa = Exa(settings.exa_api_key)
        self.openai = AsyncOpenAI(api_key=settings.openai_api_key)
        self.rate_limiter = RateLimiter(
            max_requests=settings.requests_per_minute * 2,
            time_window=60
        )
        self.session: Optional[aiohttp.ClientSession] = None
        # Track processed articles globally to avoid duplicates across queries
        self.processed_urls = set()
        self.processed_titles = set()  # Track similar titles
    
    async def discover_companies(
        self,
        limit: int = 50,
        categories: Optional[List[str]] = None,
        regions: Optional[List[str]] = None,
        sources: Optional[List[str]] = None
    ) -> List[Company]:
        """Discover companies using comprehensive 20+ source monitoring."""
        logger.info(f"🚀 Starting discovery: {limit} companies from 20+ sources...")
        
        # Reset tracking for each discovery session
        self.processed_urls.clear()
        self.processed_titles.clear()
        
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
        """Find AI companies using Exa search with enhanced uniqueness."""
        start_time = time.time()
        year_context = f" (founded in {founded_year})" if founded_year else ""
        logger.info(f"🔍 Finding {limit} AI companies with Exa{year_context}...")
        
        categories = categories or settings.ai_categories
        
        # Generate more diverse search queries
        queries = self._generate_diverse_search_queries(categories, regions, founded_year)
        logger.info(f"📝 Running {len(queries)} diverse search queries...")
        
        all_companies = []
        # Reduce results per query but increase query diversity
        results_per_query = max(8, (limit * 2) // len(queries))
        
        # Execute searches with staggered timing and different parameters
        for i, query_config in enumerate(queries):
            query = query_config["query"]
            search_type = query_config.get("type", "neural")
            time_filter = query_config.get("time_filter")
            
            logger.info(f"🔍 [{i+1}/{len(queries)}] {query[:50]}...")
            print(f"🔍 [{i+1}/{len(queries)}] Searching: {query[:50]}...")
            
            try:
                await self.rate_limiter.acquire()
                
                # Add small delay between queries to get different results
                if i > 0:
                    await asyncio.sleep(0.5)
                
                # Perform search with varied parameters
                search_params = {
                    "query": query,
                    "type": search_type,
                    "use_autoprompt": True,
                    "num_results": results_per_query,
                    "text": {"max_characters": 2000},
                    "include_domains": self._get_diverse_domains(i)
                }
                
                # Add time filter if specified
                if time_filter:
                    search_params["start_published_date"] = time_filter
                
                try:
                    print(f"   📡 Executing {search_type} search...")
                    result = self.exa.search_and_contents(**search_params)
                    
                    if result and result.results:
                        print(f"   📊 Found {len(result.results)} articles")
                        logger.info(f"  Found {len(result.results)} articles")
                    else:
                        print(f"   ⚠️  No results for query: {query}")
                        continue
                except Exception as search_error:
                    logger.error(f"  Search failed: {search_error}")
                    continue
                
                if not result or not result.results:
                    continue
                
                # Filter out duplicate/similar articles before processing
                unique_results = self._filter_unique_articles(result.results)
                print(f"   🔍 Processing {len(unique_results)} unique articles...")
                
                companies_found_this_query = 0
                for j, item in enumerate(unique_results):
                    try:
                        print(f"     📄 [{j+1}/{len(unique_results)}] Processing: {item.title[:50]}...")
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
                
                # Early exit if we have enough companies
                if len(all_companies) >= limit * 2:
                    logger.info(f"🎯 Early exit: Found {len(all_companies)} companies (target: {limit})")
                    break
                        
            except Exception as e:
                logger.error(f"Unexpected error with query '{query}': {e}")
                continue
        
        # Enhanced deduplication
        print(f"🔄 Deduplicating {len(all_companies)} companies...")
        unique_companies = self._deduplicate_companies(all_companies)
        
        elapsed_time = time.time() - start_time
        print(f"✅ Discovery complete: {len(unique_companies)} unique companies ({elapsed_time:.1f}s)")
        logger.info(f"✅ Discovery complete: {len(unique_companies)} unique companies ({elapsed_time:.1f}s)")
        
        return unique_companies[:limit]
    
    def _filter_unique_articles(self, results) -> List:
        """Filter out duplicate and similar articles."""
        unique_results = []
        
        for item in results:
            # Skip if URL already processed
            if item.url in self.processed_urls:
                continue
            
            # Skip if title is too similar to existing ones
            title_words = set(item.title.lower().split())
            is_similar = False
            
            for existing_title in self.processed_titles:
                existing_words = set(existing_title.lower().split())
                # If 70% of words overlap, consider it similar
                overlap = len(title_words & existing_words)
                if overlap > 0.7 * min(len(title_words), len(existing_words)):
                    is_similar = True
                    break
            
            if not is_similar:
                unique_results.append(item)
                self.processed_urls.add(item.url)
                self.processed_titles.add(item.title)
        
        return unique_results
    
    def _get_diverse_domains(self, query_index: int) -> List[str]:
        """Get different domain sets for each query to increase diversity."""
        domain_sets = [
            # Tech news focused
            [
                "techcrunch.com", "venturebeat.com", "theverge.com",
                "wired.com", "arstechnica.com", "thenextweb.com"
            ],
            # Business news focused
            [
                "bloomberg.com", "reuters.com", "forbes.com",
                "businessinsider.com", "inc.com", "fastcompany.com"
            ],
            # Startup databases
            [
                "crunchbase.com", "pitchbook.com", "producthunt.com",
                "angel.co", "startupgrind.com", "founder.org"
            ],
            # Industry publications
            [
                "theinformation.com", "protocol.com", "stratechery.com",
                "recode.net", "readwrite.com", "axios.com"
            ],
            # Regional and niche
            [
                "siliconvalley.com", "sanfrancisco.com", "nyc.com",
                "boston.com", "austin.com", "seattle.com"
            ],
            # AI-specific publications
            [
                "artificialintelligence-news.com", "venturebeat.com",
                "technologyreview.com", "spectrum.ieee.org"
            ]
        ]
        
        # Rotate through domain sets
        return domain_sets[query_index % len(domain_sets)]
    
    def _generate_diverse_search_queries(
        self, 
        categories: List[str], 
        regions: Optional[List[str]] = None,
        founded_year: Optional[int] = None
    ) -> List[dict]:
        """Generate diverse search queries with different types and filters."""
        
        if founded_year:
            logger.info(f"🇺🇸 Generating diverse US-focused queries for companies founded in {founded_year}")
            
            # Current date for time filters
            current_year = datetime.now().year
            
            queries = [
                # Recent funding announcements - neural search
                {"query": f"US AI startups founded {founded_year} seed funding announcement", "type": "neural"},
                {"query": f"American artificial intelligence companies {founded_year} Series A", "type": "neural"},
                {"query": f"Silicon Valley AI startups {founded_year} venture capital", "type": "neural"},
                
                # Keyword search for specific terms
                {"query": f"AI startup {founded_year} United States funding", "type": "keyword"},
                {"query": f"machine learning company {founded_year} America investment", "type": "keyword"},
                {"query": f"artificial intelligence {founded_year} US Series A", "type": "keyword"},
                
                # Time-filtered recent news
                {"query": f"AI companies founded {founded_year} funding news", "type": "neural", 
                 "time_filter": f"{current_year-1}-01-01"},
                {"query": f"American AI startups {founded_year} investment", "type": "neural",
                 "time_filter": f"{current_year-1}-01-01"},
                
                # Geographic diversity
                {"query": f"San Francisco AI startups {founded_year} Bay Area", "type": "neural"},
                {"query": f"New York AI companies {founded_year} NYC tech", "type": "neural"},
                {"query": f"Boston AI startups {founded_year} Cambridge MIT", "type": "neural"},
                {"query": f"Austin AI companies {founded_year} Texas tech", "type": "neural"},
                {"query": f"Seattle AI startups {founded_year} Washington", "type": "neural"},
                {"query": f"Los Angeles AI companies {founded_year} California", "type": "neural"},
                {"query": f"Chicago AI startups {founded_year} Illinois", "type": "neural"},
                {"query": f"Denver AI companies {founded_year} Colorado tech", "type": "neural"},
                
                # Accelerator and incubator specific
                {"query": f"Y Combinator AI batch {founded_year} demo day", "type": "keyword"},
                {"query": f"Techstars AI companies {founded_year} accelerator", "type": "keyword"},
                {"query": f"500 Startups AI batch {founded_year}", "type": "keyword"},
                {"query": f"Plug and Play AI startups {founded_year}", "type": "keyword"},
                {"query": f"AngelList AI companies {founded_year}", "type": "keyword"},
                
                # VC-specific searches
                {"query": f"Andreessen Horowitz AI investments {founded_year}", "type": "keyword"},
                {"query": f"Sequoia Capital AI startups {founded_year}", "type": "keyword"},
                {"query": f"Google Ventures AI companies {founded_year}", "type": "keyword"},
                {"query": f"Kleiner Perkins AI startups {founded_year}", "type": "keyword"},
                {"query": f"Accel Partners AI investments {founded_year}", "type": "keyword"},
                
                # Industry verticals
                {"query": f"healthcare AI startups {founded_year} United States", "type": "neural"},
                {"query": f"fintech AI companies {founded_year} American", "type": "neural"},
                {"query": f"enterprise AI startups {founded_year} B2B US", "type": "neural"},
                {"query": f"consumer AI apps {founded_year} American B2C", "type": "neural"},
                {"query": f"autonomous vehicle AI {founded_year} US self-driving", "type": "neural"},
                {"query": f"cybersecurity AI startups {founded_year} American", "type": "neural"},
                {"query": f"robotics AI companies {founded_year} US automation", "type": "neural"},
                {"query": f"drug discovery AI {founded_year} American biotech", "type": "neural"},
                
                # University spinoffs
                {"query": f"Stanford AI startup {founded_year} university", "type": "keyword"},
                {"query": f"MIT AI company {founded_year} research", "type": "keyword"},
                {"query": f"Carnegie Mellon AI startup {founded_year} CMU", "type": "keyword"},
                {"query": f"UC Berkeley AI company {founded_year}", "type": "keyword"},
                {"query": f"Harvard AI startup {founded_year}", "type": "keyword"},
                
                # Alternative search terms
                {"query": f"generative AI startup {founded_year} United States", "type": "neural"},
                {"query": f"computer vision company {founded_year} American", "type": "neural"},
                {"query": f"natural language processing startup {founded_year} US", "type": "neural"},
                {"query": f"deep learning company {founded_year} America", "type": "neural"},
                {"query": f"AI infrastructure startup {founded_year} US cloud", "type": "neural"},
                {"query": f"AI developer tools {founded_year} American programming", "type": "neural"},
                
                # Recent news angles
                {"query": f"emerging AI companies {founded_year} United States", "type": "neural"},
                {"query": f"AI unicorn startup {founded_year} American billion", "type": "neural"},
                {"query": f"stealth AI company {founded_year} US launch", "type": "neural"},
                {"query": f"AI acquisition {founded_year} American startup", "type": "neural"},
                
                # Product launches and announcements
                {"query": f"AI product launch {founded_year} American startup", "type": "neural"},
                {"query": f"AI beta launch {founded_year} US company", "type": "neural"},
                {"query": f"AI platform launch {founded_year} American", "type": "neural"}
            ]
            
            return queries[:45]  # Return more diverse queries
    
    def _normalize_company_name(self, name: str) -> str:
        """Normalize company name for comparison."""
        # Remove common suffixes and prefixes
        name = name.lower().strip()
        name = re.sub(r'\b(inc|llc|corp|corporation|ltd|limited|ai|technologies|tech|systems|solutions|labs|lab)\b', '', name)
        name = re.sub(r'[^\w\s]', '', name)  # Remove special characters
        name = re.sub(r'\s+', ' ', name).strip()  # Normalize whitespace
        return name
    
    def _is_similar_name(self, name: str, seen_names: set) -> bool:
        """Check if name is similar to any seen names."""
        name_words = set(name.split())
        for seen_name in seen_names:
            seen_words = set(seen_name.split())
            if len(name_words) > 0 and len(seen_words) > 0:
                # If 80% of words match, consider it similar
                overlap = len(name_words & seen_words)
                if overlap >= 0.8 * min(len(name_words), len(seen_words)):
                    return True
        return False
    
    def _is_similar_description(self, description: str, seen_descriptions: set) -> bool:
        """Check if description is similar to any seen descriptions."""
        desc_words = set(description.lower().split()[:20])  # First 20 words
        for seen_desc in seen_descriptions:
            seen_words = set(seen_desc.lower().split()[:20])
            if len(desc_words) > 0 and len(seen_words) > 0:
                overlap = len(desc_words & seen_words)
                if overlap >= 0.7 * min(len(desc_words), len(seen_words)):
                    return True
        return False
    
    def _extract_domain(self, url) -> str:
        """Extract domain from URL."""
        try:
            from urllib.parse import urlparse
            # Convert HttpUrl object to string if needed
            url_str = str(url) if url else ""
            return urlparse(url_str).netloc.lower()
        except:
            # Fallback: convert to string and extract domain
            url_str = str(url) if url else ""
            return url_str.lower()
    
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
    
    async def _get_crunchbase_url(self, company_name: str, website: str = None) -> Optional[str]:
        """Use Serper (Google Search) to find the crunchbase URL for a company."""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Try multiple search strategies
            search_queries = [
                f"{company_name} crunchbase",  # Basic search
                f'"{company_name}" site:crunchbase.com',  # Exact match on Crunchbase
                f"{company_name} crunchbase.com organization"  # More specific
            ]
            
            # If website provided, add site-specific search but as fallback only
            if website:
                domain = website.replace("https://", "").replace("http://", "").replace("www.", "").split("/")[0]
                search_queries.append(f"{company_name} crunchbase site:{domain}")
            
            for query in search_queries:
                logger.info(f"Searching Google for Crunchbase URL: {query}")
                
                # SerpApi API call
                url = "https://serpapi.com/search"
                payload = {
                    "q": query,
                    "num": 10,
                    "api_key": settings.serpapi_key,
                    "engine": "google"
                }
                headers = {
                    "Content-Type": "application/json"
                }
                
                async with self.session.get(url, params=payload, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.debug(f"Search response keys: {list(data.keys())}")
                        
                        # Look for Crunchbase URLs in organic results
                        organic_results = data.get("organic_results", [])
                        logger.debug(f"Found {len(organic_results)} organic results")
                        
                        for i, result in enumerate(organic_results):
                            link = result.get("link", "")
                            title = result.get("title", "")
                            logger.debug(f"  {i+1}. {title[:50]}... - {link}")
                            
                            if "crunchbase.com/organization/" in link:
                                # Clean up the URL
                                crunchbase_url = link.split("?")[0]  # Remove query parameters
                                logger.debug(f"✅ Found Crunchbase URL for {company_name}: {crunchbase_url}")
                                return crunchbase_url
                        
                        # Also check knowledge graph if available
                        knowledge_graph = data.get("knowledgeGraph", {})
                        if knowledge_graph:
                            logger.debug(f"Knowledge graph found: {knowledge_graph.keys()}")
                            kg_links = knowledge_graph.get("links", [])
                            for link_obj in kg_links:
                                if isinstance(link_obj, dict):
                                    link = link_obj.get("link", "")
                                    if "crunchbase.com/organization/" in link:
                                        crunchbase_url = link.split("?")[0]
                                        logger.debug(f"✅ Found Crunchbase URL in knowledge graph for {company_name}: {crunchbase_url}")
                                        return crunchbase_url
                        
                        logger.debug(f"No Crunchbase URL found with query: {query}")
                    else:
                        logger.warning(f"SerpApi search failed for query '{query}': {response.status}")
                        response_text = await response.text()
                        logger.debug(f"Error response: {response_text[:200]}...")
                        continue
            
            logger.info(f"❌ No Crunchbase URL found for {company_name} after trying {len(search_queries)} search strategies")
            return None
                    
        except Exception as e:
            logger.error(f"Error getting crunchbase URL for {company_name}: {e}")
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
    "sector": "detailed searchable sector description (less than 10 words)",
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
            
            def validate_founding_year(year_value):
                """Validate and clean founding year."""
                if not year_value:
                    return None
                
                current_year = datetime.now().year
                
                try:
                    # Handle string years
                    if isinstance(year_value, str):
                        # Remove any non-numeric characters and convert
                        year_clean = re.sub(r'[^\d]', '', year_value)
                        if not year_clean:
                            return None
                        year_value = int(year_clean)
                    
                    # Handle float years (round down)
                    if isinstance(year_value, float):
                        year_value = int(year_value)
                    
                    # Validate year range
                    if not isinstance(year_value, int):
                        return None
                    
                    if year_value < 1900 or year_value > current_year:
                        logger.warning(f"Invalid founding year {year_value}, must be between 1900 and {current_year}")
                        return None
                    
                    return year_value
                    
                except (ValueError, TypeError):
                    logger.warning(f"Could not parse founding year: {year_value}")
                    return None
            
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
                
                # Get centralized sector description
                sector_description = await sector_description_service.get_sector_description(
                    company_name=company_name,
                    company_description=safe_get("description", ""),
                    website_content=content[:1000] if content else "",
                    additional_context=f"AI Focus: {safe_get('ai_focus', '')}"
                )
                if not sector_description:
                    sector_description = "AI Software Solutions"
                
                company = Company(
                    uuid=f"comp_{hash(company_name)}", # Generate simple UUID
                    name=clean_text(company_name),
                    description=clean_text(safe_get("description", "")),
                    short_description=clean_text(safe_get("short_description", "")),
                    founded_year=validate_founding_year(safe_get("founded_year")),
                    funding_total_usd=funding_total_usd,
                    funding_stage=funding_stage,
                    founders=safe_get("founders", []),
                    investors=safe_get("investors", []),
                    categories=safe_get("categories", []),
                    city=clean_text(safe_get("city", "")),
                    region=clean_text(safe_get("region", "")),
                    country=clean_text(safe_get("country", "")),
                    ai_focus=clean_text(safe_get("ai_focus", "")),
                    sector=sector_description,
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
        """Enhanced deduplication with fuzzy matching and domain similarity."""
        if not companies:
            return []
        
        unique_companies = []
        seen_names = set()
        seen_domains = set()
        seen_descriptions = set()
        
        for company in companies:
            # Normalize company name for comparison
            normalized_name = self._normalize_company_name(company.name)
            
            # Skip if name is too similar to existing ones
            if self._is_similar_name(normalized_name, seen_names):
                continue
            
            # Skip if domain is already seen
            if company.website and self._extract_domain(company.website) in seen_domains:
                continue
            
            # Skip if description is too similar (for cases where same company has different names)
            if company.description and self._is_similar_description(company.description, seen_descriptions):
                continue
            
            unique_companies.append(company)
            seen_names.add(normalized_name)
            if company.website:
                seen_domains.add(self._extract_domain(company.website))
            if company.description:
                seen_descriptions.add(company.description[:100])  # First 100 chars
        
        return unique_companies