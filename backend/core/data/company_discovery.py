"""Comprehensive company discovery service implementing 20+ source monitoring."""

import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import json
import aiohttp
import re
from dataclasses import dataclass

from exa_py import Exa

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
        self.rate_limiter = RateLimiter(
            max_requests=settings.requests_per_minute * 2,
            time_window=60
        )
        self.session: Optional[aiohttp.ClientSession] = None
        # Webset management
        self.created_websets = {}  # Store webset IDs by category
        self.webset_polling_active = False
    
    async def discover_companies(
        self,
        limit: int = 50,
        categories: Optional[List[str]] = None,
        regions: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
        founded_year: Optional[int] = None
    ) -> List[Company]:
        """Discover companies using Exa websets for maximum coverage and quality."""
        logger.info(f"🚀 Starting webset-powered discovery: {limit} companies...")
        
        if not settings.webset_enabled:
            raise ValueError("Websets are disabled. Enable websets in configuration to use company discovery.")
        
        companies = await self._discover_companies_via_websets(
            limit=limit,
            categories=categories,
            regions=regions,
            founded_year=founded_year
        )
        
        logger.info(f"✅ Webset discovery complete: {len(companies)} companies found")
        return companies
    
    async def _discover_companies_via_websets(
        self,
        limit: int = 50,
        categories: Optional[List[str]] = None,
        regions: Optional[List[str]] = None,
        founded_year: Optional[int] = None
    ) -> List[Company]:
        """Discover companies using the webset workflow."""
        start_time = time.time()
        logger.info(f"🔍 Starting webset discovery for {limit} companies...")
        
        try:
            # Step 1: Create websets for different categories
            webset_ids = await self._create_websets(
                categories=categories,
                founded_year=founded_year
            )
            
            if not webset_ids:
                logger.error("No websets created - cannot proceed with discovery")
                raise RuntimeError("Failed to create websets for company discovery")
            
            # Step 2: Wait for websets to populate (websets are asynchronous)
            logger.info("⏳ Waiting for websets to populate...")
            print("⏳ Waiting for websets to populate...")
            await asyncio.sleep(10)  # Give websets time to find items
            
            # Step 3: Poll webset items
            webset_items = await self._poll_webset_items(
                webset_ids=webset_ids,
                limit=limit * 2  # Get more items for better filtering
            )
            
            if not webset_items:
                logger.error("No items found in websets after waiting")
                raise RuntimeError("Websets did not return any items for processing")
            
            # Step 4: Process webset items to extract companies
            logger.info(f"🔄 Processing {len(webset_items)} webset items...")
            print(f"🔄 Processing {len(webset_items)} webset items...")
            
            all_companies = []
            
            for i, item in enumerate(webset_items):
                try:
                    print(f"   📄 [{i+1}/{len(webset_items)}] Processing: {item.title[:50]}...")
                    
                    # Use enrichment processing if available
                    company = await self._process_webset_enrichments(
                        webset_item=item,
                        target_year=founded_year
                    )
                    
                    if company:
                        all_companies.append(company)
                        print(f"   ✅ {company.name}")
                        logger.info(f"  ✅ {company.name}")
                
                except Exception as e:
                    logger.debug(f"Failed to process webset item {item.url}: {e}")
                    continue
            
            # Step 5: Setup monitors for continuous updates (optional)
            if settings.webset_monitoring_enabled:
                try:
                    monitor_ids = await self._setup_webset_monitors(
                        webset_ids=webset_ids,
                        founded_year=founded_year
                    )
                    if monitor_ids:
                        logger.info(f"📅 Setup continuous monitoring for {len(monitor_ids)} websets")
                        print(f"📅 Setup continuous monitoring for {len(monitor_ids)} websets")
                except Exception as e:
                    logger.warning(f"Failed to setup webset monitors: {e}")
            
            elapsed_time = time.time() - start_time
            logger.info(f"✅ Webset discovery complete: {len(all_companies)} companies ({elapsed_time:.1f}s)")
            print(f"✅ Webset discovery complete: {len(all_companies)} companies ({elapsed_time:.1f}s)")
            
            return all_companies[:limit]
            
        except Exception as e:
            logger.error(f"Error in webset discovery: {e}")
            print(f"❌ Webset discovery failed: {e}")
            raise RuntimeError(f"Webset discovery failed: {e}") from e
    
    
    async def _create_websets(
        self, 
        categories: Optional[List[str]] = None,
        founded_year: Optional[int] = None
    ) -> Dict[str, str]:
        """Create websets for different AI company categories using Exa Websets API."""
        logger.info("🔧 Creating websets for AI company discovery...")
        
        if not settings.webset_enabled:
            logger.info("Websets disabled, falling back to traditional search")
            return {}
        
        created_websets = {}
        
        # Get search categories from settings or filter by provided categories
        search_categories = settings.webset_search_categories
        if categories:
            # Filter to only requested categories
            search_categories = [
                cat for cat in search_categories 
                if any(req_cat.lower() in cat["name"].lower() for req_cat in categories)
            ]
        
        for category_config in search_categories:
            try:
                category_name = category_config["name"]
                base_query = category_config["query"]
                count = category_config["count"]
                
                # Modify query if specific founded year is requested
                if founded_year:
                    query = f"{base_query} founded in {founded_year} OR established {founded_year} OR started {founded_year}"
                else:
                    query = base_query
                
                logger.info(f"📝 Creating webset for {category_name}: {query[:80]}...")
                
                # Prepare webset creation parameters
                webset_params = {
                    "search": {
                        "query": query,
                        "count": min(count, settings.webset_max_items_per_webset)
                    }
                }
                
                # Add enrichments if enabled
                if settings.webset_enrichment_enabled and settings.webset_enrichments:
                    webset_params["enrichments"] = []
                    for enrichment_config in settings.webset_enrichments:
                        webset_params["enrichments"].append({
                            "description": enrichment_config["description"],
                            "format": enrichment_config["format"]
                        })
                
                # Add external ID for tracking
                webset_params["externalId"] = f"ai_discovery_{category_name}_{founded_year or 'all'}"
                
                await self.rate_limiter.acquire()
                
                # Create webset using Exa API
                webset = self.exa.websets.create(params=webset_params)
                
                if webset and hasattr(webset, 'id'):
                    webset_id = webset.id
                    created_websets[category_name] = webset_id
                    logger.info(f"✅ Created webset {category_name}: {webset_id}")
                    print(f"✅ Created webset {category_name}: {webset_id}")
                else:
                    logger.warning(f"Failed to create webset for {category_name}")
                    print(f"⚠️ Failed to create webset for {category_name}")
                
                # Small delay between creations
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error creating webset for {category_name}: {e}")
                print(f"❌ Error creating webset for {category_name}: {e}")
                continue
        
        # Store created websets for later polling
        self.created_websets.update(created_websets)
        
        logger.info(f"🎯 Created {len(created_websets)} websets: {list(created_websets.keys())}")
        print(f"🎯 Created {len(created_websets)} websets: {list(created_websets.keys())}")
        
        return created_websets
    
    async def _poll_webset_items(
        self, 
        webset_ids: Optional[Dict[str, str]] = None,
        limit: int = 50
    ) -> List[Any]:
        """Poll webset items from created websets and return articles for processing."""
        logger.info("📡 Polling webset items...")
        
        if not webset_ids:
            webset_ids = self.created_websets
        
        if not webset_ids:
            logger.warning("No websets available for polling")
            return []
        
        all_items = []
        items_per_webset = max(1, limit // len(webset_ids))
        
        for category_name, webset_id in webset_ids.items():
            try:
                logger.info(f"📊 Polling webset {category_name} ({webset_id})")
                print(f"📊 Polling webset {category_name} ({webset_id})")
                
                await self.rate_limiter.acquire()
                
                # Get webset items using Exa API
                webset_items = self.exa.websets.get_items(
                    webset_id=webset_id,
                    limit=items_per_webset
                )
                
                if webset_items and hasattr(webset_items, 'items'):
                    items = webset_items.items
                    logger.info(f"  📄 Retrieved {len(items)} items from {category_name}")
                    print(f"  📄 Retrieved {len(items)} items from {category_name}")
                    
                    # Convert webset items to format compatible with existing extraction logic
                    for item in items:
                        # Create a structure similar to Exa search results
                        converted_item = type('WebsetItem', (), {
                            'title': getattr(item, 'title', ''),
                            'url': getattr(item, 'url', ''),
                            'text': getattr(item, 'text', ''),
                            'published_date': getattr(item, 'published_date', None),
                            'enrichments': getattr(item, 'enrichments', {}),
                            'category': category_name  # Add category for tracking
                        })()
                        
                        all_items.append(converted_item)
                else:
                    logger.warning(f"No items found in webset {category_name}")
                    print(f"⚠️ No items found in webset {category_name}")
                
                # Small delay between webset polls
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error polling webset {category_name} ({webset_id}): {e}")
                print(f"❌ Error polling webset {category_name}: {e}")
                continue
        
        # Remove duplicates based on URL (websets should handle this, but extra safety)
        unique_items = []
        seen_urls = set()
        
        for item in all_items:
            if item.url not in seen_urls:
                unique_items.append(item)
                seen_urls.add(item.url)
        
        logger.info(f"✅ Webset polling complete: {len(unique_items)} unique items from {len(webset_ids)} websets")
        print(f"✅ Webset polling complete: {len(unique_items)} unique items")
        
        return unique_items
    
    async def _process_webset_enrichments(
        self, 
        webset_item: Any,
        target_year: Optional[int] = None
    ) -> Optional[Company]:
        """Process webset item with enrichments to extract company data more efficiently."""
        try:
            # Get basic item data
            title = getattr(webset_item, 'title', '')
            url = getattr(webset_item, 'url', '')
            text = getattr(webset_item, 'text', '')
            enrichments = getattr(webset_item, 'enrichments', {})
            category = getattr(webset_item, 'category', 'general')
            
            logger.debug(f"Processing webset item with enrichments: {title[:50]}...")
            
            # If enrichments are available, use them to pre-populate company data
            if enrichments and settings.webset_enrichment_enabled:
                logger.debug(f"Found enrichments: {list(enrichments.keys())}")
                
                # Extract pre-enriched data
                enriched_data = {}
                
                # Process each enrichment based on its description (comprehensive 9-field extraction)
                for enrichment_key, enrichment_value in enrichments.items():
                    if not enrichment_value:
                        continue
                        
                    # Determine what type of data this enrichment contains
                    key_lower = enrichment_key.lower()
                    value_str = str(enrichment_value).strip()
                    
                    if 'company name' in key_lower or key_lower == 'name':
                        if len(value_str) > 2:
                            enriched_data['name'] = value_str
                    elif 'description' in key_lower:
                        if len(value_str) > 10:
                            enriched_data['description'] = value_str
                    elif 'website' in key_lower and ('http' in value_str or '.com' in value_str or '.ai' in value_str or '.io' in value_str):
                        enriched_data['website'] = value_str if value_str.startswith('http') else f'https://{value_str}'
                    elif 'location' in key_lower or 'city' in key_lower:
                        # Parse location string into city, region, country
                        location_parts = [part.strip() for part in value_str.split(',')]
                        if len(location_parts) >= 3:
                            enriched_data['city'] = location_parts[0]
                            enriched_data['region'] = location_parts[1] 
                            enriched_data['country'] = location_parts[2]
                        elif len(location_parts) == 2:
                            enriched_data['city'] = location_parts[0]
                            enriched_data['region'] = location_parts[1]
                        else:
                            enriched_data['city'] = value_str
                    elif 'funding stage' in key_lower or 'stage' in key_lower:
                        enriched_data['funding_stage'] = value_str.lower()
                    elif 'funding' in key_lower and ('usd' in key_lower or 'total' in key_lower or 'raised' in key_lower):
                        # Extract funding amount from various formats
                        import re
                        # Look for patterns like "$5M", "5 million", "$5.2M USD", etc.
                        funding_patterns = [
                            r'\$?(\d+(?:\.\d+)?)\s*(?:million|m)\b',
                            r'\$?(\d+(?:\.\d+)?)\s*(?:billion|b)\b',
                            r'\$(\d+(?:,\d{3})*(?:\.\d+)?)',
                        ]
                        for pattern in funding_patterns:
                            match = re.search(pattern, value_str.lower())
                            if match:
                                amount = float(match.group(1).replace(',', ''))
                                if 'billion' in value_str.lower() or 'b' in value_str.lower():
                                    amount *= 1_000_000_000
                                elif 'million' in value_str.lower() or 'm' in value_str.lower():
                                    amount *= 1_000_000
                                enriched_data['funding_total_usd'] = amount
                                break
                    elif 'founder' in key_lower:
                        # Parse founders list from various formats
                        founders = []
                        if ',' in value_str:
                            founders = [name.strip() for name in value_str.split(',')]
                        elif ' and ' in value_str:
                            founders = [name.strip() for name in value_str.split(' and ')]
                        elif '|' in value_str:
                            founders = [name.strip() for name in value_str.split('|')]
                        else:
                            founders = [value_str.strip()]
                        enriched_data['founders'] = [f for f in founders if len(f) > 2]
                    elif 'investor' in key_lower or 'vc' in key_lower or 'venture capital' in key_lower:
                        # Parse investors list from various formats
                        investors = []
                        if ',' in value_str:
                            investors = [name.strip() for name in value_str.split(',')]
                        elif ' and ' in value_str:
                            investors = [name.strip() for name in value_str.split(' and ')]
                        elif '|' in value_str:
                            investors = [name.strip() for name in value_str.split('|')]
                        else:
                            investors = [value_str.strip()]
                        enriched_data['investors'] = [i for i in investors if len(i) > 2]
                    elif 'crunchbase' in key_lower and 'crunchbase.com' in value_str:
                        enriched_data['crunchbase_url'] = value_str
                
                # Use enriched data to create company with less GPT processing
                company = await self._extract_company_from_enriched_data(
                    text=text,
                    url=url,
                    title=title,
                    enriched_data=enriched_data,
                    target_year=target_year
                )
                
                if company:
                    logger.debug(f"✅ Extracted company from enrichments: {company.name}")
                    return company
            
            # No fallback needed - webset enrichments should provide all required data
            logger.warning(f"No company extracted from enrichments for {url}")
            return None
            
        except Exception as e:
            logger.error(f"Error processing webset enrichments for {url}: {e}")
            return None
    
    async def _extract_company_from_enriched_data(
        self,
        text: str,
        url: str,
        title: str,
        enriched_data: Dict[str, Any],
        target_year: Optional[int] = None
    ) -> Optional[Company]:
        """Extract company data using pre-enriched information with minimal GPT processing."""
        
        # Check if we have minimum required data from enrichments
        if 'name' not in enriched_data:
            logger.warning(f"No company name found in enrichments for {url}")
            return None
        
        # Create company directly from enriched data
        return await self._create_company_from_enriched_data(enriched_data, text, url, title, target_year)
    
    async def _create_company_from_enriched_data(
        self,
        data: Dict[str, Any],
        text: str,
        url: str,
        title: str,
        target_year: Optional[int] = None
    ) -> Optional[Company]:
        """Create Company object from enriched data."""
        try:
            company_name = data.get('name')
            if not company_name:
                return None
            
            # Use target_year if provided, otherwise leave as None for Crunchbase to populate
            founded_year = target_year
            
            # Check if company is still active
            website = data.get('website')
            if not await self._is_company_alive(company_name, website):
                return None
            
            # Use enriched Crunchbase URL if available
            crunchbase_url = data.get('crunchbase_url')
            
            # Get sector description
            sector_description = await sector_description_service.get_sector_description(
                company_name=company_name,
                company_description=data.get('description', ''),
                website_content=text[:1000],
                additional_context=f"Category: {getattr(data, 'category', 'AI')}"
            )
            
            if not sector_description:
                sector_description = "AI Software Solutions"
            
            # Use funding data from enrichments
            funding_stage = None
            funding_stage_raw = data.get('funding_stage')
            if funding_stage_raw:
                stage_mappings = {
                    "pre-seed": FundingStage.PRE_SEED,
                    "seed": FundingStage.SEED,
                    "series a": FundingStage.SERIES_A,
                    "series-a": FundingStage.SERIES_A,
                    "series b": FundingStage.SERIES_B,
                    "series-b": FundingStage.SERIES_B,
                    "series c": FundingStage.SERIES_C,
                    "series-c": FundingStage.SERIES_C,
                }
                funding_stage = stage_mappings.get(funding_stage_raw.lower().strip(), FundingStage.UNKNOWN)
            
            # Use funding amount from enrichments
            funding_total_usd = data.get('funding_total_usd')
            
            company = Company(
                uuid=f"comp_{hash(company_name)}",
                name=clean_text(company_name),
                description=clean_text(data.get('description', '')),
                short_description=clean_text(data.get('description', '')[:100] + '...' if data.get('description') else ''),
                founded_year=founded_year,
                funding_total_usd=funding_total_usd,
                funding_stage=funding_stage,
                founders=data.get('founders', []),
                investors=data.get('investors', []),
                categories=['artificial intelligence'],
                city=clean_text(data.get('city', '')),
                region=clean_text(data.get('region', '')),
                country=clean_text(data.get('country', 'United States')),
                ai_focus='Artificial Intelligence',
                sector=sector_description,
                website=website,
                linkedin_url=None,  # Removed as requested
                crunchbase_url=crunchbase_url,
                source_url=None,  # Removed as requested
                extraction_date=None  # Removed as requested
            )
            
            return company
            
        except Exception as e:
            logger.error(f"Error creating company from enriched data: {e}")
            return None
    
    async def _setup_webset_monitors(
        self,
        webset_ids: Dict[str, str],
        founded_year: Optional[int] = None
    ) -> Dict[str, str]:
        """Setup monitors for websets to enable continuous updates."""
        if not settings.webset_monitoring_enabled:
            logger.info("Webset monitoring disabled")
            return {}
        
        logger.info("🔧 Setting up webset monitors for continuous monitoring...")
        
        monitor_ids = {}
        
        for category_name, webset_id in webset_ids.items():
            try:
                logger.info(f"📅 Creating monitor for webset {category_name} ({webset_id})")
                
                await self.rate_limiter.acquire()
                
                # Create monitor configuration
                monitor_config = {
                    "webset_id": webset_id,
                    "schedule": settings.webset_monitor_cron,  # e.g., "0 */6 * * *" for every 6 hours
                    "search_params": {
                        "count": 20  # Find 20 new items per monitoring cycle
                    }
                }
                
                # Create monitor using Exa API
                monitor = self.exa.websets.create_monitor(
                    webset_id=webset_id,
                    schedule=settings.webset_monitor_cron,
                    search_params=monitor_config["search_params"]
                )
                
                if monitor and hasattr(monitor, 'id'):
                    monitor_id = monitor.id
                    monitor_ids[category_name] = monitor_id
                    logger.info(f"✅ Created monitor for {category_name}: {monitor_id}")
                    print(f"✅ Created monitor for {category_name}: {monitor_id}")
                else:
                    logger.warning(f"Failed to create monitor for {category_name}")
                    print(f"⚠️ Failed to create monitor for {category_name}")
                
                # Small delay between monitor creations
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error creating monitor for {category_name}: {e}")
                print(f"❌ Error creating monitor for {category_name}: {e}")
                continue
        
        if monitor_ids:
            logger.info(f"🎯 Setup {len(monitor_ids)} webset monitors with schedule: {settings.webset_monitor_cron}")
            print(f"🎯 Setup {len(monitor_ids)} webset monitors")
        
        return monitor_ids
    
    async def _get_webset_updates(
        self,
        webset_ids: Dict[str, str],
        since_timestamp: Optional[str] = None
    ) -> List[Any]:
        """Get new items from monitored websets since last update."""
        logger.info("📡 Checking for webset updates...")
        
        all_new_items = []
        
        for category_name, webset_id in webset_ids.items():
            try:
                await self.rate_limiter.acquire()
                
                # Get new items from webset
                params = {"limit": 50}
                if since_timestamp:
                    params["since"] = since_timestamp
                
                webset_items = self.exa.websets.get_items(
                    webset_id=webset_id,
                    **params
                )
                
                if webset_items and hasattr(webset_items, 'items'):
                    new_items = webset_items.items
                    logger.info(f"📄 Found {len(new_items)} new items in {category_name}")
                    
                    # Convert and add category info
                    for item in new_items:
                        converted_item = type('WebsetItem', (), {
                            'title': getattr(item, 'title', ''),
                            'url': getattr(item, 'url', ''),
                            'text': getattr(item, 'text', ''),
                            'published_date': getattr(item, 'published_date', None),
                            'enrichments': getattr(item, 'enrichments', {}),
                            'category': category_name,
                            'updated_at': getattr(item, 'updated_at', None)
                        })()
                        
                        all_new_items.append(converted_item)
                        
            except Exception as e:
                logger.error(f"Error getting updates from webset {category_name}: {e}")
                continue
        
        logger.info(f"✅ Found {len(all_new_items)} total new items across all websets")
        return all_new_items
    
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
        """Simplified company validation - assume companies from websets are valid."""
        # Since webset enrichments provide current company data, we can assume they're active
        # This removes the need for additional Perplexity API calls
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