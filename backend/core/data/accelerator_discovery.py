"""Accelerator company discovery service for YC, Techstars, and 500.co AI/ML companies."""

import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any
import json
import aiohttp
import re
import time
import logging

from openai import AsyncOpenAI

from .. import CompanyDiscoveryService, settings
from .company_discovery import ExaCompanyDiscovery
from ...models import Company, FundingStage
from ...utils.data_processing import clean_text
from ..analysis.sector_classification import sector_description_service

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


class AcceleratorCompanyDiscovery(CompanyDiscoveryService):
    """Discovery service for AI/ML companies from YC, Techstars, and 500.co accelerators."""
    
    def __init__(self):
        self.rate_limiter = RateLimiter(
            max_requests=settings.requests_per_minute,
            time_window=60
        )
        self.session: Optional[aiohttp.ClientSession] = None
        # Initialize Exa discovery service to reuse utility methods
        self.exa_discovery = ExaCompanyDiscovery()
    
    async def discover_companies(
        self,
        limit: int = 50,
        categories: Optional[List[str]] = None,
        regions: Optional[List[str]] = None,
        sources: Optional[List[str]] = None
    ) -> List[Company]:
        """Discover AI/ML companies from selected accelerators."""
        logger.info(f"🚀 Starting accelerator discovery: {limit} AI/ML companies...")
        
        all_companies = []
        
        # Determine which accelerators to use (default to all if none specified)
        use_yc = sources is None or "yc" in sources or "ycombinator" in sources
        use_techstars = sources is None or "techstars" in sources
        use_500co = sources is None or "500co" in sources or "500global" in sources
        
        # Collect companies from selected accelerators in parallel
        tasks = []
        
        if use_yc:
            tasks.append(self.discover_yc_ai_companies())
        
        if use_techstars:
            tasks.append(self.discover_techstars_ai_companies())
            
        if use_500co:
            tasks.append(self.discover_500co_ai_companies())
        
        if not tasks:
            logger.warning("No accelerator sources specified")
            return []
        
        # Execute all discovery tasks concurrently
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Map results back to source names
            source_names = []
            if use_yc:
                source_names.append("Y Combinator")
            if use_techstars:
                source_names.append("Techstars")
            if use_500co:
                source_names.append("500 Global")
            
            for i, result in enumerate(results):
                source_name = source_names[i] if i < len(source_names) else "Unknown"
                if isinstance(result, Exception):
                    logger.error(f"Error from {source_name} discovery: {result}")
                elif isinstance(result, list):
                    all_companies.extend(result)
                    logger.info(f"✅ {source_name}: {len(result)} AI/ML companies")
        
        except Exception as e:
            logger.error(f"Error in parallel accelerator discovery: {e}")
            # Fallback to sequential discovery
            if use_yc:
                try:
                    yc_companies = await self.discover_yc_ai_companies()
                    all_companies.extend(yc_companies)
                    logger.info(f"✅ Y Combinator (fallback): {len(yc_companies)} companies")
                except Exception as yc_error:
                    logger.error(f"YC discovery failed: {yc_error}")
            
            if use_techstars:
                try:
                    techstars_companies = await self.discover_techstars_ai_companies()
                    all_companies.extend(techstars_companies)
                    logger.info(f"✅ Techstars (fallback): {len(techstars_companies)} companies")
                except Exception as ts_error:
                    logger.error(f"Techstars discovery failed: {ts_error}")
            
            if use_500co:
                try:
                    companies_500co = await self.discover_500co_ai_companies()
                    all_companies.extend(companies_500co)
                    logger.info(f"✅ 500 Global (fallback): {len(companies_500co)} companies")
                except Exception as co_error:
                    logger.error(f"500 Global discovery failed: {co_error}")
        
        # Deduplicate companies across all accelerators
        logger.info(f"🔄 Deduplicating {len(all_companies)} companies from accelerators...")
        unique_companies = self._deduplicate_companies(all_companies)
        
        # Apply limit to final results
        final_companies = unique_companies[:limit]
        
        logger.info(f"✅ Accelerator discovery complete: {len(final_companies)} unique AI/ML companies")
        return final_companies
    
    async def find_companies(
        self, 
        limit: int = 50,
        categories: Optional[List[str]] = None,
        regions: Optional[List[str]] = None,
        founded_year: Optional[int] = None
    ) -> List[Company]:
        """Implementation of abstract method - delegates to discover_companies."""
        # For accelerator discovery, we ignore categories, regions, and founded_year
        # and use the accelerator-specific discovery method
        return await self.discover_companies(limit=limit)
    
    async def _is_company_alive(self, company_name: str, website: str = None) -> bool:
        """Use existing method from ExaCompanyDiscovery to verify if a company is still active."""
        return await self.exa_discovery._is_company_alive(company_name, website)
    
    async def _get_crunchbase_url(self, company_name: str, website: str = None) -> Optional[str]:
        """Use existing method from ExaCompanyDiscovery to find the crunchbase URL for a company."""
        return await self.exa_discovery._get_crunchbase_url(company_name, website)
    
    async def discover_yc_ai_companies(self) -> List[Company]:
        """Discover AI/ML companies from Y Combinator using unofficial API with specific AI/ML tags."""
        logger.info("🚀 Fetching AI/ML companies from Y Combinator API...")
        
        companies = []
        
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # YC AI/ML specific endpoints
            yc_ai_ml_endpoints = [
                "https://yc-oss.github.io/api/tags/ai.json",
                "https://yc-oss.github.io/api/tags/generative-ai.json", 
                "https://yc-oss.github.io/api/tags/ai-assistant.json",
                "https://yc-oss.github.io/api/tags/ml.json",
                "https://yc-oss.github.io/api/tags/data-science.json",
                "https://yc-oss.github.io/api/tags/data-engineering.json",
                "https://yc-oss.github.io/api/tags/big-data.json",
                "https://yc-oss.github.io/api/tags/conversational-ai.json",
                "https://yc-oss.github.io/api/tags/ai-enhanced-learning.json",
                "https://yc-oss.github.io/api/tags/ai-powered-drug-discovery.json",
                "https://yc-oss.github.io/api/tags/aiops.json",
                "https://yc-oss.github.io/api/tags/data-visualization.json",
                "https://yc-oss.github.io/api/tags/data-labeling.json"
            ]
            
            all_yc_companies = []
            seen_companies = set()  # Track company IDs to avoid duplicates
            
            for endpoint in yc_ai_ml_endpoints:
                try:
                    async with self.session.get(endpoint) as response:
                        if response.status == 200:
                            tag_companies = await response.json()
                            logger.info(f"📊 Fetched {len(tag_companies)} companies from {endpoint.split('/')[-1]}")
                            
                            # Add unique companies only
                            for company in tag_companies:
                                company_id = company.get('id') or company.get('slug') or company.get('name')
                                if company_id and company_id not in seen_companies:
                                    all_yc_companies.append(company)
                                    seen_companies.add(company_id)
                        else:
                            logger.debug(f"Failed to fetch from {endpoint}: {response.status}")
                except Exception as endpoint_error:
                    logger.debug(f"Error fetching from {endpoint}: {endpoint_error}")
                    continue
            
            logger.info(f"📊 Total {len(all_yc_companies)} unique AI/ML companies found from YC")
            
            ai_companies_count = 0
            for yc_company in all_yc_companies:
                try:
                    company = await self._convert_yc_to_company(yc_company)
                    if company:
                        companies.append(company)
                        ai_companies_count += 1
                except Exception as e:
                    logger.warning(f"Error processing YC company {yc_company.get('name', 'Unknown')}: {e}")
                    continue
            
            logger.info(f"✅ Found {ai_companies_count} AI/ML companies from YC")
                
        except Exception as e:
            logger.error(f"Error fetching YC companies: {e}")
        
        return companies
    
    async def discover_techstars_ai_companies(self) -> List[Company]:
        """Discover AI/ML companies from Techstars using unofficial API."""
        logger.info("🚀 Fetching AI/ML companies from Techstars API...")
        
        companies = []
        
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Fetch Techstars AI/ML companies (pre-filtered)
            url = "https://veerdosi.github.io/techstars-api/api/industries/ai-ml.json"
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch Techstars AI/ML companies: {response.status}")
                    return companies
                
                techstars_companies = await response.json()
                logger.info(f"📊 Fetched {len(techstars_companies)} AI/ML companies from Techstars")
                
                for techstars_company in techstars_companies:
                    try:
                        # Filter only active companies
                        if techstars_company.get('status') in ['Active', 'active']:
                            company = await self._convert_techstars_to_company(techstars_company)
                            if company:
                                companies.append(company)
                    except Exception as e:
                        logger.warning(f"Error processing Techstars company {techstars_company.get('name', 'Unknown')}: {e}")
                        continue
                
                logger.info(f"✅ Found {len(companies)} active AI/ML companies from Techstars")
                
        except Exception as e:
            logger.error(f"Error fetching Techstars companies: {e}")
        
        return companies

    async def discover_500co_ai_companies(self) -> List[Company]:
        """Discover AI/ML companies from 500 Global using unofficial API."""
        logger.info("🚀 Fetching AI/ML companies from 500 Global API...")
        
        companies = []
        
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            # Fetch AI/ML companies from 500.co - try multiple industry endpoints
            ai_industry_endpoints = [
                "https://veerdosi.github.io/500co-api/api/industries/aimachine-learning.json",
                "https://veerdosi.github.io/500co-api/api/industries/artificial-intelligence--machine-learning.json",
                "https://veerdosi.github.io/500co-api/api/industries/data--analytics.json"
            ]
            
            all_500co_companies = []
            
            for endpoint in ai_industry_endpoints:
                try:
                    async with self.session.get(endpoint) as response:
                        if response.status == 200:
                            industry_companies = await response.json()
                            all_500co_companies.extend(industry_companies)
                            logger.info(f"📊 Fetched {len(industry_companies)} companies from {endpoint.split('/')[-1]}")
                        else:
                            logger.debug(f"Failed to fetch from {endpoint}: {response.status}")
                except Exception as endpoint_error:
                    logger.debug(f"Error fetching from {endpoint}: {endpoint_error}")
                    continue
            
            # If industry-specific endpoints fail, try fetching all and filtering
            if not all_500co_companies:
                logger.info("📊 Trying to fetch all 500.co companies and filter for AI/ML...")
                url = "https://veerdosi.github.io/500co-api/api/companies/all.json"
                async with self.session.get(url) as response:
                    if response.status == 200:
                        all_companies = await response.json()
                        logger.info(f"📊 Fetched {len(all_companies)} total 500.co companies")

                        # Filter for AI/ML companies
                        ai_ml_keywords = [
                            "ai", "artificial intelligence", "ml", "machine learning", 
                            "deep learning", "neural", "nlp", "natural language processing",
                            "computer vision", "robotics", "automation", "llm", "gpt",
                            "language model", "chatbot", "autonomous", "predictive analytics",
                            "data science", "algorithm", "voice recognition", "image recognition",
                            "data analytics", "big data", "predictive modeling", "statistical analysis"
                        ]
                        
                        for company_500co in all_companies:
                            if self._is_ai_ml_company(company_500co, ai_ml_keywords):
                                all_500co_companies.append(company_500co)
                    else:
                        logger.error(f"Failed to fetch 500.co companies: {response.status}")
                        return companies

            logger.info(f"📊 Total {len(all_500co_companies)} AI/ML companies found from 500 Global")
            
            ai_companies_count = 0
            for company_500co in all_500co_companies:
                try:
                    company = await self._convert_500co_to_company(company_500co)
                    if company:
                        companies.append(company)
                        ai_companies_count += 1
                except Exception as e:
                    logger.warning(f"Error processing 500.co company {company_500co.get('name', 'Unknown')}: {e}")
                    continue
            
            logger.info(f"✅ Found {ai_companies_count} AI/ML companies from 500 Global")

        except Exception as e:
            logger.error(f"Error fetching 500.co companies: {e}")

        return companies
    
    def _is_ai_ml_company(self, company: dict, ai_ml_keywords: List[str]) -> bool:
        """Check if a company is AI/ML/Data Science related based on multiple fields."""
        # Fields to check for AI/ML keywords
        fields_to_check = [
            company.get('one_liner', ''),
            company.get('long_description', ''),
            company.get('description', ''),
            company.get('industry', ''),
            company.get('subindustry', ''),
            ' '.join(company.get('industries', [])),
            ' '.join(company.get('tags', []))
        ]
        
        # Combine all text fields
        combined_text = ' '.join(filter(None, fields_to_check)).lower()
        
        # Check if any AI/ML keyword is present
        for keyword in ai_ml_keywords:
            if keyword in combined_text:
                return True
        
        return False
    
    async def _convert_yc_to_company(self, yc_company: dict) -> Optional[Company]:
        """Convert YC company data to Company object with enhanced data."""
        try:
            # Extract founding year from batch (e.g., "W21" -> 2021)
            founded_year = None
            batch = yc_company.get('batch', '')
            if batch and len(batch) >= 3:
                # Parse YC batch format (W21, S22, etc.)
                year_part = batch[1:]
                if year_part.isdigit():
                    year_num = int(year_part)
                    # Convert 2-digit year to 4-digit year
                    if year_num <= 30:  # Assume 00-30 means 2000-2030
                        founded_year = 2000 + year_num
                    else:  # 31-99 means 1931-1999 (unlikely for YC but just in case)
                        founded_year = 1900 + year_num
            
            # Process website URL
            website = yc_company.get('website', '')
            if website and website.strip():  # Check if not empty/whitespace
                if not website.startswith(('http://', 'https://')):
                    website = f'https://{website}'
            else:
                website = None  # Set to None for empty/invalid websites
            
            # Check if company is still alive
            company_name = yc_company.get('name', '')
            if not await self._is_company_alive(company_name, website):
                logger.debug(f"Filtered out inactive YC company: {company_name}")
                return None
            
            # Get crunchbase URL using Perplexity
            crunchbase_url = await self._get_crunchbase_url(company_name, website)
            
            # Get centralized sector description
            sector_description = await sector_description_service.get_sector_description(
                company_name=company_name,
                company_description=yc_company.get('long_description', ''),
                additional_context=f"Y Combinator company. Industry: {yc_company.get('industry', '')}. Subindustry: {yc_company.get('subindustry', '')}"
            )
            if not sector_description:
                sector_description = "AI Software Solutions"
            
            # Create Company object
            company = Company(
                uuid=f"yc_{yc_company.get('slug', hash(company_name))}",
                name=clean_text(company_name),
                description=clean_text(yc_company.get('long_description', '')),
                short_description=clean_text(yc_company.get('one_liner', '')),
                founded_year=founded_year,
                funding_total_usd=None,  # YC companies don't have funding info in API
                funding_stage=FundingStage.UNKNOWN,
                founders=[],  # YC API doesn't include founder names
                investors=["Y Combinator"],
                categories=yc_company.get('tags', []),
                city="",  # YC API doesn't include city
                region="",  # YC API doesn't include region
                country="",  # YC API doesn't include country
                ai_focus=clean_text(yc_company.get('industry', '')),
                sector=sector_description,
                website=website,
                linkedin_url=None,
                crunchbase_url=crunchbase_url,
                source_url=f"https://www.ycombinator.com/companies/{yc_company.get('slug', '')}",
                extraction_date=datetime.utcnow()
            )
            
            return company
            
        except Exception as e:
            logger.error(f"Error converting YC company {yc_company.get('name', 'Unknown')}: {e}")
            return None
    
    async def _convert_techstars_to_company(self, techstars_company: dict) -> Optional[Company]:
        """Convert Techstars company data to Company object with enhanced data."""
        try:
            # Process website URL
            website = techstars_company.get('website', '')
            if website and website.strip():  # Check if not empty/whitespace
                if not website.startswith(('http://', 'https://')):
                    website = f'https://{website}'
            else:
                website = None  # Set to None for empty/invalid websites
            
            # Parse location into city, region, country
            location = techstars_company.get('location', '')
            city, region, country = "", "", ""
            if location:
                location_parts = [part.strip() for part in location.split(',')]
                if len(location_parts) >= 3:
                    city, region, country = location_parts[0], location_parts[1], location_parts[2]
                elif len(location_parts) == 2:
                    city, country = location_parts[0], location_parts[1]
                elif len(location_parts) == 1:
                    city = location_parts[0]
            
            # Check if company is still alive
            company_name = techstars_company.get('name', '')
            if not await self._is_company_alive(company_name, website):
                logger.debug(f"Filtered out inactive Techstars company: {company_name}")
                return None
            
            # Get crunchbase URL using Perplexity
            crunchbase_url = await self._get_crunchbase_url(company_name, website)
            
            # Get centralized sector description
            sector_description = await sector_description_service.get_sector_description(
                company_name=company_name,
                company_description=techstars_company.get('description', ''),
                additional_context=f"Techstars company. Industry: {techstars_company.get('industry', '')}. Subindustry: {techstars_company.get('subindustry', '')}"
            )
            if not sector_description:
                sector_description = "AI Software Solutions"
            
            # Create Company object
            company = Company(
                uuid=f"techstars_{techstars_company.get('slug', hash(company_name))}",
                name=clean_text(company_name),
                description=clean_text(techstars_company.get('description', '')),
                short_description=clean_text(techstars_company.get('one_liner', '')),
                founded_year=techstars_company.get('founded_year'),
                funding_total_usd=None,  # Techstars API doesn't include funding info
                funding_stage=FundingStage.UNKNOWN,
                founders=[],  # Techstars API doesn't include founder names
                investors=["Techstars"],
                categories=techstars_company.get('tags', []),
                city=clean_text(city),
                region=clean_text(region),
                country=clean_text(country),
                ai_focus=clean_text(techstars_company.get('industry', '')),
                sector=sector_description,
                website=website,
                linkedin_url=None,
                crunchbase_url=crunchbase_url,
                source_url=f"https://www.techstars.com/portfolio/{techstars_company.get('slug', '')}",
                extraction_date=datetime.utcnow()
            )
            
            return company
            
        except Exception as e:
            logger.error(f"Error converting Techstars company {techstars_company.get('name', 'Unknown')}: {e}")
            return None

    async def _convert_500co_to_company(self, company_500co: dict) -> Optional[Company]:
        """Convert 500.co company data to Company object with enhanced data."""
        try:
            # Process website URL
            website = company_500co.get('website', '')
            if website and website.strip():  # Check if not empty/whitespace
                if not website.startswith(('http://', 'https://')):
                    website = f'https://{website}'
            else:
                website = None  # Set to None for empty/invalid websites
            
            # Parse location into city, region, country
            location = company_500co.get('location', '')
            city, region, country = "", "", ""
            if location:
                location_parts = [part.strip() for part in location.split(',')]
                if len(location_parts) >= 3:
                    city, region, country = location_parts[0], location_parts[1], location_parts[2]
                elif len(location_parts) == 2:
                    city, country = location_parts[0], location_parts[1]
                elif len(location_parts) == 1:
                    city = location_parts[0]
            
            # Check if company is still alive
            company_name = company_500co.get('name', '')
            if not await self._is_company_alive(company_name, website):
                logger.debug(f"Filtered out inactive 500.co company: {company_name}")
                return None
            
            # Get crunchbase URL using Perplexity
            crunchbase_url = await self._get_crunchbase_url(company_name, website)

            # Get centralized sector description
            sector_description = await sector_description_service.get_sector_description(
                company_name=company_name,
                company_description=company_500co.get('description', ''),
                additional_context=f"500 Global company. Industry: {company_500co.get('industry', '')}. Subindustry: {company_500co.get('subindustry', '')}"
            )
            if not sector_description:
                sector_description = "AI Software Solutions"

            # Create Company object
            company = Company(
                uuid=f"500co_{company_500co.get('slug', hash(company_name))}",
                name=clean_text(company_name),
                description=clean_text(company_500co.get('description', '')),
                short_description=clean_text(company_500co.get('one_liner', '')),
                founded_year=company_500co.get('founded_year'),
                funding_total_usd=None,  # 500.co API doesn't include funding info
                funding_stage=FundingStage.UNKNOWN,
                founders=[],  # 500.co API doesn't include founder names
                investors=["500 Global"],
                categories=company_500co.get('tags', []),
                city=clean_text(city),
                region=clean_text(region),
                country=clean_text(country),
                ai_focus=clean_text(company_500co.get('industry', '')),
                sector=sector_description,
                website=website,
                linkedin_url=None,
                crunchbase_url=crunchbase_url,
                source_url=f"https://500.co/companies/{company_500co.get('slug', '')}",
                extraction_date=datetime.utcnow()
            )
            
            return company
            
        except Exception as e:
            logger.error(f"Error converting 500.co company {company_500co.get('name', 'Unknown')}: {e}")
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
    
    async def __aenter__(self):
        """Async context manager entry."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
            self.session = None