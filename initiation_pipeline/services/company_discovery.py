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

from ..core import (
    CompanyDiscoveryService, 
    get_logger, 
    RateLimiter,
    clean_text,
    settings
)
from ..models import Company, FundingStage


logger = get_logger(__name__)


@dataclass
class CompanySource:
    """Configuration for a company discovery source."""
    name: str
    url: str
    source_type: str  # 'media', 'accelerator', 'vc', 'patent', 'stealth'
    check_frequency: int  # minutes
    last_checked: Optional[datetime] = None
    active: bool = True


class ExaCompanyDiscovery(CompanyDiscoveryService):
    """Comprehensive company discovery using 20+ sources from the specification."""
    
    def __init__(self):
        self.exa = Exa(settings.exa_api_key)
        self.openai = AsyncOpenAI(api_key=settings.openai_api_key)
        self.rate_limiter = RateLimiter(
            max_requests=settings.requests_per_minute,
            time_window=60
        )
        self.sources = self._initialize_sources()
        self.session: Optional[aiohttp.ClientSession] = None
    
    def _initialize_sources(self) -> List[CompanySource]:
        """Initialize all 20+ discovery sources as specified in the document."""
        return [
            # Real-time Media Monitoring (every 30 minutes)
            CompanySource("techcrunch", "https://techcrunch.com/category/startups/", "media", 30),
            CompanySource("the_information", "https://www.theinformation.com/", "media", 30),
            CompanySource("axios_pro_rata", "https://www.axios.com/newsletters/axios-pro-rata", "media", 30),
            CompanySource("venturebeat", "https://venturebeat.com/category/ai/", "media", 60),
            CompanySource("bloomberg_tech", "https://www.bloomberg.com/technology", "media", 60),
            CompanySource("wsj_tech", "https://www.wsj.com/tech", "media", 60),
            
            # VC Firm Blogs (daily sweeps)
            CompanySource("a16z_blog", "https://a16z.com/blog/", "vc", 1440),
            CompanySource("sequoia_blog", "https://medium.com/sequoia-capital", "vc", 1440),
            CompanySource("greylock_blog", "https://greylock.com/blog/", "vc", 1440),
            CompanySource("firstround_blog", "https://review.firstround.com/", "vc", 1440),
            
            # Accelerator & Demo Day Tracking (quarterly batch reviews)
            CompanySource("ycombinator", "https://www.ycombinator.com/companies", "accelerator", 10080),  # weekly
            CompanySource("techstars", "https://www.techstars.com/portfolio", "accelerator", 10080),
            CompanySource("500_startups", "https://500.co/companies/", "accelerator", 10080),
            CompanySource("stanford_spinouts", "https://otl.stanford.edu/", "university", 43200),  # monthly
            CompanySource("mit_spinouts", "https://innovation.mit.edu/", "university", 43200),
            
            # Stealth Company Detection
            CompanySource("github_new_projects", "https://github.com/trending", "stealth", 1440),
            CompanySource("linkedin_jobs", "https://www.linkedin.com/jobs/", "stealth", 1440),
            CompanySource("uspto_patents", "https://www.uspto.gov/", "patent", 10080),
            
            # International Sources
            CompanySource("eu_startups", "https://www.eu-startups.com/", "media", 1440),
            CompanySource("tech_asia", "https://www.techinasia.com/", "media", 1440),
            CompanySource("dealstreetasia", "https://www.dealstreetasia.com/", "media", 1440),
            CompanySource("sifted_eu", "https://sifted.eu/", "media", 1440),
        ]
    
    async def discover_companies(
        self,
        limit: int = 50,
        categories: Optional[List[str]] = None,
        regions: Optional[List[str]] = None,
        sources: Optional[List[str]] = None
    ) -> List[Company]:
        """Discover companies using comprehensive multi-source approach."""
        logger.info(f"🔍 Discovering {limit} companies from {len(self.sources)} sources...")
        
        # Filter active sources if specified
        active_sources = self.sources
        if sources:
            active_sources = [s for s in self.sources if s.name in sources]
        
        # Run discovery across all sources
        all_companies = []
        
        # Real-time media monitoring
        media_companies = await self._discover_from_media_sources(
            [s for s in active_sources if s.source_type == "media"],
            categories, regions, limit // 4
        )
        all_companies.extend(media_companies)
        
        # Accelerator discovery
        accelerator_companies = await self._discover_from_accelerators(
            [s for s in active_sources if s.source_type == "accelerator"],
            limit // 4
        )
        all_companies.extend(accelerator_companies)
        
        # Stealth company detection
        stealth_companies = await self._discover_stealth_companies(
            [s for s in active_sources if s.source_type == "stealth"],
            limit // 4
        )
        all_companies.extend(stealth_companies)
        
        # VC portfolio analysis
        vc_companies = await self._discover_from_vc_sources(
            [s for s in active_sources if s.source_type == "vc"],
            limit // 4
        )
        all_companies.extend(vc_companies)
        
        # Deduplicate and return
        unique_companies = self._deduplicate_companies(all_companies)
        logger.info(f"✅ Discovered {len(unique_companies)} unique companies")
        
        return unique_companies[:limit]
    
    async def find_companies(
        self, 
        limit: int = 50,
        categories: Optional[List[str]] = None,
        regions: Optional[List[str]] = None
    ) -> List[Company]:
        """Find AI companies using Exa search."""
        logger.info(f"🔍 Finding {limit} AI companies with Exa...")
        
        categories = categories or settings.ai_categories
        
        # Generate search queries
        queries = self._generate_search_queries(categories, regions)
        
        all_companies = []
        # Increase results per query to account for deduplication
        results_per_query = max(8, (limit * 2) // len(queries))  # 2x multiplier for dedup buffer
        
        # Execute searches
        for i, query in enumerate(queries):
            logger.info(f"  Query {i+1}/{len(queries)}: {query[:50]}...")
            
            try:
                await self.rate_limiter.acquire()
                
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
                
                # Extract company data from results
                for item in result.results:
                    company = await self._extract_company_data(
                        item.text,
                        item.url,
                        item.title
                    )
                    if company:
                        all_companies.append(company)
                        
            except Exception as e:
                logger.error(f"Error with query '{query}': {e}")
                continue
        
        # Deduplicate companies
        unique_companies = self._deduplicate_companies(all_companies)
        logger.info(f"✅ Found {len(unique_companies)} unique AI companies")
        
        return unique_companies[:limit]
    
    def _generate_search_queries(
        self, 
        categories: List[str], 
        regions: Optional[List[str]] = None
    ) -> List[str]:
        """Generate targeted search queries for early-stage AI companies globally."""
        
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
        title: str
    ) -> Optional[Company]:
        """Extract structured company data using GPT-4."""
        prompt = f"""
Extract company information from this content. Return valid JSON only:

Content: {content[:1500]}
Title: {title}

{{
    "name": "exact company name",
    "description": "what the company does in 1-2 sentences",
    "short_description": "brief 1-sentence description",
    "founded_year": "YYYY as integer or null",
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
            
            response = await self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            raw_content = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if raw_content.startswith("```json"):
                raw_content = raw_content[7:]
            if raw_content.startswith("```"):
                raw_content = raw_content[3:]
            if raw_content.endswith("```"):
                raw_content = raw_content[:-3]
            
            raw_content = raw_content.strip()
            
            result = json.loads(raw_content)
            
            # Map funding stage
            funding_stage = None
            if result.get("funding_stage"):
                try:
                    funding_stage = FundingStage(result["funding_stage"])
                except ValueError:
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
            
            # Create Company object
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
            
            return company if company.name else None
            
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