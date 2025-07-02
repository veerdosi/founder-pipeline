"""Comprehensive 20+ source monitoring system as specified in architecture."""

import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import aiohttp
import logging
from dataclasses import dataclass
from enum import Enum

from exa_py import Exa
from openai import AsyncOpenAI

from ..core import settings, RateLimiter, get_logger
from ..models import Company, FundingStage

logger = get_logger(__name__)


class SourceType(str, Enum):
    """Source types as specified in the architecture."""
    MEDIA = "media"
    ACCELERATOR = "accelerator" 
    VC = "vc"
    PATENT = "patent"
    STEALTH = "stealth"
    UNIVERSITY = "university"


@dataclass
class MonitoringSource:
    """Source configuration for monitoring."""
    name: str
    url: str
    source_type: SourceType
    check_frequency_minutes: int
    active: bool = True
    last_checked: Optional[datetime] = None
    
    # Monitoring configuration
    rate_limit_per_minute: int = 30
    timeout_seconds: int = 60
    custom_headers: Dict[str, str] = None
    
    def __post_init__(self):
        if self.custom_headers is None:
            self.custom_headers = {}


class ComprehensiveSourceMonitor:
    """24/7 monitoring system for 20+ sources as specified."""
    
    def __init__(self):
        self.exa = Exa(settings.exa_api_key)
        self.openai = AsyncOpenAI(api_key=settings.openai_api_key)
        self.rate_limiter = RateLimiter(
            max_requests=settings.requests_per_minute,
            time_window=60
        )
        self.sources = self._initialize_comprehensive_sources()
        self.session: Optional[aiohttp.ClientSession] = None
    
    def _initialize_comprehensive_sources(self) -> List[MonitoringSource]:
        """Initialize all 20+ sources as specified in the document."""
        return [
            # Real-time Media Monitoring (every 30 minutes)
            MonitoringSource("techcrunch", "https://techcrunch.com/category/startups/", SourceType.MEDIA, 30),
            MonitoringSource("the_information", "https://www.theinformation.com/", SourceType.MEDIA, 30),
            MonitoringSource("axios_pro_rata", "https://www.axios.com/newsletters/axios-pro-rata", SourceType.MEDIA, 30),
            MonitoringSource("venturebeat", "https://venturebeat.com/category/ai/", SourceType.MEDIA, 60),
            MonitoringSource("bloomberg_tech", "https://www.bloomberg.com/technology", SourceType.MEDIA, 60),
            MonitoringSource("wsj_tech", "https://www.wsj.com/tech", SourceType.MEDIA, 60),
            
            # VC Firm Blogs (daily sweeps)
            MonitoringSource("a16z_blog", "https://a16z.com/blog/", SourceType.VC, 1440),
            MonitoringSource("sequoia_blog", "https://medium.com/sequoia-capital", SourceType.VC, 1440),
            MonitoringSource("greylock_blog", "https://greylock.com/blog/", SourceType.VC, 1440),
            MonitoringSource("firstround_blog", "https://review.firstround.com/", SourceType.VC, 1440),
            MonitoringSource("nea_blog", "https://www.nea.com/blog", SourceType.VC, 1440),
            MonitoringSource("benchmark_blog", "https://www.benchmark.com/blog/", SourceType.VC, 1440),
            
            # Accelerator & Demo Day Tracking (weekly batch reviews)
            MonitoringSource("ycombinator", "https://www.ycombinator.com/companies", SourceType.ACCELERATOR, 10080),
            MonitoringSource("techstars", "https://www.techstars.com/portfolio", SourceType.ACCELERATOR, 10080),
            MonitoringSource("500_startups", "https://500.co/companies/", SourceType.ACCELERATOR, 10080),
            MonitoringSource("plug_and_play", "https://www.plugandplaytechcenter.com/startups/", SourceType.ACCELERATOR, 10080),
            MonitoringSource("mass_challenge", "https://masschallenge.org/startups", SourceType.ACCELERATOR, 10080),
            
            # University Spinouts (monthly discovery)
            MonitoringSource("stanford_spinouts", "https://otl.stanford.edu/", SourceType.UNIVERSITY, 43200),
            MonitoringSource("mit_spinouts", "https://innovation.mit.edu/", SourceType.UNIVERSITY, 43200),
            MonitoringSource("harvard_spinouts", "https://otd.harvard.edu/", SourceType.UNIVERSITY, 43200),
            MonitoringSource("berkeley_spinouts", "https://ipira.berkeley.edu/", SourceType.UNIVERSITY, 43200),
            
            # Stealth Company Detection (daily)
            MonitoringSource("github_trending", "https://github.com/trending", SourceType.STEALTH, 1440),
            MonitoringSource("linkedin_jobs", "https://www.linkedin.com/jobs/", SourceType.STEALTH, 1440),
            MonitoringSource("uspto_patents", "https://www.uspto.gov/", SourceType.PATENT, 10080),
            
            # International Sources
            MonitoringSource("eu_startups", "https://www.eu-startups.com/", SourceType.MEDIA, 1440),
            MonitoringSource("tech_asia", "https://www.techinasia.com/", SourceType.MEDIA, 1440),
            MonitoringSource("dealstreetasia", "https://www.dealstreetasia.com/", SourceType.MEDIA, 1440),
            MonitoringSource("sifted_eu", "https://sifted.eu/", SourceType.MEDIA, 1440),
            MonitoringSource("tech_eu", "https://tech.eu/", SourceType.MEDIA, 1440),
        ]
    
    async def discover_companies_comprehensive(
        self,
        limit: int = 50,
        categories: Optional[List[str]] = None,
        regions: Optional[List[str]] = None,
        sources: Optional[List[str]] = None
    ) -> List[Company]:
        """Comprehensive company discovery using all 20+ sources."""
        logger.info(f"🔍 Starting comprehensive discovery from {len(self.sources)} sources...")
        
        # Filter sources if specified
        active_sources = [s for s in self.sources if s.active]
        if sources:
            active_sources = [s for s in active_sources if s.name in sources]
        
        all_companies = []
        
        # Process sources by type for optimal efficiency
        source_groups = self._group_sources_by_type(active_sources)
        
        # Real-time media monitoring (priority)
        if SourceType.MEDIA in source_groups:
            media_companies = await self._monitor_media_sources(
                source_groups[SourceType.MEDIA], 
                categories, regions, limit // 4
            )
            all_companies.extend(media_companies)
        
        # Accelerator discovery
        if SourceType.ACCELERATOR in source_groups:
            accelerator_companies = await self._monitor_accelerator_sources(
                source_groups[SourceType.ACCELERATOR], 
                limit // 4
            )
            all_companies.extend(accelerator_companies)
        
        # VC portfolio tracking
        if SourceType.VC in source_groups:
            vc_companies = await self._monitor_vc_sources(
                source_groups[SourceType.VC], 
                limit // 4
            )
            all_companies.extend(vc_companies)
        
        # Stealth company detection
        if SourceType.STEALTH in source_groups:
            stealth_companies = await self._detect_stealth_companies(
                source_groups[SourceType.STEALTH], 
                limit // 4
            )
            all_companies.extend(stealth_companies)
        
        # University spinout monitoring
        if SourceType.UNIVERSITY in source_groups:
            university_companies = await self._monitor_university_sources(
                source_groups[SourceType.UNIVERSITY], 
                limit // 4
            )
            all_companies.extend(university_companies)
        
        # Deduplicate and return
        unique_companies = self._deduplicate_companies(all_companies)
        logger.info(f"✅ Comprehensive discovery complete: {len(unique_companies)} unique companies")
        
        return unique_companies[:limit]
    
    def _group_sources_by_type(self, sources: List[MonitoringSource]) -> Dict[SourceType, List[MonitoringSource]]:
        """Group sources by type for efficient processing."""
        groups = {}
        for source in sources:
            if source.source_type not in groups:
                groups[source.source_type] = []
            groups[source.source_type].append(source)
        return groups
    
    async def _monitor_media_sources(
        self, 
        sources: List[MonitoringSource], 
        categories: Optional[List[str]], 
        regions: Optional[List[str]], 
        limit: int
    ) -> List[Company]:
        """Monitor real-time media sources."""
        logger.info(f"📰 Monitoring {len(sources)} media sources...")
        
        companies = []
        categories = categories or settings.ai_categories
        
        # Generate targeted media queries
        queries = self._generate_media_queries(categories, regions)
        
        for query in queries[:10]:  # Limit queries for efficiency
            try:
                await self.rate_limiter.acquire()
                
                # Use Exa for intelligent content discovery
                result = self.exa.search_and_contents(
                    query,
                    type="neural",
                    use_autoprompt=True,
                    num_results=limit // len(queries[:10]),
                    text={"max_characters": 2000},
                    include_domains=[source.url.split('/')[2] for source in sources]
                )
                
                # Extract companies from results
                for item in result.results:
                    company = await self._extract_company_from_content(
                        item.text, item.url, item.title, "media"
                    )
                    if company:
                        companies.append(company)
                        
            except Exception as e:
                logger.error(f"Media monitoring error for query '{query}': {e}")
                continue
        
        return companies
    
    async def _monitor_accelerator_sources(
        self, 
        sources: List[MonitoringSource], 
        limit: int
    ) -> List[Company]:
        """Monitor accelerator and demo day sources."""
        logger.info(f"🚀 Monitoring {len(sources)} accelerator sources...")
        
        companies = []
        current_year = datetime.now().year
        
        # Accelerator-specific queries
        queries = [
            f"Y Combinator {current_year} batch companies AI startups",
            f"Techstars {current_year} demo day artificial intelligence",
            f"500 Startups portfolio AI machine learning {current_year}",
            f"new accelerator companies {current_year} AI funding",
            f"demo day {current_year} artificial intelligence startups"
        ]
        
        for query in queries:
            try:
                await self.rate_limiter.acquire()
                
                result = self.exa.search_and_contents(
                    query,
                    type="neural",
                    num_results=limit // len(queries),
                    text={"max_characters": 1500},
                    include_domains=[
                        "ycombinator.com", "techstars.com", "500.co", 
                        "masschallenge.org", "plugandplaytechcenter.com"
                    ]
                )
                
                for item in result.results:
                    company = await self._extract_company_from_content(
                        item.text, item.url, item.title, "accelerator"
                    )
                    if company:
                        companies.append(company)
                        
            except Exception as e:
                logger.error(f"Accelerator monitoring error: {e}")
                continue
        
        return companies
    
    async def _monitor_vc_sources(
        self, 
        sources: List[MonitoringSource], 
        limit: int
    ) -> List[Company]:
        """Monitor VC firm blogs and portfolio announcements."""
        logger.info(f"💰 Monitoring {len(sources)} VC sources...")
        
        companies = []
        current_year = datetime.now().year
        
        # VC-specific queries
        queries = [
            f"a16z new investment {current_year} AI startup",
            f"Sequoia portfolio company AI {current_year}",
            f"Greylock investment artificial intelligence {current_year}",
            f"First Round investment AI startup {current_year}",
            f"venture capital new investment AI {current_year}"
        ]
        
        for query in queries:
            try:
                await self.rate_limiter.acquire()
                
                result = self.exa.search_and_contents(
                    query,
                    type="neural",
                    num_results=limit // len(queries),
                    text={"max_characters": 1500},
                    include_domains=[
                        "a16z.com", "medium.com", "greylock.com", 
                        "review.firstround.com", "nea.com", "benchmark.com"
                    ]
                )
                
                for item in result.results:
                    company = await self._extract_company_from_content(
                        item.text, item.url, item.title, "vc_portfolio"
                    )
                    if company:
                        companies.append(company)
                        
            except Exception as e:
                logger.error(f"VC monitoring error: {e}")
                continue
        
        return companies
    
    async def _detect_stealth_companies(
        self, 
        sources: List[MonitoringSource], 
        limit: int
    ) -> List[Company]:
        """Detect stealth companies through GitHub, LinkedIn, and patents."""
        logger.info(f"🕵️ Detecting stealth companies from {len(sources)} sources...")
        
        companies = []
        
        # Stealth detection queries
        queries = [
            "stealth AI startup hiring engineers machine learning",
            "GitHub trending AI projects with significant activity",
            "LinkedIn stealth mode AI company hiring",
            "new AI patents filed startups",
            "early stage AI company building in stealth"
        ]
        
        for query in queries:
            try:
                await self.rate_limiter.acquire()
                
                result = self.exa.search_and_contents(
                    query,
                    type="neural",
                    num_results=limit // len(queries),
                    text={"max_characters": 1200}
                )
                
                for item in result.results:
                    company = await self._extract_company_from_content(
                        item.text, item.url, item.title, "stealth_detection"
                    )
                    if company:
                        companies.append(company)
                        
            except Exception as e:
                logger.error(f"Stealth detection error: {e}")
                continue
        
        return companies
    
    async def _monitor_university_sources(
        self, 
        sources: List[MonitoringSource], 
        limit: int
    ) -> List[Company]:
        """Monitor university technology transfer and spinouts."""
        logger.info(f"🎓 Monitoring {len(sources)} university sources...")
        
        companies = []
        current_year = datetime.now().year
        
        # University spinout queries
        queries = [
            f"Stanford technology transfer AI spinout {current_year}",
            f"MIT AI research commercialization {current_year}",
            f"Harvard university spinout artificial intelligence",
            f"Berkeley AI technology transfer startup {current_year}",
            f"university research AI startup {current_year}"
        ]
        
        for query in queries:
            try:
                await self.rate_limiter.acquire()
                
                result = self.exa.search_and_contents(
                    query,
                    type="neural",
                    num_results=limit // len(queries),
                    text={"max_characters": 1500}
                )
                
                for item in result.results:
                    company = await self._extract_company_from_content(
                        item.text, item.url, item.title, "university_spinout"
                    )
                    if company:
                        companies.append(company)
                        
            except Exception as e:
                logger.error(f"University monitoring error: {e}")
                continue
        
        return companies
    
    def _generate_media_queries(self, categories: List[str], regions: Optional[List[str]]) -> List[str]:
        """Generate targeted media monitoring queries."""
        current_year = datetime.now().year
        base_queries = [
            f"AI startup funding announcement {current_year}",
            f"artificial intelligence company raised seed {current_year}",
            f"machine learning startup series A {current_year}",
            f"new AI company launched {current_year}",
            f"AI startup news funding {current_year}"
        ]
        
        # Add category-specific queries
        for category in categories[:3]:
            base_queries.append(f"{category} startup funding {current_year}")
        
        # Add region-specific queries
        if regions:
            for region in regions[:3]:
                base_queries.append(f"AI startup {region} funding {current_year}")
        
        return base_queries
    
    async def _extract_company_from_content(
        self, 
        content: str, 
        url: str, 
        title: str, 
        source_type: str
    ) -> Optional[Company]:
        """Extract company data using GPT-4 with source-specific prompts."""
        
        prompt = f"""
Extract AI company information from this {source_type} content. Return valid JSON only:

Content: {content[:1200]}
Title: {title}

{{
    "name": "exact company name",
    "description": "what the company does in 1-2 sentences",
    "founded_year": "YYYY as integer or null",
    "funding_amount_millions": "funding amount in millions USD as number or null",
    "funding_stage": "seed/series-a/series-b/series-c/ipo or null",
    "founders": ["founder names if mentioned"],
    "investors": ["investor names if mentioned"],
    "city": "city name only",
    "country": "country name",
    "ai_focus": "specific AI area like NLP, computer vision, robotics, etc",
    "website": "company website if mentioned"
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
            
            # Clean and parse JSON
            if raw_content.startswith("```json"):
                raw_content = raw_content[7:-3]
            
            import json
            result = json.loads(raw_content)
            
            if not result.get("name"):
                return None
            
            # Create Company object
            company = Company(
                uuid=f"comp_{hash(result.get('name', ''))}_{source_type}",
                name=result.get("name", ""),
                description=result.get("description", ""),
                founded_year=result.get("founded_year"),
                funding_total_usd=result.get("funding_amount_millions", 0) * 1_000_000 if result.get("funding_amount_millions") else None,
                funding_stage=FundingStage(result.get("funding_stage", "unknown")) if result.get("funding_stage") else None,
                founders=result.get("founders", []),
                investors=result.get("investors", []),
                city=result.get("city", ""),
                country=result.get("country", ""),
                ai_focus=result.get("ai_focus", ""),
                website=result.get("website"),
                source_url=url,
                extraction_date=datetime.utcnow()
            )
            
            return company
            
        except Exception as e:
            logger.error(f"Failed to extract company from {source_type} content: {e}")
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
    
    async def get_source_status(self) -> List[Dict[str, Any]]:
        """Get status of all monitoring sources."""
        status_list = []
        
        for source in self.sources:
            status = {
                "name": source.name,
                "url": source.url,
                "type": source.source_type.value,
                "active": source.active,
                "check_frequency_minutes": source.check_frequency_minutes,
                "last_checked": source.last_checked.isoformat() if source.last_checked else None,
                "next_check_due": self._calculate_next_check(source)
            }
            status_list.append(status)
        
        return status_list
    
    def _calculate_next_check(self, source: MonitoringSource) -> Optional[str]:
        """Calculate when the next check is due for a source."""
        if not source.last_checked:
            return "immediate"
        
        next_check = source.last_checked + timedelta(minutes=source.check_frequency_minutes)
        if next_check <= datetime.now():
            return "overdue"
        
        return next_check.isoformat()
