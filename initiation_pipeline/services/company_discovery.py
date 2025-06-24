"""Exa-based company discovery service."""

import asyncio
from datetime import datetime
from typing import List, Optional
import json

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


class ExaCompanyDiscovery(CompanyDiscoveryService):
    """Company discovery using Exa search API."""
    
    def __init__(self):
        self.exa = Exa(settings.exa_api_key)
        self.openai = AsyncOpenAI(api_key=settings.openai_api_key)
        self.rate_limiter = RateLimiter(
            max_requests=settings.requests_per_minute,
            time_window=60
        )
    
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
    "founded_year": "YYYY as integer or null",
    "funding_amount_millions": "funding amount in millions USD as number or null",
    "funding_stage": "seed/series-a/series-b/series-c/ipo or null",
    "founders": ["founder names if mentioned"],
    "location": "city, country if mentioned",
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
            
            # Create Company object
            company = Company(
                name=clean_text(result.get("name", "")),
                description=clean_text(result.get("description", "")),
                founded_year=result.get("founded_year"),
                funding_total_usd=result.get("funding_amount_millions"),
                funding_stage=funding_stage,
                founders=result.get("founders", []),
                city=clean_text(result.get("location", "")),
                ai_focus=clean_text(result.get("ai_focus", "")),
                website=website,
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