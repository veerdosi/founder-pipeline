"""LinkedIn profile enrichment service."""

import asyncio
import re
from datetime import datetime
from typing import List, Optional
from urllib.parse import urlparse
import requests

from apify_client import ApifyClient
from openai import AsyncOpenAI
from pydantic import BaseModel

from ..core import (
    ProfileEnrichmentService,
    get_logger,
    RateLimiter,
    clean_text,
    settings
)
from ..validators import validate_linkedin_url
from ..models import Company, LinkedInProfile


logger = get_logger(__name__)


class ProfileSearchResult(BaseModel):
    """Search result for profile discovery."""
    name: str
    url: str


class ProfileListResult(BaseModel):
    """List of profile search results."""
    linkedin: List[ProfileSearchResult]


class LinkedInEnrichmentService(ProfileEnrichmentService):
    """LinkedIn profile enrichment using multiple providers."""
    
    def __init__(self):
        self.apify = ApifyClient(settings.apify_api_key)
        self.openai = AsyncOpenAI(api_key=settings.openai_api_key)
        self.rate_limiter = RateLimiter(
            max_requests=settings.requests_per_minute,
            time_window=60
        )
        # Slow down SerpAPI to 10 requests per minute to avoid blocks
        self.search_rate_limiter = RateLimiter(max_requests=10, time_window=60)
    
    async def find_profiles(self, company: Company) -> List[LinkedInProfile]:
        """Find LinkedIn profiles for company executives."""
        logger.info(f"🔍 Finding LinkedIn profiles for {company.name}")
        
        if not company.name:
            return []
        
        # Use different strategies based on available founder data
        founder_names = [name.strip() for name in company.founders if name.strip()]
        
        if len(founder_names) >= 4:
            # Use known names approach
            profiles = await self._find_profiles_by_names(company, founder_names)
        else:
            # Use mixed approach
            profiles = await self._find_profiles_mixed_approach(company, founder_names)
        
        logger.info(f"✅ Found {len(profiles)} LinkedIn profiles for {company.name}")
        return profiles
    
    async def enrich_profile(self, profile: LinkedInProfile) -> LinkedInProfile:
        """Enrich profile with additional data from LinkedIn."""
        if not validate_linkedin_url(profile.linkedin_url):
            logger.warning(f"Invalid LinkedIn URL: {profile.linkedin_url}")
            return profile
        
        try:
            enriched_data = await self._scrape_linkedin_profile(str(profile.linkedin_url))
            if enriched_data:
                # Update profile with enriched data
                for field, value in enriched_data.items():
                    if hasattr(profile, field) and value:
                        setattr(profile, field, value)
            
            return profile
            
        except Exception as e:
            logger.error(f"Error enriching profile {profile.linkedin_url}: {e}")
            return profile
    
    async def _find_profiles_by_names(
        self, 
        company: Company, 
        founder_names: List[str]
    ) -> List[LinkedInProfile]:
        """Find profiles using known founder names."""
        profiles = []
        
        for name in founder_names[:6]:  # Limit to 6 names
            query = f'{name} {company.name} LinkedIn'
            
            try:
                results = await self._search_google(query, limit=3)
                extracted = await self._extract_profiles_with_names(
                    results, company, [name]
                )
                profiles.extend(extracted)
                
            except Exception as e:
                logger.error(f"Error searching for {name}: {e}")
                continue
        
        return self._deduplicate_profiles(profiles)
    
    async def _find_profiles_mixed_approach(
        self, 
        company: Company, 
        known_names: List[str]
    ) -> List[LinkedInProfile]:
        """Find profiles using mixed approach."""
        profiles = []
        
        # Step 1: Search for known names if any
        if known_names:
            for name in known_names[:3]:
                query = f'{name} {company.name} site:linkedin.com/in'
                try:
                    results = await self._search_google(query, limit=3)
                    extracted = await self._extract_profiles_with_names(
                        results, company, [name]
                    )
                    profiles.extend(extracted)
                except Exception as e:
                    logger.error(f"Error searching for {name}: {e}")
        
        # Step 2: General search for company executives
        titles = ["CEO", "CTO", "Co-founder", "Founder", "President"]
        for title in titles[:3]:
            query = f'{company.name} {title} LinkedIn'
            
            try:
                results = await self._search_google(query, limit=5)
                extracted = await self._extract_profiles_general(
                    results, company, titles
                )
                profiles.extend(extracted)
                
            except Exception as e:
                logger.error(f"Error searching for {title}: {e}")
                continue
        
        return self._deduplicate_profiles(profiles)
    
    async def _search_google(self, query: str, limit: int = 10) -> List[dict]:
        """Search Google for LinkedIn profiles using Serper API."""
        logger.debug(f"Searching for: {query}")
        await self.search_rate_limiter.acquire()
        
        url = "https://google.serper.dev/search"
        payload = {
            "q": query,
            "num": limit
        }
        headers = {
            "X-API-KEY": settings.serper_api_key,
            "Content-Type": "application/json"
        }
        
        try:
            # Use requests with timeout
            import asyncio
            import functools
            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None, 
                    functools.partial(
                        requests.post, 
                        url, 
                        json=payload, 
                        headers=headers, 
                        timeout=30
                    )
                ),
                timeout=35.0
            )
            
            if response.status_code == 200:
                data = response.json()
                organic_results = data.get("organic", [])
                logger.debug(f"Got {len(organic_results)} organic results for query: {query}")
                
                linkedin_results = [
                    {
                        "title": r.get("title", ""),
                        "link": r.get("link", ""),
                        "snippet": r.get("snippet", "")
                    }
                    for r in organic_results
                    if "linkedin.com/in/" in r.get("link", "")
                ]
                logger.debug(f"Filtered to {len(linkedin_results)} LinkedIn results")
                return linkedin_results
            else:
                logger.error(f"Serper API error {response.status_code}: {response.text}")
                return []
        except asyncio.TimeoutError:
            logger.warning(f"Search timeout for query: {query}")
            return []
        except Exception as e:
            logger.error(f"Search error for query '{query}': {e}")
            return []
    
    async def _extract_profiles_with_names(
        self, 
        results: List[dict], 
        company: Company, 
        known_names: List[str]
    ) -> List[LinkedInProfile]:
        """Extract profiles using known names."""
        if not results:
            logger.debug(f"No search results for {company.name} with known names")
            return []
        
        logger.debug(f"Processing {len(results)} search results for {company.name}")
        
        combined = "\n".join(
            f"{r['title']} - {r['link']}\n{r['snippet']}" 
            for r in results
        )
        
        if not combined.strip():
            logger.debug(f"Empty combined results for {company.name}")
            return []
        
        prompt = f"""
You are filtering LinkedIn profiles for a specific company.

Company Name: {company.name}
Company Description: {company.description or "N/A"}
Known founder names: {', '.join(known_names)}

Below are search results. Extract only LinkedIn profiles that:
- Match the known names provided
- Are likely to belong to this company

Search Results:
{combined}

Return a JSON object with this structure:
{{
    "linkedin": [
        {{"name": "Person Name", "url": "https://linkedin.com/in/profile"}}
    ]
}}
"""
        
        try:
            await self.rate_limiter.acquire()
            
            response = await self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            logger.debug(f"OpenAI response for profiles with names: {content}")
            
            if not content or content.strip() == "":
                logger.warning("Empty response from OpenAI for profiles with names")
                return []
            
            # Strip markdown code blocks if present
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]  # Remove ```json
            if content.endswith("```"):
                content = content[:-3]  # Remove ```
            content = content.strip()
            
            import json
            result = json.loads(content)
            
            profiles = []
            for profile_data in result.get("linkedin", []):
                profile = LinkedInProfile(
                    person_name=clean_text(profile_data.get("name", "")),
                    linkedin_url=profile_data.get("url", ""),
                    company_name=company.name,
                    role=self._extract_role_from_title(profile_data.get("name", ""))
                )
                profiles.append(profile)
            
            return profiles
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for profiles with names. Content: '{content}'. Error: {e}")
            # Fallback: extract profiles directly from results
            return self._extract_profiles_fallback(results, company.name)
        except Exception as e:
            logger.error(f"Error extracting profiles with names: {e}")
            # Fallback: extract profiles directly from results  
            return self._extract_profiles_fallback(results, company.name)
    
    async def _extract_profiles_general(
        self, 
        results: List[dict], 
        company: Company, 
        titles: List[str]
    ) -> List[LinkedInProfile]:
        """Extract profiles using general search."""
        if not results:
            logger.debug(f"No search results for {company.name} general search")
            return []
        
        logger.debug(f"Processing {len(results)} general search results for {company.name}")
        
        combined = "\n".join(
            f"{r['title']} - {r['link']}\n{r['snippet']}" 
            for r in results
        )
        
        if not combined.strip():
            logger.debug(f"Empty combined results for {company.name} general search")
            return []
        
        prompt = f"""
You are filtering LinkedIn profiles for company executives.

Company Name: {company.name}
Company Description: {company.description or "N/A"}
Target roles: {', '.join(titles)}

Below are search results. Extract only LinkedIn profiles that:
- Are likely employees of this company
- Hold leadership roles like the ones mentioned

Search Results:
{combined}

Return a JSON object with this structure:
{{
    "linkedin": [
        {{"name": "Person Name", "url": "https://linkedin.com/in/profile"}}
    ]
}}
"""
        
        try:
            await self.rate_limiter.acquire()
            
            response = await self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            logger.debug(f"OpenAI response for profiles general: {content}")
            
            if not content or content.strip() == "":
                logger.warning("Empty response from OpenAI for profiles general")
                return []
            
            # Strip markdown code blocks if present
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]  # Remove ```json
            if content.endswith("```"):
                content = content[:-3]  # Remove ```
            content = content.strip()
            
            import json
            result = json.loads(content)
            
            profiles = []
            for profile_data in result.get("linkedin", []):
                profile = LinkedInProfile(
                    person_name=clean_text(profile_data.get("name", "")),
                    linkedin_url=profile_data.get("url", ""),
                    company_name=company.name,
                    role=self._extract_role_from_title(profile_data.get("name", ""))
                )
                profiles.append(profile)
            
            return profiles
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for profiles general. Content: '{content}'. Error: {e}")
            # Fallback: extract profiles directly from results
            return self._extract_profiles_fallback(results, company.name)
        except Exception as e:
            logger.error(f"Error extracting profiles general: {e}")
            # Fallback: extract profiles directly from results
            return self._extract_profiles_fallback(results, company.name)
    
    def _extract_profiles_fallback(self, results: List[dict], company_name: str) -> List[LinkedInProfile]:
        """Fallback method to extract profiles without OpenAI."""
        profiles = []
        
        for result in results:
            title = result.get("title", "")
            link = result.get("link", "")
            snippet = result.get("snippet", "")
            
            if "linkedin.com/in/" not in link:
                continue
                
            # Extract name from title (usually first part before " - ")
            name_parts = title.split(" - ")
            if len(name_parts) >= 1:
                name = name_parts[0].strip()
                # Remove common LinkedIn suffixes
                name = re.sub(r'\s+\|\s+LinkedIn.*$', '', name)
                name = re.sub(r'\s+on LinkedIn.*$', '', name)
            else:
                name = title.strip()
            
            if not name or len(name) < 2:
                continue
                
            # Extract role from title or snippet
            role = "Executive"
            title_lower = title.lower()
            if "ceo" in title_lower:
                role = "CEO"
            elif "cto" in title_lower:
                role = "CTO"
            elif "founder" in title_lower or "co-founder" in title_lower:
                role = "Founder"
            elif "president" in title_lower:
                role = "President"
            
            profile = LinkedInProfile(
                person_name=clean_text(name),
                linkedin_url=link,
                company_name=company_name,
                role=role
            )
            profiles.append(profile)
        
        return profiles
    
    def _extract_role_from_title(self, title: str) -> str:
        """Extract role from LinkedIn title."""
        title_lower = title.lower()
        if "ceo" in title_lower or "chief executive" in title_lower:
            return "CEO"
        elif "cto" in title_lower or "chief technology" in title_lower:
            return "CTO"
        elif "founder" in title_lower:
            return "Founder"
        elif "president" in title_lower:
            return "President"
        else:
            return "Executive"
    
    def _deduplicate_profiles(self, profiles: List[LinkedInProfile]) -> List[LinkedInProfile]:
        """Remove duplicate profiles based on URL."""
        unique_profiles = []
        seen_urls = set()
        
        for profile in profiles:
            url = str(profile.linkedin_url)
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_profiles.append(profile)
        
        return unique_profiles
    
    async def _scrape_linkedin_profile(self, url: str) -> Optional[dict]:
        """Scrape LinkedIn profile using Apify."""
        try:
            run_input = {"profileUrls": [url]}
            
            run = self.apify.actor(settings.linkedin_actor_id).call(
                run_input=run_input
            )
            
            # Wait for completion with timeout
            await asyncio.sleep(30)  # Give it time to complete
            
            items = list(self.apify.dataset(run["defaultDatasetId"]).iterate_items())
            
            if items:
                return self._extract_linkedin_data(items[0])
            
            return None
            
        except Exception as e:
            logger.error(f"Error scraping LinkedIn profile {url}: {e}")
            return None
    
    def _extract_linkedin_data(self, profile_data: dict) -> dict:
        """Extract and structure LinkedIn profile data."""
        if not profile_data:
            return {}
        
        extracted = {
            "headline": profile_data.get("headline"),
            "location": profile_data.get("addressWithCountry"),
            "about": profile_data.get("about"),
        }
        
        # Calculate estimated age
        experience_years = []
        education_years = []
        
        for exp in profile_data.get("experiences", []):
            year = self._extract_earliest_year(exp.get("caption", ""))
            if year:
                experience_years.append(year)
        
        for edu in profile_data.get("educations", []):
            year = self._extract_earliest_year(edu.get("caption", ""))
            if year:
                education_years.append(year)
        
        if education_years or experience_years:
            earliest_edu = min(education_years) if education_years else float('inf')
            earliest_exp = min(experience_years) if experience_years else float('inf')
            
            if earliest_edu < earliest_exp:
                extracted["estimated_age"] = datetime.now().year - earliest_edu + 18
            else:
                extracted["estimated_age"] = datetime.now().year - earliest_exp + 22
        
        # Add experiences
        for i, exp in enumerate(profile_data.get("experiences", [])[:5]):
            extracted[f"experience_{i+1}_title"] = exp.get("title", "")
            extracted[f"experience_{i+1}_company"] = exp.get("subtitle", "")
        
        # Add education
        for i, edu in enumerate(profile_data.get("educations", [])[:3]):
            extracted[f"education_{i+1}_school"] = edu.get("title", "")
            extracted[f"education_{i+1}_degree"] = edu.get("subtitle", "")
        
        # Add skills
        for i, skill in enumerate(profile_data.get("skills", [])[:5]):
            extracted[f"skill_{i+1}"] = skill.get("title", "")
        
        return extracted
    
    def _extract_earliest_year(self, caption: str) -> Optional[int]:
        """Extract earliest year from a caption string."""
        if not caption or not isinstance(caption, str):
            return None
        
        year_matches = [
            int(part) for part in caption.split() 
            if part.isdigit() and 1900 <= int(part) <= datetime.now().year
        ]
        
        return min(year_matches) if year_matches else None