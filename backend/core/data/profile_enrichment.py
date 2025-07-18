"""LinkedIn profile enrichment service."""

import asyncio
import json
import re
from datetime import datetime
from typing import List, Optional
from urllib.parse import urlparse
import requests
import logging

from apify_client import ApifyClient
from openai import AsyncOpenAI
from pydantic import BaseModel

from ..interfaces import ProfileEnrichmentService
from ..config import settings
from ...utils.data_processing import clean_text
from ...utils.rate_limiter import RateLimiter
from ...utils.validators import validate_linkedin_url
from ...models import Company, LinkedInProfile, MediaCoverageData, FinancialProfileData

logger = logging.getLogger(__name__)
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
        self.perplexity = AsyncOpenAI(
            api_key=settings.perplexity_api_key,
            base_url="https://api.perplexity.ai"
        )
        self.rate_limiter = RateLimiter(
            max_requests=settings.requests_per_minute,
            time_window=60
        )
        # Slow down SerpAPI to 10 requests per minute to avoid blocks
        self.search_rate_limiter = RateLimiter(max_requests=10, time_window=60)
        # Perplexity rate limiter
        self.perplexity_rate_limiter = RateLimiter(max_requests=20, time_window=60)
    
    async def find_profiles(self, company: Company) -> List[LinkedInProfile]:
        """Find LinkedIn profiles for company executives."""
        logger.info(f"🔍 Finding LinkedIn profiles for {company.name}")
        
        if not company.name:
            return []
        
        try:
            # Add overall timeout to prevent hanging
            profiles = await asyncio.wait_for(
                self._find_profiles_with_timeout(company),
                timeout=60.0  # 1 minute max per company
            )
            logger.info(f"✅ Found {len(profiles)} LinkedIn profiles for {company.name}")
            return profiles
        except asyncio.TimeoutError:
            logger.warning(f"⏰ Profile search timeout for {company.name}")
            return []
        except Exception as e:
            logger.error(f"❌ Profile search failed for {company.name}: {e}")
            return []
    
    async def _find_profiles_with_timeout(self, company: Company) -> List[LinkedInProfile]:
        """Internal method with timeout protection."""
        # Use different strategies based on available founder data
        founder_names = [name.strip() for name in company.founders if name.strip()]        
        if len(founder_names) >= 4:
            # Use known names approach
            result = await self._find_profiles_by_names(company, founder_names)
        else:
            # Use mixed approach
            result = await self._find_profiles_mixed_approach(company, founder_names)
        
        return result
    
    async def enrich_profile(self, profile: LinkedInProfile) -> LinkedInProfile:
        """Enrich single profile - use enrich_profiles_batch for efficiency."""
        return (await self.enrich_profiles_batch([profile]))[0]
    
    async def enrich_profiles_batch(self, profiles: List[LinkedInProfile]) -> List[LinkedInProfile]:
        """Enrich multiple profiles in one Apify call to avoid Docker container overhead."""
        if not profiles:
            return profiles
            
        # Clean URLs and filter valid ones
        valid_profiles = []
        for profile in profiles:
            url = str(profile.linkedin_url).strip()
            cleaned_url = self._fix_linkedin_url(url)
            
            if not validate_linkedin_url(cleaned_url):
                logger.warning(f"Invalid LinkedIn URL for {profile.person_name}: '{cleaned_url}'")
                continue
                
            # Update the profile with the cleaned URL if it was fixed
            if cleaned_url != url:
                profile.linkedin_url = cleaned_url
            valid_profiles.append(profile)
        
        if not valid_profiles:
            return profiles
            
        urls = [str(p.linkedin_url) for p in valid_profiles]
        logger.info(f"Batch enriching {len(urls)} LinkedIn profiles")
        
        try:
            enriched_data_list = await self._scrape_linkedin_profiles_batch(urls)
            
            # Map results back to profiles
            for i, profile in enumerate(valid_profiles):
                if i < len(enriched_data_list) and enriched_data_list[i]:
                    enriched_data = enriched_data_list[i]
                    # Update profile with enriched data
                    for field, value in enriched_data.items():
                        if hasattr(profile, field) and value is not None:
                            setattr(profile, field, value)
                        elif field.startswith(('experience_', 'education_', 'skill_')) and value:
                            # Add individual fields as dynamic attributes for ranking compatibility
                            setattr(profile, field, value)
            
            # Enhance with Perplexity data
            enhanced_profiles = await self._enhance_with_perplexity(valid_profiles)
            
            return enhanced_profiles
            
        except Exception as e:
            logger.error(f"Error batch enriching profiles: {e}")
            return profiles
    
    async def _find_profiles_by_names(
        self, 
        company: Company, 
        founder_names: List[str]
    ) -> List[LinkedInProfile]:
        """Find profiles using known founder names."""
        profiles = []
        
        for name in founder_names[:3]:  # Limit to 3 names instead of 6
            query = f'{name} {company.name} LinkedIn'
            
            try:
                results = await self._search_google(query, limit=2)  # Reduced from 3
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
            for name in known_names[:2]:  # Reduced from 3
                query = f'{name} {company.name} site:linkedin.com/in'
                try:
                    results = await self._search_google(query, limit=2)  # Reduced from 3
                    extracted = await self._extract_profiles_with_names(
                        results, company, [name]
                    )
                    profiles.extend(extracted)
                except Exception as e:
                    logger.error(f"Error searching for {name}: {e}")
        
        # Step 2: General search for company executives
        titles = ["CEO", "Founder"]  # Reduced from 5 titles to 2
        for title in titles:
            query = f'{company.name} {title} LinkedIn'
            
            try:
                results = await self._search_google(query, limit=3)  # Reduced from 5
                extracted = await self._extract_profiles_general(
                    results, company, titles
                )
                profiles.extend(extracted)
                
            except Exception as e:
                logger.error(f"Error searching for {title}: {e}")
                continue
        
        return self._deduplicate_profiles(profiles)
    
    async def _search_google(self, query: str, limit: int = 10) -> List[dict]:
        """Search Google for LinkedIn profiles using SerpApi."""
        logger.debug(f"Searching for: {query}")
        await self.search_rate_limiter.acquire()
        
        url = "https://serpapi.com/search"
        payload = {
            "q": query,
            "num": limit,
            "api_key": settings.serpapi_key,
            "engine": "google"
        }
        headers = {
            "Content-Type": "application/json"
        }
        
        try:
            # Use requests with shorter timeout
            import asyncio
            import functools
            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None, 
                    functools.partial(
                        requests.get, 
                        url, 
                        params=payload, 
                        headers=headers, 
                        timeout=10
                    )
                ),
                timeout=15.0
            )
            
            if response.status_code == 200:
                data = response.json()
                organic_results = data.get("organic_results", [])
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
                logger.error(f"SerpApi API error {response.status_code}: {response.text}")
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
            
            # Add timeout to OpenAI call
            response = await asyncio.wait_for(
                self.openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                ),
                timeout=30.0
            )
            
            content = response.choices[0].message.content
            
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
                url = profile_data.get("url", "").strip()
                name = clean_text(profile_data.get("name", ""))
                
                # Clean up common URL issues
                if url and not url.startswith(('http://', 'https://')):
                    url = 'https://' + url
                
                # Try to fix common URL issues
                url = self._fix_linkedin_url(url)
                
                # Always create the profile - validation happens later in enrich_profile
                profile = LinkedInProfile(
                    person_name=name,
                    linkedin_url=url,
                    company_name=company.name,
                    title=self._extract_title_from_result(url, combined),
                    role="Executive"  # Default role, will be enriched later if needed
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
        You are filtering LinkedIn profiles for a specific company.

        Company Name: {company.name}
        Company Description: {company.description or "N/A"}

        Below are search results. Extract only LinkedIn profiles that:
        - Match the known names provided
        - Are likely to belong to this company
        - Have a valid LinkedIn URL (skip if it's missing or uncertain)
        - Are not duplicates (remove repeated entries by name or URL)

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
            
            # Add timeout to OpenAI call
            response = await asyncio.wait_for(
                self.openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                ),
                timeout=30.0
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
                url = profile_data.get("url", "").strip()
                name = clean_text(profile_data.get("name", ""))
                
                # Clean up common URL issues
                if url and not url.startswith(('http://', 'https://')):
                    url = 'https://' + url
                
                # Try to fix common URL issues
                url = self._fix_linkedin_url(url)
                
                # Always create the profile - validation happens later in enrich_profile
                profile = LinkedInProfile(
                    person_name=name,
                    linkedin_url=url,
                    company_name=company.name,
                    title=self._extract_title_from_result(url, combined),
                    role="Executive"  # Default role, will be enriched later if needed
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
                
            # Extract role from title
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
                title=title,  # Use the extracted title from search results
                role=role
            )
            profiles.append(profile)
        
        return profiles
    
    def _extract_title_from_result(self, url: str, search_results: str) -> str:
        """Extract title from search results for a specific URL."""
        # Simple extraction - look for common patterns before the URL
        lines = search_results.split('\n')
        for i, line in enumerate(lines):
            if url in line and i > 0:
                # Check previous line for title information
                prev_line = lines[i-1].strip()
                if prev_line and len(prev_line) < 100:  # Reasonable title length
                    # Clean up common patterns
                    title = prev_line.replace(' | LinkedIn', '').replace(' - LinkedIn', '')
                    return title
        return "Professional"  # Default title
    
    def _fix_linkedin_url(self, url: str) -> str:
        """Clean and normalize LinkedIn URLs."""
        if not url:
            return ""
        
        # Remove ALL whitespace characters including newlines, tabs, etc.
        url = ''.join(url.split())
        
        # Remove any invisible characters
        import re
        url = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', url)
        
        # Ensure proper protocol
        if url and not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Fix double slashes
        url = url.replace(':///', '://')
        
        # Ensure it's actually a LinkedIn URL
        if url and 'linkedin.com' not in url.lower():
            return ""
        
        return url.strip()
    
    def _deduplicate_profiles(self, profiles: List[LinkedInProfile]) -> List[LinkedInProfile]:
        """Remove duplicate profiles based on URL."""
        logger.debug(f"Deduplicating {len(profiles)} profiles")
        unique_profiles = []
        seen_urls = set()
        
        for i, profile in enumerate(profiles):
            logger.debug(f"Processing profile {i+1}/{len(profiles)}: {profile.person_name}")
            url = str(profile.linkedin_url)
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_profiles.append(profile)
                logger.debug(f"Added unique profile: {profile.person_name}")
            else:
                logger.debug(f"Skipped duplicate profile: {profile.person_name}")
        
        logger.debug(f"Deduplication complete: {len(unique_profiles)} unique profiles")
        return unique_profiles
    
    async def _scrape_linkedin_profiles_batch(self, urls: List[str]) -> List[Optional[dict]]:
        """Scrape multiple LinkedIn profiles in one Apify call."""
        try:
            logger.debug(f"Batch scraping {len(urls)} LinkedIn profiles")
            
            run_input = {"profileUrls": urls}
            
            run = self.apify.actor(settings.linkedin_actor_id).call(
                run_input=run_input
            )
            
            # Wait for completion with timeout
            await asyncio.sleep(30)  # Give it time to complete
            
            items = list(self.apify.dataset(run["defaultDatasetId"]).iterate_items())
            
            results = []
            for i, url in enumerate(urls):
                if i < len(items) and items[i]:
                    logger.debug(f"Successfully scraped profile data for {url}")
                    results.append(self._extract_linkedin_data(items[i]))
                else:
                    logger.warning(f"No data returned from Apify for {url}, using fallback")
                    results.append(self._create_fallback_profile_data(url))
            
            return results
            
        except Exception as e:
            logger.error(f"Error batch scraping LinkedIn profiles: {e}")
            # Return fallback data for all URLs
            return [self._create_fallback_profile_data(url) for url in urls]
    
    def _create_fallback_profile_data(self, url: str) -> dict:
        """Create fallback profile data when scraping fails."""
        return {
            "headline": "Profile data unavailable (Apify scraping failed)",
            "location": None,
            "about": "LinkedIn profile data could not be retrieved",
            "experience_1_title": "Data unavailable",
            "experience_1_company": "Scraping failed",
        }
    
    def _extract_linkedin_data(self, profile_data: dict) -> dict:
        """Extract and structure LinkedIn profile data."""
        if not profile_data:
            return {}
        
        extracted = {
            "title": profile_data.get("headline"),  # Map headline to title field
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
        
        # Add experiences as structured list
        experiences = []
        for exp in profile_data.get("experiences", [])[:5]:
            if exp.get("title") or exp.get("subtitle"):
                experiences.append({
                    "title": exp.get("title", ""),
                    "company": exp.get("subtitle", "")
                })
        extracted["experience"] = experiences
        
        # Add education as structured list
        education = []
        for edu in profile_data.get("educations", [])[:3]:
            if edu.get("title") or edu.get("subtitle"):
                education.append({
                    "school": edu.get("title", ""),
                    "degree": edu.get("subtitle", "")
                })
        extracted["education"] = education
        
        # Add skills as simple list
        skills = []
        for skill in profile_data.get("skills", [])[:5]:
            if skill.get("title"):
                skills.append(skill.get("title", ""))
        extracted["skills"] = skills
        
        # Add individual experience fields that ranking service expects
        for i, exp in enumerate(profile_data.get("experiences", [])[:3]):
            extracted[f"experience_{i+1}_title"] = exp.get("title", "")
            extracted[f"experience_{i+1}_company"] = exp.get("subtitle", "")
        
        # Add individual education fields that ranking service expects
        for i, edu in enumerate(profile_data.get("educations", [])[:2]):
            extracted[f"education_{i+1}_school"] = edu.get("title", "")
            extracted[f"education_{i+1}_degree"] = edu.get("subtitle", "")
        
        # Add individual skill fields that ranking service expects
        for i, skill in enumerate(profile_data.get("skills", [])[:5]):
            if skill.get("title"):
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
    
    async def _enhance_with_perplexity(self, profiles: List[LinkedInProfile]) -> List[LinkedInProfile]:
        """Enhance profiles with Perplexity data for media coverage and financial information."""
        enhanced_profiles = []
        
        for profile in profiles:
            try:
                logger.info(f"Enhancing profile with Perplexity data: {profile.person_name}")
                
                # Get media coverage data
                media_data = await self._get_media_coverage_perplexity(
                    profile.person_name, 
                    profile.company_name
                )
                profile.media_coverage = media_data
                
                # Get financial profile data
                financial_data = await self._get_financial_profile_perplexity(
                    profile.person_name
                )
                profile.financial_profile = financial_data
                
                enhanced_profiles.append(profile)
                
            except Exception as e:
                logger.error(f"Error enhancing profile {profile.person_name} with Perplexity: {e}")
                # Still append the profile without Perplexity data
                enhanced_profiles.append(profile)
        
        return enhanced_profiles
    
    async def _get_media_coverage_perplexity(
        self, 
        founder_name: str, 
        company_name: Optional[str] = None
    ) -> Optional[MediaCoverageData]:
        """Get media coverage and public presence data using Perplexity with structured JSON."""
        current_company = f" (founder/CEO of {company_name})" if company_name else ""
        
        prompt = f"""Provide comprehensive media coverage and public presence data for {founder_name}{current_company}.
        
        Please include:
        1. Media mentions: total count from major publications, interviews, podcasts, speaking engagements
        2. Awards and recognitions: business awards, industry recognitions, honors received
        3. Thought leadership: speaking engagements, conferences, articles authored, books, keynotes
        4. Social media presence: approximate total follower count across all platforms
        5. Thought leadership score: rate 1-10 based on industry influence and visibility
        6. Overall sentiment: "positive", "neutral", "negative", or "mixed"
        
        Focus on verified, factual information with specific metrics where available."""
        
        try:
            await self.perplexity_rate_limiter.acquire()
            
            response = await self.perplexity.chat.completions.create(
                model="sonar",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"schema": MediaCoverageData.model_json_schema()},
                },
            )
            
            content = response.choices[0].message.content
            data = json.loads(content)
            return MediaCoverageData(**data)
            
        except Exception as e:
            logger.error(f"Error getting media coverage from Perplexity for {founder_name}: {e}")
            return MediaCoverageData()  # Return empty model
    
    async def _get_financial_profile_perplexity(
        self, 
        founder_name: str
    ) -> Optional[FinancialProfileData]:
        """Get financial profile information using Perplexity with structured JSON."""
        
        prompt = f"""Provide comprehensive financial profile information for {founder_name}.
        
        Please include:
        1. Companies founded: list of company names and brief details (e.g., "TechCorp - Founded 2015, AI startup, $50M valuation")
        2. Investment activities: list of investment details (e.g., "Invested $100k in StartupX (2020)", "Angel investor in 15+ companies")
        3. Board positions: list of board positions (e.g., "Board member at TechFoundation since 2018", "Advisory board - FinanceAI")
        4. Notable achievements: list of achievements and recognitions
        5. Estimated net worth: provide range or specific amount if publicly known
        6. Confidence level: "high", "medium", "low" based on data availability
        
        Focus on verified, factual information. Format as simple strings in lists."""
        
        try:
            await self.perplexity_rate_limiter.acquire()
            
            response = await self.perplexity.chat.completions.create(
                model="sonar",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"schema": FinancialProfileData.model_json_schema()},
                },
            )
            
            content = response.choices[0].message.content
            data = json.loads(content)
            return FinancialProfileData(**data)
            
        except Exception as e:
            logger.error(f"Error getting financial profile from Perplexity for {founder_name}: {e}")
            return FinancialProfileData()  # Return empty model