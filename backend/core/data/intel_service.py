"""Perplexity AI search service for comprehensive founder web intelligence."""

import asyncio
import aiohttp
from datetime import datetime
from typing import List, Optional, Dict, Any
import json
import logging

from ..ranking.models import FounderWebSearchData, WebSearchResult
from ..config import settings
from ...utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class PerplexitySearchService:
    """Service for conducting intelligent web searches using Perplexity AI."""
    
    def __init__(self):
        self.api_key = getattr(settings, 'perplexity_api_key', None)
        self.base_url = "https://api.perplexity.ai"
        self.rate_limiter = RateLimiter(max_requests=20, time_window=60)  # Perplexity limits
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Search query templates for different data types
        self.query_templates = {
            'financial': [
                "What companies has {founder_name} founded or co-founded? Include founding dates, exit values, and current status.",
                "What are {founder_name}'s major business exits, IPOs, or acquisitions? Include financial details and dates.",
                "What investments has {founder_name} made as an angel investor or venture capitalist?",
                "What board positions does {founder_name} currently hold or has held in the past?"
            ],
            'media': [
                "What major media coverage, interviews, and press mentions has {founder_name} received in the last 5 years?",
                "What awards, recognitions, or industry honors has {founder_name} received?",
                "What speaking engagements, conferences, or thought leadership activities has {founder_name} participated in?",
                "What books, articles, or significant publications has {founder_name} authored or been featured in?"
            ],
            'biographical': [
                "What is {founder_name}'s educational background, including degrees and institutions?",
                "What is {founder_name}'s professional work history before founding companies?",
                "What notable achievements or career milestones has {founder_name} accomplished?",
                "What is {founder_name}'s current role and recent business activities?"
            ]
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def collect_founder_web_intelligence(
        self, 
        founder_name: str,
        current_company: str
    ) -> FounderWebSearchData:
        """Collect comprehensive web intelligence for a founder using Perplexity."""
        logger.info(f"🔍 Collecting web intelligence for {founder_name} using Perplexity")
        
        web_data = FounderWebSearchData(
            founder_name=founder_name,
            last_search_date=datetime.now()
        )
        
        try:
            # Collect data from different categories
            if self.api_key:
                # Use Perplexity API if available
                tasks = [
                    self._search_perplexity_category(founder_name, current_company, 'financial'),
                    self._search_perplexity_category(founder_name, current_company, 'media'),
                    self._search_perplexity_category(founder_name, current_company, 'biographical')
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for i, result in enumerate(results):
                    if not isinstance(result, Exception):
                        category = ['financial', 'media', 'biographical'][i]
                        for search_result in result:
                            search_result.data_type = category
                            web_data.add_search_result(search_result)
                    else:
                        logger.warning(f"Perplexity search failed for category {i}: {result}")
            else:
                logger.warning("Perplexity API key not available, using fallback search")
                # Fallback to web search without Perplexity
                fallback_results = await self._fallback_web_search(founder_name, current_company)
                for result in fallback_results:
                    web_data.add_search_result(result)
            
            # Extract insights from all search results
            web_data.verified_facts = await self._extract_verified_facts(web_data)
            web_data.data_gaps = self._identify_data_gaps(web_data)
            web_data.overall_data_quality = self._calculate_data_quality(web_data)
            
            logger.info(f"✅ Web intelligence collected for {founder_name}: "
                       f"{web_data.total_searches_performed} searches, "
                       f"{len(web_data.verified_facts)} facts verified")
            
        except Exception as e:
            logger.error(f"Error collecting web intelligence for {founder_name}: {e}")
            web_data.overall_data_quality = 0.1
        
        return web_data
    
    async def _search_perplexity_category(
        self, 
        founder_name: str, 
        current_company: str,
        category: str
    ) -> List[WebSearchResult]:
        """Search Perplexity for a specific data category."""
        results = []
        
        try:
            templates = self.query_templates.get(category, [])
            
            for template in templates:
                query = template.format(founder_name=founder_name)
                
                # Rate limiting
                await self.rate_limiter.acquire()
                
                # Make Perplexity API call
                response_data = await self._call_perplexity_api(query)
                
                if response_data:
                    # Convert Perplexity response to WebSearchResult
                    search_result = await self._convert_perplexity_response(
                        query, response_data, category
                    )
                    if search_result:
                        results.append(search_result)
                
                # Small delay between queries
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error in Perplexity search for {category}: {e}")
        
        return results
    
    async def _call_perplexity_api(self, query: str) -> Optional[Dict[str, Any]]:
        """Make API call to Perplexity."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "sonar-pro",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that provides accurate, factual information with sources. Be specific about dates, numbers, and verifiable details."
                },
                {
                    "role": "user", 
                    "content": query
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.1,
            "top_p": 0.9,
            "search_domain_filter": ["techcrunch.com", "forbes.com", "bloomberg.com", "crunchbase.com", "linkedin.com"],
            "return_citations": True,
            "search_recency_filter": "month",  # Focus on recent information
            "top_k": 0,
            "stream": False,
            "presence_penalty": 0,
            "frequency_penalty": 1
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            ) as response:
                if response.status == 200:
                    try:
                        response_data = await response.json()
                        logger.debug(f"Perplexity API response type: {type(response_data)}")
                        return response_data
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse Perplexity JSON response: {e}")
                        response_text = await response.text()
                        logger.error(f"Raw response: {response_text[:500]}...")
                        return None
                else:
                    error_text = await response.text()
                    logger.error(f"Perplexity API error {response.status}: {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error calling Perplexity API: {e}")
            return None
    
    async def _convert_perplexity_response(
        self, 
        query: str, 
        response_data: Any,
        category: str
    ) -> Optional[WebSearchResult]:
        """Convert Perplexity API response to WebSearchResult."""
        try:
            # Debug: log the response data type and structure
            logger.debug(f"Perplexity response type: {type(response_data)}")
            logger.debug(f"Perplexity response: {str(response_data)[:200]}...")
            
            # Handle case where response_data might be a string
            if isinstance(response_data, str):
                logger.error(f"Perplexity response is a string, not a dict: {response_data[:200]}...")
                return None
            
            # Ensure response_data is a dictionary
            if not isinstance(response_data, dict):
                logger.error(f"Perplexity response is not a dict: {type(response_data)}")
                return None
            
            # Check for required structure
            if 'choices' not in response_data or not response_data['choices']:
                logger.warning(f"No choices in Perplexity response: {response_data.keys()}")
                return None
            
            choice = response_data['choices'][0]
            if not isinstance(choice, dict):
                logger.error(f"Choice is not a dict: {type(choice)}")
                return None
                
            message = choice.get('message', {})
            if not isinstance(message, dict):
                logger.error(f"Message is not a dict: {type(message)}")
                return None
                
            content = message.get('content', '')
            if not isinstance(content, str):
                logger.error(f"Content is not a string: {type(content)}")
                content = str(content)
            
            # Extract citations if available
            citations = []
            if 'citations' in response_data and isinstance(response_data['citations'], list):
                citations = [
                    citation.get('url', '') 
                    for citation in response_data['citations']
                    if isinstance(citation, dict) and citation.get('url')
                ]
            
            # Extract facts from content
            extracted_facts = await self._extract_facts_from_content(content, category)
            
            search_result = WebSearchResult(
                query=query,
                title=f"Perplexity Search: {category.title()} Information",
                url=citations[0] if citations else f"perplexity://search/{hash(query)}",
                snippet=content[:500] + "..." if len(content) > 500 else content,
                source="perplexity",
                search_date=datetime.now(),
                relevance_score=0.9,  # High relevance for Perplexity results
                extracted_facts=extracted_facts,
                data_type=category
            )
            
            return search_result
            
        except Exception as e:
            logger.error(f"Error converting Perplexity response: {e}")
            logger.error(f"Response data type: {type(response_data)}")
            logger.error(f"Response data: {str(response_data)[:500]}...")
            return None
    
    async def _extract_facts_from_content(
        self, 
        content: str, 
        category: str
    ) -> List[str]:
        """Extract structured facts from Perplexity content."""
        facts = []
        
        try:
            # Use simple patterns to extract facts based on category
            if category == 'financial':
                # Look for financial facts
                import re
                
                # Company founding patterns
                founding_matches = re.findall(
                    r'(?:founded|co-founded)\s+([A-Z][a-zA-Z\s&]+?)\s+(?:in\s+)?(\d{4})?',
                    content, re.IGNORECASE
                )
                for match in founding_matches:
                    company, year = match
                    fact = f"Founded {company.strip()}"
                    if year:
                        fact += f" in {year}"
                    facts.append(fact)
                
                # Exit patterns
                exit_matches = re.findall(
                    r'([A-Z][a-zA-Z\s&]+?)\s+(?:acquired|sold|IPO).*?\$?([\d\.]+)\s*(billion|million)',
                    content, re.IGNORECASE
                )
                for match in exit_matches:
                    company, amount, unit = match
                    facts.append(f"{company.strip()} exit: ${amount} {unit}")
                    
            elif category == 'media':
                # Look for media mentions
                import re
                award_matches = re.findall(
                    r'(?:awarded|received|won)\s+([A-Z][a-zA-Z\s&]+?)\s+(?:award|recognition)',
                    content, re.IGNORECASE
                )
                for match in award_matches:
                    facts.append(f"Received {match.strip()}")
                    
            elif category == 'biographical':
                # Look for education and career facts
                import re
                edu_matches = re.findall(
                    r'(?:graduated|degree)\s+(?:from\s+)?([A-Z][a-zA-Z\s&]+?)(?:\s+University|\s+College)',
                    content, re.IGNORECASE
                )
                for match in edu_matches:
                    facts.append(f"Educated at {match.strip()}")
            
            # General fact extraction
            sentences = content.split('.')
            for sentence in sentences[:10]:  # Limit to first 10 sentences
                sentence = sentence.strip()
                if len(sentence) > 20 and len(sentence) < 200:
                    # Filter for sentences that seem factual
                    if any(keyword in sentence.lower() for keyword in 
                          ['founded', 'graduated', 'worked', 'served', 'received', 'awarded', 'invested']):
                        facts.append(sentence)
            
        except Exception as e:
            logger.warning(f"Error extracting facts: {e}")
        
        return facts[:10]  # Limit to 10 most relevant facts
    
    async def _fallback_web_search(
        self, 
        founder_name: str, 
        current_company: str
    ) -> List[WebSearchResult]:
        """Fallback web search when Perplexity is not available."""
        results = []
        
        try:
            # Use Serper API as fallback
            search_queries = [
                f'"{founder_name}" founder CEO exit IPO acquisition',
                f'"{founder_name}" awards recognition media coverage',
                f'"{founder_name}" education career background'
            ]
            
            for i, query in enumerate(search_queries):
                category = ['financial', 'media', 'biographical'][i]
                
                # Use existing search functionality
                search_results = await self._search_web_fallback(query)
                
                for result in search_results[:3]:  # Limit results
                    web_result = WebSearchResult(
                        query=query,
                        title=result.get('title', ''),
                        url=result.get('link', ''),
                        snippet=result.get('snippet', ''),
                        source="google_serper",
                        search_date=datetime.now(),
                        relevance_score=0.6,
                        extracted_facts=[],
                        data_type=category
                    )
                    results.append(web_result)
                
                await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"Error in fallback web search: {e}")
        
        return results
    
    async def _search_web_fallback(self, query: str) -> List[Dict[str, Any]]:
        """Fallback web search using Serper."""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            url = "https://google.serper.dev/search"
            headers = {
                "X-API-KEY": settings.serper_api_key,
                "Content-Type": "application/json"
            }
            
            payload = {
                "q": query,
                "num": 5,
                "gl": "us",
                "hl": "en"
            }
            
            async with self.session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("organic", [])
                else:
                    return []
                    
        except Exception as e:
            logger.error(f"Error in fallback search: {e}")
            return []
    
    async def _extract_verified_facts(self, web_data: FounderWebSearchData) -> List[str]:
        """Extract verified facts from all search results."""
        all_facts = []
        
        # Collect all extracted facts
        for result_list in [web_data.financial_search_results, 
                           web_data.media_search_results, 
                           web_data.biographical_search_results]:
            for result in result_list:
                all_facts.extend(result.extracted_facts)
        
        # Simple verification: facts mentioned in multiple sources
        fact_counts = {}
        for fact in all_facts:
            fact_lower = fact.lower().strip()
            fact_counts[fact_lower] = fact_counts.get(fact_lower, 0) + 1
        
        # Return facts mentioned more than once (verified)
        verified = [fact for fact, count in fact_counts.items() if count > 1]
        return verified[:20]  # Limit to top 20 verified facts
    
    def _identify_data_gaps(self, web_data: FounderWebSearchData) -> List[str]:
        """Identify gaps in the collected data."""
        gaps = []
        
        # Check for missing data types
        if len(web_data.financial_search_results) == 0:
            gaps.append("No financial information found")
        
        if len(web_data.media_search_results) == 0:
            gaps.append("No media coverage found")
        
        if len(web_data.biographical_search_results) == 0:
            gaps.append("Limited biographical information")
        
        # Check for specific missing information
        all_content = " ".join([
            result.snippet for result_list in [
                web_data.financial_search_results,
                web_data.media_search_results,
                web_data.biographical_search_results
            ] for result in result_list
        ]).lower()
        
        if "exit" not in all_content and "acquisition" not in all_content:
            gaps.append("No exit information found")
        
        if "education" not in all_content and "university" not in all_content:
            gaps.append("Educational background unclear")
        
        if len(web_data.verified_facts) < 5:
            gaps.append("Limited verifiable information")
        
        return gaps
    
    def _calculate_data_quality(self, web_data: FounderWebSearchData) -> float:
        """Calculate overall data quality score."""
        score = 0.0
        
        # Base score from number of searches
        if web_data.total_searches_performed > 0:
            score += 0.2
        
        # Score from verified facts
        verified_facts_score = min(len(web_data.verified_facts) * 0.05, 0.3)
        score += verified_facts_score
        
        # Score from data coverage (different categories)
        categories_covered = sum([
            1 for result_list in [web_data.financial_search_results,
                                 web_data.media_search_results,
                                 web_data.biographical_search_results]
            if len(result_list) > 0
        ])
        score += categories_covered * 0.15
        
        # Bonus for using Perplexity (higher quality)
        if "perplexity" in web_data.search_sources_used:
            score += 0.2
        
        # Penalty for data gaps
        score -= len(web_data.data_gaps) * 0.05
        
        return min(max(score, 0.0), 1.0)