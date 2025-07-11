"""Perplexity AI search service for comprehensive founder web intelligence."""

import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any
import json
import logging

from ..ranking.models import FounderWebSearchData, WebSearchResult
from .perplexity_base import PerplexityBaseService

logger = logging.getLogger(__name__)


class PerplexitySearchService(PerplexityBaseService):
    """Service for conducting intelligent web searches using Perplexity AI."""
    
    def __init__(self):
        super().__init__()
        
        # Enhanced query templates for web intelligence
        self.query_templates = {
            'financial': [
                """What companies has {founder_name} founded, co-founded, or started? For each company, provide:
                - Company name and founding year
                - {founder_name}'s role and equity stake
                - Current status and valuation
                - Exit details if applicable (IPO, acquisition, etc.)
                - Financial outcomes and transaction values""",
                
                """What are {founder_name}'s major business exits, IPOs, or acquisitions?
                Include:
                - Company names and exit dates
                - Exit values and transaction details
                - Type of exit (IPO, acquisition, merger)
                - {founder_name}'s financial outcome
                - Acquiring companies or market performance""",
                
                """What investments has {founder_name} made as an angel investor or venture capitalist?
                Provide:
                - Portfolio company names and investment amounts
                - Investment dates and rounds participated
                - Current status of investments
                - Notable returns or successful exits
                - Investment thesis and focus areas"""
            ],
            'media': [
                """What major media coverage, interviews, and press mentions has {founder_name} received in the last 5 years?
                Include:
                - Publication names and dates
                - Types of coverage (interviews, features, news)
                - Key topics and themes discussed
                - Media reach and impact
                - Quotes and key insights shared""",
                
                """What awards, recognitions, or industry honors has {founder_name} received?
                Provide:
                - Award names and awarding organizations
                - Years received and criteria
                - Significance within the industry
                - Associated media coverage
                - Impact on reputation and credibility""",
                
                """What speaking engagements, conferences, or thought leadership activities has {founder_name} participated in?
                Include:
                - Event names and organizers
                - Dates and locations
                - Topics presented
                - Audience size and composition
                - Key insights shared"""
            ],
            'biographical': [
                """What is {founder_name}'s educational background and early career?
                Include:
                - Universities attended and degrees earned
                - Graduation years and academic achievements
                - Early career positions and companies
                - Notable mentors or influences
                - Skills and expertise developed""",
                
                """What is {founder_name}'s professional work history before founding companies?
                Provide:
                - Company names and positions held
                - Employment dates and responsibilities
                - Key achievements and contributions
                - Industries and sectors worked in
                - Skills and experience gained""",
                
                """What notable achievements or career milestones has {founder_name} accomplished?
                Include:
                - Major professional accomplishments
                - Industry recognition and impact
                - Innovation and breakthrough contributions
                - Leadership roles and responsibilities
                - Personal and professional growth milestones"""
            ]
        }
    
    async def collect_data(self, founder_name: str, current_company: str) -> FounderWebSearchData:
        """Collect comprehensive web intelligence for a founder using Perplexity AI."""
        return await self.collect_founder_web_intelligence(founder_name, current_company)
    
    def get_query_templates(self) -> Dict[str, List[str]]:
        """Get query templates for web intelligence collection."""
        return self.query_templates
    
    async def collect_founder_web_intelligence(
        self, 
        founder_name: str,
        current_company: str
    ) -> FounderWebSearchData:
        """Collect comprehensive web intelligence for a founder using Perplexity."""
        logger.info(f"🔍 Collecting web intelligence for {founder_name} using Perplexity")
        logger.debug(f"📝 Parameters: company={current_company}, api_key_available={bool(self.api_key)}")
        
        web_data = FounderWebSearchData(
            founder_name=founder_name,
            last_search_date=datetime.now()
        )
        
        try:
            # Collect data from different categories
            if self.api_key:
                # Use Perplexity API if available
                logger.debug(f"🚀 Starting Perplexity search tasks for {founder_name}")
                tasks = [
                    self._search_perplexity_category(founder_name, current_company, 'financial'),
                    self._search_perplexity_category(founder_name, current_company, 'media'),
                    self._search_perplexity_category(founder_name, current_company, 'biographical')
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                categories = ['financial', 'media', 'biographical']
                for i, result in enumerate(results):
                    if not isinstance(result, Exception):
                        category = categories[i]
                        logger.debug(f"✅ {category} search: {len(result)} results found")
                        for search_result in result:
                            search_result.data_type = category
                            web_data.add_search_result(search_result)
                    else:
                        logger.error(f"❌ Perplexity search failed for category {categories[i]}: {result}")
            else:
                logger.warning("⚠️ Perplexity API key not available, using fallback search")
                # Fallback to web search without Perplexity
                fallback_results = await self._fallback_web_search(founder_name, current_company)
                logger.debug(f"🔄 Fallback search returned {len(fallback_results)} results")
                for result in fallback_results:
                    web_data.add_search_result(result)
            
            # Extract insights from all search results
            logger.debug(f"📊 Processing search results for {founder_name}")
            web_data.verified_facts = await self._extract_verified_facts(web_data)
            web_data.data_gaps = self._identify_data_gaps(web_data)
            web_data.overall_data_quality = self._calculate_data_quality(web_data)
            
            logger.info(f"✅ Web intelligence collected for {founder_name}: "
                       f"{web_data.total_searches_performed} searches, "
                       f"{len(web_data.verified_facts)} facts verified, "
                       f"quality: {web_data.overall_data_quality:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Error collecting web intelligence for {founder_name}: {e}", exc_info=True)
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
            
            system_prompt = f"""You are a comprehensive business intelligence specialist focused on {category} information.
            Provide detailed, accurate information with specific dates, numbers, and sources.
            Be thorough and factual, focusing on verifiable information about business leaders and entrepreneurs."""
            
            for template in templates:
                query = template.format(founder_name=founder_name)
                
                # Use base class method for Perplexity API call
                response_data = await self.query_perplexity(
                    query=query,
                    system_prompt=system_prompt,
                    max_tokens=1500
                )
                
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
    
    
    async def _convert_perplexity_response(
        self, 
        query: str, 
        response_data: Any,
        category: str
    ) -> Optional[WebSearchResult]:
        """Convert Perplexity API response to WebSearchResult."""
        try:
            # Use base class method to extract content
            content = self.extract_content_from_response(response_data)
            if not content:
                return None
            
            # Extract citations using base class method
            citations = self.extract_citations_from_response(response_data)
            
            # Use base class method to parse structured data
            structured_data = self.parse_structured_data(content, category)
            extracted_facts = [item.get('content', '') for item in structured_data if item.get('content')]
            
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
                
                await asyncio.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Error in fallback web search: {e}")
        
        return results
    
    async def _search_web_fallback(self, query: str) -> List[Dict[str, Any]]:
        """Fallback web search using Serper."""
        logger.debug(f"🔄 Fallback web search for query: {query}")
        
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
            
            logger.debug(f"📡 Making fallback API request to {url}")
            
            async with self.session.post(url, json=payload, headers=headers) as response:
                logger.debug(f"📡 Fallback API response status: {response.status}")
                if response.status == 200:
                    data = await response.json()
                    results = data.get("organic", [])
                    logger.debug(f"✅ Fallback search returned {len(results)} results")
                    return results
                else:
                    response_text = await response.text()
                    logger.error(f"❌ Fallback search API error {response.status}: {response_text}")
                    return []
                    
        except aiohttp.ClientError as e:
            logger.error(f"❌ HTTP client error in fallback search: {e}")
            return []
        except Exception as e:
            logger.error(f"❌ Unexpected error in fallback search: {e}", exc_info=True)
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