"""Media report and coverage collection service for founder reputation analysis."""

import asyncio
import aiohttp
import re
from datetime import datetime, date
from typing import List, Optional, Dict, Any
import logging
from urllib.parse import urlparse

from ..ranking.models import (
    FounderMediaProfile, MediaMention, Award, ThoughtLeadership, 
    MediaType
)
from ..config import settings
from ...utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class MediaCollector:
    """Service for collecting comprehensive media coverage and public presence data."""
    
    def __init__(self):
        self.rate_limiter = RateLimiter(max_requests=4, time_window=1)  # 4 requests per second to stay under Serper's 5/sec limit
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Media source priorities for determining importance
        self.media_source_weights = {
            'techcrunch.com': 0.9,
            'forbes.com': 0.95,
            'bloomberg.com': 0.9,
            'reuters.com': 0.85,
            'cnbc.com': 0.85,
            'wsj.com': 0.9,
            'ft.com': 0.85,
            'venturebeat.com': 0.7,
            'theinformation.com': 0.8,
            'axios.com': 0.75,
            'businessinsider.com': 0.7,
            'inc.com': 0.6,
            'entrepreneur.com': 0.6,
            'wired.com': 0.65,
            'linkedin.com': 0.5,
            'medium.com': 0.4,
            'twitter.com': 0.3
        }
        
        # Search patterns for different types of media coverage
        self.search_patterns = {
            'news_coverage': [
                '"{founder_name}" interview',
                '"{founder_name}" CEO founder news',
                '"{founder_name}" startup entrepreneur',
                '"{founder_name}" company funding'
            ],
            'awards': [
                '"{founder_name}" award recognition honor',
                '"{founder_name}" entrepreneur of the year',
                '"{founder_name}" 40 under 40',
                '"{founder_name}" Forbes 30 under 30'
            ],
            'thought_leadership': [
                '"{founder_name}" speaker conference keynote',
                '"{founder_name}" authored book article',
                '"{founder_name}" podcast guest',
                '"{founder_name}" TED talk presentation'
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
    
    async def collect_founder_media_profile(
        self, 
        founder_name: str,
        current_company: str
    ) -> FounderMediaProfile:
        """Collect comprehensive media profile for a founder."""
        logger.info(f"📰 Collecting media profile for {founder_name}")
        logger.debug(f"📝 Parameters: company={current_company}")
        
        profile = FounderMediaProfile(
            founder_name=founder_name,
            last_updated=datetime.now()
        )
        
        try:
            # Collect different types of media data in parallel
            logger.debug(f"🚀 Starting parallel media collection tasks for {founder_name}")
            tasks = [
                self._collect_media_mentions(founder_name, current_company),
                self._collect_awards_recognition(founder_name),
                self._collect_thought_leadership(founder_name),
                self._collect_social_media_metrics(founder_name)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            task_names = ["media_mentions", "awards", "thought_leadership", "social_metrics"]
            for i, result in enumerate(results):
                if not isinstance(result, Exception):
                    if i == 0:  # Media mentions
                        profile.media_mentions = result
                        logger.debug(f"✅ {task_names[i]} data: {len(result)} mentions found")
                    elif i == 1:  # Awards
                        profile.awards = result
                        logger.debug(f"✅ {task_names[i]} data: {len(result)} awards found")
                    elif i == 2:  # Thought leadership
                        profile.thought_leadership = result
                        logger.debug(f"✅ {task_names[i]} data: {len(result)} activities found")
                    elif i == 3:  # Social media metrics
                        if result:
                            profile.twitter_followers = result.get('twitter_followers')
                            profile.linkedin_connections = result.get('linkedin_connections')
                            logger.debug(f"✅ {task_names[i]} data: Twitter: {profile.twitter_followers}, LinkedIn: {profile.linkedin_connections}")
                        else:
                            logger.debug(f"✅ {task_names[i]} data: No social metrics found")
                else:
                    logger.error(f"❌ Task {task_names[i]} failed for {founder_name}: {result}")
            
            # Calculate derived metrics
            logger.debug(f"📊 Calculating media metrics for {founder_name}")
            profile.calculate_metrics()
            
            # Set data sources and confidence
            profile.data_sources = ['web_search', 'social_media_apis', 'news_apis']
            profile.confidence_score = self._calculate_media_confidence(profile)
            
            logger.info(f"✅ Media profile collected for {founder_name}: "
                       f"{len(profile.media_mentions)} mentions, "
                       f"{len(profile.awards)} awards, "
                       f"{len(profile.thought_leadership)} thought leadership activities, "
                       f"confidence: {profile.confidence_score:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Error collecting media profile for {founder_name}: {e}", exc_info=True)
            profile.confidence_score = 0.1
        
        return profile
    
    async def _collect_media_mentions(
        self, 
        founder_name: str, 
        current_company: str
    ) -> List[MediaMention]:
        """Collect media mentions and news coverage."""
        mentions = []
        
        try:
            search_queries = self.search_patterns['news_coverage']
            
            for query_template in search_queries:
                query = query_template.format(founder_name=founder_name)
                
                # Search for media coverage
                search_results = await self._search_media_sources(query)
                
                for result in search_results:
                    mention = await self._parse_media_mention(result, founder_name)
                    if mention:
                        mentions.append(mention)
                
                await asyncio.sleep(0.5)  # Rate limiting - reduced delay but using rate limiter
            
            # Deduplicate mentions by URL
            unique_mentions = self._deduplicate_mentions(mentions)
            
            # Sort by importance and recency
            unique_mentions.sort(
                key=lambda x: (x.importance_score, x.publication_date or date.min), 
                reverse=True
            )
            
            return unique_mentions[:50]  # Limit to top 50 mentions
            
        except Exception as e:
            logger.error(f"Error collecting media mentions for {founder_name}: {e}")
            return []
    
    async def _collect_awards_recognition(self, founder_name: str) -> List[Award]:
        """Collect awards and recognition."""
        awards = []
        
        try:
            search_queries = self.search_patterns['awards']
            
            for query_template in search_queries:
                query = query_template.format(founder_name=founder_name)
                
                search_results = await self._search_media_sources(query)
                
                for result in search_results:
                    award = await self._parse_award(result, founder_name)
                    if award:
                        awards.append(award)
                
                await asyncio.sleep(0.5)
            
            # Deduplicate awards
            unique_awards = self._deduplicate_awards(awards)
            return unique_awards
            
        except Exception as e:
            logger.error(f"Error collecting awards for {founder_name}: {e}")
            return []
    
    async def _collect_thought_leadership(self, founder_name: str) -> List[ThoughtLeadership]:
        """Collect thought leadership activities."""
        activities = []
        
        try:
            search_queries = self.search_patterns['thought_leadership']
            
            for query_template in search_queries:
                query = query_template.format(founder_name=founder_name)
                
                search_results = await self._search_media_sources(query)
                
                for result in search_results:
                    activity = await self._parse_thought_leadership(result, founder_name)
                    if activity:
                        activities.append(activity)
                
                await asyncio.sleep(0.5)
            
            return activities
            
        except Exception as e:
            logger.error(f"Error collecting thought leadership for {founder_name}: {e}")
            return []
    
    async def _collect_social_media_metrics(self, founder_name: str) -> Optional[Dict[str, int]]:
        """Collect social media follower counts and metrics."""
        try:
            # This would integrate with Twitter API, LinkedIn API, etc.
            # For now, we'll use web scraping approximations
            
            metrics = {}
            
            # Search for social media profile information
            social_query = f'"{founder_name}" twitter linkedin followers'
            search_results = await self._search_media_sources(social_query)
            
            for result in search_results:
                snippet = result.get('snippet', '').lower()
                title = result.get('title', '').lower()
                text = f"{title} {snippet}"
                
                # Extract follower counts using regex
                twitter_matches = re.findall(r'(\d+(?:,\d+)?)\s*(?:twitter\s*)?followers', text)
                linkedin_matches = re.findall(r'(\d+(?:,\d+)?)\s*(?:linkedin\s*)?connections', text)
                
                if twitter_matches:
                    try:
                        count = int(twitter_matches[0].replace(',', ''))
                        if 'twitter_followers' not in metrics or count > metrics['twitter_followers']:
                            metrics['twitter_followers'] = count
                    except ValueError:
                        pass
                
                if linkedin_matches:
                    try:
                        count = int(linkedin_matches[0].replace(',', ''))
                        if 'linkedin_connections' not in metrics or count > metrics['linkedin_connections']:
                            metrics['linkedin_connections'] = count
                    except ValueError:
                        pass
            
            return metrics if metrics else None
            
        except Exception as e:
            logger.error(f"Error collecting social media metrics for {founder_name}: {e}")
            return None
    
    async def _search_media_sources(self, query: str) -> List[Dict[str, Any]]:
        """Search media sources using web search API."""
        await self.rate_limiter.acquire()
        
        logger.debug(f"📰 Searching media sources with query: {query}")
        
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
                "num": 10,
                "gl": "us",
                "hl": "en",
                "type": "search"
            }
            
            logger.debug(f"📡 Making API request to {url} with payload: {payload}")
            
            async with self.session.post(url, json=payload, headers=headers) as response:
                logger.debug(f"📡 API response status: {response.status}")
                if response.status == 200:
                    data = await response.json()
                    results = data.get("organic", [])
                    logger.debug(f"✅ Found {len(results)} search results for media query")
                    return results
                else:
                    response_text = await response.text()
                    logger.error(f"❌ Search API error {response.status} for query: {query}. Response: {response_text}")
                    return []
                    
        except aiohttp.ClientError as e:
            logger.error(f"❌ HTTP client error searching media sources: {e}")
            return []
        except Exception as e:
            logger.error(f"❌ Unexpected error searching media sources: {e}", exc_info=True)
            return []
    
    async def _parse_media_mention(
        self, 
        search_result: Dict[str, Any], 
        founder_name: str
    ) -> Optional[MediaMention]:
        """Parse a search result into a MediaMention."""
        try:
            title = search_result.get('title', '')
            url = search_result.get('link', '')
            snippet = search_result.get('snippet', '')
            
            # Extract publication name from URL
            domain = urlparse(url).netloc.lower().replace('www.', '')
            publication = domain.split('.')[0].title() if domain else "Unknown"
            
            # Determine media type
            media_type = self._classify_media_type(title, snippet, url)
            
            # Extract publication date if possible
            pub_date = self._extract_date_from_text(snippet)
            
            # Calculate importance score based on source
            importance_score = self.media_source_weights.get(domain, 0.3)
            
            # Determine sentiment (basic keyword analysis)
            sentiment = self._analyze_sentiment(title, snippet)
            
            mention = MediaMention(
                title=title,
                publication=publication,
                media_type=media_type,
                publication_date=pub_date,
                url=url,
                summary=snippet,
                sentiment=sentiment,
                importance_score=importance_score,
                verification_sources=[url]
            )
            
            return mention
            
        except Exception as e:
            logger.warning(f"Error parsing media mention: {e}")
            return None
    
    async def _parse_award(
        self, 
        search_result: Dict[str, Any], 
        founder_name: str
    ) -> Optional[Award]:
        """Parse a search result into an Award."""
        try:
            title = search_result.get('title', '')
            snippet = search_result.get('snippet', '')
            url = search_result.get('link', '')
            
            # Extract award information using patterns
            text = f"{title} {snippet}"
            
            award_patterns = [
                r'(?:won|received|awarded|named)\s+([^.]+?(?:award|recognition|honor))',
                r'([^.]+?(?:entrepreneur|leader|innovator|award))\s+(?:of\s+the\s+year|award)',
                r'(Forbes\s+30\s+Under\s+30|40\s+Under\s+40|Time\s+100)',
            ]
            
            award_name = None
            for pattern in award_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    award_name = match.group(1).strip()
                    break
            
            if not award_name:
                return None
            
            # Extract awarding organization
            org_patterns = [
                r'(?:by|from)\s+([A-Z][a-zA-Z\s&]+?)(?:\s|,|\.)',
                r'(Forbes|Time|Fortune|Inc|TechCrunch|Bloomberg)',
            ]
            
            awarding_org = "Unknown"
            for pattern in org_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    awarding_org = match.group(1).strip()
                    break
            
            # Extract date
            award_date = self._extract_date_from_text(text)
            
            award = Award(
                award_name=award_name,
                awarding_organization=awarding_org,
                award_date=award_date,
                description=snippet,
                verification_sources=[url]
            )
            
            return award
            
        except Exception as e:
            logger.warning(f"Error parsing award: {e}")
            return None
    
    async def _parse_thought_leadership(
        self, 
        search_result: Dict[str, Any], 
        founder_name: str
    ) -> Optional[ThoughtLeadership]:
        """Parse a search result into ThoughtLeadership."""
        try:
            title = search_result.get('title', '')
            snippet = search_result.get('snippet', '')
            url = search_result.get('link', '')
            
            text = f"{title} {snippet}".lower()
            
            # Determine activity type
            activity_type = "speaking"
            if any(keyword in text for keyword in ['book', 'authored', 'published']):
                activity_type = "book"
            elif any(keyword in text for keyword in ['article', 'blog', 'post']):
                activity_type = "blog"
            elif any(keyword in text for keyword in ['podcast', 'interview']):
                activity_type = "podcast"
            elif any(keyword in text for keyword in ['keynote', 'speaker', 'conference']):
                activity_type = "keynote"
            elif any(keyword in text for keyword in ['panel', 'discussion']):
                activity_type = "panel"
            
            # Extract venue or publication
            venue_patterns = [
                r'(?:at|on)\s+([A-Z][a-zA-Z\s&]+?)(?:\s+conference|\s+summit|\s+event)',
                r'(?:in|on)\s+(Forbes|TechCrunch|Harvard Business Review|Medium)',
            ]
            
            venue = "Unknown"
            for pattern in venue_patterns:
                match = re.search(pattern, title + " " + snippet, re.IGNORECASE)
                if match:
                    venue = match.group(1).strip()
                    break
            
            # Extract date
            activity_date = self._extract_date_from_text(snippet)
            
            leadership = ThoughtLeadership(
                activity_type=activity_type,
                title=title,
                venue_or_publication=venue,
                date=activity_date,
                url=url,
                verification_sources=[url]
            )
            
            return leadership
            
        except Exception as e:
            logger.warning(f"Error parsing thought leadership: {e}")
            return None
    
    def _classify_media_type(self, title: str, snippet: str, url: str) -> MediaType:
        """Classify the type of media mention."""
        text = f"{title} {snippet}".lower()
        
        if any(keyword in text for keyword in ['interview', 'speaks with', 'sits down']):
            return MediaType.INTERVIEW
        elif any(keyword in text for keyword in ['podcast', 'show', 'episode']):
            return MediaType.PODCAST
        elif any(keyword in text for keyword in ['award', 'honor', 'recognition']):
            return MediaType.AWARD
        elif any(keyword in text for keyword in ['speaker', 'keynote', 'conference']):
            return MediaType.SPEAKING_ENGAGEMENT
        elif any(keyword in text for keyword in ['opinion', 'analysis', 'perspective']):
            return MediaType.THOUGHT_LEADERSHIP
        else:
            return MediaType.NEWS_ARTICLE
    
    def _extract_date_from_text(self, text: str) -> Optional[date]:
        """Extract date from text using various patterns."""
        try:
            import re
            from datetime import datetime
            
            # Common date patterns
            date_patterns = [
                r'(\d{4})-(\d{1,2})-(\d{1,2})',  # YYYY-MM-DD
                r'(\d{1,2})/(\d{1,2})/(\d{4})',  # MM/DD/YYYY
                r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})',
                r'(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})'
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    try:
                        if len(groups) == 3:
                            if groups[0].isdigit() and groups[1].isdigit() and groups[2].isdigit():
                                # Numeric date
                                year, month, day = int(groups[2]), int(groups[0]), int(groups[1])
                                if year < 100:
                                    year += 2000
                                return date(year, month, day)
                            else:
                                # Month name format
                                month_names = {
                                    'january': 1, 'february': 2, 'march': 3, 'april': 4,
                                    'may': 5, 'june': 6, 'july': 7, 'august': 8,
                                    'september': 9, 'october': 10, 'november': 11, 'december': 12
                                }
                                if groups[0].lower() in month_names:
                                    month = month_names[groups[0].lower()]
                                    day = int(groups[1])
                                    year = int(groups[2])
                                else:
                                    day = int(groups[0])
                                    month = month_names[groups[1].lower()]
                                    year = int(groups[2])
                                return date(year, month, day)
                    except (ValueError, KeyError):
                        continue
            
            return None
            
        except Exception:
            return None
    
    def _analyze_sentiment(self, title: str, snippet: str) -> str:
        """Basic sentiment analysis of media mention."""
        text = f"{title} {snippet}".lower()
        
        positive_words = ['success', 'achievement', 'winner', 'award', 'breakthrough', 
                         'innovation', 'leader', 'excellence', 'outstanding', 'praised']
        negative_words = ['controversy', 'criticism', 'scandal', 'failure', 'problem', 
                         'lawsuit', 'decline', 'loss', 'crisis', 'criticized']
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def _deduplicate_mentions(self, mentions: List[MediaMention]) -> List[MediaMention]:
        """Remove duplicate media mentions."""
        unique_mentions = {}
        
        for mention in mentions:
            # Use URL as primary key for deduplication
            key = mention.url or f"{mention.title}_{mention.publication}"
            if key not in unique_mentions:
                unique_mentions[key] = mention
            else:
                # Merge verification sources
                existing = unique_mentions[key]
                existing.verification_sources.extend(mention.verification_sources)
        
        return list(unique_mentions.values())
    
    def _deduplicate_awards(self, awards: List[Award]) -> List[Award]:
        """Remove duplicate awards."""
        unique_awards = {}
        
        for award in awards:
            key = f"{award.award_name}_{award.awarding_organization}".lower()
            if key not in unique_awards:
                unique_awards[key] = award
            else:
                # Merge verification sources
                existing = unique_awards[key]
                existing.verification_sources.extend(award.verification_sources)
        
        return list(unique_awards.values())
    
    def _calculate_media_confidence(self, profile: FounderMediaProfile) -> float:
        """Calculate confidence score for media profile."""
        score = 0.0
        
        # Base score for having data
        if profile.media_mentions:
            score += 0.3
        if profile.awards:
            score += 0.2
        if profile.thought_leadership:
            score += 0.2
        
        # Boost for high-quality sources
        high_quality_mentions = sum(
            1 for mention in profile.media_mentions 
            if mention.importance_score > 0.7
        )
        score += min(high_quality_mentions * 0.05, 0.2)
        
        # Boost for positive sentiment
        if profile.positive_sentiment_ratio > 0.7:
            score += 0.1
        
        # Boost for social media presence
        if profile.twitter_followers and profile.twitter_followers > 1000:
            score += 0.1
        
        return min(score, 1.0)