"""Media report and coverage collection service for founder reputation analysis using Perplexity AI."""

import asyncio
from datetime import datetime, date
from typing import List, Optional, Dict, Any
import logging
import re
from urllib.parse import urlparse

from ..ranking.models import (
    FounderMediaProfile, MediaMention, Award, ThoughtLeadership, 
    MediaType
)
from .perplexity_base import PerplexityBaseService

logger = logging.getLogger(__name__)


class MediaCollector(PerplexityBaseService):
    """Service for collecting comprehensive media coverage and public presence data."""
    
    def __init__(self):
        super().__init__()
        
        # Media source weights for determining importance
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
        
        # Enhanced query templates for media data
        self.query_templates = {
            'media_coverage': [
                """What major media appearances, interviews, and press coverage has {founder_name} received?
                For each instance, include:
                - Media type (e.g., interview, article, podcast, press release)
                - Publication/platform name and date
                - Title, topic, and context of the appearance
                - {founder_name}'s role and key quotes or insights shared
                - Links to content (if available)
                - Industry impact or significance of the coverage
                Focus on reputable sources like Forbes, Bloomberg, TechCrunch, WSJ, and widely known podcasts or outlets."""
            ],
            
            'awards_recognition': [
                """What awards, honors, and prestigious recognitions has {founder_name} received?
                For each recognition, include:
                - Award or list name, category, and year
                - Issuing organization or publication
                - Criteria or reason for selection
                - Notable peers or recipients (if applicable)
                - Industry significance and media coverage
                Include both general and industry-specific honors (e.g., Forbes 30 Under 30, innovation prizes, entrepreneurial awards)."""
            ],
            
            'thought_leadership': [
                """What thought leadership contributions has {founder_name} made through speaking, publishing, and media?
                Include:
                - Conferences, keynote talks, or panels (event, topic, date, audience)
                - Authored articles, blog posts, or whitepapers (title, platform, date, main themes)
                - Books or publications (title, publisher, topic, impact)
                - Podcasts or expert commentary (platform, host, topics discussed)
                For each, highlight audience reach, key insights, and available links or transcripts."""
            ],
            
            'social_media_presence': [
                """What is {founder_name}'s digital and social media presence?
                Include:
                - Activity and influence on Twitter/X, LinkedIn, Instagram, YouTube, etc.
                - Follower count, engagement metrics, and notable posts or content
                - Personal website or blog traffic and reach
                - Participation in online communities or forums (platforms, roles, recognition)
                Focus on thought leadership, content quality, and public influence."""
            ]
        }

    
    async def collect_data(self, founder_name: str, current_company: str) -> FounderMediaProfile:
        """Collect comprehensive media profile for a founder using Perplexity AI."""
        return await self.collect_founder_media_profile(founder_name, current_company)
    
    def get_query_templates(self) -> Dict[str, List[str]]:
        """Get query templates for media data collection."""
        return self.query_templates
    
    async def collect_founder_media_profile(
        self, 
        founder_name: str,
        current_company: str
    ) -> FounderMediaProfile:
        """Collect comprehensive media profile for a founder using Perplexity AI."""
        logger.debug(f"📰 Collecting media profile for {founder_name} using Perplexity AI")
        logger.debug(f"📝 Parameters: company={current_company}")
        
        profile = FounderMediaProfile(
            founder_name=founder_name,
            last_updated=datetime.now()
        )
        
        try:
            # Collect different types of media data using Perplexity
            logger.debug(f"🚀 Starting Perplexity media collection tasks for {founder_name}")
            tasks = [
                self._collect_media_mentions(founder_name, current_company),
                self._collect_awards_recognition(founder_name),
                self._collect_thought_leadership(founder_name),
                self._collect_social_media_metrics(founder_name)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results):
                if not isinstance(result, Exception):
                    if i == 0:  # Media mentions
                        profile.media_mentions = result
                    elif i == 1:  # Awards
                        profile.awards = result
                    elif i == 2:  # Thought leadership
                        profile.thought_leadership = result
                    elif i == 3:  # Social media metrics
                        if result:
                            profile.twitter_followers = result.get('twitter_followers')
                            profile.linkedin_connections = result.get('linkedin_connections')
                else:
                    logger.warning(f"Media collection task {i} failed for {founder_name}: {result}")
            
            # Calculate derived metrics
            profile.calculate_metrics()
            
            # Set data sources and confidence
            profile.data_sources = ['perplexity_ai', 'web_search', 'media_databases']
            profile.confidence_score = self._calculate_media_confidence(profile)
            
            logger.info(f"✅ Media profile collected for {founder_name}: "
                       f"{len(profile.media_mentions)} mentions, "
                       f"{len(profile.awards)} awards, "
                       f"{len(profile.thought_leadership)} thought leadership activities")
            
        except Exception as e:
            logger.error(f"Error collecting media profile for {founder_name}: {e}")
            profile.confidence_score = 0.1
        
        return profile
    
    async def _collect_media_mentions(
        self, 
        founder_name: str, 
        current_company: str
    ) -> List[MediaMention]:
        """Collect media mentions and news coverage using Perplexity."""
        mentions = []
        
        try:
            system_prompt = """You are a media intelligence specialist focused on tracking press coverage and media mentions.
            Provide comprehensive information about media appearances, interviews, and news coverage.
            Include publication details, dates, and context for each mention."""
            
            for query_template in self.query_templates['media_coverage']:
                query = query_template.format(founder_name=founder_name)
                
                response = await self.query_perplexity(
                    query=query,
                    system_prompt=system_prompt,
                    max_tokens=2000
                )
                
                if response:
                    content = self.extract_content_from_response(response)
                    if content:
                        mentions.extend(self._parse_media_mentions(content, founder_name))
                
                await asyncio.sleep(1)  # Rate limiting
                await asyncio.sleep(1)  # Rate limiting
            
            # Deduplicate mentions by URL and title
            unique_mentions = self._deduplicate_mentions(mentions)
            
            # Sort by importance and recency
            unique_mentions.sort(
                key=lambda x: (x.importance_score, x.publication_date or date.min), 
                reverse=True
            )
            
            logger.debug(f"📊 Found {len(unique_mentions)} unique media mentions for {founder_name}")
            return unique_mentions[:50]  # Limit to top 50 mentions
            
        except Exception as e:
            logger.error(f"Error collecting media mentions for {founder_name}: {e}")
            return []
    
    async def _collect_awards_recognition(self, founder_name: str) -> List[Award]:
        """Collect awards and recognition using Perplexity."""
        awards = []
        
        try:
            system_prompt = """You are an awards and recognition specialist focused on tracking professional honors and achievements.
            Provide detailed information about awards, recognitions, and honors received by business leaders.
            Include award names, organizations, dates, and significance."""
            
            for query_template in self.query_templates['awards_recognition']:
                query = query_template.format(founder_name=founder_name)
                
                response = await self.query_perplexity(
                    query=query,
                    system_prompt=system_prompt,
                    max_tokens=2000
                )
                
                if response:
                    content = self.extract_content_from_response(response)
                    if content:
                        awards.extend(self._parse_awards(content, founder_name))
                
                await asyncio.sleep(1)
                await asyncio.sleep(1)
            
            # Deduplicate awards
            unique_awards = self._deduplicate_awards(awards)
            logger.debug(f"📊 Found {len(unique_awards)} unique awards for {founder_name}")
            return unique_awards
            
        except Exception as e:
            logger.error(f"Error collecting awards for {founder_name}: {e}")
            return []
    
    async def _collect_thought_leadership(self, founder_name: str) -> List[ThoughtLeadership]:
        """Collect thought leadership activities using Perplexity."""
        activities = []
        
        try:
            system_prompt = """You are a thought leadership specialist focused on tracking speaking engagements, publications, and expert commentary.
            Provide comprehensive information about thought leadership activities including conferences, articles, books, and expert opinions.
            Include event details, publication information, and impact metrics."""
            
            for query_template in self.query_templates['thought_leadership']:
                query = query_template.format(founder_name=founder_name)
                
                response = await self.query_perplexity(
                    query=query,
                    system_prompt=system_prompt,
                    max_tokens=2000
                )
                
                if response:
                    content = self.extract_content_from_response(response)
                    if content:
                        activities.extend(self._parse_thought_leadership(content, founder_name))
                
                await asyncio.sleep(1)
                await asyncio.sleep(1)
            
            logger.debug(f"📊 Found {len(activities)} thought leadership activities for {founder_name}")
            return activities
            
        except Exception as e:
            logger.error(f"Error collecting thought leadership for {founder_name}: {e}")
            return []
    
    async def _collect_social_media_metrics(self, founder_name: str) -> Optional[Dict[str, int]]:
        """Collect social media follower counts and metrics using Perplexity."""
        try:
            system_prompt = """You are a social media intelligence specialist focused on tracking online presence and influence.
            Provide specific follower counts, engagement metrics, and social media presence information.
            Include platform-specific metrics and influence indicators."""
            
            for query_template in self.query_templates['social_media_presence']:
                query = query_template.format(founder_name=founder_name)
                
                response = await self.query_perplexity(
                    query=query,
                    system_prompt=system_prompt,
                    max_tokens=1500
                )
                
                if response:
                    content = self.extract_content_from_response(response)
                    if content:
                        metrics = self._parse_social_media_metrics(content, founder_name)
                        if metrics:
                            return metrics
                
                await asyncio.sleep(1)
            
            return None
            
        except Exception as e:
            logger.error(f"Error collecting social media metrics for {founder_name}: {e}")
            return None
    
    def _parse_media_mentions(self, content: str, founder_name: str) -> List[MediaMention]:
        """Parse media mentions from Perplexity response."""
        mentions = []
        
        try:
            # Split content into sentences and paragraphs
            sentences = content.split('.')
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 30:
                    continue
                
                # Look for media mention indicators
                if any(keyword in sentence.lower() for keyword in [
                    'interviewed', 'featured', 'appeared', 'spoke', 'discussed',
                    'published', 'article', 'interview', 'coverage', 'mentioned'
                ]):
                    mention = self._extract_media_mention_from_sentence(sentence, founder_name)
                    if mention:
                        mentions.append(mention)
            
            # Also look for structured information
            mentions.extend(self._extract_structured_mentions(content, founder_name))
            
        except Exception as e:
            logger.warning(f"Error parsing media mentions: {e}")
        
        return mentions
    
    def _parse_awards(self, content: str, founder_name: str) -> List[Award]:
        """Parse awards from Perplexity response."""
        awards = []
        
        try:
            sentences = content.split('.')
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 25:
                    continue
                
                # Look for award indicators
                if any(keyword in sentence.lower() for keyword in [
                    'awarded', 'won', 'received', 'honored', 'recognized',
                    'named', 'selected', 'chosen', 'recipient'
                ]):
                    award = self._extract_award_from_sentence(sentence, founder_name)
                    if award:
                        awards.append(award)
            
            # Extract structured awards
            awards.extend(self._extract_structured_awards(content, founder_name))
            
        except Exception as e:
            logger.warning(f"Error parsing awards: {e}")
        
        return awards
    
    def _parse_thought_leadership(self, content: str, founder_name: str) -> List[ThoughtLeadership]:
        """Parse thought leadership activities from Perplexity response."""
        activities = []
        
        try:
            sentences = content.split('.')
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 25:
                    continue
                
                # Look for thought leadership indicators
                if any(keyword in sentence.lower() for keyword in [
                    'spoke', 'keynote', 'presented', 'authored', 'wrote',
                    'published', 'conference', 'panel', 'podcast', 'guest'
                ]):
                    activity = self._extract_thought_leadership_from_sentence(sentence, founder_name)
                    if activity:
                        activities.append(activity)
            
            # Extract structured activities
            activities.extend(self._extract_structured_thought_leadership(content, founder_name))
            
        except Exception as e:
            logger.warning(f"Error parsing thought leadership: {e}")
        
        return activities
    
    def _parse_social_media_metrics(self, content: str, founder_name: str) -> Optional[Dict[str, int]]:
        """Parse social media metrics from Perplexity response."""
        metrics = {}
        
        try:
            # Extract follower counts using regex
            twitter_patterns = [
                r'(\d+(?:,\d+)*)\s*(?:twitter|x)\s*followers',
                r'twitter.*?(\d+(?:,\d+)*)\s*followers',
                r'(\d+(?:,\d+)*)\s*followers.*?twitter'
            ]
            
            linkedin_patterns = [
                r'(\d+(?:,\d+)*)\s*linkedin\s*connections',
                r'linkedin.*?(\d+(?:,\d+)*)\s*connections',
                r'(\d+(?:,\d+)*)\s*connections.*?linkedin'
            ]
            
            content_lower = content.lower()
            
            for pattern in twitter_patterns:
                match = re.search(pattern, content_lower)
                if match:
                    try:
                        count = int(match.group(1).replace(',', ''))
                        metrics['twitter_followers'] = count
                        break
                    except ValueError:
                        continue
            
            for pattern in linkedin_patterns:
                match = re.search(pattern, content_lower)
                if match:
                    try:
                        count = int(match.group(1).replace(',', ''))
                        metrics['linkedin_connections'] = count
                        break
                    except ValueError:
                        continue
            
            return metrics if metrics else None
            
        except Exception as e:
            logger.warning(f"Error parsing social media metrics: {e}")
            return None
    
    def _extract_media_mention_from_sentence(self, sentence: str, founder_name: str) -> Optional[MediaMention]:
        """Extract media mention from a sentence."""
        try:
            # Extract publication name (simplified)
            words = sentence.split()
            publication = "Unknown"
            
            # Look for known publications
            for word in words:
                word_lower = word.lower().strip('.,')
                if word_lower in ['forbes', 'bloomberg', 'techcrunch', 'reuters', 'wsj', 'cnbc']:
                    publication = word_lower.title()
                    break
            
            # Determine media type
            media_type = MediaType.NEWS_ARTICLE
            if 'interview' in sentence.lower():
                media_type = MediaType.INTERVIEW
            elif 'podcast' in sentence.lower():
                media_type = MediaType.PODCAST
            elif 'speaker' in sentence.lower() or 'keynote' in sentence.lower():
                media_type = MediaType.SPEAKING_ENGAGEMENT
            
            # Extract date if present
            year_match = re.search(r'\b(20\d{2})\b', sentence)
            pub_date = None
            if year_match:
                try:
                    pub_date = date(int(year_match.group()), 1, 1)
                except ValueError:
                    pass
            
            # Calculate importance score
            importance_score = self.media_source_weights.get(publication.lower() + '.com', 0.3)
            
            # Basic sentiment analysis
            sentiment = self._analyze_sentiment(sentence)
            
            return MediaMention(
                title=sentence[:100] + "..." if len(sentence) > 100 else sentence,
                publication=publication,
                media_type=media_type,
                publication_date=pub_date,
                url=f"perplexity://mention/{hash(sentence)}",
                summary=sentence,
                sentiment=sentiment,
                importance_score=importance_score,
                verification_sources=["perplexity_ai"]
            )
            
        except Exception as e:
            logger.warning(f"Error extracting media mention: {e}")
            return None
    
    def _extract_award_from_sentence(self, sentence: str, founder_name: str) -> Optional[Award]:
        """Extract award information from a sentence."""
        try:
            # Extract award name (simplified)
            award_name = "Unknown Award"
            
            # Look for common award patterns
            award_patterns = [
                r'(Forbes\s+30\s+Under\s+30)',
                r'(40\s+Under\s+40)',
                r'(Entrepreneur\s+of\s+the\s+Year)',
                r'(Innovation\s+Award)',
                r'([A-Z][a-zA-Z\s]+Award)',
                r'(EY\s+Entrepreneur\s+of\s+the\s+Year)'
            ]
            
            for pattern in award_patterns:
                match = re.search(pattern, sentence, re.IGNORECASE)
                if match:
                    award_name = match.group(1)
                    break
            
            # Extract awarding organization
            org_patterns = [
                r'(?:by|from)\s+([A-Z][a-zA-Z\s&]+?)(?:\s|,|\.)',
                r'(Forbes|Fortune|Inc|Ernst\s+&\s+Young|EY)',
            ]
            
            awarding_org = "Unknown"
            for pattern in org_patterns:
                match = re.search(pattern, sentence, re.IGNORECASE)
                if match:
                    awarding_org = match.group(1).strip()
                    break
            
            # Extract year
            year_match = re.search(r'\b(20\d{2})\b', sentence)
            award_date = None
            if year_match:
                try:
                    award_date = date(int(year_match.group()), 1, 1)
                except ValueError:
                    pass
            
            return Award(
                award_name=award_name,
                awarding_organization=awarding_org,
                award_date=award_date,
                description=sentence,
                verification_sources=["perplexity_ai"]
            )
            
        except Exception as e:
            logger.warning(f"Error extracting award: {e}")
            return None
    
    def _extract_thought_leadership_from_sentence(self, sentence: str, founder_name: str) -> Optional[ThoughtLeadership]:
        """Extract thought leadership activity from a sentence."""
        try:
            # Determine activity type
            activity_type = "speaking"
            if any(keyword in sentence.lower() for keyword in ['book', 'authored', 'published', 'wrote']):
                activity_type = "book"
            elif any(keyword in sentence.lower() for keyword in ['article', 'blog', 'post']):
                activity_type = "article"
            elif any(keyword in sentence.lower() for keyword in ['podcast', 'interview']):
                activity_type = "podcast"
            elif any(keyword in sentence.lower() for keyword in ['keynote', 'speaker', 'conference']):
                activity_type = "keynote"
            elif any(keyword in sentence.lower() for keyword in ['panel', 'discussion']):
                activity_type = "panel"
            
            # Extract venue or publication
            venue = "Unknown"
            venue_patterns = [
                r'(?:at|on)\s+([A-Z][a-zA-Z\s&]+?)(?:\s+conference|\s+summit|\s+event)',
                r'(?:in|on)\s+(Forbes|TechCrunch|Harvard\s+Business\s+Review|Medium)',
            ]
            
            for pattern in venue_patterns:
                match = re.search(pattern, sentence, re.IGNORECASE)
                if match:
                    venue = match.group(1).strip()
                    break
            
            # Extract date
            year_match = re.search(r'\b(20\d{2})\b', sentence)
            activity_date = None
            if year_match:
                try:
                    activity_date = date(int(year_match.group()), 1, 1)
                except ValueError:
                    pass
            
            return ThoughtLeadership(
                activity_type=activity_type,
                title=sentence[:100] + "..." if len(sentence) > 100 else sentence,
                venue_or_publication=venue,
                date=activity_date,
                url=f"perplexity://thought-leadership/{hash(sentence)}",
                verification_sources=["perplexity_ai"]
            )
            
        except Exception as e:
            logger.warning(f"Error extracting thought leadership: {e}")
            return None
    
    def _extract_structured_mentions(self, content: str, founder_name: str) -> List[MediaMention]:
        """Extract structured media mentions from content."""
        mentions = []
        
        try:
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line and ('•' in line or '-' in line or line.startswith(('1.', '2.', '3.'))):
                    mention = self._extract_media_mention_from_sentence(line, founder_name)
                    if mention:
                        mentions.append(mention)
        
        except Exception as e:
            logger.warning(f"Error extracting structured mentions: {e}")
        
        return mentions
    
    def _extract_structured_awards(self, content: str, founder_name: str) -> List[Award]:
        """Extract structured awards from content."""
        awards = []
        
        try:
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line and ('•' in line or '-' in line or line.startswith(('1.', '2.', '3.'))):
                    award = self._extract_award_from_sentence(line, founder_name)
                    if award:
                        awards.append(award)
        
        except Exception as e:
            logger.warning(f"Error extracting structured awards: {e}")
        
        return awards
    
    def _extract_structured_thought_leadership(self, content: str, founder_name: str) -> List[ThoughtLeadership]:
        """Extract structured thought leadership from content."""
        activities = []
        
        try:
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line and ('•' in line or '-' in line or line.startswith(('1.', '2.', '3.'))):
                    activity = self._extract_thought_leadership_from_sentence(line, founder_name)
                    if activity:
                        activities.append(activity)
        
        except Exception as e:
            logger.warning(f"Error extracting structured thought leadership: {e}")
        
        return activities
    
    def _analyze_sentiment(self, text: str) -> str:
        """Basic sentiment analysis of text."""
        text_lower = text.lower()
        
        positive_words = ['success', 'achievement', 'winner', 'award', 'breakthrough', 
                         'innovation', 'leader', 'excellence', 'outstanding', 'praised']
        negative_words = ['controversy', 'criticism', 'scandal', 'failure', 'problem', 
                         'lawsuit', 'decline', 'loss', 'crisis', 'criticized']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
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
            # Use title as primary key for deduplication
            key = mention.title[:50].lower().strip()
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
        
        # Base score for using Perplexity (higher quality source)
        if 'perplexity_ai' in profile.data_sources:
            score += 0.4
        
        # Boost for media mentions
        if profile.media_mentions:
            score += 0.3
        
        # Boost for awards
        if profile.awards:
            score += 0.2
        
        # Boost for thought leadership
        if profile.thought_leadership:
            score += 0.2
        
        # Boost for high-quality sources
        high_quality_mentions = sum(
            1 for mention in profile.media_mentions 
            if mention.importance_score > 0.7
        )
        score += min(high_quality_mentions * 0.05, 0.15)
        
        # Boost for positive sentiment
        if profile.positive_sentiment_ratio > 0.7:
            score += 0.1
        
        # Boost for social media presence
        if profile.twitter_followers and profile.twitter_followers > 1000:
            score += 0.05
        
        return min(score, 1.0)