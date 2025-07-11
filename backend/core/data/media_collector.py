"""Media report and coverage collection service for founder reputation analysis using Perplexity AI."""

import asyncio
from datetime import datetime, date
from typing import List, Optional, Dict, Any
import logging
import json

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
    
    async def collect_data(self, founder_name: str, current_company: str) -> FounderMediaProfile:
        """Collect comprehensive media profile for a founder using Perplexity AI."""
        return await self.collect_founder_media_profile(founder_name, current_company)
    
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
            # Get all media data in one comprehensive JSON response
            media_data = await self._get_comprehensive_media_data(founder_name, current_company)
            
            if media_data:
                # Parse JSON response into profile
                profile.media_mentions = self._parse_media_mentions_from_json(media_data.get('media_mentions', []))
                profile.awards = self._parse_awards_from_json(media_data.get('awards', []))
                profile.thought_leadership = self._parse_thought_leadership_from_json(media_data.get('thought_leadership', []))
                
                # Social media metrics
                social_metrics = media_data.get('social_media', {})
                profile.twitter_followers = social_metrics.get('twitter_followers')
                profile.linkedin_connections = social_metrics.get('linkedin_connections')
                
                # Calculate derived metrics
                profile.calculate_metrics()
                
                # Set data sources and confidence
                profile.data_sources = ['perplexity_ai']
                profile.confidence_score = self._calculate_media_confidence(profile)
                
                logger.info(f"✅ Media profile collected for {founder_name}: "
                           f"{len(profile.media_mentions)} mentions, "
                           f"{len(profile.awards)} awards, "
                           f"{len(profile.thought_leadership)} thought leadership activities")
            else:
                logger.warning(f"No media data found for {founder_name}")
                profile.confidence_score = 0.1
                
        except Exception as e:
            logger.error(f"Error collecting media profile for {founder_name}: {e}")
            profile.confidence_score = 0.1
        
        return profile
    
    async def _get_comprehensive_media_data(self, founder_name: str, current_company: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive media data in JSON format from Perplexity."""
        try:
            system_prompt = """You are a media intelligence specialist. Provide comprehensive media and public presence data in JSON format only.
            Return valid JSON with the following structure:
            {
                "media_mentions": [
                    {
                        "title": "Article/Interview title",
                        "publication": "Publication name",
                        "media_type": "news_article|interview|podcast|speaking_engagement",
                        "date": "YYYY-MM-DD",
                        "url": "URL if available",
                        "summary": "Brief summary",
                        "sentiment": "positive|negative|neutral",
                        "importance_score": 0.0-1.0
                    }
                ],
                "awards": [
                    {
                        "award_name": "Award name",
                        "awarding_organization": "Organization name",
                        "date": "YYYY-MM-DD",
                        "description": "Award description",
                        "category": "Award category"
                    }
                ],
                "thought_leadership": [
                    {
                        "activity_type": "speaking|book|article|podcast|keynote|panel",
                        "title": "Title of activity",
                        "venue_or_publication": "Venue/Publication name",
                        "date": "YYYY-MM-DD",
                        "url": "URL if available",
                        "description": "Brief description"
                    }
                ],
                "social_media": {
                    "twitter_followers": 123456,
                    "linkedin_connections": 12345,
                    "instagram_followers": 12345,
                    "youtube_subscribers": 12345
                },
                "summary_metrics": {
                    "total_media_mentions": 25,
                    "major_publications": 8,
                    "total_awards": 5,
                    "speaking_engagements": 15,
                    "media_sentiment": "positive|negative|neutral"
                }
            }
            
            Only return valid JSON. No additional text or explanations."""
            
            query = f"""Provide comprehensive media coverage, awards, thought leadership, and social media data for {founder_name} (founder/CEO of {current_company}).

            Include:
            1. Media mentions from major publications (Forbes, TechCrunch, Bloomberg, WSJ, etc.)
            2. Awards and recognitions received
            3. Speaking engagements, conferences, podcasts
            4. Articles, books, or content authored
            5. Current social media follower counts
            6. Summary metrics
            
            Return only valid JSON in the specified format."""
            
            response = await self.query_perplexity(
                query=query,
                system_prompt=system_prompt,
                max_tokens=3000
            )
            
            if response:
                content = self.extract_content_from_response(response)
                if content:
                    # Try to parse as JSON
                    try:
                        # Clean the content to extract JSON
                        content = content.strip()
                        if content.startswith('```json'):
                            content = content[7:]
                        if content.endswith('```'):
                            content = content[:-3]
                        content = content.strip()
                        
                        media_data = json.loads(content)
                        logger.debug(f"📊 Successfully parsed JSON media data for {founder_name}")
                        return media_data
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON response: {e}")
                        logger.debug(f"Raw content: {content[:500]}...")
                        return None
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting comprehensive media data for {founder_name}: {e}")
            return None
    
    def _parse_media_mentions_from_json(self, mentions_data: List[Dict[str, Any]]) -> List[MediaMention]:
        """Parse media mentions from JSON data."""
        mentions = []
        
        for mention_data in mentions_data:
            try:
                # Parse date
                pub_date = None
                if mention_data.get('date'):
                    try:
                        pub_date = datetime.strptime(mention_data['date'], '%Y-%m-%d').date()
                    except ValueError:
                        pass
                
                # Determine media type
                media_type_str = mention_data.get('media_type', 'news_article')
                media_type = MediaType.NEWS_ARTICLE
                if media_type_str == 'interview':
                    media_type = MediaType.INTERVIEW
                elif media_type_str == 'podcast':
                    media_type = MediaType.PODCAST
                elif media_type_str == 'speaking_engagement':
                    media_type = MediaType.SPEAKING_ENGAGEMENT
                
                mention = MediaMention(
                    title=mention_data.get('title', 'Unknown'),
                    publication=mention_data.get('publication', 'Unknown'),
                    media_type=media_type,
                    publication_date=pub_date,
                    url=mention_data.get('url', ''),
                    summary=mention_data.get('summary', ''),
                    sentiment=mention_data.get('sentiment', 'neutral'),
                    importance_score=float(mention_data.get('importance_score', 0.5)),
                    verification_sources=["perplexity_ai"]
                )
                mentions.append(mention)
                
            except Exception as e:
                logger.warning(f"Error parsing media mention: {e}")
                continue
        
        return mentions
    
    def _parse_awards_from_json(self, awards_data: List[Dict[str, Any]]) -> List[Award]:
        """Parse awards from JSON data."""
        awards = []
        
        for award_data in awards_data:
            try:
                # Parse date
                award_date = None
                if award_data.get('date'):
                    try:
                        award_date = datetime.strptime(award_data['date'], '%Y-%m-%d').date()
                    except ValueError:
                        pass
                
                award = Award(
                    award_name=award_data.get('award_name', 'Unknown Award'),
                    awarding_organization=award_data.get('awarding_organization', 'Unknown'),
                    award_date=award_date,
                    description=award_data.get('description', ''),
                    verification_sources=["perplexity_ai"]
                )
                awards.append(award)
                
            except Exception as e:
                logger.warning(f"Error parsing award: {e}")
                continue
        
        return awards
    
    def _parse_thought_leadership_from_json(self, tl_data: List[Dict[str, Any]]) -> List[ThoughtLeadership]:
        """Parse thought leadership from JSON data."""
        activities = []
        
        for activity_data in tl_data:
            try:
                # Parse date
                activity_date = None
                if activity_data.get('date'):
                    try:
                        activity_date = datetime.strptime(activity_data['date'], '%Y-%m-%d').date()
                    except ValueError:
                        pass
                
                activity = ThoughtLeadership(
                    activity_type=activity_data.get('activity_type', 'speaking'),
                    title=activity_data.get('title', 'Unknown'),
                    venue_or_publication=activity_data.get('venue_or_publication', 'Unknown'),
                    date=activity_date,
                    url=activity_data.get('url', ''),
                    verification_sources=["perplexity_ai"]
                )
                activities.append(activity)
                
            except Exception as e:
                logger.warning(f"Error parsing thought leadership activity: {e}")
                continue
        
        return activities
    
    def _calculate_media_confidence(self, profile: FounderMediaProfile) -> float:
        """Calculate confidence score for media profile."""
        score = 0.0
        
        # Base score for using Perplexity (higher quality source)
        if 'perplexity_ai' in profile.data_sources:
            score += 0.4
        
        # Boost for media mentions
        if profile.media_mentions:
            score += 0.3
            # Extra boost for multiple mentions
            score += min(len(profile.media_mentions) * 0.01, 0.1)
        
        # Boost for awards
        if profile.awards:
            score += 0.2
            # Extra boost for multiple awards
            score += min(len(profile.awards) * 0.02, 0.1)
        
        # Boost for thought leadership
        if profile.thought_leadership:
            score += 0.2
            # Extra boost for multiple activities
            score += min(len(profile.thought_leadership) * 0.01, 0.1)
        
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