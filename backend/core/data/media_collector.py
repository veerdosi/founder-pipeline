"""Media report and coverage collection service for founder reputation analysis using Perplexity AI."""

import asyncio
from datetime import datetime, date
from typing import List, Optional, Dict, Any
import logging

from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

from ..ranking.models import (
    FounderMediaProfile, MediaMention, Award, ThoughtLeadership, 
    MediaType
)
from .perplexity_base import PerplexityBaseService

logger = logging.getLogger(__name__)


class MediaMentionData(BaseModel):
    """Pydantic model for media mention data."""
    title: str = Field(description="Title of the article or interview")
    publication: str = Field(description="Name of the publication or media outlet")
    media_type: str = Field(description="Type of media: news_article/interview/podcast/speaking_engagement")
    date: Optional[str] = Field(description="Publication date in YYYY-MM-DD format")
    url: Optional[str] = Field(description="URL of the article or content")
    summary: Optional[str] = Field(description="Brief summary of the content")
    sentiment: str = Field(description="Sentiment: positive/negative/neutral")
    importance_score: float = Field(description="Importance score between 0.0 and 1.0")


class AwardData(BaseModel):
    """Pydantic model for award data."""
    award_name: str = Field(description="Name of the award")
    awarding_organization: str = Field(description="Organization that gave the award")
    date: Optional[str] = Field(description="Award date in YYYY-MM-DD format")
    description: Optional[str] = Field(description="Description of the award")
    category: Optional[str] = Field(description="Category of the award")


class ThoughtLeadershipData(BaseModel):
    """Pydantic model for thought leadership activities."""
    activity_type: str = Field(description="Type of activity: speaking/book/article/podcast/keynote/panel")
    title: str = Field(description="Title of the activity or content")
    venue_or_publication: str = Field(description="Venue or publication name")
    date: Optional[str] = Field(description="Date in YYYY-MM-DD format")
    url: Optional[str] = Field(description="URL if available")
    description: Optional[str] = Field(description="Brief description")


class SocialMediaData(BaseModel):
    """Pydantic model for social media metrics."""
    twitter_followers: Optional[int] = Field(description="Number of Twitter followers")
    linkedin_connections: Optional[int] = Field(description="Number of LinkedIn connections")
    instagram_followers: Optional[int] = Field(description="Number of Instagram followers")
    youtube_subscribers: Optional[int] = Field(description="Number of YouTube subscribers")


class SummaryMetricsData(BaseModel):
    """Pydantic model for summary metrics."""
    total_media_mentions: int = Field(description="Total number of media mentions")
    major_publications: int = Field(description="Number of major publication mentions")
    total_awards: int = Field(description="Total number of awards")
    speaking_engagements: int = Field(description="Number of speaking engagements")
    media_sentiment: str = Field(description="Overall media sentiment: positive/negative/neutral")


class MediaProfileData(BaseModel):
    """Pydantic model for comprehensive media profile data."""
    media_mentions: List[MediaMentionData] = Field(description="List of media mentions")
    awards: List[AwardData] = Field(description="List of awards received")
    thought_leadership: List[ThoughtLeadershipData] = Field(description="List of thought leadership activities")
    social_media: SocialMediaData = Field(description="Social media metrics")
    summary_metrics: SummaryMetricsData = Field(description="Summary metrics")


class MediaCollector(PerplexityBaseService):
    """Service for collecting comprehensive media coverage and public presence data using LangChain structured output."""
    
    def __init__(self):
        super().__init__()
        
        # Set up the structured output parser
        self.parser = JsonOutputParser(pydantic_object=MediaProfileData)
        
        # Content-focused query template
        self.query_template = """
        Provide comprehensive media coverage and public presence data for {founder_name} (founder/CEO of {current_company}).
        
        Please include:
        1. Media mentions: articles, interviews, podcasts, speaking engagements from major publications
        2. Awards and recognitions: business awards, industry recognitions, honors received
        3. Thought leadership: speaking engagements, conferences, articles authored, books, keynotes
        4. Social media presence: current follower counts across platforms
        5. Summary metrics: total counts and overall sentiment
        
        Focus on verified, factual information with specific dates, publication names, and metrics.
        """
    
    async def collect_data(self, founder_name: str, current_company: str) -> FounderMediaProfile:
        """Collect comprehensive media profile for a founder using Perplexity AI."""
        return await self.collect_founder_media_profile(founder_name, current_company)
    
    async def collect_founder_media_profile(
        self, 
        founder_name: str,
        current_company: str
    ) -> FounderMediaProfile:
        """Collect comprehensive media profile for a founder using Perplexity AI."""
        # Input validation
        if not founder_name or not isinstance(founder_name, str):
            logger.error("Invalid founder_name provided")
            return FounderMediaProfile(
                founder_name=founder_name or "Unknown",
                last_updated=datetime.now(),
                confidence_score=0.0
            )
        
        logger.debug(f"📰 Collecting media profile for {founder_name} using Perplexity AI")
        logger.debug(f"📝 Parameters: company={current_company}")
        
        profile = FounderMediaProfile(
            founder_name=founder_name,
            last_updated=datetime.now()
        )
        
        try:
            # Get all media data using structured output
            media_data = await self._get_comprehensive_media_data(founder_name, current_company)
            
            if media_data:
                # Parse structured response into profile
                profile.media_mentions = self._parse_media_mentions_from_structured_data(media_data.media_mentions)
                profile.awards = self._parse_awards_from_structured_data(media_data.awards)
                profile.thought_leadership = self._parse_thought_leadership_from_structured_data(media_data.thought_leadership)
                
                # Social media metrics
                profile.twitter_followers = media_data.social_media.twitter_followers
                profile.linkedin_connections = media_data.social_media.linkedin_connections
                
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
    
    async def _get_comprehensive_media_data(self, founder_name: str, current_company: str) -> Optional[MediaProfileData]:
        """Get comprehensive media data using LangChain structured output."""
        try:
            system_prompt = """You are a media intelligence specialist focusing on founder and executive media coverage.
            
            Provide comprehensive, factual information about the person's media presence and public profile.
            Focus on verified data with specific dates, publication names, and follower counts.
            If information is not available, use null values rather than guessing."""
            
            # Get format instructions from the parser
            format_instructions = self.parser.get_format_instructions()
            
            query = self.query_template.format(founder_name=founder_name, current_company=current_company)
            full_query = f"{query}\n\n{format_instructions}"
            
            response = await self.query_perplexity(
                query=full_query,
                system_prompt=system_prompt,
                max_tokens=3000
            )
            
            if response:
                content = self.extract_content_from_response(response)
                if content:
                    try:
                        # First, try to extract JSON from markdown code blocks
                        cleaned_content = self._extract_json_from_markdown(content)
                        
                        # Use the LangChain parser to parse the response
                        parsed_data = self.parser.parse(cleaned_content)
                        
                        # Validate the parsed data
                        if self._validate_media_data(parsed_data):
                            logger.debug(f"📊 Successfully parsed structured media data for {founder_name}")
                            return parsed_data
                        else:
                            logger.warning(f"Invalid media data structure for {founder_name}")
                            return None
                    except Exception as parse_error:
                        logger.warning(f"Failed to parse structured output for {founder_name}: {parse_error}")
                        logger.debug(f"Raw response content: {content[:500]}...")
                        # Try fallback parsing if available
                        return self._try_fallback_parsing(content, founder_name)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting comprehensive media data for {founder_name}: {e}")
            return None

    def _parse_media_mentions_from_structured_data(self, mentions_data: List[MediaMentionData]) -> List[MediaMention]:
        """Parse media mentions from structured data."""
        mentions = []
        
        if not mentions_data:
            return mentions
        
        for mention_data in mentions_data:
            try:
                # Parse date
                pub_date = None
                if mention_data.date:
                    try:
                        pub_date = datetime.strptime(mention_data.date, '%Y-%m-%d').date()
                    except ValueError:
                        pass
                
                # Determine media type
                media_type = MediaType.NEWS_ARTICLE
                if mention_data.media_type == 'interview':
                    media_type = MediaType.INTERVIEW
                elif mention_data.media_type == 'podcast':
                    media_type = MediaType.PODCAST
                elif mention_data.media_type == 'speaking_engagement':
                    media_type = MediaType.SPEAKING_ENGAGEMENT
                
                mention = MediaMention(
                    title=mention_data.title,
                    publication=mention_data.publication,
                    media_type=media_type,
                    publication_date=pub_date,
                    url=mention_data.url or '',
                    summary=mention_data.summary or '',
                    sentiment=mention_data.sentiment,
                    importance_score=mention_data.importance_score,
                    verification_sources=["perplexity_ai"]
                )
                mentions.append(mention)
                
            except Exception as e:
                logger.warning(f"Error parsing media mention: {e}")
                continue
        
        return mentions
    
    def _parse_awards_from_structured_data(self, awards_data: List[AwardData]) -> List[Award]:
        """Parse awards from structured data."""
        awards = []
        
        for award_data in awards_data:
            try:
                # Parse date
                award_date = None
                if award_data.date:
                    try:
                        award_date = datetime.strptime(award_data.date, '%Y-%m-%d').date()
                    except ValueError:
                        pass
                
                award = Award(
                    award_name=award_data.award_name,
                    awarding_organization=award_data.awarding_organization,
                    award_date=award_date,
                    description=award_data.description or '',
                    verification_sources=["perplexity_ai"]
                )
                awards.append(award)
                
            except Exception as e:
                logger.warning(f"Error parsing award: {e}")
                continue
        
        return awards
    
    def _parse_thought_leadership_from_structured_data(self, tl_data: List[ThoughtLeadershipData]) -> List[ThoughtLeadership]:
        """Parse thought leadership from structured data."""
        activities = []
        
        for activity_data in tl_data:
            try:
                # Parse date
                activity_date = None
                if activity_data.date:
                    try:
                        activity_date = datetime.strptime(activity_data.date, '%Y-%m-%d').date()
                    except ValueError:
                        pass
                
                activity = ThoughtLeadership(
                    activity_type=activity_data.activity_type,
                    title=activity_data.title,
                    venue_or_publication=activity_data.venue_or_publication,
                    date=activity_date,
                    url=activity_data.url or '',
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
    
    def _extract_json_from_markdown(self, content: str) -> str:
        """Extract JSON from markdown code blocks or clean up the content."""
        import re
        
        # Try to find JSON in markdown code blocks
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            return json_match.group(1).strip()
        
        # Try to find JSON in regular code blocks
        json_match = re.search(r'```\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            return json_match.group(1).strip()
        
        # Look for JSON object without markdown
        json_match = re.search(r'\{.*?\}', content, re.DOTALL)
        if json_match:
            return json_match.group(0).strip()
        
        # Return content as-is if no JSON found
        return content.strip()
    
    def _validate_media_data(self, data: MediaProfileData) -> bool:
        """Validate the parsed media data structure."""
        try:
            # Check required fields exist
            if not hasattr(data, 'media_mentions') or not hasattr(data, 'social_media'):
                return False
            
            # Check social media data is valid
            if not hasattr(data.social_media, 'twitter_followers'):
                return False
            
            # Check that we have some meaningful data
            has_meaningful_data = (
                len(data.media_mentions) > 0 or 
                len(data.awards) > 0 or 
                len(data.thought_leadership) > 0 or
                (data.social_media.twitter_followers is not None and data.social_media.twitter_followers > 0) or
                (data.social_media.linkedin_connections is not None and data.social_media.linkedin_connections > 0)
            )
            
            # Validate importance scores in media mentions
            for mention in data.media_mentions:
                if not isinstance(mention.importance_score, (int, float)) or mention.importance_score < 0 or mention.importance_score > 1:
                    return False
            
            return has_meaningful_data
            
        except Exception as e:
            logger.warning(f"Error validating media data: {e}")
            return False
    
    def _try_fallback_parsing(self, content: str, founder_name: str) -> Optional[MediaProfileData]:
        """Try fallback parsing methods if structured parsing fails."""
        try:
            # Try to extract JSON manually and parse with a more lenient approach
            from backend.utils.data_processing import extract_and_parse_json
            
            json_data = extract_and_parse_json(content)
            if json_data and not json_data.get('error'):
                # Try to create a minimal valid structure
                fallback_data = MediaProfileData(
                    media_mentions=[],
                    awards=[],
                    thought_leadership=[],
                    social_media=SocialMediaData(
                        twitter_followers=json_data.get('social_media', {}).get('twitter_followers'),
                        linkedin_connections=json_data.get('social_media', {}).get('linkedin_connections'),
                        instagram_followers=json_data.get('social_media', {}).get('instagram_followers'),
                        youtube_subscribers=json_data.get('social_media', {}).get('youtube_subscribers')
                    ),
                    summary_metrics=SummaryMetricsData(
                        total_media_mentions=json_data.get('summary_metrics', {}).get('total_media_mentions', 0),
                        major_publications=json_data.get('summary_metrics', {}).get('major_publications', 0),
                        total_awards=json_data.get('summary_metrics', {}).get('total_awards', 0),
                        speaking_engagements=json_data.get('summary_metrics', {}).get('speaking_engagements', 0),
                        media_sentiment=json_data.get('summary_metrics', {}).get('media_sentiment', 'neutral')
                    )
                )
                
                # Try to parse media mentions if available
                if 'media_mentions' in json_data and isinstance(json_data['media_mentions'], list):
                    try:
                        mentions = []
                        for mention_data in json_data['media_mentions']:
                            if isinstance(mention_data, dict):
                                mention = MediaMentionData(
                                    title=mention_data.get('title', 'Unknown'),
                                    publication=mention_data.get('publication', 'Unknown'),
                                    media_type=mention_data.get('media_type', 'news_article'),
                                    date=mention_data.get('date'),
                                    url=mention_data.get('url'),
                                    summary=mention_data.get('summary'),
                                    sentiment=mention_data.get('sentiment', 'neutral'),
                                    importance_score=max(0.0, min(1.0, float(mention_data.get('importance_score', 0.5))))
                                )
                                mentions.append(mention)
                        fallback_data.media_mentions = mentions
                    except Exception as e:
                        logger.warning(f"Error parsing media mentions in fallback: {e}")
                
                logger.info(f"Successfully used fallback parsing for {founder_name}")
                return fallback_data
                
        except Exception as e:
            logger.warning(f"Fallback parsing failed for {founder_name}: {e}")
        
        return None