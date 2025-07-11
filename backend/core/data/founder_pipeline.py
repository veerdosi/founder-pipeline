"""Founder data collection pipeline orchestrating multiple intelligence sources."""

import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from ..ranking.models import FounderProfile
from .financial_collector import FinancialDataCollector
from .intel_service import PerplexitySearchService
from .media_collector import MediaCollector
from ...utils.rate_limiter import RateLimiter
from ...models import LinkedInProfile

logger = logging.getLogger(__name__)


class FounderDataPipeline:
    """Orchestrates comprehensive founder data collection from multiple sources."""
    
    def __init__(self):
        self.financial_collector = FinancialDataCollector()
        self.media_collector = MediaCollector()
        self.rate_limiter = RateLimiter(max_requests=5, time_window=60)  # Overall pipeline rate limiting
    
    async def __aenter__(self):
        """Async context manager entry."""
        # Initialize all services with individual timeouts
        logger.info("🔧 Initializing founder pipeline services...")
        
        try:
            await asyncio.wait_for(self.financial_collector.__aenter__(), timeout=30)
        except Exception as e:
            logger.error(f"❌ Financial collector initialization failed: {e}")
            
        try:
            await asyncio.wait_for(self.perplexity_service.__aenter__(), timeout=30)
        except Exception as e:
            logger.error(f"❌ Perplexity service initialization failed: {e}")
            
        try:
            await asyncio.wait_for(self.media_collector.__aenter__(), timeout=30)
        except Exception as e:
            logger.error(f"❌ Media collector initialization failed: {e}")
            
        logger.info("🔧 Founder pipeline services initialization complete")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # Cleanup all services
        try:
            await self.financial_collector.__aexit__(exc_type, exc_val, exc_tb)
            await self.perplexity_service.__aexit__(exc_type, exc_val, exc_tb)
            await self.media_collector.__aexit__(exc_type, exc_val, exc_tb)
        except Exception as e:
            logger.warning(f"Error during pipeline cleanup: {e}")
    
    async def collect_founder_data(
        self, 
        founder_profiles: List[FounderProfile],
        collection_options: Optional[Dict[str, bool]] = None
    ) -> List[FounderProfile]:
        """Collect comprehensive data for multiple founder profiles sequentially."""
        
        # Default collection options
        if collection_options is None:
            collection_options = {
                'collect_financial': True,
                'collect_media': True,
                'collect_web_intelligence': True
            }
        
        logger.info(f"🚀 Starting data collection for {len(founder_profiles)} founders")
        
        processed_profiles = []
        
        # Sequential processing
        for i, profile in enumerate(founder_profiles):
            logger.info(f"Processing founder {i+1}/{len(founder_profiles)}: {profile.name}")
            
            try:
                processed_profile = await self.collect_single_founder_data(profile, collection_options)
                processed_profiles.append(processed_profile)
            except Exception as e:
                logger.error(f"Error processing {profile.name}: {e}")
                processed_profiles.append(profile)  # Use original profile as fallback
            
            # Rate limiting
            await asyncio.sleep(1)
        
        logger.info(f"✅ Data collection complete. {len(processed_profiles)} profiles processed")
        return processed_profiles
    
    async def collect_single_founder_data(
        self, 
        founder_profile: FounderProfile,
        collection_options: Dict[str, bool]
    ) -> FounderProfile:
        """Collect comprehensive data for a single founder profile sequentially."""
        
        await self.rate_limiter.acquire()
        
        logger.debug(f"🔍 Collecting data for {founder_profile.name}")
        
        try:
            financial_profile = None
            media_profile = None
            web_search_data = None

            if collection_options.get('collect_financial', True):
                financial_profile = await self._collect_financial_data(founder_profile.name, founder_profile.company_name, founder_profile.linkedin_url)
            
            if collection_options.get('collect_media', True):
                media_profile = await self._collect_media_data(founder_profile.name, founder_profile.company_name)
            
            if collection_options.get('collect_web_intelligence', True):
                web_search_data = await self._collect_web_intelligence(founder_profile.name, founder_profile.company_name)

            # Update the founder profile with collected data
            founder_profile.financial_profile = financial_profile
            founder_profile.media_profile = media_profile
            founder_profile.web_search_data = web_search_data
            founder_profile.data_collected = True
            founder_profile.data_collection_timestamp = datetime.now()
            
            success_indicators = []
            if financial_profile:
                success_indicators.append(f"{len(financial_profile.company_exits)} exits")
            if media_profile:
                success_indicators.append(f"{len(media_profile.media_mentions)} media mentions")
            if web_search_data:
                success_indicators.append(f"{len(web_search_data.verified_facts)} verified facts")
            
            logger.info(f"✅ Collected data for {founder_profile.name}: {', '.join(success_indicators) if success_indicators else 'basic data only'}")
            
            return founder_profile
            
        except Exception as e:
            logger.error(f"Error collecting data for {founder_profile.name}: {e}")
            # Return original profile with error indicator
            founder_profile.data_collected = False
            return founder_profile
    
    async def _collect_financial_data(
        self, 
        founder_name: str, 
        company_name: str, 
        linkedin_url: Optional[str]
    ):
        """Collect financial data with error handling."""
        try:
            return await asyncio.wait_for(
                self.financial_collector.collect_founder_financial_data(
                    founder_name, company_name, linkedin_url
                ),
                timeout=120  # 2 minute timeout for financial data
            )
        except asyncio.TimeoutError:
            logger.error(f"Financial data collection timeout for {founder_name}")
            return None
        except Exception as e:
            logger.error(f"Financial data collection failed for {founder_name}: {e}")
            return None
    
    async def _collect_media_data(self, founder_name: str, company_name: str):
        """Collect media data with error handling."""
        try:
            return await asyncio.wait_for(
                self.media_collector.collect_founder_media_profile(
                    founder_name, company_name
                ),
                timeout=120  # 2 minute timeout for media data
            )
        except asyncio.TimeoutError:
            logger.error(f"Media data collection timeout for {founder_name}")
            return None
        except Exception as e:
            logger.error(f"Media data collection failed for {founder_name}: {e}")
            return None
    
    async def _collect_web_intelligence(self, founder_name: str, company_name: str):
        """Collect web intelligence with error handling."""
        try:
            return await asyncio.wait_for(
                self.perplexity_service.collect_founder_web_intelligence(
                    founder_name, company_name
                ),
                timeout=120  # 2 minute timeout for web intelligence
            )
        except asyncio.TimeoutError:
            logger.error(f"Web intelligence collection timeout for {founder_name}")
            return None
        except Exception as e:
            logger.error(f"Web intelligence collection failed for {founder_name}: {e}")
            return None
    
    async def validate_profiles(
        self, 
        profiles: List[FounderProfile]
    ) -> Dict[str, Any]:
        """Validate the quality of data collection."""
        
        validation_report = {
            'total_profiles': len(profiles),
            'processed_profiles': 0,
            'financial_data_collected': 0,
            'media_data_collected': 0,
            'web_intelligence_collected': 0,
            'high_confidence_profiles': 0,
            'data_quality_issues': [],
            'summary': {}
        }
        
        for profile in profiles:
            if profile.data_collected:
                validation_report['processed_profiles'] += 1
            
            if profile.financial_profile:
                validation_report['financial_data_collected'] += 1
            
            if profile.media_profile:
                validation_report['media_data_collected'] += 1
            
            if profile.web_search_data:
                validation_report['web_intelligence_collected'] += 1
            
            # Check overall confidence
            if hasattr(profile, 'calculate_overall_confidence'):
                confidence = profile.calculate_overall_confidence()
                if confidence > 0.7:
                    validation_report['high_confidence_profiles'] += 1
            
            # Check for data quality issues
            if profile.data_collected and not any([
                profile.financial_profile, 
                profile.media_profile, 
                profile.web_search_data
            ]):
                validation_report['data_quality_issues'].append(
                    f"{profile.name}: Data collection flag set but no data found"
                )
        
        # Calculate summary statistics
        total = validation_report['total_profiles']
        validation_report['summary'] = {
            'processing_rate': validation_report['processed_profiles'] / total if total > 0 else 0,
            'financial_coverage': validation_report['financial_data_collected'] / total if total > 0 else 0,
            'media_coverage': validation_report['media_data_collected'] / total if total > 0 else 0,
            'web_intelligence_coverage': validation_report['web_intelligence_collected'] / total if total > 0 else 0,
            'high_confidence_rate': validation_report['high_confidence_profiles'] / total if total > 0 else 0,
            'data_quality_score': 1.0 - (len(validation_report['data_quality_issues']) / total) if total > 0 else 0
        }
        
        logger.info(f"📊 Data Collection Validation Complete:")
        logger.info(f"   Processing Rate: {validation_report['summary']['processing_rate']:.1%}")
        logger.info(f"   Financial Coverage: {validation_report['summary']['financial_coverage']:.1%}")
        logger.info(f"   Media Coverage: {validation_report['summary']['media_coverage']:.1%}")
        logger.info(f"   Web Intelligence Coverage: {validation_report['summary']['web_intelligence_coverage']:.1%}")
        logger.info(f"   High Confidence Rate: {validation_report['summary']['high_confidence_rate']:.1%}")
        
        return validation_report
    
    async def generate_collection_report(
        self, 
        profiles: List[FounderProfile]
    ) -> Dict[str, Any]:
        """Generate a comprehensive report on the data collection process."""
        
        """Founder data collection pipeline orchestrating multiple intelligence sources."""

import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from ..ranking.models import FounderProfile
from .financial_collector import FinancialDataCollector
from .media_collector import MediaCollector
from ...utils.rate_limiter import RateLimiter
from ...models import LinkedInProfile

logger = logging.getLogger(__name__)


class FounderDataPipeline:
    """Orchestrates comprehensive founder data collection from multiple sources."""
    
    def __init__(self):
        self.financial_collector = FinancialDataCollector()
        self.media_collector = MediaCollector()
        self.rate_limiter = RateLimiter(max_requests=5, time_window=60)  # Overall pipeline rate limiting
    
    async def __aenter__(self):
        """Async context manager entry."""
        # Initialize all services with individual timeouts
        logger.info("🔧 Initializing founder pipeline services...")
        
        try:
            await asyncio.wait_for(self.financial_collector.__aenter__(), timeout=30)
        except Exception as e:
            logger.error(f"❌ Financial collector initialization failed: {e}")
            
        try:
            await asyncio.wait_for(self.media_collector.__aenter__(), timeout=30)
        except Exception as e:
            logger.error(f"❌ Media collector initialization failed: {e}")
            
        logger.info("🔧 Founder pipeline services initialization complete")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # Cleanup all services
        try:
            await self.financial_collector.__aexit__(exc_type, exc_val, exc_tb)
            await self.media_collector.__aexit__(exc_type, exc_val, exc_tb)
        except Exception as e:
            logger.warning(f"Error during pipeline cleanup: {e}")
    
    async def collect_founder_data(
        self, 
        founder_profiles: List[FounderProfile],
        collection_options: Optional[Dict[str, bool]] = None
    ) -> List[FounderProfile]:
        """Collect comprehensive data for multiple founder profiles sequentially."""
        
        # Default collection options
        if collection_options is None:
            collection_options = {
                'collect_financial': True,
                'collect_media': True,
            }
        
        logger.info(f"🚀 Starting data collection for {len(founder_profiles)} founders")
        
        processed_profiles = []
        
        # Sequential processing
        for i, profile in enumerate(founder_profiles):
            logger.info(f"Processing founder {i+1}/{len(founder_profiles)}: {profile.name}")
            
            try:
                processed_profile = await self.collect_single_founder_data(profile, collection_options)
                processed_profiles.append(processed_profile)
            except Exception as e:
                logger.error(f"Error processing {profile.name}: {e}")
                processed_profiles.append(profile)  # Use original profile as fallback
            
            # Rate limiting
            await asyncio.sleep(1)
        
        logger.info(f"✅ Data collection complete. {len(processed_profiles)} profiles processed")
        return processed_profiles
    
    async def collect_single_founder_data(
        self, 
        founder_profile: FounderProfile,
        collection_options: Dict[str, bool]
    ) -> FounderProfile:
        """Collect comprehensive data for a single founder profile sequentially."""
        
        await self.rate_limiter.acquire()
        
        logger.debug(f"🔍 Collecting data for {founder_profile.name}")
        
        try:
            financial_profile = None
            media_profile = None

            if collection_options.get('collect_financial', True):
                financial_profile = await self._collect_financial_data(founder_profile.name, founder_profile.company_name, founder_profile.linkedin_url)
            
            if collection_options.get('collect_media', True):
                media_profile = await self._collect_media_data(founder_profile.name, founder_profile.company_name)
            
            # Update the founder profile with collected data
            founder_profile.financial_profile = financial_profile
            founder_profile.media_profile = media_profile
            founder_profile.data_collected = True
            founder_profile.data_collection_timestamp = datetime.now()
            
            success_indicators = []
            if financial_profile:
                success_indicators.append(f"{len(financial_profile.company_exits)} exits")
            if media_profile:
                success_indicators.append(f"{len(media_profile.media_mentions)} media mentions")
            
            logger.info(f"✅ Collected data for {founder_profile.name}: {', '.join(success_indicators) if success_indicators else 'basic data only'}")
            
            return founder_profile
            
        except Exception as e:
            logger.error(f"Error collecting data for {founder_profile.name}: {e}")
            # Return original profile with error indicator
            founder_profile.data_collected = False
            return founder_profile
    
    async def _collect_financial_data(
        self, 
        founder_name: str, 
        company_name: str, 
        linkedin_url: Optional[str]
    ):
        """Collect financial data with error handling."""
        try:
            return await asyncio.wait_for(
                self.financial_collector.collect_founder_financial_data(
                    founder_name, company_name, linkedin_url
                ),
                timeout=180  # 3 minute timeout for financial data
            )
        except asyncio.TimeoutError:
            logger.error(f"Financial data collection timeout for {founder_name}")
            return None
        except Exception as e:
            logger.error(f"Financial data collection failed for {founder_name}: {e}")
            return None
    
    async def _collect_media_data(self, founder_name: str, company_name: str):
        """Collect media data with error handling."""
        try:
            return await asyncio.wait_for(
                self.media_collector.collect_founder_media_profile(
                    founder_name, company_name
                ),
                timeout=180  # 3 minute timeout for media data
            )
        except asyncio.TimeoutError:
            logger.error(f"Media data collection timeout for {founder_name}")
            return None
        except Exception as e:
            logger.error(f"Media data collection failed for {founder_name}: {e}")
            return None
    
    async def validate_profiles(
        self, 
        profiles: List[FounderProfile]
    ) -> Dict[str, Any]:
        """Validate the quality of data collection."""
        
        validation_report = {
            'total_profiles': len(profiles),
            'processed_profiles': 0,
            'financial_data_collected': 0,
            'media_data_collected': 0,
            'high_confidence_profiles': 0,
            'data_quality_issues': [],
            'summary': {}
        }
        
        for profile in profiles:
            if profile.data_collected:
                validation_report['processed_profiles'] += 1
            
            if profile.financial_profile:
                validation_report['financial_data_collected'] += 1
            
            if profile.media_profile:
                validation_report['media_data_collected'] += 1
            
            # Check overall confidence
            if hasattr(profile, 'calculate_overall_confidence'):
                confidence = profile.calculate_overall_confidence()
                if confidence > 0.7:
                    validation_report['high_confidence_profiles'] += 1
            
            # Check for data quality issues
            if profile.data_collected and not any([
                profile.financial_profile, 
                profile.media_profile
            ]):
                validation_report['data_quality_issues'].append(
                    f"{profile.name}: Data collection flag set but no data found"
                )
        
        # Calculate summary statistics
        total = validation_report['total_profiles']
        validation_report['summary'] = {
            'processing_rate': validation_report['processed_profiles'] / total if total > 0 else 0,
            'financial_coverage': validation_report['financial_data_collected'] / total if total > 0 else 0,
            'media_coverage': validation_report['media_data_collected'] / total if total > 0 else 0,
            'high_confidence_rate': validation_report['high_confidence_profiles'] / total if total > 0 else 0,
            'data_quality_score': 1.0 - (len(validation_report['data_quality_issues']) / total) if total > 0 else 0
        }
        
        logger.info(f"📊 Data Collection Validation Complete:")
        logger.info(f"   Processing Rate: {validation_report['summary']['processing_rate']:.1%}")
        logger.info(f"   Financial Coverage: {validation_report['summary']['financial_coverage']:.1%}")
        logger.info(f"   Media Coverage: {validation_report['summary']['media_coverage']:.1%}")
        logger.info(f"   High Confidence Rate: {validation_report['summary']['high_confidence_rate']:.1%}")
        
        return validation_report
    
    async def generate_collection_report(
        self, 
        profiles: List[FounderProfile]
    ) -> Dict[str, Any]:
        """Generate a comprehensive report on the data collection process."""
        
        report = {
            'collection_timestamp': datetime.now().isoformat(),
            'total_profiles_processed': len(profiles),
            'collection_summary': {},
            'top_founders_by_data_quality': [],
            'data_gaps_analysis': {},
            'recommendations': []
        }
        
        # Validation data
        validation_report = await self.validate_profiles(profiles)
        report['validation'] = validation_report
        
        # Analyze top founders by data completeness
        founders_with_scores = []
        for profile in profiles:
            if hasattr(profile, 'calculate_overall_confidence'):
                confidence = profile.calculate_overall_confidence()
                data_points = 0
                if profile.financial_profile:
                    data_points += 1
                if profile.media_profile:
                    data_points += 1
                
                founders_with_scores.append({
                    'name': profile.name,
                    'company': profile.company_name,
                    'confidence': confidence,
                    'data_points': data_points,
                    'financial_exits': len(profile.financial_profile.company_exits) if profile.financial_profile else 0,
                    'media_mentions': len(profile.media_profile.media_mentions) if profile.media_profile else 0
                })
        
        # Sort by confidence and take top 10
        founders_with_scores.sort(key=lambda x: x['confidence'], reverse=True)
        report['top_founders_by_data_quality'] = founders_with_scores[:10]
        
        # Analyze data gaps
        total_profiles = len(profiles)
        report['data_gaps_analysis'] = {
            'no_financial_data': sum(1 for p in profiles if not p.financial_profile),
            'no_media_data': sum(1 for p in profiles if not p.media_profile),
            'completely_unprocessed': sum(1 for p in profiles if not p.data_collected)
        }
        
        # Generate recommendations
        financial_gap_rate = report['data_gaps_analysis']['no_financial_data'] / total_profiles
        media_gap_rate = report['data_gaps_analysis']['no_media_data'] / total_profiles
        
        if financial_gap_rate > 0.5:
            report['recommendations'].append("High financial data gap rate suggests need for additional financial data sources")
        
        if media_gap_rate > 0.3:
            report['recommendations'].append("Consider expanding media source coverage or improving search strategies")
        
        if validation_report['summary']['high_confidence_rate'] < 0.4:
            report['recommendations'].append("Low confidence rates suggest need for data source quality improvements")
        
        return report
    
    @staticmethod
    def convert_linkedin_profile_to_founder_profile(linkedin_profile: LinkedInProfile, company_name: str = None) -> FounderProfile:
        """Convert LinkedInProfile to FounderProfile for intelligence collection."""
        
        # Use company name from profile or provided parameter
        profile_company = linkedin_profile.company_name or company_name or "Unknown Company"
        
        # Extract education information
        education_1_school = None
        education_1_degree = None
        if linkedin_profile.education:
            education_1_school = linkedin_profile.education[0] if len(linkedin_profile.education) > 0 else None
            # Try to extract degree from education string if it contains degree info
            if education_1_school and ("-" in education_1_school or "," in education_1_school):
                parts = education_1_school.replace("-", ",").split(",")
                if len(parts) >= 2:
                    education_1_school = parts[0].strip()
                    education_1_degree = parts[1].strip()
        
        # Extract skills
        skill_1 = linkedin_profile.skills[0] if linkedin_profile.skills and len(linkedin_profile.skills) > 0 else None
        skill_2 = linkedin_profile.skills[1] if linkedin_profile.skills and len(linkedin_profile.skills) > 1 else None
        skill_3 = linkedin_profile.skills[2] if linkedin_profile.skills and len(linkedin_profile.skills) > 2 else None
        
        # Extract experience from previous companies
        experience_1_company = None
        experience_1_title = None
        experience_2_company = None
        experience_2_title = None
        
        if linkedin_profile.previous_companies:
            experience_1_company = linkedin_profile.previous_companies[0] if len(linkedin_profile.previous_companies) > 0 else None
            experience_1_title = "Previous Role"  # Generic title since we don't have detailed experience data
            
            if len(linkedin_profile.previous_companies) > 1:
                experience_2_company = linkedin_profile.previous_companies[1]
                experience_2_title = "Previous Role"
        
        # Create FounderProfile
        founder_profile = FounderProfile(
            name=linkedin_profile.person_name,
            company_name=profile_company,
            title=linkedin_profile.current_position or "Founder",
            linkedin_url=linkedin_profile.linkedin_url,
            location=linkedin_profile.location,
            about=linkedin_profile.summary,
            estimated_age=None,  # Not available from LinkedIn profile
            
            # Experience data
            experience_1_title=experience_1_title,
            experience_1_company=experience_1_company,
            experience_2_title=experience_2_title,
            experience_2_company=experience_2_company,
            experience_3_title=None,  # Not available
            experience_3_company=None,  # Not available
            
            # Education data
            education_1_school=education_1_school,
            education_1_degree=education_1_degree,
            education_2_school=None,  # Could be expanded if needed
            education_2_degree=None,   # Could be expanded if needed
            
            # Skills
            skill_1=skill_1,
            skill_2=skill_2,
            skill_3=skill_3,
            
            # Data collection metadata
            data_collection_timestamp=datetime.now(),
            data_collected=False  # Will be set to True after intelligence collection
        )
        
        return founder_profile
    
    async def collect_founder_intelligence_from_linkedin_profiles(
        self, 
        linkedin_profiles: List[LinkedInProfile],
        company_name: str = None,
        collection_options: Optional[Dict[str, bool]] = None
    ) -> List[FounderProfile]:
        """Convert LinkedIn profiles to FounderProfiles and collect intelligence data."""
        
        logger.info(f"🔄 Converting {len(linkedin_profiles)} LinkedIn profiles to FounderProfiles")
        
        # Convert LinkedIn profiles to FounderProfiles
        founder_profiles = []
        for linkedin_profile in linkedin_profiles:
            founder_profile = self.convert_linkedin_profile_to_founder_profile(
                linkedin_profile, 
                company_name or linkedin_profile.company_name
            )
            founder_profiles.append(founder_profile)
            logger.debug(f"✅ Converted {linkedin_profile.person_name} to FounderProfile")
        
        # Collect intelligence data for all founder profiles
        enriched_profiles = await self.collect_founder_data(founder_profiles, collection_options)
        
        logger.info(f"🎯 Founder intelligence collection complete for {len(enriched_profiles)} profiles")
        return enriched_profiles