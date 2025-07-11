"""Founder data collection pipeline orchestrating financial and media intelligence sources."""

import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from ..ranking.models import FounderProfile, FounderFinancialProfile, FounderMediaProfile
from .financial_collector import FinancialDataCollector
from .media_collector import MediaCollector
from ...utils.rate_limiter import RateLimiter
from ...models import LinkedInProfile

logger = logging.getLogger(__name__)

class FounderDataPipeline:
    """Orchestrates comprehensive founder data collection from financial and media sources."""

    def __init__(self):
        self.financial_collector = FinancialDataCollector()
        self.media_collector = MediaCollector()
        self.rate_limiter = RateLimiter(max_requests=5, time_window=60)  # Overall pipeline rate limiting

    async def __aenter__(self):
        """Async context manager entry."""
        logger.info("🔧 Initializing founder pipeline services (Financial & Media)...")
        # Services are initialized in __init__, this is for resource management if needed in future
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        logger.info("🔧 Founder pipeline services cleaned up.")
        pass

    async def collect_founder_data(
        self,
        founder_profiles: List[FounderProfile],
        collection_options: Optional[Dict[str, bool]] = None
    ) -> List[FounderProfile]:
        """Collect comprehensive data for multiple founder profiles sequentially."""
        if collection_options is None:
            collection_options = {
                'collect_financial': True,
                'collect_media': True,
            }

        logger.info(f"🚀 Starting data collection for {len(founder_profiles)} founders")
        processed_profiles = []

        for i, profile in enumerate(founder_profiles):
            logger.info(f"Processing founder {i+1}/{len(founder_profiles)}: {profile.name}")
            try:
                processed_profile = await self.collect_single_founder_data(profile, collection_options)
                processed_profiles.append(processed_profile)
            except Exception as e:
                logger.error(f"Error processing {profile.name}: {e}")
                processed_profiles.append(profile)
            await asyncio.sleep(1) # Rate limiting between founders

        logger.info(f"✅ Data collection complete. {len(processed_profiles)} profiles processed")
        return processed_profiles

    async def collect_single_founder_data(
        self,
        founder_profile: FounderProfile,
        collection_options: Dict[str, bool]
    ) -> FounderProfile:
        """Collect comprehensive data for a single founder profile."""
        await self.rate_limiter.acquire()
        logger.debug(f"🔍 Collecting financial and media data for {founder_profile.name}")

        try:
            financial_profile = None
            media_profile = None

            if collection_options.get('collect_financial', True):
                financial_profile = await self._collect_financial_data(founder_profile.name, founder_profile.company_name)

            if collection_options.get('collect_media', True):
                media_profile = await self._collect_media_data(founder_profile.name, founder_profile.company_name)

            # Update the founder profile with collected data
            founder_profile.financial_profile = financial_profile
            founder_profile.media_profile = media_profile
            founder_profile.data_collected = True
            founder_profile.data_collection_timestamp = datetime.now()

            success_indicators = []
            if financial_profile and financial_profile.company_exits:
                success_indicators.append(f"{len(financial_profile.company_exits)} exits")
            if media_profile and media_profile.media_mentions:
                success_indicators.append(f"{len(media_profile.media_mentions)} media mentions")

            logger.info(f"✅ Collected data for {founder_profile.name}: {', '.join(success_indicators) if success_indicators else 'basic data only'}")
            return founder_profile

        except Exception as e:
            logger.error(f"Error collecting data for {founder_profile.name}: {e}")
            founder_profile.data_collected = False
            return founder_profile

    async def _collect_financial_data(self, founder_name: str, company_name: str) -> Optional[FounderFinancialProfile]:
        """Collect financial data with error handling."""
        try:
            # CORRECTED: No longer passes linkedin_url
            return await asyncio.wait_for(
                self.financial_collector.collect_founder_financial_data(
                    founder_name, company_name
                ),
                timeout=180  # 3 minute timeout for financial data
            )
        except asyncio.TimeoutError:
            logger.error(f"Financial data collection timeout for {founder_name}")
            return None
        except Exception as e:
            logger.error(f"Financial data collection failed for {founder_name}: {e}")
            return None

    async def _collect_media_data(self, founder_name: str, company_name: str) -> Optional[FounderMediaProfile]:
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

    @staticmethod
    def convert_linkedin_profile_to_founder_profile(linkedin_profile: LinkedInProfile, company_name: str = None) -> FounderProfile:
        """Convert LinkedInProfile to FounderProfile for intelligence collection."""
        profile_company = linkedin_profile.company_name or company_name or "Unknown Company"
        return FounderProfile(
            name=linkedin_profile.person_name,
            company_name=profile_company,
            title=linkedin_profile.current_position or "Founder",
            linkedin_url=linkedin_profile.linkedin_url,
            about=linkedin_profile.summary,
            location=linkedin_profile.location,
            data_collected=False
        )

    async def collect_founder_intelligence_from_linkedin_profiles(
        self,
        linkedin_profiles: List[LinkedInProfile],
        company_name: str = None,
        collection_options: Optional[Dict[str, bool]] = None
    ) -> List[FounderProfile]:
        """Convert LinkedIn profiles to FounderProfiles and collect financial and media intelligence."""
        logger.info(f"🔄 Converting {len(linkedin_profiles)} LinkedIn profiles to FounderProfiles for intelligence collection.")

        founder_profiles = [
            self.convert_linkedin_profile_to_founder_profile(lp, company_name or lp.company_name)
            for lp in linkedin_profiles
        ]
        logger.debug(f"✅ Converted {len(founder_profiles)} profiles.")

        enriched_profiles = await self.collect_founder_data(founder_profiles, collection_options)

        logger.info(f"🎯 Founder intelligence collection complete for {len(enriched_profiles)} profiles.")
        return enriched_profiles