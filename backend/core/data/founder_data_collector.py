"""Comprehensive founder data collection service integrating all data sources."""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

from ...core import get_logger
from ..ranking.models import FounderProfile
from .founder_financial_data import FounderFinancialProfile, ExitEvent, CompanyFinancials
from .sec_filings_collector import SECFilingsCollector
from .university_data_collector import UniversityDataCollector
from .accelerator_data_collector import AcceleratorDataCollector


logger = get_logger(__name__)


class FounderDataCollector:
    """Orchestrates comprehensive data collection for founders."""
    
    def __init__(self):
        self.sec_collector = SECFilingsCollector()
        self.university_collector = UniversityDataCollector()
        self.accelerator_collector = AcceleratorDataCollector()
    
    async def enhance_founder_profile(
        self, 
        profile: FounderProfile,
        collect_financial: bool = True,
        collect_education: bool = True,
        collect_accelerator: bool = True,
        collect_sec: bool = True
    ) -> FounderProfile:
        """Enhance a founder profile with comprehensive data collection."""
        logger.info(f"🔍 Enhancing profile for {profile.name}")
        
        try:
            company_names = profile.get_company_names()
            claimed_degrees = profile.get_claimed_degrees()
            
            # Collect data from all sources concurrently
            tasks = []
            
            if collect_financial:
                tasks.append(self._collect_financial_data(profile, company_names))
            
            if collect_education:
                tasks.append(self._collect_education_data(profile, claimed_degrees))
            
            if collect_accelerator:
                tasks.append(self._collect_accelerator_data(profile, company_names))
            
            if collect_sec:
                tasks.append(self._collect_sec_data(profile, company_names))
            
            # Execute all collections concurrently with timeout
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            financial_data, education_data, accelerator_data, sec_data = None, None, None, None
            
            if collect_financial and len(results) > 0 and not isinstance(results[0], Exception):
                financial_data = results[0]
            
            if collect_education and len(results) > (1 if collect_financial else 0):
                idx = 1 if collect_financial else 0
                if not isinstance(results[idx], Exception):
                    education_data = results[idx]
            
            if collect_accelerator and len(results) > sum([collect_financial, collect_education]):
                idx = sum([collect_financial, collect_education])
                if not isinstance(results[idx], Exception):
                    accelerator_data = results[idx]
            
            if collect_sec and len(results) > sum([collect_financial, collect_education, collect_accelerator]):
                idx = sum([collect_financial, collect_education, collect_accelerator])
                if not isinstance(results[idx], Exception):
                    sec_data = results[idx]
            
            # Update profile with collected data
            if financial_data:
                profile.financial_profile = financial_data.to_dict()
            
            if education_data:
                profile.education_profile = self._education_profile_to_dict(education_data)
            
            if accelerator_data:
                profile.accelerator_profile = self._accelerator_profile_to_dict(accelerator_data)
            
            if sec_data:
                profile.sec_profile = self._sec_profile_to_dict(sec_data)
            
            # Mark as enhanced
            profile.enhanced_data_collected = True
            profile.data_collection_timestamp = datetime.now()
            
            logger.info(f"✅ Enhanced profile for {profile.name} - Financial: {bool(financial_data)}, Education: {bool(education_data)}, Accelerator: {bool(accelerator_data)}, SEC: {bool(sec_data)}")
            
            return profile
            
        except Exception as e:
            logger.error(f"Error enhancing profile for {profile.name}: {e}")
            profile.enhanced_data_collected = False
            return profile
    
    async def enhance_founder_profiles_batch(
        self, 
        profiles: List[FounderProfile],
        batch_size: int = 5
    ) -> List[FounderProfile]:
        """Enhance multiple founder profiles in batches."""
        enhanced_profiles = []
        
        for i in range(0, len(profiles), batch_size):
            batch = profiles[i:i + batch_size]
            logger.info(f"Processing enhancement batch {i//batch_size + 1}/{(len(profiles) + batch_size - 1)//batch_size}")
            
            # Process each profile in the batch
            batch_tasks = [
                self.enhance_founder_profile(profile) 
                for profile in batch
            ]
            
            try:
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Batch processing error: {result}")
                        # Add original profile without enhancement
                        enhanced_profiles.append(batch[batch_results.index(result)])
                    else:
                        enhanced_profiles.append(result)
                
                # Small delay between batches
                if i + batch_size < len(profiles):
                    await asyncio.sleep(2)
                    
            except Exception as e:
                logger.error(f"Batch enhancement failed: {e}")
                # Add original profiles without enhancement
                enhanced_profiles.extend(batch)
        
        logger.info(f"Enhanced {len(enhanced_profiles)} founder profiles")
        return enhanced_profiles
    
    async def _collect_financial_data(
        self, 
        profile: FounderProfile, 
        company_names: List[str]
    ) -> Optional[FounderFinancialProfile]:
        """Collect financial data for a founder."""
        try:
            # Create financial profile from existing data
            financial_profile = FounderFinancialProfile(founder_name=profile.name)
            
            # Add companies from profile
            for company_name in company_names:
                if company_name and company_name.strip():
                    company_financial = CompanyFinancials(
                        company_name=company_name.strip(),
                        founder_role="founder" if company_name == profile.company_name else "co-founder",
                        is_active=True,  # Assume active unless proven otherwise
                        verification_sources=["linkedin_profile"]
                    )
                    financial_profile.companies_founded.append(company_financial)
            
            # Calculate metrics
            financial_profile.calculate_metrics()
            
            return financial_profile
            
        except Exception as e:
            logger.error(f"Financial data collection failed for {profile.name}: {e}")
            return None
    
    async def _collect_education_data(
        self, 
        profile: FounderProfile, 
        claimed_degrees: List[Dict[str, str]]
    ) -> Optional[Any]:
        """Collect education verification data."""
        try:
            async with self.university_collector:
                education_profile = await self.university_collector.verify_education(
                    profile.name, claimed_degrees
                )
                return education_profile
                
        except Exception as e:
            logger.error(f"Education data collection failed for {profile.name}: {e}")
            return None
    
    async def _collect_accelerator_data(
        self, 
        profile: FounderProfile, 
        company_names: List[str]
    ) -> Optional[Any]:
        """Collect accelerator participation data."""
        try:
            async with self.accelerator_collector:
                accelerator_profile = await self.accelerator_collector.get_accelerator_participation(
                    profile.name, company_names
                )
                return accelerator_profile
                
        except Exception as e:
            logger.error(f"Accelerator data collection failed for {profile.name}: {e}")
            return None
    
    async def _collect_sec_data(
        self, 
        profile: FounderProfile, 
        company_names: List[str]
    ) -> Optional[Any]:
        """Collect SEC filings data."""
        try:
            async with self.sec_collector:
                sec_profile = await self.sec_collector.get_founder_sec_filings(
                    profile.name, company_names
                )
                return sec_profile
                
        except Exception as e:
            logger.error(f"SEC data collection failed for {profile.name}: {e}")
            return None
    
    def _education_profile_to_dict(self, education_profile) -> Dict[str, Any]:
        """Convert education profile to dictionary."""
        return {
            "education_records": [
                {
                    "institution": record.institution,
                    "degree_type": record.degree_type,
                    "field_of_study": record.field_of_study,
                    "graduation_year": record.graduation_year,
                    "verification_status": record.verification_status,
                    "confidence_score": record.confidence_score
                }
                for record in education_profile.education_records
            ],
            "phd_degrees": [
                {
                    "institution": record.institution,
                    "field_of_study": record.field_of_study,
                    "verification_status": record.verification_status
                }
                for record in education_profile.phd_degrees
            ],
            "highest_degree": education_profile.highest_degree,
            "technical_field_background": education_profile.technical_field_background,
            "top_tier_institution": education_profile.top_tier_institution,
            "academic_publications": education_profile.academic_publications,
            "l3_criteria": education_profile.meets_l3_criteria()
        }
    
    def _accelerator_profile_to_dict(self, accelerator_profile) -> Dict[str, Any]:
        """Convert accelerator profile to dictionary."""
        return {
            "accelerator_participations": [
                {
                    "accelerator_name": program.accelerator_name,
                    "program_name": program.program_name,
                    "batch_info": program.batch_info,
                    "company_name": program.company_name,
                    "participation_year": program.participation_year,
                    "initial_funding": program.initial_funding,
                    "verification_status": program.verification_status
                }
                for program in accelerator_profile.accelerator_participations
            ],
            "total_programs": accelerator_profile.total_programs,
            "has_top_accelerator": accelerator_profile.has_top_accelerator,
            "total_accelerator_funding": accelerator_profile.total_accelerator_funding,
            "accelerator_network_strength": accelerator_profile.accelerator_network_strength,
            "l2_criteria": accelerator_profile.meets_l2_criteria()
        }
    
    def _sec_profile_to_dict(self, sec_profile) -> Dict[str, Any]:
        """Convert SEC profile to dictionary."""
        return {
            "verified_exits": [
                {
                    "company_name": exit_event.company_name,
                    "exit_type": exit_event.exit_type,
                    "exit_date": exit_event.exit_date.isoformat(),
                    "exit_value_usd": exit_event.exit_value_usd,
                    "verification_score": exit_event.verification_score
                }
                for exit_event in sec_profile.verified_exits
            ],
            "total_verified_exit_value": sec_profile.total_verified_exit_value,
            "highest_exit_value": sec_profile.highest_exit_value,
            "exit_count": sec_profile.exit_count,
            "l7_plus_criteria": sec_profile.meets_l7_plus_criteria()
        }


class FounderDataEnhancementOrchestrator:
    """High-level orchestrator for founder data enhancement workflows."""
    
    def __init__(self):
        self.collector = EnhancedFounderDataCollector()
    
    async def enhance_founders_for_ranking(
        self, 
        profiles: List[FounderProfile],
        prioritize_levels: List[str] = None
    ) -> List[FounderProfile]:
        """Enhance founder profiles optimized for L-level ranking."""
        
        if prioritize_levels is None:
            prioritize_levels = ["L7", "L8", "L9", "L10"]  # High-value levels
        
        logger.info(f"Enhancing {len(profiles)} founders for ranking (priority levels: {prioritize_levels})")
        
        # Determine which data to collect based on priority levels
        collect_sec = any(level in ["L7", "L8", "L9", "L10"] for level in prioritize_levels)
        collect_education = "L3" in prioritize_levels
        collect_accelerator = "L2" in prioritize_levels
        collect_financial = True  # Always collect financial data
        
        enhanced_profiles = []
        
        for profile in profiles:
            try:
                enhanced_profile = await self.collector.enhance_founder_profile(
                    profile,
                    collect_financial=collect_financial,
                    collect_education=collect_education,
                    collect_accelerator=collect_accelerator,
                    collect_sec=collect_sec
                )
                enhanced_profiles.append(enhanced_profile)
                
                # Brief delay between profiles
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Failed to enhance {profile.name}: {e}")
                enhanced_profiles.append(profile)  # Add original profile
        
        logger.info(f"Enhanced {len(enhanced_profiles)} profiles for ranking")
        return enhanced_profiles
    
    async def selective_enhancement(
        self, 
        profiles: List[FounderProfile],
        enhancement_strategy: str = "adaptive"
    ) -> List[FounderProfile]:
        """Selectively enhance profiles based on strategy."""
        
        if enhancement_strategy == "adaptive":
            # Enhance based on available profile data
            enhanced_profiles = []
            
            for profile in profiles:
                # Determine what to collect based on existing data
                collect_sec = bool(profile.get_company_names())
                collect_education = bool(profile.get_claimed_degrees())
                collect_accelerator = bool(profile.get_company_names())
                
                enhanced_profile = await self.collector.enhance_founder_profile(
                    profile,
                    collect_financial=True,
                    collect_education=collect_education,
                    collect_accelerator=collect_accelerator,
                    collect_sec=collect_sec
                )
                enhanced_profiles.append(enhanced_profile)
            
            return enhanced_profiles
        
        elif enhancement_strategy == "full":
            # Enhance all profiles with all data sources
            return await self.collector.enhance_founder_profiles_batch(profiles)
        
        elif enhancement_strategy == "minimal":
            # Only enhance with financial data
            enhanced_profiles = []
            for profile in profiles:
                enhanced_profile = await self.collector.enhance_founder_profile(
                    profile,
                    collect_financial=True,
                    collect_education=False,
                    collect_accelerator=False,
                    collect_sec=False
                )
                enhanced_profiles.append(enhanced_profile)
            return enhanced_profiles
        
        else:
            logger.warning(f"Unknown enhancement strategy: {enhancement_strategy}")
            return profiles
