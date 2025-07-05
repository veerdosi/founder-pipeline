"""Accelerator participation data collection for founder tracking (L2 criteria)."""

import asyncio
import aiohttp
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
from urllib.parse import quote

from ...core import get_logger


logger = get_logger(__name__)


@dataclass
class AcceleratorProgram:
    """Individual accelerator program participation."""
    accelerator_name: str
    program_name: str
    batch_info: str  # "W21", "S22", etc. for YC
    company_name: str
    founder_names: List[str]
    participation_year: Optional[int] = None
    demo_day_date: Optional[datetime] = None
    initial_funding: Optional[float] = None  # In USD
    program_duration_weeks: Optional[int] = None
    program_focus: str = ""  # "General", "AI/ML", "Fintech", etc.
    verification_status: str = "unverified"  # "verified", "likely", "unverified"
    verification_sources: List[str] = field(default_factory=list)
    additional_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AcceleratorProfile:
    """Profile of an accelerator for data collection."""
    name: str
    website_url: str
    api_endpoint: Optional[str] = None
    company_list_url: Optional[str] = None
    typical_funding_amount: Optional[float] = None
    program_duration_weeks: int = 12
    application_batch_format: str = ""  # "YY" for YC, "Cohort N" for others
    tier: str = "tier_1"  # "tier_1", "tier_2", "tier_3"


@dataclass
class FounderAcceleratorProfile:
    """Complete accelerator participation profile for a founder."""
    founder_name: str
    accelerator_participations: List[AcceleratorProgram] = field(default_factory=list)
    tier_1_accelerators: List[AcceleratorProgram] = field(default_factory=list)  # YC, Techstars, etc.
    tier_2_accelerators: List[AcceleratorProgram] = field(default_factory=list)
    tier_3_accelerators: List[AcceleratorProgram] = field(default_factory=list)
    total_programs: int = 0
    earliest_program_year: Optional[int] = None
    most_recent_program_year: Optional[int] = None
    total_accelerator_funding: float = 0.0
    has_top_accelerator: bool = False
    accelerator_network_strength: float = 0.0  # 0-1 based on accelerator quality
    last_updated: datetime = field(default_factory=datetime.now)
    
    def calculate_accelerator_metrics(self):
        """Calculate derived accelerator metrics."""
        if not self.accelerator_participations:
            return
        
        self.total_programs = len(self.accelerator_participations)
        
        # Categorize by tier
        tier_1_names = ["y combinator", "techstars", "500 startups", "plug and play"]
        tier_2_names = ["angelpad", "seedcamp", "rocket internet", "entrepreneur first"]
        
        for program in self.accelerator_participations:
            acc_name_lower = program.accelerator_name.lower()
            if any(tier1 in acc_name_lower for tier1 in tier_1_names):
                self.tier_1_accelerators.append(program)
            elif any(tier2 in acc_name_lower for tier2 in tier_2_names):
                self.tier_2_accelerators.append(program)
            else:
                self.tier_3_accelerators.append(program)
        
        # Calculate years
        years = [
            p.participation_year for p in self.accelerator_participations 
            if p.participation_year
        ]
        if years:
            self.earliest_program_year = min(years)
            self.most_recent_program_year = max(years)
        
        # Calculate total funding
        self.total_accelerator_funding = sum(
            p.initial_funding or 0 
            for p in self.accelerator_participations
        )
        
        # Check for top accelerators
        self.has_top_accelerator = len(self.tier_1_accelerators) > 0
        
        # Calculate network strength score
        tier_1_score = len(self.tier_1_accelerators) * 1.0
        tier_2_score = len(self.tier_2_accelerators) * 0.6
        tier_3_score = len(self.tier_3_accelerators) * 0.3
        
        self.accelerator_network_strength = min(
            (tier_1_score + tier_2_score + tier_3_score) / 3.0, 1.0
        )
    
    def meets_l2_criteria(self) -> Dict[str, bool]:
        """Check if founder meets L2 accelerator criteria."""
        return {
            "accelerator_graduate": self.total_programs > 0,
            "top_tier_accelerator": self.has_top_accelerator,
            "recent_participation": (
                self.most_recent_program_year and 
                self.most_recent_program_year >= 2018
            ),
            "accelerator_funding": self.total_accelerator_funding >= 25000,  # $25K typical
            "verified_participation": any(
                p.verification_status == "verified" 
                for p in self.accelerator_participations
            )
        }


class AcceleratorDataCollector:
    """Collects accelerator participation data for founders."""
    
    def __init__(self):
        self.session = None
        self.rate_limit_delay = 1.0
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
        
        # Load accelerator profiles
        self.accelerator_profiles = self._load_accelerator_profiles()
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def get_accelerator_participation(
        self, 
        founder_name: str,
        company_names: List[str] = None
    ) -> FounderAcceleratorProfile:
        """Get comprehensive accelerator participation data for a founder."""
        logger.info(f"🚀 Collecting accelerator data for {founder_name}")
        
        profile = FounderAcceleratorProfile(founder_name=founder_name)
        
        try:
            # Search Y Combinator database
            yc_participations = await self._search_y_combinator(founder_name, company_names)
            profile.accelerator_participations.extend(yc_participations)
            
            # Search Techstars database
            techstars_participations = await self._search_techstars(founder_name, company_names)
            profile.accelerator_participations.extend(techstars_participations)
            
            # Search 500 Startups
            five_hundred_participations = await self._search_500_startups(founder_name, company_names)
            profile.accelerator_participations.extend(five_hundred_participations)
            
            # Search AngelList for accelerator tags
            angellist_participations = await self._search_angellist_accelerators(founder_name, company_names)
            profile.accelerator_participations.extend(angellist_participations)
            
            # Search Crunchbase for accelerator data
            crunchbase_participations = await self._search_crunchbase_accelerators(founder_name, company_names)
            profile.accelerator_participations.extend(crunchbase_participations)
            
            # Remove duplicates
            profile.accelerator_participations = self._deduplicate_participations(
                profile.accelerator_participations
            )
            
            # Calculate metrics
            profile.calculate_accelerator_metrics()
            
            logger.info(f"✅ Found {profile.total_programs} accelerator participations for {founder_name}")
            return profile
            
        except Exception as e:
            logger.error(f"Error collecting accelerator data for {founder_name}: {e}")
            return profile
    
    async def _search_y_combinator(
        self, 
        founder_name: str, 
        company_names: List[str] = None
    ) -> List[AcceleratorProgram]:
        """Search Y Combinator company database using unofficial but public API."""
        participations = []
        
        try:
            # Use unofficial but public Y Combinator API
            url = "https://yc-oss.github.io/api/all.json"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    companies = await response.json()
                    
                    # Search by founder name and company names
                    for company in companies:
                        company_name = company.get("name", "")
                        batch = company.get("batch", "")
                        website = company.get("website", "")
                        
                        # Check if company name matches
                        company_match = company_names and any(
                            comp_name.lower() in company_name.lower() or 
                            company_name.lower() in comp_name.lower()
                            for comp_name in company_names
                        )
                        
                        if company_match:
                            # Extract batch year
                            batch_year = None
                            if batch:
                                # Parse YC batch format (W21, S22, etc.)
                                yc_match = re.search(r'([WS])(\d{2})', batch)
                                if yc_match:
                                    year_suffix = int(yc_match.group(2))
                                    batch_year = 2000 + year_suffix if year_suffix > 50 else 2020 + year_suffix
                            
                            program = AcceleratorProgram(
                                accelerator_name="Y Combinator",
                                program_name="YC Core Program",
                                batch_info=batch,
                                company_name=company_name,
                                founder_names=[founder_name],  # We don't have founder details in this API
                                participation_year=batch_year,
                                initial_funding=250000.0,  # Updated YC standard funding
                                program_duration_weeks=12,
                                program_focus="General",
                                verification_status="verified",
                                verification_sources=["yc_unofficial_api"],
                                additional_info={
                                    "yc_url": website,
                                    "description": company.get("one_liner", "")
                                }
                            )
                            participations.append(program)
            
            await asyncio.sleep(self.rate_limit_delay)
            
        except Exception as e:
            logger.warning(f"Y Combinator search failed for {founder_name}: {e}")
        
        return participations
    
    async def _search_techstars(
        self, 
        founder_name: str, 
        company_names: List[str] = None
    ) -> List[AcceleratorProgram]:
        """Search Techstars portfolio companies."""
        participations = []
        
        try:
            # Techstars companies page (web scraping approach)
            url = "https://www.techstars.com/portfolio"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    
                    # Look for company and founder mentions
                    if company_names:
                        for company_name in company_names:
                            if company_name.lower() in content.lower():
                                # Try to extract program info from surrounding context
                                company_pattern = re.escape(company_name)
                                context_match = re.search(
                                    f'.*{company_pattern}.*?(\d{{4}}).*',
                                    content,
                                    re.IGNORECASE | re.DOTALL
                                )
                                
                                year = None
                                if context_match:
                                    year = int(context_match.group(1))
                                
                                program = AcceleratorProgram(
                                    accelerator_name="Techstars",
                                    program_name="Techstars Accelerator",
                                    batch_info=f"Class of {year}" if year else "Unknown",
                                    company_name=company_name,
                                    founder_names=[founder_name],
                                    participation_year=year,
                                    initial_funding=20000.0,  # Typical Techstars funding
                                    program_duration_weeks=13,
                                    program_focus="General",
                                    verification_status="likely",
                                    verification_sources=["techstars_portfolio"],
                                    additional_info={"source": "portfolio_page"}
                                )
                                participations.append(program)
            
            await asyncio.sleep(self.rate_limit_delay)
            
        except Exception as e:
            logger.warning(f"Techstars search failed for {founder_name}: {e}")
        
        return participations
    
    async def _search_500_startups(
        self, 
        founder_name: str, 
        company_names: List[str] = None
    ) -> List[AcceleratorProgram]:
        """Search 500 Startups portfolio."""
        participations = []
        
        try:
            # 500 Startups portfolio endpoint
            url = "https://500.co/companies"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    
                    # Search for company matches
                    if company_names:
                        for company_name in company_names:
                            if company_name.lower() in content.lower():
                                program = AcceleratorProgram(
                                    accelerator_name="500 Startups",
                                    program_name="500 Accelerator",
                                    batch_info="Unknown",
                                    company_name=company_name,
                                    founder_names=[founder_name],
                                    initial_funding=50000.0,  # Typical 500 Startups funding
                                    program_duration_weeks=16,
                                    program_focus="General",
                                    verification_status="likely",
                                    verification_sources=["500_portfolio"],
                                    additional_info={"source": "portfolio_page"}
                                )
                                participations.append(program)
            
            await asyncio.sleep(self.rate_limit_delay)
            
        except Exception as e:
            logger.warning(f"500 Startups search failed for {founder_name}: {e}")
        
        return participations
    
    async def _search_angellist_accelerators(
        self, 
        founder_name: str, 
        company_names: List[str] = None
    ) -> List[AcceleratorProgram]:
        """Search AngelList for accelerator tags."""
        participations = []
        
        try:
            # AngelList search (simplified approach)
            if company_names:
                for company_name in company_names:
                    search_query = quote(f"{company_name} accelerator")
                    url = f"https://angel.co/search?q={search_query}&type=Company"
                    
                    async with self.session.get(url) as response:
                        if response.status == 200:
                            content = await response.text()
                            
                            # Look for accelerator mentions
                            accelerator_keywords = [
                                "y combinator", "techstars", "500 startups",
                                "angelpad", "seedcamp", "plug and play"
                            ]
                            
                            for keyword in accelerator_keywords:
                                if keyword in content.lower():
                                    program = AcceleratorProgram(
                                        accelerator_name=keyword.title(),
                                        program_name="Accelerator Program",
                                        batch_info="Unknown",
                                        company_name=company_name,
                                        founder_names=[founder_name],
                                        verification_status="likely",
                                        verification_sources=["angellist"],
                                        additional_info={"source": "angellist_search"}
                                    )
                                    participations.append(program)
                                    break
                    
                    await asyncio.sleep(self.rate_limit_delay)
            
        except Exception as e:
            logger.warning(f"AngelList search failed for {founder_name}: {e}")
        
        return participations
    
    async def _search_crunchbase_accelerators(
        self, 
        founder_name: str, 
        company_names: List[str] = None
    ) -> List[AcceleratorProgram]:
        """Search Crunchbase for accelerator data."""
        # This would integrate with existing Crunchbase service
        # For now, return empty as Crunchbase data should already include accelerator info
        return []
    
    def _load_accelerator_profiles(self) -> List[AcceleratorProfile]:
        """Load predefined accelerator profiles."""
        return [
            AcceleratorProfile(
                name="Y Combinator",
                website_url="https://www.ycombinator.com",
                api_endpoint="https://yc-oss.github.io/api/all.json",
                typical_funding_amount=250000.0,  # Updated YC standard
                program_duration_weeks=12,
                application_batch_format="W/S YY",
                tier="tier_1"
            ),
            AcceleratorProfile(
                name="Techstars",
                website_url="https://www.techstars.com",
                company_list_url="https://www.techstars.com/portfolio",
                typical_funding_amount=20000.0,
                program_duration_weeks=13,
                tier="tier_1"
            ),
            AcceleratorProfile(
                name="500 Startups",
                website_url="https://500.co",
                company_list_url="https://500.co/companies",
                typical_funding_amount=50000.0,
                program_duration_weeks=16,
                tier="tier_1"
            ),
            AcceleratorProfile(
                name="Plug and Play",
                website_url="https://www.plugandplaytechcenter.com",
                typical_funding_amount=25000.0,
                program_duration_weeks=12,
                tier="tier_1"
            ),
            AcceleratorProfile(
                name="AngelPad",
                website_url="https://angelpad.com",
                typical_funding_amount=120000.0,
                program_duration_weeks=12,
                tier="tier_2"
            ),
            AcceleratorProfile(
                name="Seedcamp",
                website_url="https://seedcamp.com",
                typical_funding_amount=75000.0,
                program_duration_weeks=4,
                tier="tier_2"
            )
        ]
    
    def _deduplicate_participations(
        self, 
        participations: List[AcceleratorProgram]
    ) -> List[AcceleratorProgram]:
        """Remove duplicate accelerator participations."""
        seen = set()
        unique_participations = []
        
        for participation in participations:
            key = (
                participation.accelerator_name.lower(), 
                participation.company_name.lower(),
                participation.batch_info
            )
            if key not in seen:
                seen.add(key)
                unique_participations.append(participation)
        
        return unique_participations
