"""Crunchbase API integration for enhanced company data."""

import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from ...core import settings
from ...utils.rate_limiter import RateLimiter
from ...utils.validators import validate_url, validate_funding_amount, validate_year

import logging
logger = logging.getLogger(__name__)


@dataclass
class CrunchbaseCompany:
    """Structured Crunchbase company data."""
    name: str
    description: str
    website: Optional[str]
    founded_date: Optional[str]
    employee_count: Optional[int]
    funding_total: Optional[float]
    funding_stage: Optional[str]
    last_funding_date: Optional[str]
    last_funding_amount: Optional[float]
    investor_count: Optional[int]
    categories: List[str]
    headquarters_location: Optional[str]
    linkedin_url: Optional[str]
    founder_names: List[str]
    key_investors: List[str]
    crunchbase_url: str
    data_quality_score: float


class CrunchbaseService:
    """Crunchbase API integration for comprehensive company data."""
    
    def __init__(self):
        self.api_key = settings.crunchbase_api_key
        self.base_url = "https://api.crunchbase.com/v4"
        # Conservative rate limiting: 150 requests per minute (75% of 200 limit) with buffer
        self.rate_limiter = RateLimiter(max_requests=150, time_window=60)
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Crunchbase category UUIDs for AI companies
        self.ai_category_uuids = {
            'artificial_intelligence': 'c4d8caf3-5fe7-359b-f9f2-2d708378e4ee',
            'machine_learning': '5ea0cdb7-c9a6-47fc-50f8-c9b0fac04863',
            'computer_vision': 'fdf2e811-0311-4b8c-b3b0-12c5cf21c54d',
            'natural_language_processing': '3a4e9e8b-5c4d-4a12-8f9c-1b2e3f4a5b6c',
            'robotics': 'c4d62b58-0e8a-4c8d-9b7f-4e5a1b2c3d4f',
            'autonomous_vehicles': '3e42c36c-20ee-42c3-aa98-eb57bbeeca82',
            'deep_learning': 'f4a1b2c3-d4e5-4f6a-9b8c-1e2f3a4b5c6d'
        }
        
        # Standard field sets for different queries
        self.field_ids = [
            "identifier", "name", "short_description", "description", 
            "website", "founded_on", "categories", "location_identifiers",
            "funding_total", "last_equity_funding_type", "last_equity_funding_total",
            "num_employees_enum", "linkedin", "founder_identifiers",
            "investor_identifiers", "last_funding_at"
        ]
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    
    async def enrich_existing_company(
        self, 
        company_name: str, 
        website: Optional[str] = None,
        crunchbase_url: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Enrich an existing company with Crunchbase data using direct URL lookup only."""
        logger.info(f"🔍 Enriching Crunchbase data for: {company_name}")
        
        # Validate API key
        if not self.api_key:
            logger.error("Crunchbase API key is not configured")
            return None
        
        # Only proceed if we have a Crunchbase URL
        if not crunchbase_url:
            logger.info(f"❌ No Crunchbase URL provided for: {company_name}")
            return None
        
        logger.info(f"📋 Using Crunchbase URL: {crunchbase_url}")
        try:
            # Extract UUID from URL like https://www.crunchbase.com/organization/company-name
            import re
            uuid_match = re.search(r'/organization/([^/?]+)', crunchbase_url)
            if uuid_match:
                uuid = uuid_match.group(1)
                logger.info(f"📋 Extracted UUID: {uuid}")
                company_data = await self._get_company_by_uuid(uuid)
                if company_data:
                    logger.info(f"✅ Successfully enriched {company_name} from Crunchbase")
                    return company_data
                else:
                    logger.info(f"❌ Failed to fetch data for UUID: {uuid}")
                    return None
            else:
                logger.warning(f"Failed to extract UUID from URL: {crunchbase_url}")
                return None
        except Exception as e:
            logger.error(f"Error enriching company {company_name}: {e}")
            return None
    
    
    async def _get_company_by_uuid(self, uuid: str) -> Optional[CrunchbaseCompany]:
        """Get detailed company data by UUID."""
        try:
            await self.rate_limiter.acquire()
            
            if not self.session:
                self.session = aiohttp.ClientSession()
                
            url = f"{self.base_url}/data/entities/organizations/{uuid}"
            params = {
                'user_key': self.api_key,
                'field_ids': ','.join(self.field_ids)
            }
            
            logger.debug(f"Fetching company data for UUID: {uuid}")
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    properties = data.get('properties', {})
                    company_data = await self._process_company_data({'properties': properties, 'uuid': uuid})
                    if company_data:
                        logger.info(f"✅ Successfully fetched Crunchbase data for UUID: {uuid}")
                    return company_data
                else:
                    error_text = await response.text()
                    logger.warning(f"Company lookup failed {response.status} for UUID {uuid}: {error_text}")
                    return None
        except Exception as e:
            logger.error(f"Error getting company by UUID {uuid}: {e}")
            return None

    
    async def _process_company_data(self, entity_data: Dict[str, Any]) -> Optional[CrunchbaseCompany]:
        """Process raw Crunchbase entity data into structured format."""
        try:
            props = entity_data.get('properties', {})
            
            # Ensure props is a dict
            if not isinstance(props, dict):
                logger.warning(f"Properties is not a dict: {type(props)}")
                return None
            
            # Extract basic information
            name = props.get('name', '')
            if not name:
                return None
            
            # Process funding information
            funding_total = self._safe_extract_funding(props.get('funding_total'))
            last_funding = self._safe_extract_funding(props.get('last_equity_funding_total'))
            
            # Extract categories with safe dict access
            categories = []
            categories_data = props.get('categories')
            if categories_data and isinstance(categories_data, list):
                categories = [cat.get('value', '') for cat in categories_data if isinstance(cat, dict) and cat.get('value')]
            
            # Extract location with safe dict access
            location = None
            location_data = props.get('location_identifiers')
            if location_data and isinstance(location_data, list):
                for loc in location_data:
                    if isinstance(loc, dict) and loc.get('value'):
                        location = loc.get('value', '')
                        break
            
            # Extract founders and investors
            founders = self._extract_person_names(props.get('founder_identifiers', []))
            investors = self._extract_investor_names(props.get('investor_identifiers', []))
            
            # Calculate data quality score
            quality_score = self._calculate_data_quality(props)
            
            # Safe extraction of nested dict values
            website = None
            website_data = props.get('website')
            if isinstance(website_data, dict):
                website = website_data.get('value')
            elif isinstance(website_data, str):
                website = website_data
            
            funding_stage = None
            funding_stage_data = props.get('last_equity_funding_type')
            if isinstance(funding_stage_data, dict):
                funding_stage = funding_stage_data.get('value')
            elif isinstance(funding_stage_data, str):
                funding_stage = funding_stage_data
            
            linkedin_url = None
            linkedin_data = props.get('linkedin')
            if isinstance(linkedin_data, dict):
                linkedin_url = linkedin_data.get('value')
            elif isinstance(linkedin_data, str):
                linkedin_url = linkedin_data
            
            company = CrunchbaseCompany(
                name=name,
                description=props.get('short_description', '') or props.get('description', ''),
                website=website,
                founded_date=self._extract_date(props.get('founded_on')),
                employee_count=self._extract_employee_count(props.get('num_employees_enum')),
                funding_total=funding_total,
                funding_stage=funding_stage,
                last_funding_date=self._extract_date(props.get('last_funding_at')),
                last_funding_amount=last_funding,
                investor_count=len(investors),
                categories=categories,
                headquarters_location=location,
                linkedin_url=linkedin_url,
                founder_names=founders,
                key_investors=investors[:10],  # Top 10 investors
                crunchbase_url=f"https://www.crunchbase.com/organization/{entity_data.get('uuid', '')}",
                data_quality_score=quality_score
            )
            
            logger.debug(f"Processed company data for: {name}")
            return company
            
        except Exception as e:
            logger.error(f"Error processing company data: {e}")
            return None
    
    
    def _safe_extract_funding(self, funding_data: Optional[Dict]) -> Optional[float]:
        """Safely extract funding amount from Crunchbase data."""
        if not funding_data or not isinstance(funding_data, dict):
            return None
        
        try:
            value = funding_data.get('value_usd')
            if value and validate_funding_amount(value):
                return float(value)
        except (ValueError, TypeError):
            pass
        
        return None
    
    def _extract_date(self, date_data: Optional[Dict]) -> Optional[str]:
        """Extract and format date from Crunchbase data."""
        if not date_data or not isinstance(date_data, dict):
            return None
        
        try:
            value = date_data.get('value')
            if value:
                # Parse and validate date
                parsed_date = datetime.fromisoformat(value.replace('Z', '+00:00'))
                return parsed_date.strftime('%Y-%m-%d')
        except (ValueError, TypeError):
            pass
        
        return None
    
    def _extract_employee_count(self, employee_data: Optional[Dict]) -> Optional[int]:
        """Extract employee count from Crunchbase enum."""
        if not employee_data or not isinstance(employee_data, dict):
            return None
        
        value = employee_data.get('value', '')
        
        # Map Crunchbase employee ranges to approximate numbers
        employee_map = {
            'c_00001_00010': 5,
            'c_00011_00050': 25,
            'c_00051_00100': 75,
            'c_00101_00250': 175,
            'c_00251_00500': 375,
            'c_00501_01000': 750,
            'c_01001_05000': 2500,
            'c_05001_10000': 7500,
            'c_10001_max': 15000
        }
        
        return employee_map.get(value)
    
    def _extract_person_names(self, person_list: List[Dict]) -> List[str]:
        """Extract person names from Crunchbase identifier list."""
        names = []
        for person in person_list[:5]:  # Limit to first 5
            if isinstance(person, dict) and person.get('value'):
                # Clean up the identifier to get a readable name
                name = person['value'].replace('-', ' ').title()
                names.append(name)
        return names
    
    def _extract_investor_names(self, investor_list: List[Dict]) -> List[str]:
        """Extract investor names from Crunchbase identifier list."""
        return self._extract_person_names(investor_list)
    
    def _calculate_data_quality(self, props: Dict[str, Any]) -> float:
        """Calculate data quality score based on field completeness."""
        important_fields = [
            'name', 'short_description', 'website', 'founded_on',
            'funding_total', 'last_equity_funding_type', 'categories'
        ]
        
        available_fields = sum(1 for field in important_fields if props.get(field) is not None)
        return available_fields / len(important_fields)
    
