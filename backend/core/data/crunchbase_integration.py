"""Crunchbase API integration for enhanced company data."""

import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from ...core import settings
from ...validators import validate_url, validate_funding_amount, validate_year

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
        self.rate_limiter = RateLimiter(max_requests=200, time_window=60)  # Crunchbase limits
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
    
    async def search_ai_companies(
        self, 
        limit: int = 50,
        funding_stages: Optional[List[str]] = None,
        location_filter: Optional[str] = None,
        founded_after: Optional[int] = None
    ) -> List[CrunchbaseCompany]:
        """Search for AI companies using Crunchbase API."""
        try:
            logger.info(f"🔍 Searching Crunchbase for {limit} AI companies")
            
            # Prepare search query
            query_payload = self._build_search_query(
                limit=limit,
                funding_stages=funding_stages,
                location_filter=location_filter,
                founded_after=founded_after
            )
            
            # Execute search
            search_results = await self._execute_search(query_payload)
            
            # Process and enrich results
            companies = []
            for item in search_results.get('entities', []):
                try:
                    company = await self._process_company_data(item)
                    if company:
                        companies.append(company)
                except Exception as e:
                    logger.warning(f"Error processing company data: {e}")
                    continue
            
            logger.info(f"✅ Found {len(companies)} Crunchbase AI companies")
            return companies
            
        except Exception as e:
            logger.error(f"Error searching Crunchbase: {e}")
            return []
    
    async def enrich_existing_company(
        self, 
        company_name: str, 
        website: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Enrich an existing company with Crunchbase data."""
        try:
            # Search for the company by name
            search_query = {
                "field_ids": self.field_ids,
                "query": [
                    {
                        "type": "predicate",
                        "field_id": "facet_ids",
                        "operator_id": "includes",
                        "values": ["company"]
                    }
                ],
                "order": [
                    {
                        "field_id": "rank_org",
                        "sort": "asc"
                    }
                ],
                "limit": 10
            }
            
            # Use autocomplete API for name-based search instead of contains operator
            if company_name:
                autocomplete_results = await self._autocomplete_search(company_name)
                if autocomplete_results:
                    # Get the best match UUID and fetch detailed data
                    best_match_uuid = autocomplete_results[0].get('uuid')
                    if best_match_uuid:
                        return await self._get_company_by_uuid(best_match_uuid)
            
            # Fallback to category/website search without name filter
            search_query = {
                "field_ids": self.field_ids,
                "query": [
                    {
                        "type": "predicate",
                        "field_id": "facet_ids",
                        "operator_id": "includes",
                        "values": ["company"]
                    }
                ],
                "limit": 10
            }
            
            results = await self._execute_search(search_query)
            
            # Find best match
            best_match = self._find_best_company_match(
                results.get('entities', []), 
                company_name, 
                website
            )
            
            if best_match:
                return await self._process_company_data(best_match)
            
            return None
            
        except Exception as e:
            logger.error(f"Error enriching company {company_name}: {e}")
            return None
    
    def _build_search_query(
        self,
        limit: int,
        funding_stages: Optional[List[str]],
        location_filter: Optional[str],
        founded_after: Optional[int]
    ) -> Dict[str, Any]:
        """Build Crunchbase search query."""
        current_year = datetime.now().year
        founded_after = founded_after or (current_year - 6)  # Last 6 years by default
        
        query = {
            "field_ids": self.field_ids,
            "query": [
                {
                    "type": "predicate",
                    "field_id": "facet_ids",
                    "operator_id": "includes",
                    "values": ["company"]
                },
                {
                    "type": "predicate", 
                    "field_id": "categories",
                    "operator_id": "includes",
                    "values": list(self.ai_category_uuids.values())
                },
                {
                    "type": "predicate",
                    "field_id": "founded_on",
                    "operator_id": "gte",
                    "values": [f"{founded_after}-01-01"]
                }
            ],
            "order": [
                {
                    "field_id": "last_funding_at",
                    "sort": "desc"
                }
            ],
            "limit": min(limit, 1000)  # Crunchbase limit
        }
        
        # Add funding stage filter
        if funding_stages:
            stage_map = {
                'seed': 'seed',
                'series-a': 'series_a',
                'series-b': 'series_b', 
                'series-c': 'series_c',
                'pre-seed': 'pre_seed'
            }
            
            cb_stages = [stage_map.get(stage, stage) for stage in funding_stages]
            query["query"].append({
                "type": "predicate",
                "field_id": "last_equity_funding_type",
                "operator_id": "includes", 
                "values": cb_stages
            })
        
        # Add location filter
        if location_filter:
            query["query"].append({
                "type": "predicate",
                "field_id": "location_identifiers",
                "operator_id": "includes",
                "values": [self._get_location_uuid(location_filter)]
            })
        
        return query
    
    async def _autocomplete_search(self, query: str) -> List[Dict]:
        """Search companies using Crunchbase Autocomplete API."""
        try:
            url = f"{self.base_url}/data/autocompletes"
            params = {
                'user_key': self.api_key,
                'query': query,
                'collection_ids': 'organization.companies',
                'limit': 5
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('entities', [])
                else:
                    logger.warning(f"Autocomplete search failed {response.status}: {await response.text()}")
                    return []
        except Exception as e:
            logger.error(f"Error in autocomplete search: {e}")
            return []
    
    async def _get_company_by_uuid(self, uuid: str) -> Optional[CrunchbaseCompany]:
        """Get detailed company data by UUID."""
        try:
            url = f"{self.base_url}/data/entities/organizations/{uuid}"
            params = {
                'user_key': self.api_key,
                'field_ids': ','.join(self.field_ids)
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    properties = data.get('properties', {})
                    return await self._process_company_data({'properties': properties})
                else:
                    logger.warning(f"Company lookup failed {response.status}: {await response.text()}")
                    return None
        except Exception as e:
            logger.error(f"Error getting company by UUID: {e}")
            return None

    async def _execute_search(self, query_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Crunchbase search API call."""
        await self.rate_limiter.acquire()
        
        url = f"{self.base_url}/data/searches/organizations"
        params = {'user_key': self.api_key}
        
        async with self.session.post(url, params=params, json=query_payload) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                logger.error(f"Crunchbase search failed {response.status}: {error_text}")
                return {}
    
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
            
            return CrunchbaseCompany(
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
            
        except Exception as e:
            logger.error(f"Error processing company data: {e}")
            return None
    
    def _find_best_company_match(
        self, 
        entities: List[Dict], 
        target_name: str, 
        target_website: Optional[str]
    ) -> Optional[Dict]:
        """Find the best matching company from search results."""
        if not entities:
            return None
        
        target_name_lower = target_name.lower()
        best_match = None
        best_score = 0
        
        for entity in entities:
            props = entity.get('properties', {})
            if not isinstance(props, dict):
                continue
            
            name = props.get('name', '').lower()
            
            # Safe website extraction
            website = ''
            website_data = props.get('website')
            if isinstance(website_data, dict):
                website = website_data.get('value', '')
            elif isinstance(website_data, str):
                website = website_data
            
            score = 0
            
            # Exact name match
            if name == target_name_lower:
                score += 100
            # Partial name match
            elif target_name_lower in name or name in target_name_lower:
                score += 50
            
            # Website match
            if target_website and website:
                if target_website.lower() in website.lower() or website.lower() in target_website.lower():
                    score += 30
            
            if score > best_score:
                best_score = score
                best_match = entity
        
        return best_match if best_score > 25 else None  # Minimum threshold
    
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
    
    def _get_location_uuid(self, location: str) -> str:
        """Get Crunchbase UUID for location (simplified mapping)."""
        location_map = {
            'united states': 'f110fca2-1055-99f6-996d-011c198b3928',
            'california': 'eb879a83-c91a-121e-95a8-e65bad1f1f3c', 
            'new york': 'f6c74c55-be0b-4adf-9e0e-7d7fa4c94d43',
            'united kingdom': '8b04f56e-b3ec-400a-b2b7-3f2c2e077eba',
            'canada': '2cc3f133-e929-5bb7-5eef-4f5bd0fb0c28'
        }
        return location_map.get(location.lower(), location_map['united states'])
