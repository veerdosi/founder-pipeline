"""Crunchbase API integration for enhanced company data."""

import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from ..core import get_logger, RateLimiter, settings
from ..validators import validate_url, validate_funding_amount, validate_year


logger = get_logger(__name__)


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
        self.company_fields = [
            "identifier", "name", "short_description", "description", 
            "website", "founded_on", "categories", "location_identifiers",
            "funding_total", "last_equity_funding_type", "last_equity_funding_total",
            "num_employees_enum", "linkedin", "founder_identifiers",
            "investor_identifiers", "last_funding_on"
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
                "field_ids": self.company_fields,
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
            
            # Add name filter
            if company_name:
                search_query["query"].append({
                    "type": "predicate",
                    "field_id": "name",
                    "operator_id": "contains",
                    "values": [company_name]
                })
            
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
            "field_ids": self.company_fields,
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
                    "field_id": "last_funding_on",
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
            
            # Extract basic information
            name = props.get('name', '')
            if not name:
                return None
            
            # Process funding information
            funding_total = self._safe_extract_funding(props.get('funding_total'))
            last_funding = self._safe_extract_funding(props.get('last_equity_funding_total'))
            
            # Extract categories
            categories = []
            if props.get('categories'):
                categories = [cat.get('value', '') for cat in props['categories'] if cat.get('value')]
            
            # Extract location
            location = None
            if props.get('location_identifiers'):
                locations = props['location_identifiers']
                if locations:
                    location = locations[0].get('value', '')
            
            # Extract founders and investors
            founders = self._extract_person_names(props.get('founder_identifiers', []))
            investors = self._extract_investor_names(props.get('investor_identifiers', []))
            
            # Calculate data quality score
            quality_score = self._calculate_data_quality(props)
            
            return CrunchbaseCompany(
                name=name,
                description=props.get('short_description', '') or props.get('description', ''),
                website=props.get('website', {}).get('value') if props.get('website') else None,
                founded_date=self._extract_date(props.get('founded_on')),
                employee_count=self._extract_employee_count(props.get('num_employees_enum')),
                funding_total=funding_total,
                funding_stage=props.get('last_equity_funding_type', {}).get('value') if props.get('last_equity_funding_type') else None,
                last_funding_date=self._extract_date(props.get('last_funding_on')),
                last_funding_amount=last_funding,
                investor_count=len(investors),
                categories=categories,
                headquarters_location=location,
                linkedin_url=props.get('linkedin', {}).get('value') if props.get('linkedin') else None,
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
            name = props.get('name', '').lower()
            website = props.get('website', {}).get('value', '') if props.get('website') else ''
            
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
