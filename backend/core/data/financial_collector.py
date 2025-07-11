"""Simplified financial data collection service using Perplexity AI with direct JSON responses."""

import asyncio
from datetime import datetime, date
from typing import List, Optional, Dict, Any
import logging
import json

from ..ranking.models import (
    FounderFinancialProfile, CompanyExit, CompanyFounding, Investment, 
    BoardPosition, ExitType, InvestmentType
)
from .perplexity_base import PerplexityBaseService

logger = logging.getLogger(__name__)


class FinancialDataCollector(PerplexityBaseService):
    """Simplified service for collecting founder financial data using Perplexity AI with JSON responses."""
    
    def __init__(self):
        super().__init__()
        
        # Direct JSON request query
        self.json_query = """
        Please provide a comprehensive financial profile for {founder_name} in the following JSON format:
        
        {{
            "companies_founded": [
                {{
                    "company_name": "string",
                    "founding_year": "YYYY",
                    "role": "string",
                    "current_status": "active/acquired/closed/public",
                    "industry": "string",
                    "current_valuation_usd": number_or_null,
                    "exit_value_usd": number_or_null,
                    "exit_type": "acquisition/ipo/merger/closed/null",
                    "exit_year": "YYYY_or_null",
                    "acquirer": "string_or_null"
                }}
            ],
            "total_companies_founded": number,
            "successful_exits": number,
            "total_exit_value_usd": number,
            "investments_made": [
                {{
                    "company_name": "string",
                    "investment_amount_usd": number_or_null,
                    "investment_year": "YYYY",
                    "investment_type": "angel/seed/series_a/venture",
                    "current_status": "active/exited/failed"
                }}
            ],
            "total_investments_made": number,
            "board_positions": [
                {{
                    "company_name": "string",
                    "position_title": "string",
                    "start_year": "YYYY",
                    "current_status": "active/ended",
                    "company_industry": "string"
                }}
            ],
            "total_board_positions": number,
            "media_mentions": number,
            "notable_achievements": [
                {{
                    "achievement": "string",
                    "year": "YYYY",
                    "category": "award/recognition/milestone"
                }}
            ],
            "estimated_net_worth_usd": number_or_null,
            "confidence_score": number_between_0_and_1
        }}
        
        Include only verified, factual information. If information is not available, use null values.
        Be thorough but accurate. Focus on quantifiable metrics and major achievements.
        """
    
    async def collect_data(self, founder_name: str, current_company: str) -> FounderFinancialProfile:
        """Collect comprehensive financial data for a founder using direct JSON queries."""
        return await self.collect_founder_financial_data(founder_name, current_company)
    
    async def collect_founder_financial_data(
        self, 
        founder_name: str,
        current_company: str,
        linkedin_url: Optional[str] = None
    ) -> FounderFinancialProfile:
        """Collect comprehensive financial data using simplified JSON approach."""
        logger.debug(f"🔍 Collecting financial data for {founder_name} using JSON queries")
        
        profile = FounderFinancialProfile(
            founder_name=founder_name,
            last_updated=datetime.now()
        )
        
        try:
            # Get comprehensive profile
            comprehensive_data = await self._get_comprehensive_profile(founder_name)
            
            if comprehensive_data:
                profile = self._populate_profile_from_json(profile, comprehensive_data, current_company)
                profile.confidence_score = comprehensive_data.get('confidence_score', 0.7)
            else:
                logger.warning(f"No financial data found for {founder_name}")
                profile.confidence_score = 0.1
            
            # Set data sources
            profile.data_sources = ['perplexity_ai', 'web_search']
            
            # Calculate final metrics
            profile.calculate_metrics()
            
            logger.info(f"✅ Financial data collected for {founder_name}: "
                       f"{len(profile.company_exits)} exits, "
                       f"{len(profile.companies_founded)} companies founded")
            
        except Exception as e:
            logger.error(f"Error collecting financial data for {founder_name}: {e}")
            profile.confidence_score = 0.1
        
        return profile
    
    async def _get_comprehensive_profile(self, founder_name: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive financial profile as JSON."""
        try:
            system_prompt = """You are a financial intelligence analyst. 
            Provide accurate, factual information about entrepreneurs and business leaders.
            Respond ONLY with valid JSON. No additional text or formatting.
            If you don't have information, use null values rather than guessing."""
            
            query = self.json_query.format(founder_name=founder_name)
            
            response = await self.query_perplexity(
                query=query,
                system_prompt=system_prompt,
                max_tokens=3000
            )
            
            if response:
                content = self.extract_content_from_response(response)
                if content:
                    # Clean and parse JSON
                    json_data = self._extract_json_from_content(content)
                    if json_data:
                        logger.debug(f"📊 Got comprehensive profile for {founder_name}")
                        return json_data
            
        except Exception as e:
            logger.error(f"Error getting comprehensive profile for {founder_name}: {e}")
        
        return None
    
    def _extract_json_from_content(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON from content."""
        try:
            # Remove any markdown formatting
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            # Try to parse JSON
            return json.loads(content)
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {e}")
            # Try to find JSON object in the content
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
        
        return None
    
    def _populate_profile_from_json(
        self, 
        profile: FounderFinancialProfile, 
        data: Dict[str, Any], 
        current_company: str
    ) -> FounderFinancialProfile:
        """Populate profile from comprehensive JSON data."""
        try:
            # Companies founded
            if 'companies_founded' in data:
                profile.companies_founded = self._create_company_founding_objects(
                    data['companies_founded'], current_company
                )
            
            # Company exits
            if 'companies_founded' in data:
                profile.company_exits = self._create_company_exit_objects(data['companies_founded'])
            
            # Investments
            if 'investments_made' in data:
                profile.investments_made = self._create_investment_objects(data['investments_made'])
            
            # Board positions
            if 'board_positions' in data:
                profile.board_positions = self._create_board_position_objects(data['board_positions'])
            
            # Direct metrics
            profile.total_companies_founded = data.get('total_companies_founded', 0)
            profile.successful_exits = data.get('successful_exits', 0)
            profile.total_exit_value_usd = data.get('total_exit_value_usd', 0)
            profile.estimated_net_worth_usd = data.get('estimated_net_worth_usd')
            profile.confidence_score = data.get('confidence_score', 0.7)
            
            # Additional metrics
            if 'media_mentions' in data:
                profile.media_mentions = data['media_mentions']
            
        except Exception as e:
            logger.error(f"Error populating profile from JSON: {e}")
        
        return profile
    
    def _create_company_founding_objects(
        self, 
        companies_data: List[Dict[str, Any]], 
        current_company: str
    ) -> List[CompanyFounding]:
        """Create CompanyFounding objects from JSON data."""
        companies = []
        
        for company_data in companies_data:
            try:
                founding_date = None
                if company_data.get('founding_year'):
                    try:
                        founding_date = date(int(company_data['founding_year']), 1, 1)
                    except (ValueError, TypeError):
                        pass
                
                company = CompanyFounding(
                    company_name=company_data.get('company_name', 'Unknown'),
                    founding_date=founding_date,
                    founder_role=company_data.get('role', 'Founder'),
                    industry=company_data.get('industry', 'Unknown'),
                    current_status=company_data.get('current_status', 'unknown'),
                    current_valuation_usd=company_data.get('current_valuation_usd'),
                    is_current_company=(company_data.get('company_name', '').lower() == current_company.lower()),
                    verification_sources=["perplexity_ai"]
                )
                companies.append(company)
                
            except Exception as e:
                logger.warning(f"Error creating company founding object: {e}")
        
        return companies
    
    def _create_company_exit_objects(self, companies_data: List[Dict[str, Any]]) -> List[CompanyExit]:
        """Create CompanyExit objects from JSON data."""
        exits = []
        
        for company_data in companies_data:
            try:
                # Only create exit if there's exit information
                if company_data.get('exit_value_usd') or company_data.get('exit_type'):
                    exit_date = None
                    if company_data.get('exit_year'):
                        try:
                            exit_date = date(int(company_data['exit_year']), 1, 1)
                        except (ValueError, TypeError):
                            pass
                    
                    # Map exit type
                    exit_type = ExitType.ACQUISITION
                    if company_data.get('exit_type') == 'ipo':
                        exit_type = ExitType.IPO
                    elif company_data.get('exit_type') == 'merger':
                        exit_type = ExitType.MERGER
                    
                    exit_obj = CompanyExit(
                        company_name=company_data.get('company_name', 'Unknown'),
                        exit_type=exit_type,
                        exit_value_usd=company_data.get('exit_value_usd'),
                        exit_date=exit_date,
                        acquirer_name=company_data.get('acquirer'),
                        verification_sources=["perplexity_ai"]
                    )
                    exits.append(exit_obj)
                    
            except Exception as e:
                logger.warning(f"Error creating company exit object: {e}")
        
        return exits
    
    def _create_investment_objects(self, investments_data: List[Dict[str, Any]]) -> List[Investment]:
        """Create Investment objects from JSON data."""
        investments = []
        
        for investment_data in investments_data:
            try:
                investment_date = None
                if investment_data.get('investment_year'):
                    try:
                        investment_date = date(int(investment_data['investment_year']), 1, 1)
                    except (ValueError, TypeError):
                        pass
                
                # Map investment type
                investment_type = InvestmentType.ANGEL
                type_str = investment_data.get('investment_type', '').lower()
                if type_str == 'seed':
                    investment_type = InvestmentType.ANGEL
                elif type_str == 'series_a':
                    investment_type = InvestmentType.SERIES_A
                elif type_str == 'venture':
                    investment_type = InvestmentType.VENTURE
                
                investment = Investment(
                    company_name=investment_data.get('company_name', 'Unknown'),
                    investment_type=investment_type,
                    investment_amount_usd=investment_data.get('investment_amount_usd'),
                    investment_date=investment_date,
                    current_status=investment_data.get('current_status', 'active'),
                    verification_sources=["perplexity_ai"]
                )
                investments.append(investment)
                
            except Exception as e:
                logger.warning(f"Error creating investment object: {e}")
        
        return investments
    
    def _create_board_position_objects(self, positions_data: List[Dict[str, Any]]) -> List[BoardPosition]:
        """Create BoardPosition objects from JSON data."""
        positions = []
        
        for position_data in positions_data:
            try:
                start_date = None
                if position_data.get('start_year'):
                    try:
                        start_date = date(int(position_data['start_year']), 1, 1)
                    except (ValueError, TypeError):
                        pass
                
                position = BoardPosition(
                    company_name=position_data.get('company_name', 'Unknown'),
                    position_title=position_data.get('position_title', 'Board Member'),
                    start_date=start_date,
                    is_current=position_data.get('current_status', 'active') == 'active',
                    company_industry=position_data.get('company_industry', 'Unknown'),
                    verification_sources=["perplexity_ai"]
                )
                positions.append(position)
                
            except Exception as e:
                logger.warning(f"Error creating board position object: {e}")
        
        return positions