"""Simplified financial data collection service using Perplexity AI with LangChain structured output."""

import asyncio
from datetime import datetime, date
from typing import List, Optional, Dict, Any
import logging

from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

from ..ranking.models import (
    FounderFinancialProfile, CompanyExit, CompanyFounding, Investment, 
    BoardPosition, ExitType, InvestmentType
)
from .perplexity_base import PerplexityBaseService

logger = logging.getLogger(__name__)


class CompanyFoundingData(BaseModel):
    """Pydantic model for company founding data."""
    company_name: str = Field(description="Name of the company")
    founding_year: Optional[str] = Field(description="Year company was founded (YYYY format)")
    role: str = Field(description="Founder's role in the company")
    current_status: str = Field(description="Current status: active/acquired/closed/public")
    industry: str = Field(description="Industry sector of the company")
    current_valuation_usd: Optional[float] = Field(description="Current valuation in USD")
    exit_value_usd: Optional[float] = Field(description="Exit value in USD if company was sold")
    exit_type: Optional[str] = Field(description="Type of exit: acquisition/ipo/merger/closed")
    exit_year: Optional[str] = Field(description="Year of exit (YYYY format)")
    acquirer: Optional[str] = Field(description="Name of acquiring company")


class InvestmentData(BaseModel):
    """Pydantic model for investment data."""
    company_name: str = Field(description="Name of the company invested in")
    investment_amount_usd: Optional[float] = Field(description="Investment amount in USD")
    investment_year: str = Field(description="Year of investment (YYYY format)")
    investment_type: str = Field(description="Type of investment: angel/seed/series_a/venture")
    current_status: str = Field(description="Current status: active/exited/failed")


class BoardPositionData(BaseModel):
    """Pydantic model for board position data."""
    company_name: str = Field(description="Name of the company")
    position_title: str = Field(description="Board position title")
    start_year: str = Field(description="Year started on board (YYYY format)")
    current_status: str = Field(description="Current status: active/ended")
    company_industry: str = Field(description="Industry of the company")


class NotableAchievementData(BaseModel):
    """Pydantic model for notable achievements."""
    achievement: str = Field(description="Description of the achievement")
    year: str = Field(description="Year of achievement (YYYY format)")
    category: str = Field(description="Category: award/recognition/milestone")


class FinancialProfileData(BaseModel):
    """Pydantic model for comprehensive financial profile data."""
    companies_founded: List[CompanyFoundingData] = Field(description="List of companies founded by the person")
    total_companies_founded: int = Field(description="Total number of companies founded")
    successful_exits: int = Field(description="Number of successful exits")
    total_exit_value_usd: float = Field(description="Total exit value in USD")
    investments_made: List[InvestmentData] = Field(description="List of investments made")
    total_investments_made: int = Field(description="Total number of investments made")
    board_positions: List[BoardPositionData] = Field(description="List of board positions held")
    total_board_positions: int = Field(description="Total number of board positions")
    media_mentions: int = Field(description="Number of media mentions")
    notable_achievements: List[NotableAchievementData] = Field(description="List of notable achievements")
    estimated_net_worth_usd: Optional[float] = Field(description="Estimated net worth in USD")
    confidence_score: float = Field(description="Confidence score between 0 and 1")


class FinancialDataCollector(PerplexityBaseService):
    """Simplified service for collecting founder financial data using Perplexity AI with LangChain structured output."""
    
    def __init__(self):
        super().__init__()
        
        # Set up the structured output parser
        self.parser = JsonOutputParser(pydantic_object=FinancialProfileData)
        
        # Content-focused query template
        self.query_template = """
        Provide comprehensive financial profile information for {founder_name}.
        
        Please include:
        1. Companies founded: name, founding year, role, current status, industry, valuation/exit details
        2. Investment activities: companies invested in, amounts, years, types, current status
        3. Board positions: companies, position titles, start years, current status, industries
        4. Notable achievements: awards, recognitions, milestones with years and categories
        5. Estimated net worth and confidence in the information
        
        Focus on verified, factual information. Use specific numbers where available.
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
        # Input validation
        if not founder_name or not isinstance(founder_name, str):
            logger.error("Invalid founder_name provided")
            return FounderFinancialProfile(
                founder_name=founder_name or "Unknown",
                last_updated=datetime.now(),
                confidence_score=0.0
            )
        
        logger.debug(f"🔍 Collecting financial data for {founder_name} using JSON queries")
        
        profile = FounderFinancialProfile(
            founder_name=founder_name,
            last_updated=datetime.now()
        )
        
        try:
            # Get comprehensive profile
            comprehensive_data = await self._get_comprehensive_profile(founder_name)
            
            if comprehensive_data:
                profile = self._populate_profile_from_structured_data(profile, comprehensive_data, current_company or '')
                profile.confidence_score = comprehensive_data.confidence_score
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
    
    async def _get_comprehensive_profile(self, founder_name: str) -> Optional[FinancialProfileData]:
        """Get comprehensive financial profile using LangChain structured output."""
        try:
            system_prompt = """You are a financial intelligence analyst specializing in founder and executive profiles.
            
            Provide comprehensive, factual information about the person's financial background.
            Focus on verified data and be specific with numbers, dates, and company names.
            If information is not available, use null values rather than guessing."""
            
            # Get format instructions from the parser
            format_instructions = self.parser.get_format_instructions()
            
            query = self.query_template.format(founder_name=founder_name)
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
                        if self._validate_financial_data(parsed_data):
                            logger.debug(f"📊 Got comprehensive profile for {founder_name}")
                            return parsed_data
                        else:
                            logger.warning(f"Invalid financial data structure for {founder_name}")
                            return None
                    except Exception as parse_error:
                        logger.warning(f"Failed to parse structured output for {founder_name}: {parse_error}")
                        logger.debug(f"Raw response content: {content[:500]}...")
                        # Try fallback parsing if available
                        return self._try_fallback_parsing(content, founder_name)
            
        except Exception as e:
            logger.error(f"Error getting comprehensive profile for {founder_name}: {e}")
        
        return None
    
    def _populate_profile_from_structured_data(
        self, 
        profile: FounderFinancialProfile, 
        data: FinancialProfileData, 
        current_company: str
    ) -> FounderFinancialProfile:
        """Populate profile from structured Pydantic data."""
        try:
            # Companies founded
            profile.companies_founded = self._create_company_founding_objects(
                data.companies_founded, current_company
            )
            
            # Company exits
            profile.company_exits = self._create_company_exit_objects(data.companies_founded)
            
            # Investments
            profile.investments_made = self._create_investment_objects(data.investments_made)
            
            # Board positions
            profile.board_positions = self._create_board_position_objects(data.board_positions)
            
            # Direct metrics
            profile.total_companies_founded = data.total_companies_founded
            profile.successful_exits = data.successful_exits
            profile.total_exit_value_usd = data.total_exit_value_usd
            profile.estimated_net_worth_usd = data.estimated_net_worth_usd
            profile.media_mentions = data.media_mentions
            
        except Exception as e:
            logger.error(f"Error populating profile from structured data: {e}")
        
        return profile
    
    def _create_company_founding_objects(
        self, 
        companies_data: List[CompanyFoundingData], 
        current_company: str
    ) -> List[CompanyFounding]:
        """Create CompanyFounding objects from structured data."""
        companies = []
        
        if not companies_data:
            return companies
        
        for company_data in companies_data:
            try:
                founding_date = None
                if company_data.founding_year:
                    try:
                        founding_date = date(int(company_data.founding_year), 1, 1)
                    except (ValueError, TypeError):
                        pass
                
                company = CompanyFounding(
                    company_name=company_data.company_name,
                    founding_date=founding_date,
                    founder_role=company_data.role,
                    industry=company_data.industry,
                    current_status=company_data.current_status,
                    current_valuation_usd=company_data.current_valuation_usd,
                    is_current_company=(company_data.company_name.lower() == (current_company or '').lower()),
                    verification_sources=["perplexity_ai"]
                )
                companies.append(company)
                
            except Exception as e:
                logger.warning(f"Error creating company founding object: {e}")
        
        return companies
    
    def _create_company_exit_objects(self, companies_data: List[CompanyFoundingData]) -> List[CompanyExit]:
        """Create CompanyExit objects from structured data."""
        exits = []
        
        for company_data in companies_data:
            try:
                # Only create exit if there's exit information
                if company_data.exit_value_usd or company_data.exit_type:
                    exit_date = None
                    if company_data.exit_year:
                        try:
                            exit_date = date(int(company_data.exit_year), 1, 1)
                        except (ValueError, TypeError):
                            pass
                    
                    # Map exit type
                    exit_type = ExitType.ACQUISITION
                    if company_data.exit_type == 'ipo':
                        exit_type = ExitType.IPO
                    elif company_data.exit_type == 'merger':
                        exit_type = ExitType.MERGER
                    
                    exit_obj = CompanyExit(
                        company_name=company_data.company_name,
                        exit_type=exit_type,
                        exit_value_usd=company_data.exit_value_usd,
                        exit_date=exit_date,
                        acquirer_name=company_data.acquirer,
                        verification_sources=["perplexity_ai"]
                    )
                    exits.append(exit_obj)
                    
            except Exception as e:
                logger.warning(f"Error creating company exit object: {e}")
        
        return exits
    
    def _create_investment_objects(self, investments_data: List[InvestmentData]) -> List[Investment]:
        """Create Investment objects from structured data."""
        investments = []
        
        if not investments_data:
            return investments
        
        for investment_data in investments_data:
            try:
                investment_date = None
                if investment_data.investment_year:
                    try:
                        investment_date = date(int(investment_data.investment_year), 1, 1)
                    except (ValueError, TypeError):
                        pass
                
                # Map investment type
                investment_type = InvestmentType.ANGEL
                type_str = (investment_data.investment_type or '').lower()
                if type_str == 'seed':
                    investment_type = InvestmentType.ANGEL
                elif type_str == 'series_a':
                    investment_type = InvestmentType.SERIES_A
                elif type_str == 'venture':
                    investment_type = InvestmentType.VENTURE
                
                investment = Investment(
                    company_name=investment_data.company_name,
                    investment_type=investment_type,
                    investment_amount_usd=investment_data.investment_amount_usd,
                    investment_date=investment_date,
                    current_status=investment_data.current_status,
                    verification_sources=["perplexity_ai"]
                )
                investments.append(investment)
                
            except Exception as e:
                logger.warning(f"Error creating investment object: {e}")
        
        return investments
    
    def _create_board_position_objects(self, positions_data: List[BoardPositionData]) -> List[BoardPosition]:
        """Create BoardPosition objects from structured data."""
        positions = []
        
        for position_data in positions_data:
            try:
                start_date = None
                if position_data.start_year:
                    try:
                        start_date = date(int(position_data.start_year), 1, 1)
                    except (ValueError, TypeError):
                        pass
                
                position = BoardPosition(
                    company_name=position_data.company_name,
                    position_title=position_data.position_title,
                    start_date=start_date,
                    is_current=position_data.current_status == 'active',
                    company_industry=position_data.company_industry,
                    verification_sources=["perplexity_ai"]
                )
                positions.append(position)
                
            except Exception as e:
                logger.warning(f"Error creating board position object: {e}")
        
        return positions
    
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
    
    def _validate_financial_data(self, data: FinancialProfileData) -> bool:
        """Validate the parsed financial data structure."""
        try:
            # Check required fields exist
            if not hasattr(data, 'companies_founded') or not hasattr(data, 'confidence_score'):
                return False
            
            # Check confidence score is valid
            if not isinstance(data.confidence_score, (int, float)) or data.confidence_score < 0 or data.confidence_score > 1:
                return False
            
            # Check numeric fields are valid
            if data.total_companies_founded < 0 or data.successful_exits < 0:
                return False
            
            # Check that we have some meaningful data
            has_meaningful_data = (
                len(data.companies_founded) > 0 or 
                len(data.investments_made) > 0 or 
                len(data.board_positions) > 0 or
                data.estimated_net_worth_usd is not None
            )
            
            return has_meaningful_data
            
        except Exception as e:
            logger.warning(f"Error validating financial data: {e}")
            return False
    
    def _try_fallback_parsing(self, content: str, founder_name: str) -> Optional[FinancialProfileData]:
        """Try fallback parsing methods if structured parsing fails."""
        try:
            # Try to extract JSON manually and parse with a more lenient approach
            from backend.utils.data_processing import extract_and_parse_json
            
            json_data = extract_and_parse_json(content)
            if json_data and not json_data.get('error'):
                # Try to create a minimal valid structure
                fallback_data = FinancialProfileData(
                    companies_founded=[],
                    total_companies_founded=json_data.get('total_companies_founded', 0),
                    successful_exits=json_data.get('successful_exits', 0),
                    total_exit_value_usd=json_data.get('total_exit_value_usd', 0),
                    investments_made=[],
                    total_investments_made=json_data.get('total_investments_made', 0),
                    board_positions=[],
                    total_board_positions=json_data.get('total_board_positions', 0),
                    media_mentions=json_data.get('media_mentions', 0),
                    notable_achievements=[],
                    estimated_net_worth_usd=json_data.get('estimated_net_worth_usd'),
                    confidence_score=max(0.1, min(1.0, json_data.get('confidence_score', 0.3)))
                )
                
                # Try to parse companies if available
                if 'companies_founded' in json_data and isinstance(json_data['companies_founded'], list):
                    try:
                        companies = []
                        for company_data in json_data['companies_founded']:
                            if isinstance(company_data, dict):
                                company = CompanyFoundingData(
                                    company_name=company_data.get('company_name', 'Unknown'),
                                    founding_year=company_data.get('founding_year'),
                                    role=company_data.get('role', 'Founder'),
                                    current_status=company_data.get('current_status', 'unknown'),
                                    industry=company_data.get('industry', 'Unknown'),
                                    current_valuation_usd=company_data.get('current_valuation_usd'),
                                    exit_value_usd=company_data.get('exit_value_usd'),
                                    exit_type=company_data.get('exit_type'),
                                    exit_year=company_data.get('exit_year'),
                                    acquirer=company_data.get('acquirer')
                                )
                                companies.append(company)
                        fallback_data.companies_founded = companies
                    except Exception as e:
                        logger.warning(f"Error parsing companies in fallback: {e}")
                
                logger.info(f"Successfully used fallback parsing for {founder_name}")
                return fallback_data
                
        except Exception as e:
            logger.warning(f"Fallback parsing failed for {founder_name}: {e}")
        
        return None