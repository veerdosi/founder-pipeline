"""Financial data collection service for comprehensive founder financial intelligence."""

import asyncio
import re
from datetime import datetime, date
from typing import List, Optional, Dict, Any, Tuple
import aiohttp
from urllib.parse import quote_plus
import logging

from ..ranking.models import (
    FounderFinancialProfile, CompanyExit, CompanyFounding, Investment, 
    BoardPosition, ExitType, InvestmentType
)
from ..config import settings
from ...utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class FinancialDataCollector:
    """Service for collecting comprehensive founder financial data."""
    
    def __init__(self):
        self.rate_limiter = RateLimiter(max_requests=4, time_window=1)  # 4 requests per second to stay under Serper's 5/sec limit
        self.session: Optional[aiohttp.ClientSession] = None
        
        # API endpoints for different data sources
        self.endpoints = {
            'crunchbase': 'https://api.crunchbase.com/api/v4',
            'sec': 'https://data.sec.gov/api',
            'pitchbook': 'https://api.pitchbook.com',  # If available
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def collect_founder_financial_data(
        self, 
        founder_name: str,
        current_company: str,
        linkedin_url: Optional[str] = None
    ) -> FounderFinancialProfile:
        """Collect comprehensive financial data for a founder."""
        logger.info(f"🔍 Collecting financial data for {founder_name}")
        logger.debug(f"📋 Parameters: company={current_company}, linkedin_url={linkedin_url}")
        
        profile = FounderFinancialProfile(
            founder_name=founder_name,
            last_updated=datetime.now()
        )
        
        try:
            # Collect data from multiple sources in parallel
            logger.debug(f"🚀 Starting parallel data collection tasks for {founder_name}")
            tasks = [
                self._collect_company_founding_data(founder_name, current_company),
                self._collect_exit_data(founder_name),
                self._collect_investment_data(founder_name),
                self._collect_board_positions(founder_name)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            companies_founded, exits, investments, board_positions = [], [], [], []
            task_names = ["founding", "exits", "investments", "board_positions"]
            
            for i, result in enumerate(results):
                if not isinstance(result, Exception):
                    if i == 0:  # Company founding data
                        companies_founded = result
                        logger.debug(f"✅ {task_names[i]} data: {len(companies_founded)} companies found")
                    elif i == 1:  # Exit data
                        exits = result
                        logger.debug(f"✅ {task_names[i]} data: {len(exits)} exits found")
                    elif i == 2:  # Investment data
                        investments = result
                        logger.debug(f"✅ {task_names[i]} data: {len(investments)} investments found")
                    elif i == 3:  # Board positions
                        board_positions = result
                        logger.debug(f"✅ {task_names[i]} data: {len(board_positions)} positions found")
                else:
                    logger.error(f"❌ Task {task_names[i]} failed for {founder_name}: {result}")
            
            # Update profile
            profile.companies_founded = companies_founded
            profile.company_exits = exits
            profile.investments_made = investments
            profile.board_positions = board_positions
            
            # Calculate derived metrics
            logger.debug(f"📊 Calculating metrics for {founder_name}")
            profile.calculate_metrics()
            
            # Set data sources and confidence
            profile.data_sources = ['web_search', 'sec_filings', 'crunchbase']
            profile.confidence_score = self._calculate_financial_confidence(profile)
            
            logger.info(f"✅ Financial data collected for {founder_name}: "
                       f"{len(exits)} exits, {len(companies_founded)} companies founded, "
                       f"{len(investments)} investments, {len(board_positions)} board positions, "
                       f"confidence: {profile.confidence_score:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Error collecting financial data for {founder_name}: {e}", exc_info=True)
            profile.confidence_score = 0.1
        
        return profile
    
    async def _collect_company_founding_data(
        self, 
        founder_name: str, 
        current_company: str
    ) -> List[CompanyFounding]:
        """Collect data about companies founded by the founder."""
        companies = []
        
        try:
            # Search for companies founded by this person
            search_queries = [
                f'"{founder_name}" founder "{current_company}"',
                f'"{founder_name}" co-founder companies founded',
                f'"{founder_name}" CEO founder startup'
            ]
            
            for query in search_queries:
                results = await self._search_web_for_financial_data(query, "founding")
                companies.extend(await self._extract_founding_data(results, founder_name))
                
                # Rate limiting - increased delay to prevent 429 errors
                await asyncio.sleep(0.5)
            
            # Add current company
            current_company_data = CompanyFounding(
                company_name=current_company,
                founder_role="Founder/CEO",
                is_current_company=True,
                verification_sources=["linkedin_profile"]
            )
            companies.append(current_company_data)
            
            # Deduplicate by company name
            unique_companies = {}
            for company in companies:
                if company.company_name not in unique_companies:
                    unique_companies[company.company_name] = company
                else:
                    # Merge data from multiple sources
                    existing = unique_companies[company.company_name]
                    if company.founding_date and not existing.founding_date:
                        existing.founding_date = company.founding_date
                    if company.current_valuation_usd and not existing.current_valuation_usd:
                        existing.current_valuation_usd = company.current_valuation_usd
                    existing.verification_sources.extend(company.verification_sources)
            
            return list(unique_companies.values())
            
        except Exception as e:
            logger.error(f"Error collecting founding data for {founder_name}: {e}")
            return []
    
    async def _collect_exit_data(self, founder_name: str) -> List[CompanyExit]:
        """Collect data about company exits."""
        exits = []
        
        try:
            # Search for exits
            search_queries = [
                f'"{founder_name}" IPO exit acquisition',
                f'"{founder_name}" company sold acquired',
                f'"{founder_name}" founder exit billion million'
            ]
            
            for query in search_queries:
                results = await self._search_web_for_financial_data(query, "exit")
                exits.extend(await self._extract_exit_data(results, founder_name))
                await asyncio.sleep(1)
            
            # Deduplicate and validate
            unique_exits = self._deduplicate_exits(exits)
            return unique_exits
            
        except Exception as e:
            logger.error(f"Error collecting exit data for {founder_name}: {e}")
            return []
    
    async def _collect_investment_data(self, founder_name: str) -> List[Investment]:
        """Collect data about investments made by founder."""
        investments = []
        
        try:
            search_queries = [
                f'"{founder_name}" angel investor investments',
                f'"{founder_name}" venture capital portfolio',
                f'"{founder_name}" invested companies'
            ]
            
            for query in search_queries:
                results = await self._search_web_for_financial_data(query, "investment")
                investments.extend(await self._extract_investment_data(results, founder_name))
                await asyncio.sleep(1)
            
            return investments
            
        except Exception as e:
            logger.error(f"Error collecting investment data for {founder_name}: {e}")
            return []
    
    async def _collect_board_positions(self, founder_name: str) -> List[BoardPosition]:
        """Collect data about board positions."""
        positions = []
        
        try:
            search_queries = [
                f'"{founder_name}" board member director',
                f'"{founder_name}" advisory board',
                f'"{founder_name}" chairman board'
            ]
            
            for query in search_queries:
                results = await self._search_web_for_financial_data(query, "board")
                positions.extend(await self._extract_board_data(results, founder_name))
                await asyncio.sleep(1)
            
            return positions
            
        except Exception as e:
            logger.error(f"Error collecting board data for {founder_name}: {e}")
            return []
    
    async def _search_web_for_financial_data(
        self, 
        query: str, 
        data_type: str
    ) -> List[Dict[str, Any]]:
        """Search web for financial data using available search APIs."""
        await self.rate_limiter.acquire()
        
        logger.debug(f"🔍 Searching for {data_type} data with query: {query}")
        
        try:
            # Use Serper API for web search
            url = "https://google.serper.dev/search"
            headers = {
                "X-API-KEY": settings.serper_api_key,
                "Content-Type": "application/json"
            }
            
            payload = {
                "q": query,
                "num": 10,
                "gl": "us",
                "hl": "en"
            }
            
            logger.debug(f"📡 Making API request to {url} with payload: {payload}")
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.post(url, json=payload, headers=headers) as response:
                logger.debug(f"📡 API response status: {response.status}")
                if response.status == 200:
                    data = await response.json()
                    results = data.get("organic", [])
                    logger.debug(f"✅ Found {len(results)} search results for {data_type} query")
                    return results
                else:
                    response_text = await response.text()
                    logger.error(f"❌ Search API error {response.status} for query: {query}. Response: {response_text}")
                    return []
                    
        except aiohttp.ClientError as e:
            logger.error(f"❌ HTTP client error searching for {data_type} data: {e}")
            return []
        except Exception as e:
            logger.error(f"❌ Unexpected error searching for {data_type} data: {e}", exc_info=True)
            return []
    
    async def _extract_founding_data(
        self, 
        search_results: List[Dict[str, Any]], 
        founder_name: str
    ) -> List[CompanyFounding]:
        """Extract company founding data from search results."""
        companies = []
        
        for result in search_results:
            try:
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                url = result.get("link", "")
                
                # Look for founding patterns
                text = f"{title} {snippet}".lower()
                
                # Extract company names and founding information
                founding_patterns = [
                    r'(?:co-)?founded?\s+([A-Z][a-zA-Z\s&]+?)(?:\s+in\s+(\d{4}))?',
                    r'([A-Z][a-zA-Z\s&]+?)\s+(?:co-)?founder',
                    r'(?:CEO|founder)\s+(?:of\s+)?([A-Z][a-zA-Z\s&]+?)'
                ]
                
                for pattern in founding_patterns:
                    matches = re.finditer(pattern, title + " " + snippet, re.IGNORECASE)
                    for match in matches:
                        company_name = match.group(1).strip()
                        founding_year = match.group(2) if len(match.groups()) > 1 else None
                        
                        if len(company_name) > 2 and company_name not in [founder_name]:
                            founding_date = None
                            if founding_year:
                                try:
                                    founding_date = date(int(founding_year), 1, 1)
                                except ValueError:
                                    pass
                            
                            company = CompanyFounding(
                                company_name=company_name,
                                founding_date=founding_date,
                                founder_role="Founder",
                                verification_sources=[url]
                            )
                            companies.append(company)
                            
            except Exception as e:
                logger.warning(f"Error extracting founding data from result: {e}")
                continue
        
        return companies
    
    async def _extract_exit_data(
        self, 
        search_results: List[Dict[str, Any]], 
        founder_name: str
    ) -> List[CompanyExit]:
        """Extract exit data from search results."""
        exits = []
        
        for result in search_results:
            try:
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                url = result.get("link", "")
                
                text = f"{title} {snippet}".lower()
                
                # Look for exit patterns
                exit_patterns = [
                    r'([A-Z][a-zA-Z\s&]+?)\s+(?:acquired|bought)\s+(?:by\s+)?([A-Z][a-zA-Z\s&]+?)\s+(?:for\s+)?\$?([\d\.]+)\s*(billion|million)',
                    r'([A-Z][a-zA-Z\s&]+?)\s+IPO.*?\$?([\d\.]+)\s*(billion|million)',
                    r'\$?([\d\.]+)\s*(billion|million).*?(?:acquisition|exit|sale).*?([A-Z][a-zA-Z\s&]+?)'
                ]
                
                for pattern in exit_patterns:
                    matches = re.finditer(pattern, title + " " + snippet, re.IGNORECASE)
                    for match in matches:
                        groups = match.groups()
                        
                        # Extract exit information
                        if "ipo" in text:
                            exit_type = ExitType.IPO
                            company_name = groups[0] if len(groups) > 2 else "Unknown"
                            value_str = groups[1] if len(groups) > 2 else groups[0]
                            unit = groups[2] if len(groups) > 2 else groups[1]
                        else:
                            exit_type = ExitType.ACQUISITION
                            company_name = groups[0] if groups else "Unknown"
                            acquiring_company = groups[1] if len(groups) > 3 else None
                            value_str = groups[2] if len(groups) > 3 else groups[1] if len(groups) > 1 else groups[0]
                            unit = groups[3] if len(groups) > 3 else groups[2] if len(groups) > 2 else groups[1]
                        
                        # Convert value to USD
                        try:
                            value = float(value_str.replace(',', ''))
                            if unit.lower() == 'billion':
                                value *= 1_000_000_000
                            elif unit.lower() == 'million':
                                value *= 1_000_000
                        except (ValueError, AttributeError):
                            value = None
                        
                        exit = CompanyExit(
                            company_name=company_name.strip(),
                            exit_type=exit_type,
                            exit_value_usd=value,
                            acquiring_company=acquiring_company.strip() if acquiring_company else None,
                            verification_sources=[url]
                        )
                        exits.append(exit)
                        
            except Exception as e:
                logger.warning(f"Error extracting exit data from result: {e}")
                continue
        
        return exits
    
    async def _extract_investment_data(
        self, 
        search_results: List[Dict[str, Any]], 
        founder_name: str
    ) -> List[Investment]:
        """Extract investment data from search results."""
        investments = []
        
        for result in search_results:
            try:
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                url = result.get("link", "")
                
                # Look for investment patterns
                investment_patterns = [
                    r'(?:invested|angel|funding).*?([A-Z][a-zA-Z\s&]+?).*?\$?([\d\.]+)\s*(million|thousand)',
                    r'([A-Z][a-zA-Z\s&]+?).*?(?:angel investor|investment).*?\$?([\d\.]+)\s*(million|thousand)',
                ]
                
                for pattern in investment_patterns:
                    matches = re.finditer(pattern, title + " " + snippet, re.IGNORECASE)
                    for match in matches:
                        company_name = match.group(1).strip()
                        value_str = match.group(2)
                        unit = match.group(3)
                        
                        # Convert value to USD
                        try:
                            value = float(value_str.replace(',', ''))
                            if unit.lower() == 'million':
                                value *= 1_000_000
                            elif unit.lower() == 'thousand':
                                value *= 1_000
                        except ValueError:
                            value = None
                        
                        investment = Investment(
                            company_name=company_name,
                            investment_type=InvestmentType.ANGEL,
                            investment_amount_usd=value,
                            verification_sources=[url]
                        )
                        investments.append(investment)
                        
            except Exception as e:
                logger.warning(f"Error extracting investment data: {e}")
                continue
        
        return investments
    
    async def _extract_board_data(
        self, 
        search_results: List[Dict[str, Any]], 
        founder_name: str
    ) -> List[BoardPosition]:
        """Extract board position data from search results."""
        positions = []
        
        for result in search_results:
            try:
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                url = result.get("link", "")
                
                # Look for board position patterns
                board_patterns = [
                    r'board\s+(?:member|director).*?(?:at\s+)?([A-Z][a-zA-Z\s&]+?)(?:\s|,|\.)',
                    r'([A-Z][a-zA-Z\s&]+?).*?board\s+(?:member|director)',
                    r'(?:chairman|chair).*?(?:of\s+)?([A-Z][a-zA-Z\s&]+?)'
                ]
                
                for pattern in board_patterns:
                    matches = re.finditer(pattern, title + " " + snippet, re.IGNORECASE)
                    for match in matches:
                        company_name = match.group(1).strip()
                        
                        position_title = "Board Member"
                        if "chairman" in (title + snippet).lower():
                            position_title = "Chairman"
                        elif "advisory" in (title + snippet).lower():
                            position_title = "Advisory Board Member"
                        
                        position = BoardPosition(
                            company_name=company_name,
                            position_title=position_title,
                            verification_sources=[url]
                        )
                        positions.append(position)
                        
            except Exception as e:
                logger.warning(f"Error extracting board data: {e}")
                continue
        
        return positions
    
    def _deduplicate_exits(self, exits: List[CompanyExit]) -> List[CompanyExit]:
        """Remove duplicate exits and merge similar ones."""
        unique_exits = {}
        
        for exit in exits:
            key = exit.company_name.lower().strip()
            if key not in unique_exits:
                unique_exits[key] = exit
            else:
                # Merge verification sources
                existing = unique_exits[key]
                existing.verification_sources.extend(exit.verification_sources)
                
                # Update with more specific data if available
                if exit.exit_value_usd and not existing.exit_value_usd:
                    existing.exit_value_usd = exit.exit_value_usd
                if exit.exit_date and not existing.exit_date:
                    existing.exit_date = exit.exit_date
        
        return list(unique_exits.values())
    
    def _calculate_financial_confidence(self, profile: FounderFinancialProfile) -> float:
        """Calculate confidence score for financial profile."""
        score = 0.0
        
        # Base score for data sources
        if 'crunchbase' in profile.data_sources:
            score += 0.3
        if 'sec_filings' in profile.data_sources:
            score += 0.4
        if 'web_search' in profile.data_sources:
            score += 0.2
        
        # Boost for verified exits
        verified_exits = sum(1 for exit in profile.company_exits 
                           if len(exit.verification_sources) > 1)
        score += min(verified_exits * 0.1, 0.3)
        
        # Boost for multiple companies
        if len(profile.companies_founded) > 1:
            score += 0.2
        
        # Boost for financial data availability
        if profile.total_exit_value_usd:
            score += 0.2
        
        return min(score, 1.0)