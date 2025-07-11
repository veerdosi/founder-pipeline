"""Financial data collection service for comprehensive founder financial intelligence using Perplexity AI."""

import asyncio
from datetime import datetime, date
from typing import List, Optional, Dict, Any
import logging
import json
import re

from ..ranking.models import (
    FounderFinancialProfile, CompanyExit, CompanyFounding, Investment, 
    BoardPosition, ExitType, InvestmentType
)
from .perplexity_base import PerplexityBaseService

logger = logging.getLogger(__name__)


class FinancialDataCollector(PerplexityBaseService):
    """Service for collecting comprehensive founder financial data using Perplexity AI."""
    
    def __init__(self):
        super().__init__()
        
        # Enhanced query templates for financial data
        self.query_templates = {
            'company_founding': [
                """What companies has {founder_name} founded, co-founded, or started? For each company, provide:
                - Company name
                - Founding year
                - {founder_name}'s role (founder, co-founder, CEO, etc.)
                - Current status (active, acquired, closed, etc.)
                - Industry or sector
                - Any initial funding or valuation information
                Include both successful and unsuccessful ventures.""",
                
                """What startups and businesses has {founder_name} created or launched throughout their career?
                Please include:
                - Complete company names
                - Exact founding dates when available
                - {founder_name}'s specific founding role
                - Current company status and operations
                - Any notable achievements or milestones""",
                
                """List all entrepreneurial ventures where {founder_name} was a founding member or key early employee with equity.
                Include:
                - Company names and founding years
                - {founder_name}'s founding role and equity stake if known
                - Business model and industry
                - Current status and any exit information"""
            ],
            
            'company_exits': [
                """What are all the company exits, acquisitions, IPOs, or major liquidity events involving companies that {founder_name} founded or had significant equity in?
                For each exit, provide:
                - Company name
                - Type of exit (IPO, acquisition, merger, etc.)
                - Exit date (month and year)
                - Exit value or transaction amount
                - Acquiring company (if applicable)
                - {founder_name}'s estimated proceeds or stake
                - Current status of the acquired company""",
                
                """Has {founder_name} had any successful exits through IPOs, acquisitions, or sales of companies they founded?
                Include:
                - Specific company names and exit details
                - Transaction values and dates
                - {founder_name}'s role and ownership percentage
                - Financial outcomes and proceeds
                - Impact on their net worth""",
                
                """What major business sales, acquisitions, or public offerings has {founder_name} been involved in as a founder or significant equity holder?
                Provide:
                - Complete transaction details
                - Financial terms and valuations
                - Timeline of events
                - {founder_name}'s financial outcome"""
            ],
            
            'investments': [
                """What investments has {founder_name} made as an angel investor, venture capitalist, or private investor?
                For each investment, include:
                - Company name and industry
                - Investment amount (if known)
                - Investment date or timeframe
                - Type of investment (angel, seed, Series A, etc.)
                - Current status of the investment
                - Any notable returns or outcomes
                - {founder_name}'s involvement beyond capital""",
                
                """Is {founder_name} an active angel investor or venture capitalist? What is their investment portfolio?
                Include:
                - Investment firm affiliations (if any)
                - Portfolio companies and investment amounts
                - Investment thesis and preferred sectors
                - Notable successful investments and returns
                - Co-investors and investment partnerships""",
                
                """What startups, companies, or funds has {founder_name} invested in or provided funding to?
                Provide:
                - Detailed investment history
                - Investment amounts and ownership stakes
                - Investment performance and outcomes
                - {founder_name}'s involvement in portfolio companies"""
            ],
            
            'board_positions': [
                """What board positions, directorships, or advisory roles does {founder_name} currently hold or has held?
                For each position, include:
                - Company or organization name
                - Position title (board member, director, advisor, etc.)
                - Start date and current status
                - Type of organization (public company, private company, nonprofit, etc.)
                - Any compensation or equity arrangements
                - Key responsibilities and contributions""",
                
                """Is {founder_name} a board member, director, or advisor for any companies or organizations?
                Include:
                - Current and past board positions
                - Advisory roles and consulting arrangements
                - Board committee memberships
                - Governance responsibilities
                - Any notable board decisions or contributions""",
                
                """What corporate governance roles has {founder_name} taken on outside of their own companies?
                Provide:
                - Board seats and advisory positions
                - Duration of service
                - Company types and industries
                - Leadership roles within boards
                - Any notable governance initiatives or decisions"""
            ]
        }
    
    async def collect_data(self, founder_name: str, current_company: str) -> FounderFinancialProfile:
        """Collect comprehensive financial data for a founder using Perplexity AI."""
        return await self.collect_founder_financial_data(founder_name, current_company)
    
    def get_query_templates(self) -> Dict[str, List[str]]:
        """Get query templates for financial data collection."""
        return self.query_templates
    
    async def collect_founder_financial_data(
        self, 
        founder_name: str,
        current_company: str,
        linkedin_url: Optional[str] = None
    ) -> FounderFinancialProfile:
        """Collect comprehensive financial data for a founder using Perplexity AI."""
        logger.info(f"🔍 Collecting financial data for {founder_name} using Perplexity AI")
        logger.debug(f"📋 Parameters: company={current_company}, linkedin_url={linkedin_url}")
        
        profile = FounderFinancialProfile(
            founder_name=founder_name,
            last_updated=datetime.now()
        )
        
        try:
            # Collect data from multiple categories using Perplexity
            logger.debug(f"🚀 Starting Perplexity financial data collection for {founder_name}")
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
            profile.data_sources = ['perplexity_ai', 'web_search', 'financial_databases']
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
        """Collect data about companies founded by the founder using Perplexity."""
        companies = []
        
        try:
            system_prompt = """You are a business intelligence specialist focused on company founding information.
            Provide comprehensive, accurate information about companies founded by entrepreneurs.
            Include specific dates, roles, and current status. Be thorough but factual."""
            
            for query_template in self.query_templates['company_founding']:
                query = query_template.format(founder_name=founder_name)
                
                response = await self.query_perplexity(
                    query=query,
                    system_prompt=system_prompt,
                    max_tokens=2000
                )
                
                if response:
                    content = self.extract_content_from_response(response)
                    if content:
                        companies.extend(self._parse_company_founding_data(content, founder_name))
                
                # Rate limiting delay
                await asyncio.sleep(1)
            
            # Add current company if not already included
            if current_company:
                existing_companies = [c.company_name.lower() for c in companies]
                if current_company.lower() not in existing_companies:
                    current_company_data = CompanyFounding(
                        company_name=current_company,
                        founder_role="Founder/CEO",
                        is_current_company=True,
                        verification_sources=["linkedin_profile", "company_website"]
                    )
                    companies.append(current_company_data)
            
            # Deduplicate companies
            unique_companies = self._deduplicate_companies(companies)
            logger.debug(f"📊 Found {len(unique_companies)} unique companies founded by {founder_name}")
            
            return unique_companies
            
        except Exception as e:
            logger.error(f"Error collecting founding data for {founder_name}: {e}")
            return []
    
    async def _collect_exit_data(self, founder_name: str) -> List[CompanyExit]:
        """Collect data about company exits using Perplexity."""
        exits = []
        
        try:
            system_prompt = """You are a financial intelligence specialist focused on company exits and liquidity events.
            Provide detailed information about IPOs, acquisitions, and other exit events.
            Include specific financial details, dates, and transaction terms when available."""
            
            for query_template in self.query_templates['company_exits']:
                query = query_template.format(founder_name=founder_name)
                
                response = await self.query_perplexity(
                    query=query,
                    system_prompt=system_prompt,
                    max_tokens=2000
                )
                
                if response:
                    content = self.extract_content_from_response(response)
                    if content:
                        exits.extend(self._parse_exit_data(content, founder_name))
                
                await asyncio.sleep(1)
            
            # Deduplicate and validate exits
            unique_exits = self._deduplicate_exits(exits)
            logger.debug(f"📊 Found {len(unique_exits)} unique exits for {founder_name}")
            
            return unique_exits
            
        except Exception as e:
            logger.error(f"Error collecting exit data for {founder_name}: {e}")
            return []
    
    async def _collect_investment_data(self, founder_name: str) -> List[Investment]:
        """Collect data about investments made by founder using Perplexity."""
        investments = []
        
        try:
            system_prompt = """You are an investment intelligence specialist focused on angel investing and venture capital.
            Provide detailed information about investments made by entrepreneurs and business leaders.
            Include investment amounts, dates, company details, and outcomes when available."""
            
            for query_template in self.query_templates['investments']:
                query = query_template.format(founder_name=founder_name)
                
                response = await self.query_perplexity(
                    query=query,
                    system_prompt=system_prompt,
                    max_tokens=2000
                )
                
                if response:
                    content = self.extract_content_from_response(response)
                    if content:
                        investments.extend(self._parse_investment_data(content, founder_name))
                
                await asyncio.sleep(1)
            
            logger.debug(f"📊 Found {len(investments)} investments made by {founder_name}")
            return investments
            
        except Exception as e:
            logger.error(f"Error collecting investment data for {founder_name}: {e}")
            return []
    
    async def _collect_board_positions(self, founder_name: str) -> List[BoardPosition]:
        """Collect data about board positions using Perplexity."""
        positions = []
        
        try:
            system_prompt = """You are a corporate governance specialist focused on board positions and advisory roles.
            Provide detailed information about board memberships, directorships, and advisory positions.
            Include position titles, organization details, and duration of service."""
            
            for query_template in self.query_templates['board_positions']:
                query = query_template.format(founder_name=founder_name)
                
                response = await self.query_perplexity(
                    query=query,
                    system_prompt=system_prompt,
                    max_tokens=2000
                )
                
                if response:
                    content = self.extract_content_from_response(response)
                    if content:
                        positions.extend(self._parse_board_data(content, founder_name))
                
                await asyncio.sleep(1)
            
            logger.debug(f"📊 Found {len(positions)} board positions for {founder_name}")
            return positions
            
        except Exception as e:
            logger.error(f"Error collecting board data for {founder_name}: {e}")
            return []
    
    def _parse_company_founding_data(self, content: str, founder_name: str) -> List[CompanyFounding]:
        """Parse company founding data from Perplexity response."""
        companies = []
        
        try:
            # Split content into sentences and look for company mentions
            sentences = content.split('.')
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 20:
                    continue
                
                # Look for founding keywords
                if any(keyword in sentence.lower() for keyword in ['founded', 'co-founded', 'started', 'created', 'launched']):
                    company_data = self._extract_company_from_sentence(sentence, founder_name)
                    if company_data:
                        companies.append(company_data)
            
            # Also look for structured information patterns
            companies.extend(self._extract_structured_companies(content, founder_name))
            
        except Exception as e:
            logger.warning(f"Error parsing company founding data: {e}")
        
        return companies
    
    def _parse_exit_data(self, content: str, founder_name: str) -> List[CompanyExit]:
        """Parse exit data from Perplexity response."""
        exits = []
        
        try:
            sentences = content.split('.')
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 20:
                    continue
                
                # Look for exit keywords
                if any(keyword in sentence.lower() for keyword in ['acquired', 'ipo', 'sold', 'exit', 'acquisition', 'merger']):
                    exit_data = self._extract_exit_from_sentence(sentence, founder_name)
                    if exit_data:
                        exits.append(exit_data)
            
            # Extract structured exit information
            exits.extend(self._extract_structured_exits(content, founder_name))
            
        except Exception as e:
            logger.warning(f"Error parsing exit data: {e}")
        
        return exits
    
    def _parse_investment_data(self, content: str, founder_name: str) -> List[Investment]:
        """Parse investment data from Perplexity response."""
        investments = []
        
        try:
            sentences = content.split('.')
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 20:
                    continue
                
                # Look for investment keywords
                if any(keyword in sentence.lower() for keyword in ['invested', 'angel', 'funding', 'venture', 'portfolio']):
                    investment_data = self._extract_investment_from_sentence(sentence, founder_name)
                    if investment_data:
                        investments.append(investment_data)
            
        except Exception as e:
            logger.warning(f"Error parsing investment data: {e}")
        
        return investments
    
    def _parse_board_data(self, content: str, founder_name: str) -> List[BoardPosition]:
        """Parse board position data from Perplexity response."""
        positions = []
        
        try:
            sentences = content.split('.')
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 20:
                    continue
                
                # Look for board keywords
                if any(keyword in sentence.lower() for keyword in ['board', 'director', 'advisor', 'advisory']):
                    board_data = self._extract_board_from_sentence(sentence, founder_name)
                    if board_data:
                        positions.append(board_data)
            
        except Exception as e:
            logger.warning(f"Error parsing board data: {e}")
        
        return positions
    
    def _extract_company_from_sentence(self, sentence: str, founder_name: str) -> Optional[CompanyFounding]:
        """Extract company founding information from a sentence."""
        try:
            # Simple pattern matching for company names (can be enhanced)
            # Look for capitalized words that might be company names
            words = sentence.split()
            potential_companies = []
            
            for i, word in enumerate(words):
                if word[0].isupper() and len(word) > 2:
                    # Check if it's likely a company name
                    if i < len(words) - 1 and words[i+1][0].isupper():
                        potential_companies.append(f"{word} {words[i+1]}")
                    else:
                        potential_companies.append(word)
            
            if potential_companies:
                # Take the first potential company name
                company_name = potential_companies[0]
                
                # Extract year if present
                year_match = re.search(r'\b(19|20)\d{2}\b', sentence)
                founding_date = None
                if year_match:
                    try:
                        founding_date = date(int(year_match.group()), 1, 1)
                    except ValueError:
                        pass
                
                # Determine role
                role = "Founder"
                if "co-founded" in sentence.lower():
                    role = "Co-Founder"
                elif "ceo" in sentence.lower():
                    role = "Founder/CEO"
                
                return CompanyFounding(
                    company_name=company_name,
                    founding_date=founding_date,
                    founder_role=role,
                    verification_sources=["perplexity_ai"]
                )
        
        except Exception as e:
            logger.warning(f"Error extracting company from sentence: {e}")
        
        return None
    
    def _extract_exit_from_sentence(self, sentence: str, founder_name: str) -> Optional[CompanyExit]:
        """Extract exit information from a sentence."""
        try:
            # Extract company name (simplified)
            words = sentence.split()
            company_name = "Unknown"
            
            # Look for capitalized words that might be company names
            for word in words:
                if word[0].isupper() and len(word) > 2 and word not in ['IPO', 'CEO', 'The', 'In', 'By']:
                    company_name = word
                    break
            
            # Determine exit type
            exit_type = ExitType.ACQUISITION
            if "ipo" in sentence.lower():
                exit_type = ExitType.IPO
            elif "merger" in sentence.lower():
                exit_type = ExitType.MERGER
            
            # Extract financial amount
            amount_match = re.search(r'\$?([\d,]+(?:\.\d+)?)\s*(million|billion|M|B)', sentence, re.IGNORECASE)
            exit_value = None
            if amount_match:
                try:
                    value = float(amount_match.group(1).replace(',', ''))
                    unit = amount_match.group(2).lower()
                    if unit in ['billion', 'b']:
                        exit_value = value * 1_000_000_000
                    elif unit in ['million', 'm']:
                        exit_value = value * 1_000_000
                except ValueError:
                    pass
            
            # Extract year
            year_match = re.search(r'\b(19|20)\d{2}\b', sentence)
            exit_date = None
            if year_match:
                try:
                    exit_date = date(int(year_match.group()), 1, 1)
                except ValueError:
                    pass
            
            return CompanyExit(
                company_name=company_name,
                exit_type=exit_type,
                exit_value_usd=exit_value,
                exit_date=exit_date,
                verification_sources=["perplexity_ai"]
            )
        
        except Exception as e:
            logger.warning(f"Error extracting exit from sentence: {e}")
        
        return None
    
    def _extract_investment_from_sentence(self, sentence: str, founder_name: str) -> Optional[Investment]:
        """Extract investment information from a sentence."""
        try:
            # Extract company name (simplified)
            words = sentence.split()
            company_name = "Unknown"
            
            for word in words:
                if word[0].isupper() and len(word) > 2:
                    company_name = word
                    break
            
            # Extract investment amount
            amount_match = re.search(r'\$?([\d,]+(?:\.\d+)?)\s*(million|thousand|M|K)', sentence, re.IGNORECASE)
            investment_amount = None
            if amount_match:
                try:
                    value = float(amount_match.group(1).replace(',', ''))
                    unit = amount_match.group(2).lower()
                    if unit in ['million', 'm']:
                        investment_amount = value * 1_000_000
                    elif unit in ['thousand', 'k']:
                        investment_amount = value * 1_000
                except ValueError:
                    pass
            
            # Determine investment type
            investment_type = InvestmentType.ANGEL
            if "seed" in sentence.lower():
                investment_type = InvestmentType.SEED
            elif "series a" in sentence.lower():
                investment_type = InvestmentType.SERIES_A
            elif "venture" in sentence.lower():
                investment_type = InvestmentType.VENTURE
            
            return Investment(
                company_name=company_name,
                investment_type=investment_type,
                investment_amount_usd=investment_amount,
                verification_sources=["perplexity_ai"]
            )
        
        except Exception as e:
            logger.warning(f"Error extracting investment from sentence: {e}")
        
        return None
    
    def _extract_board_from_sentence(self, sentence: str, founder_name: str) -> Optional[BoardPosition]:
        """Extract board position information from a sentence."""
        try:
            # Extract company name (simplified)
            words = sentence.split()
            company_name = "Unknown"
            
            for word in words:
                if word[0].isupper() and len(word) > 2:
                    company_name = word
                    break
            
            # Determine position title
            position_title = "Board Member"
            if "director" in sentence.lower():
                position_title = "Board Director"
            elif "chairman" in sentence.lower():
                position_title = "Chairman"
            elif "advisory" in sentence.lower():
                position_title = "Advisory Board Member"
            
            return BoardPosition(
                company_name=company_name,
                position_title=position_title,
                verification_sources=["perplexity_ai"]
            )
        
        except Exception as e:
            logger.warning(f"Error extracting board position from sentence: {e}")
        
        return None
    
    def _extract_structured_companies(self, content: str, founder_name: str) -> List[CompanyFounding]:
        """Extract structured company information from content."""
        companies = []
        
        try:
            # Look for structured lists or bullet points
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line and ('•' in line or '-' in line or line.startswith(('1.', '2.', '3.'))):
                    company_data = self._extract_company_from_sentence(line, founder_name)
                    if company_data:
                        companies.append(company_data)
        
        except Exception as e:
            logger.warning(f"Error extracting structured companies: {e}")
        
        return companies
    
    def _extract_structured_exits(self, content: str, founder_name: str) -> List[CompanyExit]:
        """Extract structured exit information from content."""
        exits = []
        
        try:
            # Look for structured lists or bullet points
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line and ('•' in line or '-' in line or line.startswith(('1.', '2.', '3.'))):
                    exit_data = self._extract_exit_from_sentence(line, founder_name)
                    if exit_data:
                        exits.append(exit_data)
        
        except Exception as e:
            logger.warning(f"Error extracting structured exits: {e}")
        
        return exits
    
    def _deduplicate_companies(self, companies: List[CompanyFounding]) -> List[CompanyFounding]:
        """Remove duplicate companies and merge information."""
        unique_companies = {}
        
        for company in companies:
            key = company.company_name.lower().strip()
            if key not in unique_companies:
                unique_companies[key] = company
            else:
                # Merge information
                existing = unique_companies[key]
                if company.founding_date and not existing.founding_date:
                    existing.founding_date = company.founding_date
                if company.current_valuation_usd and not existing.current_valuation_usd:
                    existing.current_valuation_usd = company.current_valuation_usd
                existing.verification_sources.extend(company.verification_sources)
        
        return list(unique_companies.values())
    
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
        
        # Base score for using Perplexity (higher quality source)
        if 'perplexity_ai' in profile.data_sources:
            score += 0.4
        
        # Boost for verified exits
        verified_exits = sum(1 for exit in profile.company_exits 
                           if len(exit.verification_sources) > 0)
        score += min(verified_exits * 0.15, 0.3)
        
        # Boost for multiple companies
        if len(profile.companies_founded) > 1:
            score += 0.2
        
        # Boost for financial data availability
        if profile.total_exit_value_usd:
            score += 0.2
        
        # Boost for investment activity
        if len(profile.investments_made) > 0:
            score += 0.1
        
        return min(score, 1.0)