"""SEC filings data collection for founder exit tracking (L7+ criteria)."""

import asyncio
import aiohttp
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import xml.etree.ElementTree as ET

from ...core import get_logger


logger = get_logger(__name__)


@dataclass
class SECFiling:
    """Individual SEC filing record."""
    filing_type: str  # 10-K, 8-K, S-1, etc.
    company_name: str
    cik: str  # Central Index Key
    filing_date: datetime
    document_url: str
    accession_number: str
    description: str
    form_data: Dict[str, Any] = field(default_factory=dict)
    mentioned_founders: List[str] = field(default_factory=list)
    exit_indicators: List[str] = field(default_factory=list)  # IPO, acquisition signals
    financial_data: Dict[str, float] = field(default_factory=dict)  # Revenue, valuation etc.


@dataclass
class ExitEventFromSEC:
    """Exit event extracted from SEC filings."""
    company_name: str
    exit_type: str  # "IPO", "Acquisition", "Merger"
    exit_date: datetime
    exit_value_usd: Optional[float] = None
    founder_involvement: List[str] = field(default_factory=list)
    sec_filing_accessions: List[str] = field(default_factory=list)
    key_documents: List[str] = field(default_factory=list)
    verification_score: float = 0.0  # 0-1 based on filing quality


@dataclass
class FounderSECProfile:
    """SEC filings profile for a founder.""" 
    founder_name: str
    related_companies: List[str] = field(default_factory=list)
    sec_filings: List[SECFiling] = field(default_factory=list)
    verified_exits: List[ExitEventFromSEC] = field(default_factory=list)
    ipo_events: List[ExitEventFromSEC] = field(default_factory=list)
    acquisition_events: List[ExitEventFromSEC] = field(default_factory=list)
    total_verified_exit_value: float = 0.0
    highest_exit_value: float = 0.0
    exit_count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def calculate_exit_metrics(self):
        """Calculate derived exit metrics from SEC data."""
        all_exits = self.ipo_events + self.acquisition_events
        
        self.exit_count = len(all_exits)
        
        verified_values = [
            exit_event.exit_value_usd 
            for exit_event in all_exits 
            if exit_event.exit_value_usd and exit_event.verification_score > 0.7
        ]
        
        if verified_values:
            self.total_verified_exit_value = sum(verified_values)
            self.highest_exit_value = max(verified_values)
    
    def meets_l7_plus_criteria(self) -> Dict[str, bool]:
        """Check if founder meets L7+ SEC-verified exit criteria."""
        return {
            "l7_multiple_100m_exits": len([
                e for e in self.verified_exits 
                if e.exit_value_usd and e.exit_value_usd >= 100
            ]) >= 2,
            "l8_unicorn_exit": self.highest_exit_value >= 1000,
            "l9_billion_exit": self.highest_exit_value >= 1000,
            "l10_multiple_billion_exits": len([
                e for e in self.verified_exits 
                if e.exit_value_usd and e.exit_value_usd >= 1000
            ]) >= 2,
            "sec_verified_exits": len(self.verified_exits) > 0
        }


class SECFilingsCollector:
    """Collects SEC filings data for founder exit verification."""
    
    def __init__(self):
        self.session = None
        self.edgar_base_url = "https://data.sec.gov"
        self.rate_limit_delay = 0.1  # SEC requires 10 requests/second max
        self.headers = {
            "User-Agent": "InitiationPipeline/1.0 (contact@example.com)",  # SEC requires identification
            "Accept": "application/json"
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def get_founder_sec_filings(
        self, 
        founder_name: str,
        company_names: List[str] = None
    ) -> FounderSECProfile:
        """Get comprehensive SEC filings data for a founder."""
        logger.info(f"🏛️ Collecting SEC filings for {founder_name}")
        
        profile = FounderSECProfile(
            founder_name=founder_name,
            related_companies=company_names or []
        )
        
        try:
            # Search for companies associated with founder
            if company_names:
                for company_name in company_names:
                    # Get company CIK
                    cik = await self._get_company_cik(company_name)
                    if cik:
                        # Get recent filings for the company
                        filings = await self._get_company_filings(cik, company_name)
                        
                        # Filter filings that mention the founder
                        relevant_filings = await self._filter_founder_mentions(
                            filings, founder_name
                        )
                        
                        profile.sec_filings.extend(relevant_filings)
                        
                        await asyncio.sleep(self.rate_limit_delay)
            
            # Extract exit events from filings
            profile.ipo_events = self._extract_ipo_events(profile.sec_filings)
            profile.acquisition_events = self._extract_acquisition_events(profile.sec_filings)
            profile.verified_exits = profile.ipo_events + profile.acquisition_events
            
            # Calculate metrics
            profile.calculate_exit_metrics()
            
            logger.info(f"✅ Found {len(profile.sec_filings)} SEC filings, {profile.exit_count} verified exits for {founder_name}")
            return profile
            
        except Exception as e:
            logger.error(f"Error collecting SEC data for {founder_name}: {e}")
            return profile
    
    async def _get_company_cik(self, company_name: str) -> Optional[str]:
        """Get CIK (Central Index Key) for a company."""
        try:
            # Use SEC company tickers endpoint
            url = f"{self.edgar_base_url}/files/company_tickers.json"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Search for company name match
                    for cik, company_info in data.items():
                        if isinstance(company_info, dict):
                            title = company_info.get("title", "").lower()
                            if company_name.lower() in title or title in company_name.lower():
                                return str(company_info.get("cik_str", "")).zfill(10)
            
            await asyncio.sleep(self.rate_limit_delay)
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get CIK for {company_name}: {e}")
            return None
    
    async def _get_company_filings(self, cik: str, company_name: str) -> List[SECFiling]:
        """Get recent filings for a company by CIK."""
        filings = []
        
        try:
            # Get company facts to find recent filings
            url = f"{self.edgar_base_url}/submissions/CIK{cik}.json"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Extract recent filings
                    recent_filings = data.get("filings", {}).get("recent", {})
                    
                    if recent_filings:
                        accession_numbers = recent_filings.get("accessionNumber", [])
                        filing_dates = recent_filings.get("filingDate", [])
                        forms = recent_filings.get("form", [])
                        
                        for i, accession in enumerate(accession_numbers[:50]):  # Limit to 50 recent
                            if i < len(filing_dates) and i < len(forms):
                                # Focus on exit-relevant forms
                                form_type = forms[i]
                                if form_type in ["8-K", "S-1", "S-1/A", "10-K", "10-Q", "DEF 14A"]:
                                    filing = SECFiling(
                                        filing_type=form_type,
                                        company_name=company_name,
                                        cik=cik,
                                        filing_date=datetime.strptime(filing_dates[i], "%Y-%m-%d"),
                                        document_url=f"{self.edgar_base_url}/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{accession}-index.htm",
                                        accession_number=accession,
                                        description=f"{form_type} filing for {company_name}"
                                    )
                                    filings.append(filing)
            
            await asyncio.sleep(self.rate_limit_delay)
            
        except Exception as e:
            logger.warning(f"Failed to get filings for CIK {cik}: {e}")
        
        return filings
    
    async def _filter_founder_mentions(
        self, 
        filings: List[SECFiling], 
        founder_name: str
    ) -> List[SECFiling]:
        """Filter filings that mention the founder."""
        relevant_filings = []
        
        for filing in filings[:10]:  # Limit detailed analysis to 10 most recent
            try:
                # Get filing document content
                content = await self._get_filing_content(filing)
                
                if content and self._mentions_founder(content, founder_name):
                    filing.mentioned_founders = [founder_name]
                    filing.form_data = {"content_snippet": content[:1000]}
                    relevant_filings.append(filing)
                
                await asyncio.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.warning(f"Error analyzing filing {filing.accession_number}: {e}")
                continue
        
        return relevant_filings
    
    async def _get_filing_content(self, filing: SECFiling) -> Optional[str]:
        """Get text content of SEC filing."""
        try:
            # Try to get the primary document
            primary_doc_url = filing.document_url.replace("-index.htm", ".txt")
            
            async with self.session.get(primary_doc_url) as response:
                if response.status == 200:
                    return await response.text()
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get content for filing {filing.accession_number}: {e}")
            return None
    
    def _mentions_founder(self, content: str, founder_name: str) -> bool:
        """Check if filing content mentions the founder."""
        if not content:
            return False
        
        content_lower = content.lower()
        name_parts = founder_name.lower().split()
        
        # Check for full name
        if founder_name.lower() in content_lower:
            return True
        
        # Check for last name + first initial
        if len(name_parts) >= 2:
            first_initial = name_parts[0][0]
            last_name = name_parts[-1]
            pattern = f"{first_initial}.? {last_name}"
            if re.search(pattern, content_lower):
                return True
        
        return False
    
    def _extract_ipo_events(self, filings: List[SECFiling]) -> List[ExitEventFromSEC]:
        """Extract IPO events from SEC filings."""
        ipo_events = []
        
        for filing in filings:
            if filing.filing_type in ["S-1", "S-1/A"] and filing.form_data.get("content_snippet"):
                content = filing.form_data["content_snippet"].lower()
                
                # Look for IPO indicators
                ipo_indicators = [
                    "initial public offering",
                    "public offering",
                    "registration statement",
                    "shares of common stock"
                ]
                
                if any(indicator in content for indicator in ipo_indicators):
                    # Extract potential valuation/offering size
                    valuation = self._extract_financial_figures(content)
                    
                    ipo_event = ExitEventFromSEC(
                        company_name=filing.company_name,
                        exit_type="IPO",
                        exit_date=filing.filing_date,
                        exit_value_usd=valuation.get("offering_value"),
                        founder_involvement=filing.mentioned_founders,
                        sec_filing_accessions=[filing.accession_number],
                        key_documents=[filing.document_url],
                        verification_score=0.9  # High confidence for S-1 filings
                    )
                    ipo_events.append(ipo_event)
        
        return ipo_events
    
    def _extract_acquisition_events(self, filings: List[SECFiling]) -> List[ExitEventFromSEC]:
        """Extract acquisition events from SEC filings."""
        acquisition_events = []
        
        for filing in filings:
            if filing.filing_type in ["8-K", "DEF 14A"] and filing.form_data.get("content_snippet"):
                content = filing.form_data["content_snippet"].lower()
                
                # Look for acquisition indicators
                acquisition_indicators = [
                    "merger agreement",
                    "acquisition",
                    "purchase agreement",
                    "definitive agreement",
                    "tender offer"
                ]
                
                if any(indicator in content for indicator in acquisition_indicators):
                    # Extract potential transaction value
                    valuation = self._extract_financial_figures(content)
                    
                    acquisition_event = ExitEventFromSEC(
                        company_name=filing.company_name,
                        exit_type="Acquisition",
                        exit_date=filing.filing_date,
                        exit_value_usd=valuation.get("transaction_value"),
                        founder_involvement=filing.mentioned_founders,
                        sec_filing_accessions=[filing.accession_number],
                        key_documents=[filing.document_url],
                        verification_score=0.8  # Good confidence for 8-K acquisition filings
                    )
                    acquisition_events.append(acquisition_event)
        
        return acquisition_events
    
    def _extract_financial_figures(self, content: str) -> Dict[str, Optional[float]]:
        """Extract financial figures from filing content."""
        financial_data = {}
        
        # Pattern for monetary amounts
        money_patterns = [
            r'\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(million|billion|m|b)',
            r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*million',
            r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*billion'
        ]
        
        for pattern in money_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                amount_str = match.group(1).replace(",", "")
                amount = float(amount_str)
                
                unit = match.group(2).lower() if len(match.groups()) > 1 else ""
                if "billion" in unit or unit == "b":
                    amount *= 1000  # Convert to millions
                
                # Determine context
                before_match = content[max(0, match.start() - 100):match.start()].lower()
                after_match = content[match.end():match.end() + 100].lower()
                context = before_match + after_match
                
                if any(term in context for term in ["offering", "raise", "proceeds"]):
                    financial_data["offering_value"] = amount
                elif any(term in context for term in ["acquisition", "purchase", "transaction"]):
                    financial_data["transaction_value"] = amount
                elif any(term in context for term in ["valuation", "valued"]):
                    financial_data["valuation"] = amount
        
        return financial_data
