"""Comprehensive founder financial data collection and management."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class ExitType(Enum):
    """Types of company exits."""
    IPO = "ipo"
    ACQUISITION = "acquisition" 
    MERGER = "merger"
    BUYOUT = "buyout"
    UNKNOWN = "unknown"


class FundingStage(Enum):
    """Company funding stages."""
    PRE_SEED = "pre_seed"
    SEED = "seed"
    SERIES_A = "series_a"
    SERIES_B = "series_b"
    SERIES_C = "series_c"
    SERIES_D = "series_d"
    SERIES_E_PLUS = "series_e_plus"
    IPO = "ipo"
    UNKNOWN = "unknown"


@dataclass
class ExitEvent:
    """Individual company exit event."""
    company_name: str
    exit_type: ExitType
    exit_value_usd: Optional[float]  # In millions
    exit_date: Optional[datetime]
    acquirer_name: Optional[str] = None
    founder_stake_percent: Optional[float] = None
    founder_payout_usd: Optional[float] = None  # In millions
    verified: bool = False
    verification_sources: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate exit data."""
        if self.exit_value_usd and self.exit_value_usd < 0:
            raise ValueError("Exit value cannot be negative")
        if self.founder_stake_percent and (self.founder_stake_percent < 0 or self.founder_stake_percent > 100):
            raise ValueError("Founder stake must be between 0-100%")


@dataclass  
class CompanyFinancials:
    """Financial data for a company founded/co-founded."""
    company_name: str
    founder_role: str  # "founder", "co-founder", "ceo"
    current_valuation_usd: Optional[float] = None  # In millions
    total_funding_raised_usd: Optional[float] = None  # In millions
    funding_stage: Optional[FundingStage] = None
    latest_round_amount_usd: Optional[float] = None  # In millions
    latest_round_date: Optional[datetime] = None
    founded_date: Optional[datetime] = None
    employee_count: Optional[int] = None
    is_active: bool = True
    exit_event: Optional[ExitEvent] = None
    verification_sources: List[str] = field(default_factory=list)


@dataclass
class FounderFinancialProfile:
    """Comprehensive financial profile for a founder."""
    founder_name: str
    companies_founded: List[CompanyFinancials] = field(default_factory=list)
    total_exits: int = 0
    total_exit_value_usd: float = 0.0  # Sum of all verified exits
    total_value_created_usd: float = 0.0  # Exits + current valuations
    highest_exit_value_usd: float = 0.0
    unicorn_companies_count: int = 0  # Companies valued >$1B
    companies_with_major_exits_count: int = 0  # Exits >$100M
    current_portfolio_value_usd: float = 0.0  # Current active companies
    years_entrepreneurship: Optional[int] = None
    first_company_founded_year: Optional[int] = None
    last_updated: datetime = field(default_factory=datetime.now)
    
    def calculate_metrics(self):
        """Calculate derived financial metrics."""
        active_companies = [c for c in self.companies_founded if c.is_active]
        exited_companies = [c for c in self.companies_founded if c.exit_event]
        
        # Calculate totals
        self.total_exits = len(exited_companies)
        self.total_exit_value_usd = sum(
            c.exit_event.exit_value_usd or 0 
            for c in exited_companies 
            if c.exit_event and c.exit_event.verified
        )
        
        # Current portfolio value
        self.current_portfolio_value_usd = sum(
            c.current_valuation_usd or 0 
            for c in active_companies
        )
        
        # Total value created
        self.total_value_created_usd = self.total_exit_value_usd + self.current_portfolio_value_usd
        
        # Highest exit
        if exited_companies:
            self.highest_exit_value_usd = max(
                c.exit_event.exit_value_usd or 0 
                for c in exited_companies 
                if c.exit_event
            )
        
        # Unicorn count
        self.unicorn_companies_count = len([
            c for c in self.companies_founded 
            if (c.current_valuation_usd or 0) >= 1000 or 
               (c.exit_event and (c.exit_event.exit_value_usd or 0) >= 1000)
        ])
        
        # Major exits count (>$100M)
        self.companies_with_major_exits_count = len([
            c for c in exited_companies
            if c.exit_event and (c.exit_event.exit_value_usd or 0) >= 100
        ])
        
        # Calculate years in entrepreneurship
        founded_years = [
            c.founded_date.year for c in self.companies_founded 
            if c.founded_date
        ]
        if founded_years:
            self.first_company_founded_year = min(founded_years)
            self.years_entrepreneurship = datetime.now().year - self.first_company_founded_year
    
    def meets_level_criteria(self, level: str) -> Dict[str, bool]:
        """Check if founder meets financial criteria for specific L-level."""
        criteria_met = {}
        
        if level == "L7":  # Elite Serial Entrepreneurs
            criteria_met["multiple_exits_100m"] = (
                self.companies_with_major_exits_count >= 2 or
                self.unicorn_companies_count >= 2
            )
            criteria_met["total_value_500m"] = self.total_value_created_usd >= 500
            
        elif level == "L8":  # Proven Unicorn Builders  
            criteria_met["unicorn_built"] = self.unicorn_companies_count >= 1
            criteria_met["funding_100m"] = any(
                (c.total_funding_raised_usd or 0) >= 100 
                for c in self.companies_founded
            )
            
        elif level == "L9":  # Transformational Leaders
            criteria_met["billion_exit"] = self.highest_exit_value_usd >= 1000
            criteria_met["current_company_500m"] = any(
                (c.current_valuation_usd or 0) >= 500 
                for c in self.companies_founded if c.is_active
            )
            
        elif level == "L10":  # Legendary Entrepreneurs
            criteria_met["multiple_billion_exits"] = (
                len([c for c in self.companies_founded 
                     if c.exit_event and (c.exit_event.exit_value_usd or 0) >= 1000]) >= 2
            )
            criteria_met["total_value_5b"] = self.total_value_created_usd >= 5000
            
        elif level == "L5":  # Growth-Stage Entrepreneurs
            criteria_met["funding_50m"] = any(
                (c.total_funding_raised_usd or 0) >= 50 
                for c in self.companies_founded
            )
            criteria_met["valuation_200m"] = any(
                (c.current_valuation_usd or 0) >= 200 
                for c in self.companies_founded
            )
            
        elif level == "L4":  # Proven Operators
            criteria_met["exit_10m_100m"] = any(
                c.exit_event and 10 <= (c.exit_event.exit_value_usd or 0) <= 100
                for c in self.companies_founded
            )
            
        elif level == "L2":  # Early-Stage
            criteria_met["seed_funding"] = any(
                0.5 <= (c.total_funding_raised_usd or 0) <= 5 
                for c in self.companies_founded
            )
            
        elif level == "L1":  # Nascent
            criteria_met["limited_funding"] = all(
                (c.total_funding_raised_usd or 0) <= 1 
                for c in self.companies_founded
            )
        
        return criteria_met
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/API responses."""
        return {
            "founder_name": self.founder_name,
            "companies_founded": [
                {
                    "company_name": c.company_name,
                    "founder_role": c.founder_role,
                    "current_valuation_usd": c.current_valuation_usd,
                    "total_funding_raised_usd": c.total_funding_raised_usd,
                    "funding_stage": c.funding_stage.value if c.funding_stage else None,
                    "latest_round_amount_usd": c.latest_round_amount_usd,
                    "latest_round_date": c.latest_round_date.isoformat() if c.latest_round_date else None,
                    "founded_date": c.founded_date.isoformat() if c.founded_date else None,
                    "employee_count": c.employee_count,
                    "is_active": c.is_active,
                    "exit_event": {
                        "company_name": c.exit_event.company_name,
                        "exit_type": c.exit_event.exit_type.value,
                        "exit_value_usd": c.exit_event.exit_value_usd,
                        "exit_date": c.exit_event.exit_date.isoformat() if c.exit_event.exit_date else None,
                        "acquirer_name": c.exit_event.acquirer_name,
                        "verified": c.exit_event.verified
                    } if c.exit_event else None,
                    "verification_sources": c.verification_sources
                }
                for c in self.companies_founded
            ],
            "metrics": {
                "total_exits": self.total_exits,
                "total_exit_value_usd": self.total_exit_value_usd,
                "total_value_created_usd": self.total_value_created_usd,
                "highest_exit_value_usd": self.highest_exit_value_usd,
                "unicorn_companies_count": self.unicorn_companies_count,
                "companies_with_major_exits_count": self.companies_with_major_exits_count,
                "current_portfolio_value_usd": self.current_portfolio_value_usd,
                "years_entrepreneurship": self.years_entrepreneurship,
                "first_company_founded_year": self.first_company_founded_year
            },
            "last_updated": self.last_updated.isoformat()
        }


class FounderFinancialDataValidator:
    """Validates founder financial data for consistency."""
    
    @staticmethod
    def validate_financial_profile(profile: FounderFinancialProfile) -> Dict[str, List[str]]:
        """Validate financial profile and return warnings/errors."""
        errors = []
        warnings = []
        
        for company in profile.companies_founded:
            # Check for unrealistic valuations
            if company.current_valuation_usd and company.current_valuation_usd > 100000:  # >$100B
                warnings.append(f"{company.company_name}: Valuation >$100B seems unrealistic")
            
            # Check funding vs valuation consistency
            if (company.total_funding_raised_usd and company.current_valuation_usd and 
                company.total_funding_raised_usd > company.current_valuation_usd):
                warnings.append(f"{company.company_name}: Funding raised > current valuation")
            
            # Check exit event consistency
            if company.exit_event:
                if company.is_active:
                    errors.append(f"{company.company_name}: Cannot be active with exit event")
                
                if (company.current_valuation_usd and company.exit_event.exit_value_usd and
                    company.current_valuation_usd > company.exit_event.exit_value_usd * 2):
                    warnings.append(f"{company.company_name}: Current valuation >> exit value")
        
        # Check for duplicate companies
        company_names = [c.company_name.lower() for c in profile.companies_founded]
        if len(company_names) != len(set(company_names)):
            errors.append("Duplicate company names found")
        
        return {"errors": errors, "warnings": warnings}
