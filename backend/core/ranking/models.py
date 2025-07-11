"""Data models for founder ranking system."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime, date


# Founder Intelligence Data Models

class ExitType(Enum):
    """Types of company exits."""
    IPO = "ipo"
    ACQUISITION = "acquisition"
    MERGER = "merger"
    SPINOFF = "spinoff"
    UNKNOWN = "unknown"


class InvestmentType(Enum):
    """Types of investments made by founder."""
    ANGEL = "angel"
    SERIES_A = "series_a"
    VENTURE = "venture"
    ADVISORY = "advisory"
    BOARD_SEAT = "board_seat"
    UNKNOWN = "unknown"


class MediaType(Enum):
    """Types of media mentions."""
    NEWS_ARTICLE = "news_article"
    INTERVIEW = "interview"
    PODCAST = "podcast"
    AWARD = "award"
    SPEAKING_ENGAGEMENT = "speaking_engagement"
    THOUGHT_LEADERSHIP = "thought_leadership"
    PRESS_RELEASE = "press_release"
    UNKNOWN = "unknown"


@dataclass
class CompanyExit:
    """Information about a company exit."""
    company_name: str
    exit_type: ExitType
    exit_value_usd: Optional[float] = None
    exit_date: Optional[date] = None
    founder_stake_percent: Optional[float] = None
    founder_payout_usd: Optional[float] = None
    acquiring_company: Optional[str] = None
    exchange: Optional[str] = None  # For IPOs
    ticker_symbol: Optional[str] = None  # For IPOs
    verification_sources: List[str] = field(default_factory=list)
    confidence_score: float = 0.0


@dataclass
class CompanyFounding:
    """Information about companies founded."""
    company_name: str
    founding_date: Optional[date] = None
    founder_role: Optional[str] = None  # "co-founder", "sole founder", etc.
    equity_stake_percent: Optional[float] = None
    current_valuation_usd: Optional[float] = None
    total_funding_raised_usd: Optional[float] = None
    current_status: Optional[str] = None  # "active", "acquired", "closed"
    is_current_company: bool = False
    verification_sources: List[str] = field(default_factory=list)


@dataclass
class Investment:
    """Information about investments made by founder."""
    company_name: str
    investment_type: InvestmentType
    investment_amount_usd: Optional[float] = None
    investment_date: Optional[date] = None
    current_value_usd: Optional[float] = None
    return_multiple: Optional[float] = None
    verification_sources: List[str] = field(default_factory=list)


@dataclass
class BoardPosition:
    """Information about board positions."""
    company_name: str
    position_title: str  # "Board Member", "Chairman", "Advisory Board"
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    is_current: bool = True
    company_valuation_usd: Optional[float] = None
    verification_sources: List[str] = field(default_factory=list)


@dataclass
class FounderFinancialProfile:
    """Comprehensive financial profile for a founder."""
    founder_name: str
    
    # Company founding history
    companies_founded: List[CompanyFounding] = field(default_factory=list)
    
    # Exit history
    company_exits: List[CompanyExit] = field(default_factory=list)
    
    # Investment activity
    investments_made: List[Investment] = field(default_factory=list)
    
    # Board positions
    board_positions: List[BoardPosition] = field(default_factory=list)
    
    # Calculated financial metrics
    total_exit_value_usd: Optional[float] = None
    estimated_net_worth_usd: Optional[float] = None
    total_funding_raised_usd: Optional[float] = None
    number_of_exits: int = 0
    number_of_companies_founded: int = 0
    number_of_investments: int = 0
    
    # Data collection metadata
    data_sources: List[str] = field(default_factory=list)
    last_updated: Optional[datetime] = None
    confidence_score: float = 0.0
    
    def calculate_metrics(self) -> None:
        """Calculate derived financial metrics."""
        self.number_of_exits = len(self.company_exits)
        self.number_of_companies_founded = len(self.companies_founded)
        self.number_of_investments = len(self.investments_made)
        
        # Calculate total exit value
        exit_values = [exit.exit_value_usd for exit in self.company_exits if exit.exit_value_usd]
        self.total_exit_value_usd = sum(exit_values) if exit_values else None
        
        # Calculate total funding raised across all companies
        funding_amounts = [company.total_funding_raised_usd for company in self.companies_founded if company.total_funding_raised_usd]
        self.total_funding_raised_usd = sum(funding_amounts) if funding_amounts else None


@dataclass
class MediaMention:
    """Individual media mention or coverage."""
    title: str
    publication: str
    media_type: MediaType
    publication_date: Optional[date] = None
    url: Optional[str] = None
    summary: Optional[str] = None
    sentiment: Optional[str] = None  # "positive", "neutral", "negative"
    reach_estimate: Optional[int] = None  # Estimated audience reach
    importance_score: float = 0.0  # 0-1 score based on publication prominence
    verification_sources: List[str] = field(default_factory=list)


@dataclass
class Award:
    """Awards and recognition received."""
    award_name: str
    awarding_organization: str
    award_date: Optional[date] = None
    category: Optional[str] = None
    description: Optional[str] = None
    significance_level: Optional[str] = None  # "local", "national", "international"
    verification_sources: List[str] = field(default_factory=list)


@dataclass
class ThoughtLeadership:
    """Thought leadership activities."""
    activity_type: str  # "keynote", "panel", "book", "paper", "blog"
    title: str
    venue_or_publication: str
    date: Optional[date] = None
    audience_size: Optional[int] = None
    topic: Optional[str] = None
    url: Optional[str] = None
    verification_sources: List[str] = field(default_factory=list)


@dataclass
class FounderMediaProfile:
    """Comprehensive media and public presence profile."""
    founder_name: str
    
    # Media coverage
    media_mentions: List[MediaMention] = field(default_factory=list)
    
    # Awards and recognition
    awards: List[Award] = field(default_factory=list)
    
    # Thought leadership
    thought_leadership: List[ThoughtLeadership] = field(default_factory=list)
    
    # Social media presence
    twitter_followers: Optional[int] = None
    linkedin_connections: Optional[int] = None
    
    # Calculated metrics
    total_media_mentions: int = 0
    positive_sentiment_ratio: float = 0.0
    thought_leader_score: float = 0.0  # Derived from speaking, writing, etc.
    public_profile_score: float = 0.0  # Overall public visibility
    
    # Data collection metadata
    data_sources: List[str] = field(default_factory=list)
    last_updated: Optional[datetime] = None
    confidence_score: float = 0.0
    
    def calculate_metrics(self) -> None:
        """Calculate derived media metrics."""
        self.total_media_mentions = len(self.media_mentions)
        
        # Calculate sentiment ratio
        sentiments = [mention.sentiment for mention in self.media_mentions if mention.sentiment]
        if sentiments:
            positive_count = sum(1 for s in sentiments if s == "positive")
            self.positive_sentiment_ratio = positive_count / len(sentiments)
        
        # Calculate thought leader score based on activities
        leadership_points = len(self.thought_leadership) * 10 + len(self.awards) * 15
        self.thought_leader_score = min(leadership_points / 100, 1.0)
        
        # Calculate overall public profile score
        media_points = min(self.total_media_mentions * 2, 50)
        social_points = min((self.twitter_followers or 0) / 10000 * 20, 20)
        self.public_profile_score = min((media_points + social_points + leadership_points) / 100, 1.0)


@dataclass
class WebSearchResult:
    """Individual web search result."""
    query: str
    title: str
    url: str
    snippet: str
    source: str  # "perplexity", "google", "bing"
    search_date: datetime
    relevance_score: float = 0.0
    extracted_facts: List[str] = field(default_factory=list)
    data_type: Optional[str] = None  # "financial", "media", "biographical"


@dataclass
class FounderWebSearchData:
    """Web search intelligence for founder."""
    founder_name: str
    
    # Search results by category
    financial_search_results: List[WebSearchResult] = field(default_factory=list)
    media_search_results: List[WebSearchResult] = field(default_factory=list)
    biographical_search_results: List[WebSearchResult] = field(default_factory=list)
    
    # Extracted insights
    verified_facts: List[str] = field(default_factory=list)
    conflicting_information: List[Dict[str, Any]] = field(default_factory=list)
    data_gaps: List[str] = field(default_factory=list)
    
    # Search metadata
    total_searches_performed: int = 0
    last_search_date: Optional[datetime] = None
    search_sources_used: List[str] = field(default_factory=list)
    overall_data_quality: float = 0.0
    
    def add_search_result(self, result: WebSearchResult) -> None:
        """Add a search result to appropriate category."""
        if result.data_type == "financial":
            self.financial_search_results.append(result)
        elif result.data_type == "media":
            self.media_search_results.append(result)
        else:
            self.biographical_search_results.append(result)
        
        self.total_searches_performed += 1
        self.last_search_date = datetime.now()
        
        if result.source not in self.search_sources_used:
            self.search_sources_used.append(result.source)


# Main Ranking Models


class ExperienceLevel(Enum):
    """L1-L10 experience levels."""
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    L4 = "L4"
    L5 = "L5"
    L6 = "L6"
    L7 = "L7" 
    L8 = "L8"  
    L9 = "L9"  
    L10 = "L10"  
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"  # Confidence <75%


@dataclass
class LevelClassification:
    """Classification result for a founder's experience level."""
    level: ExperienceLevel
    confidence_score: float  # 0.0 to 1.0
    reasoning: str
    evidence: List[str]
    verification_sources: List[str]


@dataclass
class FounderProfile:
    """Founder profile data for ranking."""
    name: str
    company_name: str
    title: str
    linkedin_url: Optional[str] = None
    location: Optional[str] = None
    about: Optional[str] = None
    estimated_age: Optional[int] = None
    
    # Experience data
    experience_1_title: Optional[str] = None
    experience_1_company: Optional[str] = None
    experience_2_title: Optional[str] = None
    experience_2_company: Optional[str] = None
    experience_3_title: Optional[str] = None
    experience_3_company: Optional[str] = None
    
    # Education data
    education_1_school: Optional[str] = None
    education_1_degree: Optional[str] = None
    education_2_school: Optional[str] = None
    education_2_degree: Optional[str] = None
    
    # Skills
    skill_1: Optional[str] = None
    skill_2: Optional[str] = None
    skill_3: Optional[str] = None
    
    # Comprehensive founder intelligence data
    financial_profile: Optional[FounderFinancialProfile] = None
    media_profile: Optional[FounderMediaProfile] = None
    web_search_data: Optional[FounderWebSearchData] = None
    
    # Data collection metadata
    data_collection_timestamp: Optional[datetime] = None
    data_collected: bool = False
    
    @classmethod
    def from_csv_row(cls, row: Dict[str, Any]) -> "FounderProfile":
        """Create FounderProfile from CSV row data."""
        return cls(
            name=row.get("person_name", ""),
            company_name=row.get("company_name", ""),
            title=row.get("title", ""),
            linkedin_url=row.get("linkedin_url"),
            location=row.get("location"),
            about=row.get("about"),
            estimated_age=int(row["estimated_age"]) if row.get("estimated_age") and str(row.get("estimated_age")).isdigit() else None,
            experience_1_title=row.get("experience_1_title"),
            experience_1_company=row.get("experience_1_company"),
            experience_2_title=row.get("experience_2_title"),
            experience_2_company=row.get("experience_2_company"),
            experience_3_title=row.get("experience_3_title"),
            experience_3_company=row.get("experience_3_company"),
            education_1_school=row.get("education_1_school"),
            education_1_degree=row.get("education_1_degree"),
            education_2_school=row.get("education_2_school"),
            education_2_degree=row.get("education_2_degree"),
            skill_1=row.get("skill_1"),
            skill_2=row.get("skill_2"),
            skill_3=row.get("skill_3"),
            data_collection_timestamp=datetime.now(),
            data_collected=False
        )
    
    def get_company_names(self) -> List[str]:
        """Extract all company names associated with this founder."""
        companies = [self.company_name]
        
        # Add companies from experience
        if self.experience_1_company:
            companies.append(self.experience_1_company)
        if self.experience_2_company:
            companies.append(self.experience_2_company)
        if self.experience_3_company:
            companies.append(self.experience_3_company)
        
        # Add companies from financial profile
        if self.financial_profile and hasattr(self.financial_profile, 'companies_founded'):
            financial_companies = [
                comp.company_name 
                for comp in self.financial_profile.companies_founded
                if comp.company_name
            ]
            companies.extend(financial_companies)
        
        # Remove duplicates and clean
        unique_companies = []
        seen = set()
        for company in companies:
            if company and company.strip().lower() not in seen:
                seen.add(company.strip().lower())
                unique_companies.append(company.strip())
        
        return unique_companies
    
    def get_claimed_degrees(self) -> List[Dict[str, str]]:
        """Extract claimed degree information for verification."""
        degrees = []
        
        if self.education_1_school and self.education_1_degree:
            degrees.append({
                "institution": self.education_1_school,
                "degree": self.education_1_degree,
                "field": "",  
                "year": ""   
            })
        
        if self.education_2_school and self.education_2_degree:
            degrees.append({
                "institution": self.education_2_school,
                "degree": self.education_2_degree,
                "field": "",
                "year": ""
            })
        
        return degrees
    
    def has_intelligence_data(self) -> bool:
        """Check if founder has any intelligence data collected."""
        return any([
            self.financial_profile is not None,
            self.media_profile is not None,
            self.web_search_data is not None,
            self.data_collected
        ])
    
    def has_financial_data(self) -> bool:
        """Check if founder has meaningful financial data."""
        if not self.financial_profile:
            return False
        
        if hasattr(self.financial_profile, 'company_exits'):
            return (len(self.financial_profile.company_exits) > 0 or 
                   len(self.financial_profile.companies_founded) > 1 or
                   self.financial_profile.total_exit_value_usd is not None)
        
        return False
    
    def has_media_presence(self) -> bool:
        """Check if founder has significant media presence."""
        if not self.media_profile:
            return False
        
        if hasattr(self.media_profile, 'media_mentions'):
            return (len(self.media_profile.media_mentions) > 5 or
                   len(self.media_profile.awards) > 0 or
                   self.media_profile.thought_leader_score > 0.3)
        
        return False
    
    def get_exit_summary(self) -> Dict[str, Any]:
        """Get summary of founder's exit history."""
        if not self.financial_profile or not hasattr(self.financial_profile, 'company_exits'):
            return {"total_exits": 0, "total_value": None, "largest_exit": None}
        
        exits = self.financial_profile.company_exits
        return {
            "total_exits": len(exits),
            "total_value": self.financial_profile.total_exit_value_usd,
            "largest_exit": max(exits, key=lambda x: x.exit_value_usd or 0) if exits else None
        }
    
    def get_media_summary(self) -> Dict[str, Any]:
        """Get summary of founder's media presence."""
        if not self.media_profile:
            return {"total_mentions": 0, "awards": 0, "thought_leader_score": 0.0}
        
        if hasattr(self.media_profile, 'media_mentions'):
            return {
                "total_mentions": len(self.media_profile.media_mentions),
                "awards": len(self.media_profile.awards),
                "thought_leader_score": self.media_profile.thought_leader_score
            }
        
        return {"total_mentions": 0, "awards": 0, "thought_leader_score": 0.0}
    
    def get_web_intelligence_summary(self) -> Dict[str, Any]:
        """Get summary of web intelligence data."""
        if not self.web_search_data:
            return {"verified_facts": 0, "data_quality": 0.0, "searches_performed": 0}
        
        if hasattr(self.web_search_data, 'verified_facts'):
            return {
                "verified_facts": len(self.web_search_data.verified_facts),
                "data_quality": self.web_search_data.overall_data_quality,
                "searches_performed": self.web_search_data.total_searches_performed
            }
        
        return {"verified_facts": 0, "data_quality": 0.0, "searches_performed": 0}
    
    def calculate_overall_confidence(self) -> float:
        """Calculate overall confidence score across all data sources."""
        scores = []
        
        # Basic LinkedIn data confidence (always available)
        basic_score = 0.3 if self.linkedin_url else 0.1
        scores.append(basic_score)
        
        # Enhanced data confidence scores
        if self.financial_profile and hasattr(self.financial_profile, 'confidence_score'):
            scores.append(self.financial_profile.confidence_score)
        
        if self.media_profile and hasattr(self.media_profile, 'confidence_score'):
            scores.append(self.media_profile.confidence_score)
        
        if self.web_search_data and hasattr(self.web_search_data, 'overall_data_quality'):
            scores.append(self.web_search_data.overall_data_quality)
        
        return sum(scores) / len(scores) if scores else 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary including all data for export."""
        base_dict = {
            # Core profile data
            "name": self.name,
            "company_name": self.company_name,
            "title": self.title,
            "linkedin_url": self.linkedin_url,
            "location": self.location,
            "about": self.about,
            "estimated_age": self.estimated_age,
            
            # Experience
            "experience_1_title": self.experience_1_title,
            "experience_1_company": self.experience_1_company,
            "experience_2_title": self.experience_2_title,
            "experience_2_company": self.experience_2_company,
            "experience_3_title": self.experience_3_title,
            "experience_3_company": self.experience_3_company,
            
            # Education
            "education_1_school": self.education_1_school,
            "education_1_degree": self.education_1_degree,
            "education_2_school": self.education_2_school,
            "education_2_degree": self.education_2_degree,
            
            # Skills
            "skill_1": self.skill_1,
            "skill_2": self.skill_2,
            "skill_3": self.skill_3,
            
            # Metadata
            "data_collected": self.data_collected,
            "data_collection_timestamp": self.data_collection_timestamp.isoformat() if self.data_collection_timestamp else None,
        }
        
        # Add financial data summary
        if self.financial_profile:
            financial_summary = self.get_exit_summary()
            base_dict.update({
                "total_exits": financial_summary["total_exits"],
                "total_exit_value_usd": financial_summary["total_value"],
                "companies_founded_count": len(self.financial_profile.companies_founded) if hasattr(self.financial_profile, 'companies_founded') else 0,
                "total_funding_raised_usd": self.financial_profile.total_funding_raised_usd if hasattr(self.financial_profile, 'total_funding_raised_usd') else None,
                "financial_confidence": self.financial_profile.confidence_score if hasattr(self.financial_profile, 'confidence_score') else 0.0
            })
        else:
            base_dict.update({
                "total_exits": 0,
                "total_exit_value_usd": None,
                "companies_founded_count": 0,
                "total_funding_raised_usd": None,
                "financial_confidence": 0.0
            })
        
        # Add media data summary
        if self.media_profile:
            media_summary = self.get_media_summary()
            base_dict.update({
                "media_mentions_count": media_summary["total_mentions"],
                "awards_count": media_summary["awards"],
                "thought_leader_score": media_summary["thought_leader_score"],
                "media_confidence": self.media_profile.confidence_score if hasattr(self.media_profile, 'confidence_score') else 0.0
            })
        else:
            base_dict.update({
                "media_mentions_count": 0,
                "awards_count": 0,
                "thought_leader_score": 0.0,
                "media_confidence": 0.0
            })
        
        # Add web intelligence summary
        web_summary = self.get_web_intelligence_summary()
        base_dict.update({
            "verified_facts_count": web_summary["verified_facts"],
            "web_data_quality": web_summary["data_quality"],
            "searches_performed": web_summary["searches_performed"]
        })
        
        # Overall metrics
        base_dict["overall_confidence"] = self.calculate_overall_confidence()
        
        return base_dict


@dataclass
class FounderRanking:
    """Complete founder ranking result."""
    profile: FounderProfile
    classification: LevelClassification
    timestamp: str
    processing_metadata: Dict[str, Any]
