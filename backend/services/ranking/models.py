"""Data models for founder ranking system."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum


class ExperienceLevel(Enum):
    """L1-L10 experience levels."""
    L1 = "L1"  # Nascent Founders with Potential
    L2 = "L2"  # Early-Stage Entrepreneurs  
    L3 = "L3"  # Technical and Management Veterans
    L4 = "L4"  # Proven Operators with Exits or Executive Experience
    L5 = "L5"  # Growth-Stage Entrepreneurs
    L6 = "L6"  # Market Innovators and Thought Leaders
    L7 = "L7"  # Elite Serial Entrepreneurs
    L8 = "L8"  # Proven Unicorn Builders
    L9 = "L9"  # Transformational Leaders
    L10 = "L10"  # Legendary Entrepreneurs
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
            skill_3=row.get("skill_3")
        )


@dataclass
class FounderRanking:
    """Complete founder ranking result."""
    profile: FounderProfile
    classification: LevelClassification
    timestamp: str
    processing_metadata: Dict[str, Any]
