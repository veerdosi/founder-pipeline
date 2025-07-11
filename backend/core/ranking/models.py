"""Data models for founder ranking system."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime, date


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
    
    def calculate_overall_confidence(self) -> float:
        """Calculate overall confidence score based on LinkedIn data only."""
        # Basic LinkedIn data confidence (always available)
        basic_score = 0.3 if self.linkedin_url else 0.1
        return basic_score
    
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
