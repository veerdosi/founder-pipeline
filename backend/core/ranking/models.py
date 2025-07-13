"""Data models for founder ranking system."""

from dataclasses import dataclass
from typing import List, Dict, Any
from enum import Enum


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
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass
class LevelClassification:
    """Classification result for a founder's experience level."""
    level: ExperienceLevel
    confidence_score: float  # 0.0 to 1.0
    reasoning: str
    evidence: List[str]
    verification_sources: List[str]


@dataclass
class FounderRanking:
    """Complete founder ranking result."""
    profile: Any  # LinkedInProfile from models
    classification: LevelClassification
    timestamp: str
    processing_metadata: Dict[str, Any]
