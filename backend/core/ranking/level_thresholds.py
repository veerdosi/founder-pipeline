"""L1-L10 experience level thresholds and validation criteria."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum


class SourceType(Enum):
    """Source credibility hierarchy."""
    CRUNCHBASE = "crunchbase"
    LINKEDIN = "linkedin"
    MEDIA_REPORT = "media_report"  # Lowest credibility
    
    @property
    def credibility_score(self) -> float:
        """Source credibility score (0.0-1.0)."""
        scores = {
            SourceType.CRUNCHBASE: 0.8,
            SourceType.LINKEDIN: 0.6,
            SourceType.MEDIA_REPORT: 0.4
        }
        return scores[self]


@dataclass
class DataPoint:
    """Individual data point with source verification."""
    value: Any
    source_type: SourceType
    source_url: str
    verified: bool = False
    confidence: float = 0.0


@dataclass
class LevelThreshold:
    """Specific thresholds for each L-level."""
    level: str
    description: str
    primary_criteria: List[str]
    financial_thresholds: Dict[str, float]  # In millions USD
    experience_requirements: Dict[str, int]
    verification_requirements: List[SourceType]
    minimum_sources: int = 2


class LevelThresholds:
    """L1-L10 threshold definitions with specific validation criteria."""
    
    @staticmethod
    def get_level_definitions() -> Dict[str, LevelThreshold]:
        """Get complete L-level threshold definitions."""
        return {
            "L10": LevelThreshold(
                level="L10",
                description="Legendary Entrepreneurs",
                primary_criteria=[
                    "Multiple IPOs or exits >$1B",
                    "Created entire industries or market categories",
                    "Recognized industry pioneer status"
                ],
                financial_thresholds={
                    "min_exits": 2,
                    "min_exit_value": 1000.0,  # $1B+
                    "total_value_created": 5000.0  # $5B+
                },
                experience_requirements={
                    "min_companies_founded": 2,
                    "min_years_experience": 20
                },
                verification_requirements=[
                    SourceType.MEDIA_REPORT
                ],
                minimum_sources=3
            ),
            
            "L9": LevelThreshold(
                level="L9",
                description="Transformational Leaders",
                primary_criteria=[
                    "1 major IPO or exit >$1B",
                    "Building second major company",
                    "Recognized as industry visionary"
                ],
                financial_thresholds={
                    "min_exits": 1,
                    "min_exit_value": 1000.0,  # $1B+
                    "current_company_valuation": 500.0  # $500M+
                },
                experience_requirements={
                    "min_companies_founded": 2,
                    "min_years_experience": 15
                },
                verification_requirements=[
                    SourceType.CRUNCHBASE
                ],
                minimum_sources=2
            ),
            
            "L8": LevelThreshold(
                level="L8",
                description="Proven Unicorn Builders",
                primary_criteria=[
                    "Built 1+ companies to $1B+ valuation",
                    "Achieved unicorn status"
                ],
                financial_thresholds={
                    "min_company_valuation": 1000.0,  # $1B+
                    "min_funding_raised": 100.0  # $100M+
                },
                experience_requirements={
                    "min_companies_founded": 1,
                    "min_years_experience": 10
                },
                verification_requirements=[
                    SourceType.CRUNCHBASE,
                    SourceType.MEDIA_REPORT
                ],
                minimum_sources=2
            ),
            
            "L7": LevelThreshold(
                level="L7",
                description="Elite Serial Entrepreneurs",
                primary_criteria=[
                    "2+ exits >$100M OR 2+ unicorn companies founded",
                    "Multiple successful companies"
                ],
                financial_thresholds={
                    "min_exits": 2,
                    "min_exit_value": 100.0,  # $100M+
                    "total_value_created": 500.0  # $500M+
                },
                experience_requirements={
                    "min_companies_founded": 2,
                    "min_years_experience": 12
                },
                verification_requirements=[
                    SourceType.CRUNCHBASE,
                    SourceType.MEDIA_REPORT
                ],
                minimum_sources=2
            ),
            
            "L6": LevelThreshold(
                level="L6",
                description="Market Innovators and Thought Leaders",
                primary_criteria=[
                    "Groundbreaking innovation recognition",
                    "Disrupted or created new markets",
                    "Significant media recognition"
                ],
                financial_thresholds={
                    "min_company_valuation": 100.0,  # $100M+
                    "min_funding_raised": 25.0  # $25M+
                },
                experience_requirements={
                    "min_years_experience": 8,
                    "patents_filed": 3,
                    "awards_received": 1
                },
                verification_requirements=[
                    SourceType.PATENT_DATABASE,
                    SourceType.MEDIA_REPORT
                ],
                minimum_sources=2
            ),
            
            "L5": LevelThreshold(
                level="L5",
                description="Growth-Stage Entrepreneurs",
                primary_criteria=[
                    "Scaled companies to >$50M funding",
                    "Positioned for major exits",
                    "IPO preparation experience"
                ],
                financial_thresholds={
                    "min_funding_raised": 50.0,  # $50M+
                    "min_company_valuation": 200.0  # $200M+
                },
                experience_requirements={
                    "min_companies_founded": 1,
                    "min_years_experience": 7
                },
                verification_requirements=[
                    SourceType.CRUNCHBASE,
                ],
                minimum_sources=2
            ),
            
            "L4": LevelThreshold(
                level="L4",
                description="Proven Operators with Exits or Executive Experience",
                primary_criteria=[
                    "Exit between $10M-$100M OR C-level at notable tech company",
                    "Senior executive experience"
                ],
                financial_thresholds={
                    "min_exit_value": 10.0,  # $10M+
                    "max_exit_value": 100.0,  # <$100M
                    "company_size_employees": 1000
                },
                experience_requirements={
                    "min_years_experience": 8,
                    "executive_roles": 1
                },
                verification_requirements=[
                    SourceType.LINKEDIN,
                    SourceType.CRUNCHBASE
                ],
                minimum_sources=2
            ),
            
            "L3": LevelThreshold(
                level="L3",
                description="Technical and Management Veterans",
                primary_criteria=[
                    "10+ years combined technical/management experience",
                    "PhD in relevant field OR senior role at fast-growing company"
                ],
                financial_thresholds={
                    "company_growth_rate": 50.0  # 50%+ YoY growth
                },
                experience_requirements={
                    "min_years_experience": 10,
                    "technical_experience": 5,
                    "management_experience": 3
                },
                verification_requirements=[
                    SourceType.LINKEDIN,
                    SourceType.UNIVERSITY_RECORD
                ],
                minimum_sources=2
            ),
            
            "L2": LevelThreshold(
                level="L2",
                description="Early-Stage Entrepreneurs",
                primary_criteria=[
                    "Accelerator graduate OR 2-5 years startup experience",
                    "Seed funding raised"
                ],
                financial_thresholds={
                    "min_funding_raised": 0.5,  # $500K+
                    "max_funding_raised": 5.0   # <$5M
                },
                experience_requirements={
                    "min_years_experience": 2,
                    "max_years_experience": 5
                },
                verification_requirements=[
                    SourceType.CRUNCHBASE,
                    SourceType.LINKEDIN
                ],
                minimum_sources=1
            ),
            
            "L1": LevelThreshold(
                level="L1", 
                description="Nascent Founders with Potential",
                primary_criteria=[
                    "<2 years professional experience",
                    "First-time founder OR recent graduate"
                ],
                financial_thresholds={
                    "max_funding_raised": 1.0  # <$1M
                },
                experience_requirements={
                    "max_years_experience": 2
                },
                verification_requirements=[
                    SourceType.LINKEDIN
                ],
                minimum_sources=1
            )
        }
    
    @staticmethod
    def get_search_strategies() -> Dict[str, List[str]]:
        """Get search strategies for each L-level validation."""
        return {
            "L10": [
                "founder name + multiple IPOs",
                "industry pioneer",
                "legendary entrepreneur",
                "founder name + billion exit"
            ],
            "L9": [
                "founder name + billion exit",
                "building second company",
                "industry visionary",
                "founder name + transformational leader"
            ],
            "L8": [
                "founder name + unicorn",
                "$1B valuation",
                "unicorn founder"
            ],
            "L7": [
                "founder name + IPO",
                "founder name + acquisition + $100M+",
                "founder name + serial entrepreneur"
            ],
            "L6": [
                "founder name + innovation award",
                "founder name + thought leader",
                "founder name + Forbes",
                "founder name + TED talk"
            ],
            "L5": [
                "founder name + Series C",
                "founder name + $50M funding",
                "founder name + IPO preparation"
            ],
            "L4": [
                "founder name + exit",
                "founder name + CTO",
                "founder name + VP",
                "founder name + acquired"
            ],
            "L3": [
                "founder name + PhD",
                "founder name + senior engineer",
                "founder name + 10 years experience"
            ],
            "L2": [
                "founder name + Y Combinator",
                "founder name + Techstars",
                "founder name + accelerator",
                "founder name + seed funding"
            ],
            "L1": [
                "founder name + recent graduate",
                "founder name + first startup",
                "founder name + young entrepreneur"
            ]
        }


class LevelValidator:
    """Validates founder data against L-level thresholds."""
    
    def __init__(self):
        self.thresholds = LevelThresholds.get_level_definitions()
        self.search_strategies = LevelThresholds.get_search_strategies()
    
    def validate_level_assignment(
        self, 
        proposed_level: str, 
        data_points: List[DataPoint]
    ) -> Dict[str, Any]:
        """Validate if founder meets criteria for proposed level."""
        
        if proposed_level not in self.thresholds:
            return {"valid": False, "reason": f"Invalid level: {proposed_level}"}
        
        threshold = self.thresholds[proposed_level]
        validation_result = {
            "valid": False,
            "confidence": 0.0,
            "missing_criteria": [],
            "verified_criteria": [],
            "source_quality_score": 0.0,
            "recommendation": proposed_level
        }
        
        # Check source requirements
        source_types = [dp.source_type for dp in data_points if dp.verified]
        missing_sources = [
            req for req in threshold.verification_requirements 
            if req not in source_types
        ]
        
        if len(source_types) < threshold.minimum_sources:
            validation_result["missing_criteria"].append(
                f"Need {threshold.minimum_sources} sources, have {len(source_types)}"
            )
        
        # Calculate source quality score
        if source_types:
            avg_credibility = sum(st.credibility_score for st in source_types) / len(source_types)
            validation_result["source_quality_score"] = avg_credibility
        
        # Check financial thresholds
        financial_data = self._extract_financial_data(data_points)
        financial_validation = self._validate_financial_thresholds(
            financial_data, threshold.financial_thresholds
        )
        
        # Check experience requirements  
        experience_data = self._extract_experience_data(data_points)
        experience_validation = self._validate_experience_requirements(
            experience_data, threshold.experience_requirements
        )
        
        # Combine validation results
        all_criteria_met = (
            len(missing_sources) == 0 and
            len(source_types) >= threshold.minimum_sources and
            financial_validation["valid"] and
            experience_validation["valid"]
        )
        
        validation_result["valid"] = all_criteria_met
        validation_result["missing_criteria"].extend(financial_validation["missing"])
        validation_result["missing_criteria"].extend(experience_validation["missing"])
        validation_result["verified_criteria"].extend(financial_validation["met"])
        validation_result["verified_criteria"].extend(experience_validation["met"])
        
        # Calculate confidence based on data quality and completeness
        if all_criteria_met:
            validation_result["confidence"] = min(1.0, validation_result["source_quality_score"] + 0.2)
        else:
            # Suggest alternative level if current doesn't meet criteria
            alternative_level = self._suggest_alternative_level(data_points)
            validation_result["recommendation"] = alternative_level
            validation_result["confidence"] = 0.5 * validation_result["source_quality_score"]
        
        return validation_result
    
    def _extract_financial_data(self, data_points: List[DataPoint]) -> Dict[str, float]:
        """Extract financial data from data points."""
        financial_data = {}
        
        for dp in data_points:
            if isinstance(dp.value, dict):
                for key, value in dp.value.items():
                    if key in ["exit_value", "funding_raised", "company_valuation", "exits"]:
                        financial_data[key] = float(value) if value else 0.0
        
        return financial_data
    
    def _extract_experience_data(self, data_points: List[DataPoint]) -> Dict[str, int]:
        """Extract experience data from data points."""
        experience_data = {}
        
        for dp in data_points:
            if isinstance(dp.value, dict):
                for key, value in dp.value.items():
                    if key in ["years_experience", "companies_founded", "patents_filed"]:
                        experience_data[key] = int(value) if value else 0
        
        return experience_data
    
    def _validate_financial_thresholds(
        self, 
        financial_data: Dict[str, float], 
        thresholds: Dict[str, float]
    ) -> Dict[str, Any]:
        """Validate financial data against thresholds."""
        
        result = {"valid": True, "met": [], "missing": []}
        
        for threshold_key, threshold_value in thresholds.items():
            data_value = financial_data.get(threshold_key, 0.0)
            
            if threshold_key.startswith("min_") and data_value >= threshold_value:
                result["met"].append(f"{threshold_key}: ${data_value}M >= ${threshold_value}M")
            elif threshold_key.startswith("max_") and data_value <= threshold_value:
                result["met"].append(f"{threshold_key}: ${data_value}M <= ${threshold_value}M")
            else:
                result["missing"].append(f"{threshold_key}: need ${threshold_value}M, have ${data_value}M")
                result["valid"] = False
        
        return result
    
    def _validate_experience_requirements(
        self, 
        experience_data: Dict[str, int], 
        requirements: Dict[str, int]
    ) -> Dict[str, Any]:
        """Validate experience data against requirements."""
        
        result = {"valid": True, "met": [], "missing": []}
        
        for req_key, req_value in requirements.items():
            data_value = experience_data.get(req_key, 0)
            
            if req_key.startswith("min_") and data_value >= req_value:
                result["met"].append(f"{req_key}: {data_value} >= {req_value}")
            elif req_key.startswith("max_") and data_value <= req_value:
                result["met"].append(f"{req_key}: {data_value} <= {req_value}")
            else:
                result["missing"].append(f"{req_key}: need {req_value}, have {data_value}")
                result["valid"] = False
        
        return result
    
    def _suggest_alternative_level(self, data_points: List[DataPoint]) -> str:
        """Suggest alternative level based on available data."""
        
        # Simple heuristic: suggest lower level if current doesn't meet criteria
        financial_data = self._extract_financial_data(data_points)
        experience_data = self._extract_experience_data(data_points)
        
        # Start from L1 and work up based on data
        years_exp = experience_data.get("years_experience", 0)
        funding = financial_data.get("funding_raised", 0.0)
        exits = financial_data.get("exits", 0)
        
        if exits >= 2 and financial_data.get("exit_value", 0) >= 1000:
            return "L10"
        elif exits >= 1 and financial_data.get("exit_value", 0) >= 1000:
            return "L9"
        elif financial_data.get("company_valuation", 0) >= 1000:
            return "L8"
        elif exits >= 2 and financial_data.get("exit_value", 0) >= 100:
            return "L7"
        elif funding >= 50:
            return "L5"
        elif funding >= 10 or years_exp >= 8:
            return "L4"
        elif years_exp >= 10:
            return "L3"
        elif funding >= 0.5 and years_exp >= 2:
            return "L2"
        else:
            return "L1"
