"""Validation utilities for the pipeline."""

import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from datetime import datetime

from ..models import Company, LinkedInProfile, MarketMetrics


class ValidationError(Exception):
    """Custom validation error."""
    pass


def validate_api_keys(api_keys: Dict[str, str]) -> Dict[str, bool]:
    """Validate API keys format and presence."""
    results = {}
    
    # EXA API Key validation
    exa_key = api_keys.get("exa_api_key", "")
    results["exa_api_key"] = bool(exa_key and len(exa_key) > 10 and exa_key != "your_exa_api_key_here")
    
    # OpenAI API Key validation
    openai_key = api_keys.get("openai_api_key", "")
    results["openai_api_key"] = bool(openai_key and openai_key.startswith("sk-") and len(openai_key) > 20)
    
    # Serper API Key validation
    serper_key = api_keys.get("serper_api_key", "")
    results["serper_api_key"] = bool(serper_key and len(serper_key) > 10 and serper_key != "your_serper_api_key_here")
    
    # Apify API Key validation
    apify_key = api_keys.get("apify_api_key", "")
    results["apify_api_key"] = bool(apify_key and len(apify_key) > 10 and apify_key != "your_apify_api_key_here")
    
    return results


def validate_company_data(data: Dict[str, Any]) -> List[str]:
    """Validate company data and return list of errors."""
    errors = []
    
    # Required fields
    if not data.get("name"):
        errors.append("Company name is required")
    elif len(data["name"]) < 2:
        errors.append("Company name must be at least 2 characters")
    
    # Optional field validation
    if data.get("website") and not is_valid_url(data["website"]):
        errors.append("Invalid website URL")
    
    if data.get("linkedin_url") and not is_valid_linkedin_url(data["linkedin_url"]):
        errors.append("Invalid LinkedIn URL")
    
    if data.get("founded_year"):
        year = data["founded_year"]
        if not isinstance(year, int) or year < 1800 or year > 2025:
            errors.append("Invalid founded year")
    
    if data.get("funding_total_usd"):
        funding = data["funding_total_usd"]
        if not isinstance(funding, (int, float)) or funding < 0:
            errors.append("Invalid funding amount")
    
    return errors


def validate_profile_data(data: Dict[str, Any]) -> List[str]:
    """Validate LinkedIn profile data and return list of errors."""
    errors = []
    
    # Required fields
    if not data.get("person_name"):
        errors.append("Person name is required")
    elif len(data["person_name"]) < 2:
        errors.append("Person name must be at least 2 characters")
    
    if not data.get("linkedin_url"):
        errors.append("LinkedIn URL is required")
    elif not is_valid_linkedin_url(data["linkedin_url"]):
        errors.append("Invalid LinkedIn URL format")
    
    # Optional field validation
    if data.get("estimated_age"):
        age = data["estimated_age"]
        if not isinstance(age, int) or age < 16 or age > 100:
            errors.append("Invalid estimated age")
    
    return errors


def validate_market_metrics(data: Dict[str, Any]) -> List[str]:
    """Validate market metrics data and return list of errors."""
    errors = []
    
    # Market size validation
    if data.get("market_size_billion") is not None:
        size = data["market_size_billion"]
        if not isinstance(size, (int, float)) or size < 0:
            errors.append("Market size must be a positive number")
    
    # CAGR validation
    if data.get("cagr_percent") is not None:
        cagr = data["cagr_percent"]
        if not isinstance(cagr, (int, float)) or cagr < -100 or cagr > 1000:
            errors.append("CAGR must be between -100% and 1000%")
    
    # Score validations (1-5 scale)
    score_fields = ["timing_score", "us_sentiment", "sea_sentiment", "momentum_score"]
    for field in score_fields:
        if data.get(field) is not None:
            score = data[field]
            if not isinstance(score, (int, float)) or score < 1 or score > 5:
                errors.append(f"{field} must be between 1 and 5")
    
    # Competitor count validation
    if data.get("competitor_count") is not None:
        count = data["competitor_count"]
        if not isinstance(count, int) or count < 0:
            errors.append("Competitor count must be a non-negative integer")
    
    # Confidence score validation (0-1 scale)
    if data.get("confidence_score") is not None:
        confidence = data["confidence_score"]
        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
            errors.append("Confidence score must be between 0 and 1")
    
    return errors


def is_valid_url(url: str) -> bool:
    """Check if URL is valid."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False


from ..validators import validate_linkedin_url

# Re-export for backward compatibility  
def is_valid_linkedin_url(url: str) -> bool:
    """Check if LinkedIn URL is valid."""
    return validate_linkedin_url(url)


def is_valid_email(email: str) -> bool:
    """Check if email format is valid."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def is_valid_phone(phone: str) -> bool:
    """Check if phone number format is valid."""
    # Remove all non-digits
    digits = re.sub(r'\D', '', phone)
    
    # Check if it's a reasonable length (7-15 digits)
    return 7 <= len(digits) <= 15


def validate_funding_stage(stage: str) -> bool:
    """Validate funding stage."""
    valid_stages = [
        "seed", "series-a", "series-b", "series-c", "series-d", 
        "ipo", "acquired", "unknown"
    ]
    return stage.lower() in valid_stages


def validate_market_stage(stage: str) -> bool:
    """Validate market stage."""
    valid_stages = ["emerging", "growth", "mature", "unknown"]
    return stage.lower() in valid_stages


def validate_ai_category(category: str) -> bool:
    """Validate AI category."""
    valid_categories = [
        "artificial intelligence", "machine learning", "computer vision",
        "natural language processing", "robotics", "autonomous vehicles",
        "generative ai", "deep learning", "neural networks", "quantum computing",
        "edge ai", "ai hardware", "conversational ai", "ai ethics"
    ]
    return category.lower() in valid_categories


def sanitize_text_input(text: str, max_length: int = 1000) -> str:
    """Sanitize and validate text input."""
    if not isinstance(text, str):
        return ""
    
    # Remove dangerous characters
    text = re.sub(r'[<>"\']', '', text)
    
    # Limit length
    if len(text) > max_length:
        text = text[:max_length]
    
    # Clean whitespace
    text = ' '.join(text.split())
    
    return text.strip()


def validate_csv_headers(headers: List[str], required_headers: List[str]) -> List[str]:
    """Validate CSV headers and return missing ones."""
    missing = []
    headers_lower = [h.lower().strip() for h in headers]
    
    for required in required_headers:
        if required.lower().strip() not in headers_lower:
            missing.append(required)
    
    return missing


def validate_data_completeness(data: Dict[str, Any], required_fields: List[str]) -> float:
    """Calculate data completeness score (0-1)."""
    if not required_fields:
        return 1.0
    
    complete_fields = 0
    for field in required_fields:
        value = data.get(field)
        if value is not None and str(value).strip():
            complete_fields += 1
    
    return complete_fields / len(required_fields)


def validate_date_range(date_str: str, min_year: int = 1900, max_year: Optional[int] = None) -> bool:
    """Validate if date is within reasonable range."""
    if max_year is None:
        max_year = datetime.now().year + 1
    
    try:
        import pandas as pd
        date_obj = pd.to_datetime(date_str)
        year = date_obj.year
        return min_year <= year <= max_year
    except:
        return False


def validate_numeric_range(value: Any, min_val: float, max_val: float) -> bool:
    """Validate if numeric value is within range."""
    try:
        num_val = float(value)
        return min_val <= num_val <= max_val
    except:
        return False


class DataQualityChecker:
    """Comprehensive data quality checker."""
    
    def __init__(self):
        self.quality_scores = {}
    
    def check_company_quality(self, company_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check quality of company data."""
        quality_report = {
            "overall_score": 0.0,
            "completeness": 0.0,
            "validity": 0.0,
            "issues": [],
            "suggestions": []
        }
        
        # Required fields for quality scoring
        required_fields = ["name", "description", "ai_focus"]
        optional_fields = ["website", "founded_year", "funding_total_usd", "founders"]
        
        # Calculate completeness
        quality_report["completeness"] = validate_data_completeness(
            company_data, required_fields + optional_fields
        )
        
        # Validate required fields
        errors = validate_company_data(company_data)
        quality_report["validity"] = 1.0 if not errors else max(0.0, 1.0 - len(errors) * 0.2)
        quality_report["issues"].extend(errors)
        
        # Additional quality checks
        if company_data.get("description"):
            desc_length = len(company_data["description"])
            if desc_length < 20:
                quality_report["issues"].append("Description too short")
                quality_report["suggestions"].append("Add more detailed description")
            elif desc_length > 1000:
                quality_report["issues"].append("Description too long")
        
        if not company_data.get("ai_focus"):
            quality_report["suggestions"].append("Specify AI focus area")
        
        if not company_data.get("website"):
            quality_report["suggestions"].append("Add company website")
        
        # Calculate overall score
        quality_report["overall_score"] = (
            quality_report["completeness"] * 0.6 + 
            quality_report["validity"] * 0.4
        )
        
        return quality_report
    
    def check_profile_quality(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check quality of profile data."""
        quality_report = {
            "overall_score": 0.0,
            "completeness": 0.0,
            "validity": 0.0,
            "issues": [],
            "suggestions": []
        }
        
        # Required fields
        required_fields = ["person_name", "linkedin_url", "role"]
        optional_fields = ["headline", "location", "about", "estimated_age"]
        
        # Calculate completeness
        quality_report["completeness"] = validate_data_completeness(
            profile_data, required_fields + optional_fields
        )
        
        # Validate required fields
        errors = validate_profile_data(profile_data)
        quality_report["validity"] = 1.0 if not errors else max(0.0, 1.0 - len(errors) * 0.3)
        quality_report["issues"].extend(errors)
        
        # Additional quality checks
        if not profile_data.get("headline"):
            quality_report["suggestions"].append("Add professional headline")
        
        if not profile_data.get("about"):
            quality_report["suggestions"].append("Add about section")
        
        # Calculate overall score
        quality_report["overall_score"] = (
            quality_report["completeness"] * 0.5 + 
            quality_report["validity"] * 0.5
        )
        
        return quality_report
    
    def check_dataset_quality(self, dataset: List[Dict[str, Any]], data_type: str = "company") -> Dict[str, Any]:
        """Check quality of entire dataset."""
        if not dataset:
            return {
                "overall_score": 0.0,
                "total_records": 0,
                "valid_records": 0,
                "avg_completeness": 0.0,
                "common_issues": [],
                "recommendations": ["Dataset is empty"]
            }
        
        total_records = len(dataset)
        quality_scores = []
        all_issues = []
        valid_count = 0
        
        # Check each record
        for record in dataset:
            if data_type == "company":
                quality = self.check_company_quality(record)
            else:
                quality = self.check_profile_quality(record)
            
            quality_scores.append(quality["overall_score"])
            all_issues.extend(quality["issues"])
            
            if quality["validity"] > 0.7:
                valid_count += 1
        
        # Calculate dataset statistics
        avg_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        avg_completeness = sum(score for score in quality_scores) / len(quality_scores) if quality_scores else 0
        
        # Find common issues
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        common_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Generate recommendations
        recommendations = []
        if avg_score < 0.6:
            recommendations.append("Overall data quality needs improvement")
        if valid_count / total_records < 0.8:
            recommendations.append("Many records have validation errors")
        if common_issues:
            recommendations.append(f"Most common issue: {common_issues[0][0]}")
        
        return {
            "overall_score": avg_score,
            "total_records": total_records,
            "valid_records": valid_count,
            "validity_rate": valid_count / total_records,
            "avg_completeness": avg_completeness,
            "common_issues": [issue for issue, count in common_issues],
            "recommendations": recommendations
        }


def validate_pipeline_config(config: Dict[str, Any]) -> List[str]:
    """Validate pipeline configuration."""
    errors = []
    
    # Required configuration
    required_keys = ["exa_api_key", "openai_api_key", "serper_api_key", "apify_api_key"]
    for key in required_keys:
        if not config.get(key):
            errors.append(f"Missing required configuration: {key}")
    
    # Validate numeric settings
    if config.get("requests_per_minute"):
        if not validate_numeric_range(config["requests_per_minute"], 1, 300):
            errors.append("requests_per_minute must be between 1 and 300")
    
    if config.get("concurrent_requests"):
        if not validate_numeric_range(config["concurrent_requests"], 1, 20):
            errors.append("concurrent_requests must be between 1 and 20")
    
    if config.get("default_company_limit"):
        if not validate_numeric_range(config["default_company_limit"], 1, 1000):
            errors.append("default_company_limit must be between 1 and 1000")
    
    return errors
