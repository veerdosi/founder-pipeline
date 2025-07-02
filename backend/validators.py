"""Centralized validation functions for the pipeline."""

import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from datetime import datetime


class ValidationError(Exception):
    """Custom validation error."""
    pass


def validate_linkedin_url(url: str) -> bool:
    """
    Validate if the URL is a proper LinkedIn profile URL.
    
    Accepts:
    - linkedin.com/in/profile
    - www.linkedin.com/in/profile  
    - country.linkedin.com/in/profile (e.g., uk.linkedin.com, ch.linkedin.com)
    
    Args:
        url: The LinkedIn URL to validate
        
    Returns:
        bool: True if valid LinkedIn profile URL, False otherwise
    """
    try:
        if not url:
            return False
        
        # Normalize URL for easier checking
        url_lower = url.lower()
        
        # Check if it contains linkedin.com and /in/
        if "linkedin.com" not in url_lower or "/in/" not in url_lower:
            return False
        
        # Parse the URL
        parsed_url = urlparse(url)
        
        # LinkedIn domains: linkedin.com, www.linkedin.com, *.linkedin.com (country codes)
        domain = parsed_url.netloc.lower()
        valid_domains = (
            domain == "linkedin.com" or 
            domain == "www.linkedin.com" or
            domain.endswith(".linkedin.com")
        )
        
        if not valid_domains:
            return False
            
        # Check path structure
        path_parts = parsed_url.path.strip("/").split("/")
        if len(path_parts) < 2 or path_parts[0] != "in":
            return False
            
        # Profile ID should exist and be reasonable length
        profile_id = path_parts[1]
        if not profile_id or len(profile_id) < 2:
            return False
            
        return True
    except Exception:
        return False


def validate_url(url: str) -> bool:
    """Validate if URL is properly formatted."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def validate_email(email: str) -> bool:
    """Validate email format."""
    if not email:
        return False
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_phone(phone: str) -> bool:
    """Validate phone number format."""
    if not phone:
        return False
    # Remove all non-digits
    digits = re.sub(r'\D', '', phone)
    # Check if it's a reasonable length (7-15 digits)
    return 7 <= len(digits) <= 15


def validate_year(year: Any, min_year: int = 1800) -> bool:
    """Validate if year is reasonable."""
    try:
        year_int = int(year)
        return min_year <= year_int <= datetime.now().year + 1
    except (ValueError, TypeError):
        return False


def validate_funding_amount(amount: Any) -> bool:
    """Validate funding amount is positive number."""
    try:
        amount_float = float(amount)
        return amount_float >= 0
    except (ValueError, TypeError):
        return False


def validate_company_name(name: str) -> bool:
    """Validate company name is reasonable."""
    if not name or not isinstance(name, str):
        return False
    return 2 <= len(name.strip()) <= 200


def validate_person_name(name: str) -> bool:
    """Validate person name is reasonable."""
    if not name or not isinstance(name, str):
        return False
    return 2 <= len(name.strip()) <= 100


def validate_text_length(text: str, min_len: int = 0, max_len: int = 10000) -> bool:
    """Validate text is within reasonable length bounds."""
    if not isinstance(text, str):
        return min_len == 0  # Empty is ok if min_len is 0
    return min_len <= len(text.strip()) <= max_len


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


def validate_numeric_range(value: Any, min_val: float, max_val: float) -> bool:
    """Validate if numeric value is within range."""
    try:
        num_val = float(value)
        return min_val <= num_val <= max_val
    except:
        return False


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
        errors = []
        if not company_data.get("name"):
            errors.append("Company name is required")
        elif len(company_data["name"]) < 2:
            errors.append("Company name must be at least 2 characters")
        
        if company_data.get("website") and not validate_url(company_data["website"]):
            errors.append("Invalid website URL")
        
        if company_data.get("linkedin_url") and not validate_linkedin_url(company_data["linkedin_url"]):
            errors.append("Invalid LinkedIn URL")
        
        if company_data.get("founded_year"):
            year = company_data["founded_year"]
            if not isinstance(year, int) or year < 1800 or year > 2025:
                errors.append("Invalid founded year")
        
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
        errors = []
        if not profile_data.get("person_name"):
            errors.append("Person name is required")
        elif len(profile_data["person_name"]) < 2:
            errors.append("Person name must be at least 2 characters")
        
        if not profile_data.get("linkedin_url"):
            errors.append("LinkedIn URL is required")
        elif not validate_linkedin_url(profile_data["linkedin_url"]):
            errors.append("Invalid LinkedIn URL format")
        
        if profile_data.get("estimated_age"):
            age = profile_data["estimated_age"]
            if not isinstance(age, int) or age < 16 or age > 100:
                errors.append("Invalid estimated age")
        
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
