"""Centralized validation functions for the pipeline."""

import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from datetime import datetime


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
