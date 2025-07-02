"""Utility functions for data processing and validation."""

import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, TypeVar
from urllib.parse import urlparse

import pandas as pd

# Import centralized validators
from ..validators import validate_linkedin_url, validate_email, validate_url

T = TypeVar('T')


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    
    # Remove extra whitespace and normalize
    text = " ".join(text.split())
    
    # Remove any <think> tags from LLM output
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    return text.strip()


def extract_year_from_date(date_str: str) -> Optional[int]:
    """Extract year from various date string formats."""
    if not date_str:
        return None
        
    try:
        return pd.to_datetime(date_str).year
    except:
        # Try to extract year with regex
        year_match = re.search(r'\b(19|20)\d{2}\b', str(date_str))
        if year_match:
            return int(year_match.group())
    
    return None


def extract_name_from_linkedin_url(url: str) -> str:
    """Extract name from LinkedIn URL."""
    try:
        parts = url.split('/in/')
        if len(parts) > 1:
            name_part = parts[1].split('/')[0].replace('-', ' ').replace('%20', ' ')
            return name_part.title()
    except:
        pass
    return "Unknown"


def deduplicate_by_key(items: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    """Deduplicate a list of dictionaries by a specific key."""
    seen = set()
    unique_items = []
    
    for item in items:
        value = item.get(key)
        if value and value not in seen:
            seen.add(value)
            unique_items.append(item)
    
    return unique_items


def deduplicate_by_similarity(items: List[str], threshold: float = 0.8) -> List[str]:
    """Deduplicate strings by similarity using simple ratio."""
    unique_items = []
    
    for item in items:
        item_clean = item.lower().strip()
        is_duplicate = False
        
        for existing in unique_items:
            existing_clean = existing.lower().strip()
            
            # Simple similarity check
            similarity = calculate_similarity(item_clean, existing_clean)
            if similarity > threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_items.append(item)
    
    return unique_items


def calculate_similarity(str1: str, str2: str) -> float:
    """Calculate simple similarity ratio between two strings."""
    if not str1 or not str2:
        return 0.0
    
    # Simple character overlap ratio
    set1 = set(str1.lower())
    set2 = set(str2.lower())
    
    if not set1 and not set2:
        return 1.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0


def chunk_list(items: List[T], chunk_size: int) -> List[List[T]]:
    """Split a list into chunks of specified size."""
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float."""
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert value to int."""
    try:
        if value is None or value == "":
            return default
        return int(float(value))  # Handle strings like "123.0"
    except (ValueError, TypeError):
        return default


def format_currency(amount: float, currency: str = "USD") -> str:
    """Format currency amount."""
    if currency == "USD":
        if amount >= 1_000_000_000:
            return f"${amount/1_000_000_000:.1f}B"
        elif amount >= 1_000_000:
            return f"${amount/1_000_000:.1f}M"
        elif amount >= 1_000:
            return f"${amount/1_000:.1f}K"
        else:
            return f"${amount:.0f}"
    return f"{amount} {currency}"


def parse_funding_amount(text: str) -> Optional[float]:
    """Parse funding amount from text and return in millions USD."""
    if not text:
        return None
    
    # Patterns for different formats
    patterns = [
        r'\$([0-9,\.]+)\s*(billion|bn|b)',
        r'\$([0-9,\.]+)\s*(million|mn|m)',
        r'\$([0-9,\.]+)\s*(thousand|k)',
        r'([0-9,\.]+)\s*(billion|bn|b)\s*\$',
        r'([0-9,\.]+)\s*(million|mn|m)\s*\$',
    ]
    
    text_lower = text.lower()
    
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            try:
                amount = float(match.group(1).replace(',', ''))
                unit = match.group(2).lower()
                
                # Convert to millions
                if unit in ['billion', 'bn', 'b']:
                    return amount * 1000
                elif unit in ['million', 'mn', 'm']:
                    return amount
                elif unit in ['thousand', 'k']:
                    return amount / 1000
                
            except ValueError:
                continue
    
    return None


def extract_emails_from_text(text: str) -> List[str]:
    """Extract email addresses from text."""
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    return list(set(emails))  # Remove duplicates


def extract_urls_from_text(text: str) -> List[str]:
    """Extract URLs from text."""
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_pattern, text)
    return list(set(urls))  # Remove duplicates


def normalize_company_name(name: str) -> str:
    """Normalize company name for comparison."""
    if not name:
        return ""
    
    # Clean the name
    normalized = clean_text(name)
    
    # Convert to lowercase
    normalized = normalized.lower()
    
    # Remove common suffixes
    suffixes = [
        'inc', 'inc.', 'incorporated', 'corp', 'corp.', 'corporation',
        'ltd', 'ltd.', 'limited', 'llc', 'l.l.c.', 'co', 'co.', 'company',
        'pte', 'pte.', 'pvt', 'pvt.', 'private', 'limited'
    ]
    
    words = normalized.split()
    filtered_words = []
    
    for word in words:
        if word.rstrip('.') not in suffixes:
            filtered_words.append(word)
    
    return ' '.join(filtered_words).strip()


def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two text strings using word overlap."""
    if not text1 or not text2:
        return 0.0
    
    # Clean and tokenize
    words1 = set(clean_text(text1).lower().split())
    words2 = set(clean_text(text2).lower().split())
    
    if not words1 and not words2:
        return 1.0
    
    # Calculate Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0


def get_domain_from_url(url: str) -> Optional[str]:
    """Extract domain from URL."""
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except:
        return None


def is_recent_date(date_str: str, days_threshold: int = 365) -> bool:
    """Check if a date string represents a recent date."""
    try:
        date_obj = pd.to_datetime(date_str)
        now = datetime.now()
        diff = now - date_obj.to_pydatetime()
        return diff.days <= days_threshold
    except:
        return False


def create_slug(text: str, max_length: int = 50) -> str:
    """Create a URL-friendly slug from text."""
    if not text:
        return ""
    
    # Convert to lowercase and replace spaces/special chars with hyphens
    slug = re.sub(r'[^\w\s-]', '', text.lower())
    slug = re.sub(r'[-\s]+', '-', slug)
    slug = slug.strip('-')
    
    # Truncate if too long
    if len(slug) > max_length:
        slug = slug[:max_length].rstrip('-')
    
    return slug


def format_phone_number(phone: str) -> Optional[str]:
    """Format phone number to standard format."""
    if not phone:
        return None
    
    # Remove all non-digits
    digits = re.sub(r'\D', '', phone)
    
    # Format US numbers
    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    elif len(digits) == 11 and digits[0] == '1':
        return f"+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
    
    return phone  # Return original if can't format


class DataValidator:
    """Data validation utilities."""
    
    @staticmethod
    def validate_company_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean company data."""
        cleaned = {}
        
        # Required fields
        cleaned['name'] = clean_text(data.get('name', ''))
        if not cleaned['name']:
            raise ValueError("Company name is required")
        
        # Optional fields with validation
        cleaned['description'] = clean_text(data.get('description', ''))
        cleaned['website'] = data.get('website', '')
        if cleaned['website'] and not validate_url(cleaned['website']):
            cleaned['website'] = ''
        
        cleaned['founded_year'] = safe_int(data.get('founded_year'))
        if cleaned['founded_year'] < 1800 or cleaned['founded_year'] > datetime.now().year:
            cleaned['founded_year'] = None
        
        cleaned['funding_total_usd'] = safe_float(data.get('funding_total_usd'))
        
        return cleaned
    
    @staticmethod
    def validate_profile_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean profile data."""
        cleaned = {}
        
        cleaned['person_name'] = clean_text(data.get('person_name', ''))
        if not cleaned['person_name']:
            raise ValueError("Person name is required")
        
        cleaned['linkedin_url'] = data.get('linkedin_url', '')
        if cleaned['linkedin_url'] and not validate_linkedin_url(cleaned['linkedin_url']):
            raise ValueError("Invalid LinkedIn URL")
        
        cleaned['role'] = clean_text(data.get('role', ''))
        cleaned['company_name'] = clean_text(data.get('company_name', ''))
        
        return cleaned


def batch_process(items: List[T], batch_size: int = 100) -> List[List[T]]:
    """Process items in batches."""
    return chunk_list(items, batch_size)


def retry_with_backoff(func, max_retries: int = 3, backoff_factor: float = 2.0):
    """Retry function with exponential backoff."""
    def wrapper(*args, **kwargs):
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                
                wait_time = backoff_factor ** attempt
                time.sleep(wait_time)
        
        return None
    
    return wrapper


def merge_dicts_safely(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge dictionaries safely, handling conflicts."""
    result = {}
    
    for d in dicts:
        if not isinstance(d, dict):
            continue
            
        for key, value in d.items():
            if key not in result:
                result[key] = value
            elif value is not None and result[key] is None:
                result[key] = value
            # Keep existing non-null value if new value is null
    
    return result
