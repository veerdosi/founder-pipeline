"""Simple CSV-based company tracking system for duplicate prevention."""

import csv
import logging
from datetime import datetime
from typing import List, Optional, Tuple
from pathlib import Path

from ..config import settings
from ...models import Company

logger = logging.getLogger(__name__)


class CompanyTracker:
    """CSV-based company tracking system for duplicate prevention."""
    
    def __init__(self, csv_path: Optional[Path] = None):
        """Initialize the company tracker with CSV path."""
        if csv_path is None:
            csv_path = settings.default_output_dir / "company_registry.csv"
        
        self.csv_path = csv_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize CSV file if it doesn't exist
        self._init_csv_file()
        logger.info(f"CompanyTracker initialized with CSV: {self.csv_path}")
    
    def _init_csv_file(self):
        """Initialize the CSV file with headers if it doesn't exist."""
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['company_name', 'website'])
    
    def _normalize_company_name(self, name) -> str:
        """Normalize company name for comparison."""
        import re
        if not name:
            return ""
        
        # Convert to string in case it's not a string
        name = str(name).lower().strip()
        # Remove common suffixes and prefixes
        name = re.sub(r'\b(inc|llc|corp|corporation|ltd|limited|ai|technologies|tech|systems|solutions|labs|lab)\b', '', name)
        name = re.sub(r'[^\w\s]', '', name)  # Remove special characters
        name = re.sub(r'\s+', ' ', name).strip()  # Normalize whitespace
        return name
    
    def _extract_domain(self, website) -> str:
        """Extract domain from website URL."""
        if not website:
            return ""
        
        try:
            from urllib.parse import urlparse
            # Convert to string if it's an HttpUrl object
            website_str = str(website)
            domain = urlparse(website_str).netloc.lower()
            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except Exception:
            # Fallback: convert to string and extract domain manually
            website_str = str(website)
            return website_str.lower().replace('https://', '').replace('http://', '').replace('www.', '').split('/')[0]
    
    def _load_tracked_companies(self) -> List[Tuple[str, str]]:
        """Load tracked companies from CSV file."""
        companies = []
        try:
            with open(self.csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    companies.append((row['company_name'], row['website']))
        except FileNotFoundError:
            self._init_csv_file()
        except Exception as e:
            logger.error(f"Error loading tracked companies: {e}")
        return companies
    
    def is_duplicate_company(self, company: Company) -> Tuple[bool, Optional[str]]:
        """
        Check if company is a duplicate based on name or website.
        
        Returns:
            Tuple of (is_duplicate, reason)
        """
        # Safe attribute access with fallbacks
        company_name = getattr(company, 'name', '')
        normalized_name = self._normalize_company_name(company_name)
        if not normalized_name.strip():
            return True, "Company name is empty or invalid"
            
        website_domain = self._extract_domain(getattr(company, 'website', None)) if getattr(company, 'website', None) else ""
        
        # Load existing companies
        tracked_companies = self._load_tracked_companies()
        
        for tracked_name, tracked_website in tracked_companies:
            tracked_normalized = self._normalize_company_name(tracked_name)
            tracked_domain = self._extract_domain(tracked_website) if tracked_website else ""
            
            # Check for exact name match
            if normalized_name == tracked_normalized:
                return True, f"Duplicate name: '{company_name}' matches existing '{tracked_name}'"
            
            # Check for website domain match (if both have websites)
            if website_domain and tracked_domain and website_domain == tracked_domain:
                return True, f"Duplicate website: '{getattr(company, 'website', '')}' matches existing company '{tracked_name}'"
        
        return False, None
    
    def add_company(self, company: Company, run_id: str = "", source_url: str = "") -> bool:
        """
        Add a company to the tracking CSV file.
        
        Returns:
            True if company was added, False if it was a duplicate
        """
        # Check for duplicates first
        is_duplicate, reason = self.is_duplicate_company(company)
        if is_duplicate:
            logger.info(f"Skipping duplicate company '{getattr(company, 'name', 'unknown')}': {reason}")
            return False
        
        # Add company to CSV
        company_name = getattr(company, 'name', '')
        website = str(getattr(company, 'website', '')) if getattr(company, 'website', None) else ''
        
        try:
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([company_name, website])
                
            logger.info(f"Added new company to tracking: '{company_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error adding company '{company_name}' to tracking CSV: {e}")
            return False
    
    def get_tracked_companies(self) -> List[Tuple[str, str]]:
        """Get all tracked companies from CSV."""
        return self._load_tracked_companies()
    
    def get_company_count(self) -> int:
        """Get the total number of tracked companies."""
        return len(self._load_tracked_companies())
    
    def bulk_import_companies(self, companies: List[Company], run_id: str = "") -> Tuple[int, int]:
        """
        Bulk import companies from a list.
        
        Returns:
            Tuple of (new_companies_added, duplicates_skipped)
        """
        new_count = 0
        duplicate_count = 0
        
        for company in companies:
            if self.add_company(company, run_id):
                new_count += 1
            else:
                duplicate_count += 1
        
        logger.info(f"Bulk import completed: {new_count} new, {duplicate_count} duplicates")
        return new_count, duplicate_count


# Global instance
company_tracker = CompanyTracker()