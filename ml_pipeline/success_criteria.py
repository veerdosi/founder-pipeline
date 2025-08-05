#!/usr/bin/env python3
"""
Success Criteria Definition for US AI Founder Prediction
========================================================

This module implements the success criteria for US AI market based on
age-adjusted funding thresholds and market benchmarks.

Key Features:
- Age-adjusted funding thresholds for US AI market
- Success binary classification
- Market timing adjustments
- Funding velocity calculations
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class SuccessCriteria:
    """Defines and applies success criteria for US AI companies"""
    
    def __init__(self):
        # US AI Success Benchmarks (more aggressive than SEA)
        # Based on: Unicorn timeline avg 6 years, Series B+ = $50M+ funding
        self.success_thresholds = {
            8: 100_000_000,   # $100M+ funding at 8+ years (unicorn trajectory)
            7: 80_000_000,    # $80M+ funding at 7 years
            6: 60_000_000,    # $60M+ funding at 6 years  
            5: 40_000_000,    # $40M+ funding at 5 years
            4: 25_000_000,    # $25M+ funding at 4 years (Series B)
            3: 15_000_000,    # $15M+ funding at 3 years (Series A+)
            2: 8_000_000,     # $8M+ funding at 2 years (Seed+)
            1: 3_000_000,     # $3M+ funding at 1 year (strong seed)
            0: 1_000_000      # $1M+ funding at founding year
        }
        
        # Market cycle adjustments (funding environment changes)
        self.market_cycle_adjustments = {
            (2013, 2017): 0.8,  # Early market, lower thresholds
            (2018, 2019): 1.0,  # Normal market
            (2020, 2021): 1.3,  # Hot market, higher thresholds
            (2022, 2023): 0.9,  # Cooling market
            (2024, 2025): 1.1   # Recovery market
        }
        
        # AI sector multipliers (some sectors require more capital)
        self.sector_multipliers = {
            'robotics': 2.0,          # Hardware intensive
            'autonomous': 1.8,        # Hardware + regulatory
            'quantum': 2.5,           # Deep tech, long cycles
            'computer vision': 1.2,   # Moderate capital needs
            'nlp': 1.0,              # Software focused
            'machine learning': 1.0,  # General ML
            'data analytics': 0.8,    # Lower capital needs
            'ai software': 0.9,       # SaaS model
            'conversational ai': 1.1, # Moderate needs
            'generative ai': 1.4      # High compute costs
        }
    
    def calculate_company_age(self, founding_year: int, current_year: int = None) -> int:
        """Calculate company age in years"""
        if current_year is None:
            current_year = datetime.now().year
        
        return max(0, current_year - founding_year)
    
    def get_market_cycle_adjustment(self, founding_year: int) -> float:
        """Get market cycle adjustment factor based on founding year"""
        for (start_year, end_year), adjustment in self.market_cycle_adjustments.items():
            if start_year <= founding_year <= end_year:
                return adjustment
        
        # Default to normal market for years not covered
        return 1.0
    
    def get_sector_multiplier(self, sector: str, description: str = '') -> float:
        """Get sector-specific funding multiplier"""
        if pd.isna(sector):
            sector = ''
        if pd.isna(description):
            description = ''
        
        combined_text = f"{sector} {description}".lower()
        
        # Check for sector keywords
        for sector_key, multiplier in self.sector_multipliers.items():
            if sector_key in combined_text:
                return multiplier
        
        # Default multiplier
        return 1.0
    
    def define_success(self, 
                      company_age_years: int, 
                      total_funding_raised_millions: float,
                      founding_year: int = None,
                      sector: str = '',
                      description: str = '') -> bool:
        """
        Define success based on US AI market benchmarks
        
        Args:
            company_age_years: Age of company in years
            total_funding_raised_millions: Total funding in millions USD
            founding_year: Year company was founded (for market adjustment)
            sector: Company sector/industry
            description: Company description
            
        Returns:
            bool: True if company meets success criteria
        """
        if pd.isna(total_funding_raised_millions) or total_funding_raised_millions <= 0:
            return False
        
        # Convert to actual dollar amount
        total_funding_usd = total_funding_raised_millions * 1_000_000 if total_funding_raised_millions < 1000 else total_funding_raised_millions
        
        # Get base threshold for company age
        age_key = min(company_age_years, 8)  # Cap at 8 years
        expected_funding = self.success_thresholds.get(age_key, self.success_thresholds[8])
        
        # Apply market cycle adjustment
        if founding_year:
            market_adjustment = self.get_market_cycle_adjustment(founding_year)
            expected_funding *= market_adjustment
        
        # Apply sector multiplier
        sector_multiplier = self.get_sector_multiplier(sector, description)
        expected_funding *= sector_multiplier
        
        return total_funding_usd >= expected_funding
    
    def calculate_success_score(self, 
                               company_age_years: int, 
                               total_funding_raised_millions: float,
                               founding_year: int = None,
                               sector: str = '',
                               description: str = '') -> float:
        """
        Calculate a continuous success score (0-1)
        
        Returns:
            float: Success score where 1.0 = meets threshold exactly
        """
        if pd.isna(total_funding_raised_millions) or total_funding_raised_millions <= 0:
            return 0.0
        
        # Convert to actual dollar amount
        total_funding_usd = total_funding_raised_millions * 1_000_000 if total_funding_raised_millions < 1000 else total_funding_raised_millions
        
        # Get expected funding
        age_key = min(company_age_years, 8)
        expected_funding = self.success_thresholds.get(age_key, self.success_thresholds[8])
        
        # Apply adjustments
        if founding_year:
            market_adjustment = self.get_market_cycle_adjustment(founding_year)
            expected_funding *= market_adjustment
        
        sector_multiplier = self.get_sector_multiplier(sector, description)
        expected_funding *= sector_multiplier
        
        # Calculate score (ratio of actual to expected funding)
        score = total_funding_usd / expected_funding
        
        # Cap at reasonable maximum (10x threshold = score of 10.0)
        return min(score, 10.0)
    
    def apply_success_criteria(self, df: pd.DataFrame, current_year: int = None) -> pd.DataFrame:
        """
        Apply success criteria to a DataFrame of companies
        
        Args:
            df: DataFrame with company data
            current_year: Current year for age calculation
            
        Returns:
            DataFrame with success labels added
        """
        if current_year is None:
            current_year = datetime.now().year
        
        df = df.copy()
        
        # Calculate company age
        df['company_age_years'] = df['founding_year'].apply(
            lambda x: self.calculate_company_age(x, current_year) if pd.notna(x) else 0
        )
        
        # Apply success criteria
        df['is_successful'] = df.apply(
            lambda row: self.define_success(
                company_age_years=row.get('company_age_years', 0),
                total_funding_raised_millions=row.get('total_funding_usd', 0) / 1_000_000,
                founding_year=row.get('founding_year'),
                sector=row.get('sector', ''),
                description=row.get('description', '')
            ), axis=1
        )
        
        # Calculate success score
        df['success_score'] = df.apply(
            lambda row: self.calculate_success_score(
                company_age_years=row.get('company_age_years', 0),
                total_funding_raised_millions=row.get('total_funding_usd', 0) / 1_000_000,
                founding_year=row.get('founding_year'),
                sector=row.get('sector', ''),
                description=row.get('description', '')
            ), axis=1
        )
        
        return df
    
    def get_success_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate success statistics for the dataset"""
        if 'is_successful' not in df.columns:
            raise ValueError("Success criteria must be applied first")
        
        stats = {
            'total_companies': len(df),
            'successful_companies': df['is_successful'].sum(),
            'success_rate': df['is_successful'].mean(),
            'avg_success_score': df['success_score'].mean(),
            'median_success_score': df['success_score'].median(),
        }
        
        # Success rate by age
        if 'company_age_years' in df.columns:
            stats['success_by_age'] = df.groupby('company_age_years')['is_successful'].agg(['count', 'sum', 'mean']).to_dict()
        
        # Success rate by founding year
        if 'founding_year' in df.columns:
            stats['success_by_year'] = df.groupby('founding_year')['is_successful'].agg(['count', 'sum', 'mean']).to_dict()
        
        return stats
    
    def validate_thresholds(self, df: pd.DataFrame) -> Dict:
        """Validate that success thresholds are reasonable for the dataset"""
        validation = {}
        
        if 'is_successful' not in df.columns:
            df = self.apply_success_criteria(df)
        
        # Check overall success rate (should be 20-40% for reasonable thresholds)
        success_rate = df['is_successful'].mean()
        validation['overall_success_rate'] = success_rate
        validation['success_rate_reasonable'] = 0.15 <= success_rate <= 0.45
        
        # Check success rate by age (should increase with age)
        age_success = df.groupby('company_age_years')['is_successful'].mean().to_dict()
        validation['success_by_age'] = age_success
        
        # Check if success rates generally increase with age
        ages = sorted(age_success.keys())
        if len(ages) > 2:
            increasing = all(
                age_success[ages[i]] <= age_success[ages[i+1]] 
                for i in range(len(ages)-2)
            )
            validation['success_increases_with_age'] = increasing
        
        return validation


def main():
    """Test the success criteria implementation"""
    # Test cases
    criteria = SuccessCriteria()
    
    test_cases = [
        # (age, funding_millions, founding_year, sector, expected_success)
        (1, 5, 2023, 'AI software', True),    # $5M at 1 year - should be successful
        (2, 5, 2023, 'AI software', False),   # $5M at 2 years - should not be successful
        (3, 20, 2020, 'machine learning', True),  # $20M at 3 years in hot market
        (5, 30, 2019, 'robotics', False),     # $30M at 5 years in robotics (needs 80M)
        (6, 80, 2018, 'computer vision', True), # $80M at 6 years - should be successful
    ]
    
    print("Testing Success Criteria:")
    print("-" * 50)
    
    for i, (age, funding, year, sector, expected) in enumerate(test_cases):
        result = criteria.define_success(age, funding, year, sector)
        score = criteria.calculate_success_score(age, funding, year, sector)
        
        print(f"Test {i+1}: Age={age}, Funding=${funding}M, Year={year}, Sector={sector}")
        print(f"  Expected: {expected}, Got: {result}, Score: {score:.2f}")
        print(f"  {'✓' if result == expected else '✗'}")
        print()


if __name__ == "__main__":
    main()