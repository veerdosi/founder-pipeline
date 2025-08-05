#!/usr/bin/env python3
"""
Test Success Criteria on Consolidated Data
==========================================

This script tests the success criteria implementation on the consolidated
master dataset and provides analysis of success rates.
"""

import pandas as pd
import numpy as np
from success_criteria import SuccessCriteria
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Test success criteria on consolidated data"""
    
    # Load consolidated companies data
    logger.info("Loading companies master data...")
    companies_df = pd.read_csv("/Users/veerdosi/Documents/code/github/initiation-pipeline/datasets/companies_master.csv")
    logger.info(f"Loaded {len(companies_df)} companies")
    
    # Initialize success criteria
    criteria = SuccessCriteria()
    
    # Apply success criteria
    logger.info("Applying success criteria...")
    companies_with_success = criteria.apply_success_criteria(companies_df)
    
    # Get statistics
    logger.info("Calculating success statistics...")
    stats = criteria.get_success_statistics(companies_with_success)
    
    print("\n" + "="*60)
    print("SUCCESS CRITERIA ANALYSIS")
    print("="*60)
    
    print(f"\nOverall Statistics:")
    print(f"  Total Companies: {stats['total_companies']:,}")
    print(f"  Successful Companies: {stats['successful_companies']:,}")
    print(f"  Success Rate: {stats['success_rate']:.1%}")
    print(f"  Average Success Score: {stats['avg_success_score']:.2f}")
    print(f"  Median Success Score: {stats['median_success_score']:.2f}")
    
    # Success by age
    if 'success_by_age' in stats:
        print(f"\nSuccess Rate by Company Age:")
        print("-" * 40)
        age_data = stats['success_by_age']['mean']
        for age in sorted(age_data.keys()):
            count = stats['success_by_age']['count'][age]
            success_count = stats['success_by_age']['sum'][age]
            success_rate = age_data[age]
            print(f"  {age:2d} years: {success_rate:6.1%} ({success_count:3d}/{count:3d} companies)")
    
    # Success by founding year
    if 'success_by_year' in stats:
        print(f"\nSuccess Rate by Founding Year (top 10):")
        print("-" * 45)
        year_data = stats['success_by_year']['mean']
        sorted_years = sorted(year_data.keys(), reverse=True)[:10]
        for year in sorted_years:
            count = stats['success_by_year']['count'][year]
            success_count = stats['success_by_year']['sum'][year]
            success_rate = year_data[year]
            print(f"  {year}: {success_rate:6.1%} ({success_count:3d}/{count:3d} companies)")
    
    # Validate thresholds
    logger.info("Validating success thresholds...")
    validation = criteria.validate_thresholds(companies_with_success)
    
    print(f"\nThreshold Validation:")
    print("-" * 25)
    print(f"  Overall Success Rate: {validation['overall_success_rate']:.1%}")
    print(f"  Is Reasonable (15-45%): {'✓' if validation['success_rate_reasonable'] else '✗'}")
    
    if 'success_increases_with_age' in validation:
        print(f"  Success Increases with Age: {'✓' if validation['success_increases_with_age'] else '✗'}")
    
    # Top successful companies
    print(f"\nTop 10 Most Successful Companies (by success score):")
    print("-" * 55)
    top_companies = companies_with_success.nlargest(10, 'success_score')[
        ['company_name', 'founding_year', 'company_age_years', 'total_funding_usd', 'success_score', 'sector']
    ]
    
    for idx, row in top_companies.iterrows():
        funding_millions = row['total_funding_usd'] / 1_000_000
        print(f"  {row['company_name'][:30]:<30} | {row['founding_year']} | Age: {row['company_age_years']:2d} | ${funding_millions:6.1f}M | Score: {row['success_score']:5.2f}")
    
    # Save results
    output_path = "/Users/veerdosi/Documents/code/github/initiation-pipeline/datasets/companies_with_success.csv"
    companies_with_success.to_csv(output_path, index=False)
    logger.info(f"Saved companies with success labels to {output_path}")
    
    return companies_with_success, stats

if __name__ == "__main__":
    companies, stats = main()