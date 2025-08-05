#!/usr/bin/env python3
"""
Data Consolidation Pipeline for XGBoost Founder Success Prediction
================================================================

This module consolidates historical company and founder data (2013-2025) into
master datasets for training the founder success prediction model.

Features:
- Combines temporal datasets with founding_year column
- Filters alive companies only 
- Filters US AI companies
- Joins companies-founders data
- Handles duplicates and missing data
"""

import pandas as pd
import numpy as np
import os
import glob
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataConsolidator:
    """Consolidates and preprocesses historical startup data for ML pipeline"""
    
    def __init__(self, data_dir: str = "/Users/veerdosi/Documents/code/github/initiation-pipeline/output"):
        self.data_dir = Path(data_dir)
        self.companies_master = None
        self.founders_master = None
        
        # AI/ML related keywords for filtering
        self.ai_keywords = [
            'artificial intelligence', 'ai', 'machine learning', 'ml', 'deep learning',
            'neural network', 'computer vision', 'natural language', 'nlp', 'robotics',
            'automation', 'conversational ai', 'generative ai', 'large language model',
            'llm', 'voice model', 'speech synthesis', 'predictive analytics', 'data science',
            'big data', 'analytics platform', 'intelligent', 'smart', 'cognitive',
            'autonomous', 'algorithm', 'model', 'inference', 'training'
        ]
        
    def load_companies_data(self) -> pd.DataFrame:
        """Load and consolidate all companies CSV files"""
        logger.info("Loading companies data...")
        
        companies_files = sorted(glob.glob(str(self.data_dir / "*_companies.csv")))
        logger.info(f"Found {len(companies_files)} companies files")
        
        all_companies = []
        
        for file_path in companies_files:
            try:
                # Extract year from filename
                year = int(re.search(r'(\d{4})_companies\.csv', file_path).group(1))
                
                df = pd.read_csv(file_path)
                df['data_source_year'] = year
                
                # Add founding_year if not present
                if 'founded year' in df.columns and 'founding_year' not in df.columns:
                    df['founding_year'] = df['founded year']
                elif 'founded_year' not in df.columns and 'founding_year' not in df.columns:
                    # If no founding year info, use data source year as proxy
                    df['founding_year'] = year
                
                all_companies.append(df)
                logger.info(f"Loaded {len(df)} companies from {year}")
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue
        
        if not all_companies:
            raise ValueError("No companies data could be loaded")
        
        # Concatenate all data
        companies_df = pd.concat(all_companies, ignore_index=True, sort=False)
        logger.info(f"Total companies loaded: {len(companies_df)}")
        
        # Standardize column names
        companies_df = self._standardize_company_columns(companies_df)
        
        return companies_df
    
    def load_founders_data(self) -> pd.DataFrame:
        """Load and consolidate all founders CSV files"""
        logger.info("Loading founders data...")
        
        founders_files = sorted(glob.glob(str(self.data_dir / "*_founders.csv")))
        logger.info(f"Found {len(founders_files)} founders files")
        
        all_founders = []
        
        for file_path in founders_files:
            try:
                # Extract year from filename
                year_match = re.search(r'(\d{4})_founders\.csv', file_path)
                if not year_match:
                    logger.warning(f"Could not extract year from {file_path}")
                    continue
                    
                year = int(year_match.group(1))
                
                df = pd.read_csv(file_path)
                df['data_source_year'] = year
                
                all_founders.append(df)
                logger.info(f"Loaded {len(df)} founders from {year}")
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue
        
        if not all_founders:
            logger.warning("No founders data could be loaded")
            return pd.DataFrame()
        
        # Concatenate all data
        founders_df = pd.concat(all_founders, ignore_index=True, sort=False)
        logger.info(f"Total founders loaded: {len(founders_df)}")
        
        # Standardize column names
        founders_df = self._standardize_founder_columns(founders_df)
        
        return founders_df
    
    def _standardize_company_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize company column names across different years"""
        column_mapping = {
            'Name': 'company_name',
            'Description': 'description',
            'founded year': 'founding_year',
            'founded_year': 'founding_year',
            'total funding amount (in usd)': 'total_funding_usd',
            'number of funding rounds': 'funding_rounds',
            'last funding date': 'last_funding_date',
            'last funding amount (in usd)': 'last_funding_amount_usd',
            'city': 'city',
            'region': 'region',
            'country': 'country',
            'investor names': 'investors',
            'founder names': 'founders',
            'number of employees': 'employee_count',
            'sector': 'sector',
            'website': 'website',
            'linkedin url': 'linkedin_url',
            'operating status': 'operating_status',
            'market_size_billion': 'market_size_billion',
            'cagr_percent': 'cagr_percent',
            'timing_score': 'timing_score',
            'competitor_count': 'competitor_count',
            'market_stage': 'market_stage',
            'confidence_score_market': 'confidence_score_market',
            'us_sentiment': 'us_sentiment',
            'sea_sentiment': 'sea_sentiment',
            'total_funding_billion': 'total_funding_billion',
            'momentum_score': 'momentum_score'
        }
        
        # Rename columns that exist
        existing_mappings = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=existing_mappings)
        
        return df
    
    def _standardize_founder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize founder column names across different years"""
        column_mapping = {
            'company_name': 'company_name',
            'operating_status': 'company_operating_status',
            'name': 'founder_name',
            'title': 'title',
            'linkedin_url': 'linkedin_url',
            'location': 'location',
            'about': 'about',
            'estimated_age': 'estimated_age',
            'extraction_date': 'extraction_date',
            'l_level': 'l_level',
            'reasoning': 'reasoning',
            'confidence_score': 'confidence_score'
        }
        
        # Handle experience columns
        for i in range(1, 6):  # Up to 5 experiences
            if f'experience_{i}_title' in df.columns:
                column_mapping[f'experience_{i}_title'] = f'experience_{i}_title'
            if f'experience_{i}_company' in df.columns:
                column_mapping[f'experience_{i}_company'] = f'experience_{i}_company'
        
        # Handle education columns
        for i in range(1, 4):  # Up to 3 educations
            if f'education_{i}_school' in df.columns:
                column_mapping[f'education_{i}_school'] = f'education_{i}_school'
            if f'education_{i}_degree' in df.columns:
                column_mapping[f'education_{i}_degree'] = f'education_{i}_degree'
        
        # Handle skill columns
        for i in range(1, 6):  # Up to 5 skills
            if f'skill_{i}' in df.columns:
                column_mapping[f'skill_{i}'] = f'skill_{i}'
        
        # Rename columns that exist
        existing_mappings = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=existing_mappings)
        
        return df
    
    def filter_alive_companies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to keep only alive/active companies"""
        logger.info("Filtering alive companies...")
        
        initial_count = len(df)
        
        # Check different possible column names for operating status
        status_columns = ['operating_status', 'status', 'company_status']
        status_col = None
        
        for col in status_columns:
            if col in df.columns:
                status_col = col
                break
        
        if status_col is None:
            logger.warning("No operating status column found. Keeping all companies.")
            return df
        
        # Define alive/active status values
        alive_statuses = ['active', 'alive', 'operating', 'open', 'Active', 'Alive', 'Operating', 'Open']
        
        # Filter for alive companies
        df_filtered = df[df[status_col].isin(alive_statuses)]
        
        logger.info(f"Filtered from {initial_count} to {len(df_filtered)} alive companies")
        return df_filtered
    
    def filter_us_ai_companies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter for US AI companies"""
        logger.info("Filtering US AI companies...")
        
        initial_count = len(df)
        
        # Filter for US companies
        us_df = df[df['country'].str.upper() == 'UNITED STATES'].copy()
        logger.info(f"US companies: {len(us_df)}")
        
        # Filter for AI companies based on description and sector
        ai_mask = pd.Series([False] * len(us_df), index=us_df.index)
        
        # Check description
        if 'description' in us_df.columns:
            description_mask = us_df['description'].str.lower().str.contains(
                '|'.join(self.ai_keywords), na=False, regex=True
            )
            ai_mask = ai_mask | description_mask
        
        # Check sector
        if 'sector' in us_df.columns:
            sector_mask = us_df['sector'].str.lower().str.contains(
                '|'.join(self.ai_keywords), na=False, regex=True
            )
            ai_mask = ai_mask | sector_mask
        
        us_ai_df = us_df[ai_mask].copy()
        
        logger.info(f"Filtered from {initial_count} to {len(us_ai_df)} US AI companies")
        return us_ai_df
    
    def consolidate_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Main method to consolidate all data"""
        logger.info("Starting data consolidation...")
        
        # Load companies data
        companies_df = self.load_companies_data()
        
        # Filter alive companies
        companies_df = self.filter_alive_companies(companies_df)
        
        # Filter US AI companies
        companies_df = self.filter_us_ai_companies(companies_df)
        
        # Load founders data
        founders_df = self.load_founders_data()
        
        # Store consolidated data
        self.companies_master = companies_df
        self.founders_master = founders_df
        
        logger.info("Data consolidation completed successfully")
        return companies_df, founders_df
    
    def save_consolidated_data(self, output_dir: str = None):
        """Save consolidated master datasets"""
        if output_dir is None:
            # Save to datasets folder instead of output folder
            output_dir = Path(self.data_dir).parent / "datasets"
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)  # Create datasets folder if it doesn't exist
        
        if self.companies_master is not None:
            companies_path = output_path / "companies_master.csv"
            self.companies_master.to_csv(companies_path, index=False)
            logger.info(f"Saved companies master data to {companies_path}")
        
        if self.founders_master is not None:
            founders_path = output_path / "founders_master.csv"
            self.founders_master.to_csv(founders_path, index=False)
            logger.info(f"Saved founders master data to {founders_path}")
    
    def get_data_summary(self) -> Dict:
        """Get summary statistics of consolidated data"""
        summary = {}
        
        if self.companies_master is not None:
            summary['companies'] = {
                'total_count': len(self.companies_master),
                'years_range': (
                    self.companies_master['founding_year'].min(),
                    self.companies_master['founding_year'].max()
                ),
                'countries': self.companies_master['country'].value_counts().to_dict(),
                'funding_stats': {
                    'mean_funding': self.companies_master['total_funding_usd'].mean(),
                    'median_funding': self.companies_master['total_funding_usd'].median(),
                    'max_funding': self.companies_master['total_funding_usd'].max()
                }
            }
        
        if self.founders_master is not None:
            summary['founders'] = {
                'total_count': len(self.founders_master),
                'unique_companies': self.founders_master['company_name'].nunique(),
                'avg_age': self.founders_master['estimated_age'].mean() if 'estimated_age' in self.founders_master.columns else None
            }
        
        return summary


def main():
    """Main execution function"""
    consolidator = DataConsolidator()
    
    try:
        # Consolidate data
        companies_df, founders_df = consolidator.consolidate_data()
        
        # Save consolidated data
        consolidator.save_consolidated_data()
        
        # Print summary
        summary = consolidator.get_data_summary()
        logger.info(f"Data consolidation summary: {summary}")
        
        return companies_df, founders_df
        
    except Exception as e:
        logger.error(f"Error in data consolidation: {e}")
        raise


if __name__ == "__main__":
    companies, founders = main()