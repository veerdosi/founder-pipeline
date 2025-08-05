#!/usr/bin/env python3
"""
Feature Engineering Framework for Founder Success Prediction
============================================================

This module implements comprehensive feature engineering for the XGBoost
founder success prediction model, including founder-level and company-level features.

Features:
- Founder demographics and experience
- Education and credentials  
- Professional recognition
- Entrepreneurial track record
- Company market position
- Traction indicators
- Temporal features
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Comprehensive feature engineering for founder success prediction"""
    
    def __init__(self):
        # Top-tier universities for AI (US focused)
        self.top_ai_universities = {
            'stanford', 'mit', 'carnegie mellon', 'berkeley', 'harvard', 'caltech',
            'georgia tech', 'university of washington', 'cornell', 'princeton',
            'columbia', 'yale', 'michigan', 'texas', 'illinois', 'wisconsin',
            'purdue', 'johns hopkins', 'nyu', 'penn', 'chicago', 'duke'
        }
        
        # Big tech companies (FAANG + AI leaders)
        self.big_tech_companies = {
            'google', 'apple', 'facebook', 'meta', 'amazon', 'microsoft', 'netflix',
            'nvidia', 'openai', 'anthropic', 'deepmind', 'tesla', 'spacex',
            'uber', 'airbnb', 'stripe', 'palantir', 'databricks', 'snowflake',
            'atlassian', 'salesforce', 'oracle', 'adobe', 'intuit', 'zoom'
        }
        
        # AI-related degree keywords
        self.ai_degree_keywords = {
            'computer science', 'artificial intelligence', 'machine learning',
            'data science', 'robotics', 'electrical engineering', 'statistics',
            'mathematics', 'physics', 'cognitive science', 'neuroscience'
        }
        
        # AI verticals
        self.ai_verticals = {
            'computer vision': ['vision', 'image', 'visual', 'cv', 'opencv'],
            'nlp': ['nlp', 'natural language', 'text', 'linguistic', 'language model'],
            'robotics': ['robot', 'robotic', 'autonomous', 'hardware'],
            'generative ai': ['generative', 'gpt', 'llm', 'chatbot', 'generation'],
            'data analytics': ['analytics', 'data science', 'big data', 'insight'],
            'automation': ['automation', 'automate', 'workflow', 'process'],
            'ml platform': ['platform', 'infrastructure', 'mlops', 'deployment']
        }
        
        self.label_encoders = {}
        self.scalers = {}
    
    def extract_founder_features(self, founders_df: pd.DataFrame) -> pd.DataFrame:
        """Extract comprehensive founder-level features"""
        logger.info("Extracting founder features...")
        
        df = founders_df.copy()
        
        # Demographics & Experience
        df = self._extract_demographic_features(df)
        df = self._extract_experience_features(df)
        
        # Education & Credentials
        df = self._extract_education_features(df)
        
        # Professional Recognition
        df = self._extract_recognition_features(df)
        
        # Entrepreneurial Track Record
        df = self._extract_entrepreneurial_features(df)
        
        logger.info(f"Extracted {len([c for c in df.columns if c not in founders_df.columns])} founder features")
        return df
    
    def _extract_demographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract demographic and age-related features"""
        
        # Age features
        if 'estimated_age' in df.columns:
            df['founder_age'] = pd.to_numeric(df['estimated_age'], errors='coerce')
            df['age_squared'] = df['founder_age'] ** 2
            df['optimal_age_range'] = ((df['founder_age'] >= 30) & (df['founder_age'] <= 50)).astype(int)
            df['age_missing'] = df['founder_age'].isna().astype(int)
            df['founder_age'] = df['founder_age'].fillna(df['founder_age'].median())
        else:
            df['founder_age'] = 35  # Default age
            df['age_squared'] = 35 ** 2
            df['optimal_age_range'] = 1
            df['age_missing'] = 1
        
        return df
    
    def _extract_experience_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract professional experience features"""
        
        # Count experiences
        experience_cols = [c for c in df.columns if c.startswith('experience_') and c.endswith('_company')]
        df['total_experiences'] = df[experience_cols].notna().sum(axis=1)
        
        # Years of experience (estimate based on number of roles)
        df['years_experience'] = df['total_experiences'] * 2.5  # Assume 2.5 years per role on average
        
        # Big tech experience
        df['big_tech_experience'] = 0
        df['ai_experience'] = 0
        
        for col in experience_cols:
            if col in df.columns:
                # Big tech experience
                big_tech_mask = df[col].str.lower().str.contains(
                    '|'.join(self.big_tech_companies), na=False, regex=True
                )
                df['big_tech_experience'] += big_tech_mask.astype(int)
                
                # AI experience  
                ai_keywords = ['ai', 'artificial intelligence', 'machine learning', 'ml', 'data science', 'analytics']
                ai_mask = df[col].str.lower().str.contains(
                    '|'.join(ai_keywords), na=False, regex=True
                )
                df['ai_experience'] += ai_mask.astype(int)
        
        # Cap at 1 (binary features)
        df['big_tech_experience'] = (df['big_tech_experience'] > 0).astype(int)
        df['ai_experience'] = (df['ai_experience'] > 0).astype(int)
        
        # Previous companies count
        df['previous_companies_count'] = df['total_experiences']
        
        return df
    
    def _extract_education_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract education and credential features"""
        
        # Education degree features
        education_degree_cols = [c for c in df.columns if c.startswith('education_') and c.endswith('_degree')]
        education_school_cols = [c for c in df.columns if c.startswith('education_') and c.endswith('_school')]
        
        df['has_phd'] = 0
        df['has_masters'] = 0
        df['cs_degree'] = 0
        df['ai_related_degree'] = 0
        df['top_tier_university'] = 0
        
        # Check degrees
        for col in education_degree_cols:
            if col in df.columns:
                # PhD
                phd_mask = df[col].str.lower().str.contains(
                    'phd|ph.d|doctorate|doctor of philosophy', na=False, regex=True
                )
                df['has_phd'] += phd_mask.astype(int)
                
                # Masters
                masters_mask = df[col].str.lower().str.contains(
                    'master|ms|ma|msc|mba', na=False, regex=True
                )
                df['has_masters'] += masters_mask.astype(int)
                
                # CS degree
                cs_mask = df[col].str.lower().str.contains(
                    'computer science|cs|computer engineering', na=False, regex=True
                )
                df['cs_degree'] += cs_mask.astype(int)
                
                # AI-related degree
                ai_degree_mask = df[col].str.lower().str.contains(
                    '|'.join(self.ai_degree_keywords), na=False, regex=True
                )
                df['ai_related_degree'] += ai_degree_mask.astype(int)
        
        # Check schools
        for col in education_school_cols:
            if col in df.columns:
                # Top-tier university
                top_uni_mask = df[col].str.lower().str.contains(
                    '|'.join(self.top_ai_universities), na=False, regex=True
                )
                df['top_tier_university'] += top_uni_mask.astype(int)
        
        # Cap binary features at 1
        for feat in ['has_phd', 'has_masters', 'cs_degree', 'ai_related_degree', 'top_tier_university']:
            df[feat] = (df[feat] > 0).astype(int)
        
        return df
    
    def _extract_recognition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract professional recognition features"""
        
        # Initialize features
        df['media_mentions_count'] = 0
        df['awards_and_recognitions'] = 0
        df['speaking_engagements'] = 0
        df['social_media_followers'] = 0
        df['thought_leadership_score'] = 0
        
        # Extract from existing columns if available
        if 'media_mentions_count' in df.columns:
            df['media_mentions_count'] = pd.to_numeric(df['media_mentions_count'], errors='coerce').fillna(0)
        
        if 'awards_and_recognitions' in df.columns:
            df['awards_and_recognitions'] = df['awards_and_recognitions'].notna().astype(int)
        
        if 'speaking_engagements' in df.columns:
            df['speaking_engagements'] = df['speaking_engagements'].notna().astype(int)
        
        if 'social_media_followers' in df.columns:
            df['social_media_followers'] = pd.to_numeric(df['social_media_followers'], errors='coerce').fillna(0)
        
        if 'thought_leadership_score' in df.columns:
            df['thought_leadership_score'] = pd.to_numeric(df['thought_leadership_score'], errors='coerce').fillna(0)
        
        # Overall sentiment (if available)
        if 'overall_sentiment' in df.columns:
            sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
            df['sentiment_score'] = df['overall_sentiment'].map(sentiment_map).fillna(0)
        else:
            df['sentiment_score'] = 0
        
        return df
    
    def _extract_entrepreneurial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract entrepreneurial track record features"""
        
        # Initialize features
        df['companies_founded'] = 0
        df['previous_exits'] = 0
        df['board_positions'] = 0
        df['serial_entrepreneur'] = 0
        df['technical_cofounder'] = 0
        
        # Extract from existing columns if available
        if 'companies_founded' in df.columns:
            df['companies_founded'] = pd.to_numeric(df['companies_founded'], errors='coerce').fillna(0)
        
        if 'investment_activities' in df.columns:
            df['has_investment_activities'] = df['investment_activities'].notna().astype(int)
        else:
            df['has_investment_activities'] = 0
        
        if 'board_positions' in df.columns:
            df['board_positions'] = df['board_positions'].notna().astype(int)
        
        # Technical cofounder (based on title)
        if 'title' in df.columns:
            technical_titles = ['cto', 'chief technology', 'technical', 'engineer', 'developer', 'architect']
            tech_mask = df['title'].str.lower().str.contains(
                '|'.join(technical_titles), na=False, regex=True
            )
            df['technical_cofounder'] = tech_mask.astype(int)
        
        # Serial entrepreneur (founded multiple companies)
        df['serial_entrepreneur'] = (df['companies_founded'] > 1).astype(int)
        
        return df
    
    def extract_company_features(self, companies_df: pd.DataFrame) -> pd.DataFrame:
        """Extract comprehensive company-level features"""
        logger.info("Extracting company features...")
        
        df = companies_df.copy()
        
        # Market position features
        df = self._extract_market_features(df)
        
        # Traction indicators
        df = self._extract_traction_features(df)
        
        # Temporal features
        df = self._extract_temporal_features(df)
        
        # AI vertical classification
        df = self._extract_ai_vertical_features(df)
        
        logger.info(f"Extracted {len([c for c in df.columns if c not in companies_df.columns])} company features")
        return df
    
    def _extract_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract market position features"""
        
        # Company age (should already exist from success criteria)
        if 'company_age_years' not in df.columns:
            current_year = datetime.now().year
            df['company_age_years'] = current_year - df['founding_year']
        
        # Market timing score
        if 'timing_score' not in df.columns:
            df['timing_score'] = 3.0  # Default neutral score
        
        # Market size and growth
        if 'market_size_billion' not in df.columns:
            df['market_size_billion'] = 10.0  # Default market size
        
        if 'cagr_percent' not in df.columns:
            df['cagr_percent'] = 15.0  # Default growth rate
        
        # B2B vs B2C classification (based on description)
        if 'description' in df.columns:
            b2b_keywords = ['enterprise', 'business', 'b2b', 'saas', 'platform', 'api', 'infrastructure']
            b2b_mask = df['description'].str.lower().str.contains(
                '|'.join(b2b_keywords), na=False, regex=True
            )
            df['is_b2b'] = b2b_mask.astype(int)
            df['is_b2c'] = (~b2b_mask).astype(int)
        else:
            df['is_b2b'] = 1  # Default to B2B for AI companies
            df['is_b2c'] = 0
        
        return df
    
    def _extract_traction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract traction indicator features"""
        
        # Funding features
        if 'total_funding_usd' not in df.columns:
            df['total_funding_usd'] = 0
        
        df['total_funding_millions'] = df['total_funding_usd'] / 1_000_000
        df['log_funding'] = np.log1p(df['total_funding_usd'])
        
        # Funding rounds
        if 'funding_rounds' not in df.columns:
            df['funding_rounds'] = 1
        
        # Average funding per round
        df['avg_funding_per_round'] = df['total_funding_usd'] / np.maximum(df['funding_rounds'], 1)
        df['log_avg_funding_per_round'] = np.log1p(df['avg_funding_per_round'])
        
        # Employee count
        if 'employee_count' not in df.columns:
            df['employee_count'] = 10  # Default small team
        
        df['log_employee_count'] = np.log1p(df['employee_count'])
        
        # Funding per employee (efficiency metric)
        df['funding_per_employee'] = df['total_funding_usd'] / np.maximum(df['employee_count'], 1)
        
        # Investor quality (count of investors)
        if 'investors' in df.columns:
            df['investor_count'] = df['investors'].str.count('\\|').fillna(0) + 1
            df['investor_count'] = df['investor_count'].replace([np.inf, -np.inf], 0)
        else:
            df['investor_count'] = 1
        
        return df
    
    def _extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal and market cycle features"""
        
        # Market cycle classification
        def get_market_cycle(year):
            if year <= 2017:
                return 'early_market'
            elif 2018 <= year <= 2019:
                return 'normal_market'
            elif 2020 <= year <= 2021:
                return 'hot_market'
            elif 2022 <= year <= 2023:
                return 'cooling_market'
            else:
                return 'recovery_market'
        
        df['market_cycle'] = df['founding_year'].apply(get_market_cycle)
        
        # One-hot encode market cycle
        market_cycles = ['early_market', 'normal_market', 'hot_market', 'cooling_market', 'recovery_market']
        for cycle in market_cycles:
            df[f'founded_in_{cycle}'] = (df['market_cycle'] == cycle).astype(int)
        
        # Funding environment score
        funding_env_scores = {
            'early_market': 0.8, 'normal_market': 1.0, 'hot_market': 1.3,
            'cooling_market': 0.9, 'recovery_market': 1.1
        }
        df['funding_environment_score'] = df['market_cycle'].map(funding_env_scores)
        
        return df
    
    def _extract_ai_vertical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract AI vertical classification features"""
        
        # Initialize vertical features
        for vertical in self.ai_verticals.keys():
            df[f'vertical_{vertical.replace(" ", "_")}'] = 0
        
        # Classify based on description and sector
        if 'description' in df.columns:
            combined_text = (df['description'].fillna('') + ' ' + df.get('sector', '')).str.lower()
            
            for vertical, keywords in self.ai_verticals.items():
                vertical_col = f'vertical_{vertical.replace(" ", "_")}'
                mask = combined_text.str.contains('|'.join(keywords), na=False, regex=True)
                df[vertical_col] = mask.astype(int)
        
        return df
    
    def join_company_founder_features(self, companies_df: pd.DataFrame, founders_df: pd.DataFrame) -> pd.DataFrame:
        """Join company and founder features into a unified dataset"""
        logger.info("Joining company and founder features...")
        
        # Aggregate founder features by company
        founder_agg = self._aggregate_founder_features(founders_df)
        
        # Join with companies
        merged_df = companies_df.merge(founder_agg, on='company_name', how='left')
        
        # Fill missing founder features with defaults
        founder_feature_cols = [c for c in founder_agg.columns if c != 'company_name']
        for col in founder_feature_cols:
            if col in merged_df.columns:
                merged_df[col] = merged_df[col].fillna(0)
        
        logger.info(f"Final dataset shape: {merged_df.shape}")
        return merged_df
    
    def _aggregate_founder_features(self, founders_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate founder features by company"""
        
        # Numerical features to aggregate
        numerical_features = [
            'founder_age', 'age_squared', 'total_experiences', 'years_experience',
            'media_mentions_count', 'social_media_followers', 'thought_leadership_score',
            'companies_founded', 'sentiment_score'
        ]
        
        # Binary features to aggregate (max = any founder has this)
        binary_features = [
            'optimal_age_range', 'big_tech_experience', 'ai_experience', 'has_phd',
            'has_masters', 'cs_degree', 'ai_related_degree', 'top_tier_university',
            'awards_and_recognitions', 'speaking_engagements', 'serial_entrepreneur',
            'technical_cofounder', 'has_investment_activities', 'board_positions'
        ]
        
        agg_dict = {}
        
        # Add founder count (using any column that exists)
        if 'founder_name' in founders_df.columns:
            agg_dict['founder_name'] = 'count'
        elif 'name' in founders_df.columns:
            agg_dict['name'] = 'count'
        else:
            # Use the first column as a proxy for count
            agg_dict[founders_df.columns[0]] = 'count'
        
        # Numerical aggregations - only add if column exists
        for feat in numerical_features:
            if feat in founders_df.columns:
                agg_dict[feat] = ['mean', 'max']
        
        # Binary aggregations - only add if column exists
        for feat in binary_features:
            if feat in founders_df.columns:
                agg_dict[feat] = 'max'
        
        # Perform aggregation
        founder_agg = founders_df.groupby('company_name').agg(agg_dict).reset_index()
        
        # Flatten column names
        new_columns = []
        for col in founder_agg.columns:
            if isinstance(col, tuple):
                if col[1] == 'count':
                    new_columns.append('founder_count')
                elif len(col[1]) > 0:
                    new_columns.append(f'{col[0]}_{col[1]}')
                else:
                    new_columns.append(col[0])
            else:
                new_columns.append(col)
        
        founder_agg.columns = new_columns
        
        return founder_agg
    
    def prepare_features_for_ml(self, df: pd.DataFrame, target_col: str = 'is_successful') -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for machine learning"""
        logger.info("Preparing features for ML...")
        
        # Select feature columns (exclude target and metadata)
        exclude_cols = [
            'company_name', 'description', 'website', 'linkedin_url', 'investors',
            'founders', 'sector', 'city', 'region', 'country', 'last_funding_date',
            'market_cycle', target_col, 'success_score', 'data_source_year',
            'extraction_date', 'reasoning', 'l_level', 'about', 'location',
            'founder_name', 'title', 'linkedin_url', 'company_operating_status'
        ]
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Handle missing values
        X = df[feature_cols].copy()
        for col in X.columns:
            if X[col].dtype in ['object', 'category']:
                X[col] = X[col].fillna('unknown')
            else:
                X[col] = X[col].fillna(X[col].median())
        
        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
            else:
                # Handle new categories
                known_categories = set(self.label_encoders[col].classes_)
                new_categories = set(X[col].unique()) - known_categories
                if new_categories:
                    # Add new categories
                    self.label_encoders[col].classes_ = np.append(
                        self.label_encoders[col].classes_, 
                        list(new_categories)
                    )
                X[col] = X[col].map(lambda x: self.label_encoders[col].transform([str(x)])[0] 
                                   if str(x) in self.label_encoders[col].classes_ else 0)
        
        # Get target
        y = df[target_col] if target_col in df.columns else None
        
        logger.info(f"Final feature matrix shape: {X.shape}")
        logger.info(f"Feature columns: {len(feature_cols)}")
        
        return X, y


def main():
    """Test feature engineering pipeline"""
    # Load data
    companies_df = pd.read_csv("/Users/veerdosi/Documents/code/github/initiation-pipeline/datasets/companies_with_success.csv")
    founders_df = pd.read_csv("/Users/veerdosi/Documents/code/github/initiation-pipeline/datasets/founders_master.csv")
    
    # Initialize feature engineer
    fe = FeatureEngineer()
    
    # Extract features
    companies_with_features = fe.extract_company_features(companies_df)
    founders_with_features = fe.extract_founder_features(founders_df)
    
    # Join features
    final_dataset = fe.join_company_founder_features(companies_with_features, founders_with_features)
    
    # Prepare for ML
    X, y = fe.prepare_features_for_ml(final_dataset)
    
    print(f"Final dataset shape: {final_dataset.shape}")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution: {y.value_counts()}")
    
    # Save processed dataset
    output_path = "/Users/veerdosi/Documents/code/github/initiation-pipeline/datasets/ml_ready_dataset.csv"
    final_dataset.to_csv(output_path, index=False)
    logger.info(f"Saved ML-ready dataset to {output_path}")
    
    return final_dataset, X, y


if __name__ == "__main__":
    dataset, X, y = main()