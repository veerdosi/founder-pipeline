#!/usr/bin/env python3
"""
Founder Success Prediction Script
=================================

Use this script to predict the success probability of new founders and companies
using the trained XGBoost model.

Usage:
    python ml_pipeline/predict_founder.py
"""

import pandas as pd
import numpy as np
from xgboost_model import FounderSuccessPredictor
from feature_engineering import FeatureEngineer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def predict_new_founder(company_data: dict, founder_data: dict):
    """
    Predict success probability for a new founder/company pair
    
    Args:
        company_data: Dictionary with company information
        founder_data: Dictionary with founder information
    
    Returns:
        Dictionary with prediction results
    """
    
    try:
        # Load trained model
        logger.info("Loading trained model...")
        predictor = FounderSuccessPredictor()
        predictor.load_model('datasets/founder_success_model.pkl')
        
        # Initialize feature engineer
        fe = FeatureEngineer()
        
        # Create DataFrames from input data
        company_df = pd.DataFrame([company_data])
        founder_df = pd.DataFrame([founder_data])
        
        # Extract features
        logger.info("Extracting features...")
        company_features = fe.extract_company_features(company_df)
        founder_features = fe.extract_founder_features(founder_df)
        
        # Join features
        combined_df = fe.join_company_founder_features(company_features, founder_features)
        
        # Prepare features for ML
        X, _ = fe.prepare_features_for_ml(combined_df, target_col=None)
        
        # Make prediction
        logger.info("Making prediction...")
        predictions, probabilities = predictor.predict(X, use_ensemble=True)
        probability = probabilities[0]
        
        # Generate recommendation
        if probability >= 0.65:
            recommendation = "STRONG BUY"
            confidence = "HIGH"
        elif probability >= 0.5:
            recommendation = "BUY"
            confidence = "MEDIUM"
        elif probability >= 0.35:
            recommendation = "CONSIDER"
            confidence = "LOW"
        else:
            recommendation = "PASS"
            confidence = "HIGH"
        
        # Get feature importance for this prediction
        feature_names = predictor.feature_columns
        feature_values = X.iloc[0].values
        
        # Get top contributing features (simplified)
        importance_scores = predictor.feature_importance['importance_xgb'].values
        top_features_idx = np.argsort(importance_scores)[-10:]  # Top 10 features
        
        key_strengths = []
        risk_factors = []
        
        for idx in top_features_idx:
            feature_name = feature_names[idx]
            feature_value = feature_values[idx]
            importance = importance_scores[idx]
            
            if feature_value > 0.5:  # High feature value
                key_strengths.append(f"{feature_name}: {feature_value:.2f}")
            elif feature_value < 0.3:  # Low feature value for important feature
                risk_factors.append(f"Low {feature_name}: {feature_value:.2f}")
        
        return {
            'founder_name': founder_data.get('name', 'Unknown'),
            'company_name': company_data.get('company_name', 'Unknown'),
            'success_probability': probability,
            'recommendation': recommendation,
            'confidence': confidence,
            'key_strengths': key_strengths[:5],  # Top 5
            'risk_factors': risk_factors[:3],    # Top 3
            'probability_threshold_65%': probability >= 0.65,
            'detailed_score': {
                'probability': f"{probability:.1%}",
                'percentile': f"{probability * 100:.0f}th percentile"
            }
        }
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return {
            'error': str(e),
            'founder_name': founder_data.get('name', 'Unknown'),
            'company_name': company_data.get('company_name', 'Unknown')
        }

def predict_multiple_founders(founder_company_pairs):
    """Process multiple founder/company pairs"""
    results = []
    
    for i, (company_data, founder_data) in enumerate(founder_company_pairs):
        logger.info(f"Processing founder {i+1}/{len(founder_company_pairs)}: {founder_data.get('name', 'Unknown')}")
        result = predict_new_founder(company_data, founder_data)
        results.append(result)
    
    # Sort by probability (highest first)
    valid_results = [r for r in results if 'error' not in r]
    error_results = [r for r in results if 'error' in r]
    
    valid_results.sort(key=lambda x: x['success_probability'], reverse=True)
    
    return valid_results + error_results

def display_prediction_result(result):
    """Display a formatted prediction result"""
    if 'error' in result:
        print(f"\n❌ ERROR: {result['error']}")
        print(f"Founder: {result['founder_name']}")
        print(f"Company: {result['company_name']}")
        return
    
    print("\n" + "="*60)
    print("FOUNDER SUCCESS PREDICTION REPORT")
    print("="*60)
    print(f"Founder: {result['founder_name']}")
    print(f"Company: {result['company_name']}")
    print(f"Success Probability: {result['success_probability']:.1%}")
    print(f"Recommendation: {result['recommendation']} ({result['confidence']} confidence)")
    print(f"Meets 65% Threshold: {'✅ YES' if result['probability_threshold_65%'] else '❌ NO'}")
    
    if result['key_strengths']:
        print(f"\n✓ Key Strengths:")
        for strength in result['key_strengths']:
            print(f"  • {strength}")
    
    if result['risk_factors']:
        print(f"\n⚠️ Risk Factors:")
        for risk in result['risk_factors']:
            print(f"  • {risk}")
    
    print(f"\n📊 Detailed Score: {result['detailed_score']['probability']} ({result['detailed_score']['percentile']})")

def main():
    """Main execution with example founder prediction"""
    
    print("🚀 Founder Success Prediction System")
    print("=" * 50)
    
    # Example 1: High-potential founder (ex-FAANG with PhD)
    company_data_1 = {
        'company_name': 'NextGen AI Corp',
        'founding_year': 2024,
        'description': 'AI-powered autonomous vehicle software using computer vision and machine learning for self-driving cars',
        'total_funding_usd': 15_000_000,  # $15M Series A
        'funding_rounds': 2,
        'employee_count': 25,
        'sector': 'Artificial Intelligence and Computer Vision',
        'city': 'San Francisco',
        'region': 'California',
        'country': 'United States',
        'investor_count': 3
    }
    
    founder_data_1 = {
        'company_name': 'NextGen AI Corp',
        'name': 'Alex Chen',
        'title': 'CEO and Co-founder',
        'estimated_age': 34,
        'experience_1_company': 'Google',
        'experience_1_title': 'Senior Software Engineer',
        'experience_2_company': 'Tesla',
        'experience_2_title': 'ML Engineer - Autopilot Team',
        'education_1_school': 'Stanford University',
        'education_1_degree': 'MS Computer Science',
        'education_2_school': 'MIT',
        'education_2_degree': 'PhD Artificial Intelligence',
        'about': 'Former Google and Tesla engineer with PhD in AI from MIT, specialized in computer vision and autonomous systems. Published 15+ papers in top AI conferences.'
    }
    
    # Example 2: Moderate-potential founder (good but not exceptional background)
    company_data_2 = {
        'company_name': 'DataFlow Analytics',
        'founding_year': 2023,
        'description': 'Business intelligence platform using machine learning for predictive analytics',
        'total_funding_usd': 2_000_000,  # $2M seed
        'funding_rounds': 1,
        'employee_count': 8,
        'sector': 'Data Analytics and Business Intelligence',
        'city': 'Austin',
        'region': 'Texas',
        'country': 'United States',
        'investor_count': 2
    }
    
    founder_data_2 = {
        'company_name': 'DataFlow Analytics',
        'name': 'Maria Garcia',
        'title': 'CEO and Founder',
        'estimated_age': 28,
        'experience_1_company': 'Salesforce',
        'experience_1_title': 'Data Analyst',
        'experience_2_company': 'McKinsey & Company',
        'experience_2_title': 'Business Analyst',
        'education_1_school': 'University of Texas',
        'education_1_degree': 'MBA',
        'education_2_school': 'Rice University',
        'education_2_degree': 'BS Business Administration',
        'about': 'Former Salesforce and McKinsey analyst with strong business and data analysis background.'
    }
    
    # Example 3: Lower-potential founder (limited experience)
    company_data_3 = {
        'company_name': 'AI Assistant Pro',
        'founding_year': 2024,
        'description': 'AI chatbot for customer service automation',
        'total_funding_usd': 500_000,  # $500K pre-seed
        'funding_rounds': 1,
        'employee_count': 3,
        'sector': 'Conversational AI',
        'city': 'Denver',
        'region': 'Colorado',
        'country': 'United States',
        'investor_count': 1
    }
    
    founder_data_3 = {
        'company_name': 'AI Assistant Pro',
        'name': 'John Smith',
        'title': 'CEO and Founder',
        'estimated_age': 25,
        'experience_1_company': 'Local Tech Startup',
        'experience_1_title': 'Junior Developer',
        'education_1_school': 'Colorado State University',
        'education_1_degree': 'BS Computer Science',
        'about': 'Recent CS graduate with some startup experience, passionate about AI and customer service.'
    }
    
    # Process all examples
    founder_company_pairs = [
        (company_data_1, founder_data_1),
        (company_data_2, founder_data_2),
        (company_data_3, founder_data_3)
    ]
    
    print("Processing 3 example founders...")
    results = predict_multiple_founders(founder_company_pairs)
    
    print("\n" + "="*80)
    print("BATCH PREDICTION RESULTS - RANKED BY SUCCESS PROBABILITY")
    print("="*80)
    
    for i, result in enumerate(results):
        if 'error' not in result:
            print(f"\n🏆 RANK #{i+1}")
            display_prediction_result(result)
        else:
            print(f"\n❌ ERROR in prediction for {result.get('founder_name', 'Unknown')}: {result['error']}")
    
    print("\n" + "="*80)
    print("INVESTMENT SUMMARY")
    print("="*80)
    
    strong_buys = [r for r in results if 'error' not in r and r['success_probability'] >= 0.65]
    buys = [r for r in results if 'error' not in r and 0.5 <= r['success_probability'] < 0.65]
    considers = [r for r in results if 'error' not in r and 0.35 <= r['success_probability'] < 0.5]
    passes = [r for r in results if 'error' not in r and r['success_probability'] < 0.35]
    
    print(f"🟢 STRONG BUY (≥65%): {len(strong_buys)} founders")
    for r in strong_buys:
        print(f"   • {r['founder_name']} ({r['company_name']}) - {r['success_probability']:.1%}")
    
    print(f"\n🔵 BUY (50-64%): {len(buys)} founders")
    for r in buys:
        print(f"   • {r['founder_name']} ({r['company_name']}) - {r['success_probability']:.1%}")
    
    print(f"\n🟡 CONSIDER (35-49%): {len(considers)} founders")
    for r in considers:
        print(f"   • {r['founder_name']} ({r['company_name']}) - {r['success_probability']:.1%}")
    
    print(f"\n🔴 PASS (<35%): {len(passes)} founders")
    for r in passes:
        print(f"   • {r['founder_name']} ({r['company_name']}) - {r['success_probability']:.1%}")
    
    print(f"\n💡 To test your own founders:")
    print(f"   1. Modify the company_data and founder_data dictionaries above")
    print(f"   2. Run: python ml_pipeline/predict_founder.py")
    print(f"   3. Or import this module and use predict_new_founder() function")

if __name__ == "__main__":
    main()