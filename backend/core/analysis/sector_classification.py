"""AI-powered sector classification service for accurate company categorization."""

import json
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from openai import AsyncOpenAI

from ...core import settings
from ...validators import validate_text_length

import logging
logger = logging.getLogger(__name__)


@dataclass
class SectorClassification:
    """Structured sector classification result."""
    primary_sector: str
    sub_sectors: List[str]
    ai_focus: str
    technology_stack: List[str]
    business_model: str
    target_market: str
    confidence_score: float
    reasoning: str
    classification_timestamp: str


class SectorClassifier:
    """AI-powered classification of companies into detailed sectors and categories."""
    
    def __init__(self):
        self.openai = AsyncOpenAI(api_key=settings.openai_api_key)
        self.rate_limiter = RateLimiter(max_requests=60, time_window=60)
        
        # Define comprehensive AI sector taxonomy
        self.ai_taxonomy = {
            'machine_learning': {
                'sub_categories': [
                    'deep_learning', 'neural_networks', 'reinforcement_learning',
                    'supervised_learning', 'unsupervised_learning', 'federated_learning'
                ],
                'keywords': ['ml', 'machine learning', 'neural', 'deep learning', 'algorithms']
            },
            'computer_vision': {
                'sub_categories': [
                    'image_recognition', 'object_detection', 'facial_recognition',
                    'medical_imaging', 'autonomous_vehicles', 'augmented_reality'
                ],
                'keywords': ['computer vision', 'image', 'visual', 'opencv', 'detection']
            },
            'natural_language_processing': {
                'sub_categories': [
                    'chatbots', 'language_models', 'text_analysis', 'translation',
                    'sentiment_analysis', 'voice_recognition', 'conversational_ai'
                ],
                'keywords': ['nlp', 'language', 'text', 'chatbot', 'gpt', 'llm', 'transformer']
            },
            'generative_ai': {
                'sub_categories': [
                    'text_generation', 'image_generation', 'code_generation',
                    'music_generation', 'video_generation', 'creative_ai'
                ],
                'keywords': ['generative', 'gpt', 'dall-e', 'midjourney', 'stable diffusion']
            },
            'robotics': {
                'sub_categories': [
                    'industrial_robotics', 'service_robotics', 'humanoid_robots',
                    'drones', 'automation', 'robotic_process_automation'
                ],
                'keywords': ['robot', 'robotics', 'automation', 'drone', 'rpa']
            },
            'autonomous_systems': {
                'sub_categories': [
                    'self_driving_cars', 'autonomous_delivery', 'smart_logistics',
                    'autonomous_drones', 'smart_manufacturing'
                ],
                'keywords': ['autonomous', 'self-driving', 'adas', 'autopilot']
            },
            'ai_infrastructure': {
                'sub_categories': [
                    'ai_chips', 'ml_platforms', 'ai_cloud', 'edge_computing',
                    'model_deployment', 'mlops'
                ],
                'keywords': ['infrastructure', 'platform', 'cloud', 'edge', 'mlops', 'chips']
            },
            'ai_healthcare': {
                'sub_categories': [
                    'drug_discovery', 'medical_diagnosis', 'health_monitoring',
                    'digital_therapeutics', 'medical_imaging', 'precision_medicine'
                ],
                'keywords': ['healthcare', 'medical', 'drug', 'diagnosis', 'health']
            },
            'ai_fintech': {
                'sub_categories': [
                    'algorithmic_trading', 'fraud_detection', 'credit_scoring',
                    'robo_advisors', 'insurance_ai', 'regulatory_compliance'
                ],
                'keywords': ['fintech', 'trading', 'fraud', 'credit', 'finance', 'banking']
            },
            'ai_security': {
                'sub_categories': [
                    'cybersecurity', 'threat_detection', 'identity_verification',
                    'privacy_protection', 'data_security'
                ],
                'keywords': ['security', 'cybersecurity', 'threat', 'privacy', 'encryption']
            }
        }
        
        # Business model patterns
        self.business_models = {
            'b2b_saas': ['saas', 'software as a service', 'b2b', 'enterprise'],
            'b2c_app': ['b2c', 'consumer', 'mobile app', 'consumer-facing'],
            'api_platform': ['api', 'platform', 'developer tools', 'sdk'],
            'marketplace': ['marketplace', 'platform', 'two-sided'],
            'hardware': ['hardware', 'device', 'chip', 'sensor'],
            'consulting': ['consulting', 'services', 'implementation'],
            'freemium': ['freemium', 'free tier', 'premium'],
            'subscription': ['subscription', 'monthly', 'annual']
        }
    
    async def classify_company(
        self, 
        company_name: str,
        description: str,
        website_content: str = "",
        additional_context: str = ""
    ) -> SectorClassification:
        """Classify a company into detailed sectors using AI analysis."""
        try:
            # Combine all available text for analysis
            combined_text = self._prepare_classification_text(
                company_name, description, website_content, additional_context
            )
            
            # Get AI-powered classification
            ai_classification = await self._get_ai_classification(company_name, combined_text)
            
            # Enhance with rule-based classification
            rule_based = self._get_rule_based_classification(combined_text)
            
            # Merge and validate results
            final_classification = self._merge_classifications(ai_classification, rule_based)
            
            return final_classification
            
        except Exception as e:
            logger.error(f"Error classifying company {company_name}: {e}")
            return self._get_default_classification()
    
    def _prepare_classification_text(
        self, 
        company_name: str, 
        description: str, 
        website_content: str, 
        additional_context: str
    ) -> str:
        """Prepare combined text for classification."""
        # Combine and clean text
        texts = [
            f"Company: {company_name}",
            f"Description: {description}",
            website_content[:1000] if website_content else "",
            additional_context[:500] if additional_context else ""
        ]
        
        combined = "\n".join(filter(None, texts))
        
        # Truncate if too long
        if len(combined) > 2000:
            combined = combined[:2000] + "..."
        
        return combined
    
    async def _get_ai_classification(self, company_name: str, text: str) -> Dict:
        """Get AI-powered sector classification."""
        prompt = f"""
        Analyze this AI company and classify it into detailed sectors and categories.
        
        Company Information:
        {text}
        
        Provide a comprehensive classification including:
        
        1. Primary AI Sector (choose from: machine_learning, computer_vision, natural_language_processing, 
           generative_ai, robotics, autonomous_systems, ai_infrastructure, ai_healthcare, ai_fintech, ai_security)
        
        2. Sub-sectors (specific technologies/applications)
        
        3. AI Focus (detailed description of AI application)
        
        4. Technology Stack (programming languages, frameworks, platforms)
        
        5. Business Model (b2b_saas, b2c_app, api_platform, marketplace, hardware, etc.)
        
        6. Target Market (enterprise, consumer, developer, specific industry)
        
        7. Confidence Score (0.0 to 1.0)
        
        8. Reasoning (brief explanation)
        
        Return as JSON:
        {{
            "primary_sector": "sector_name",
            "sub_sectors": ["sub1", "sub2"],
            "ai_focus": "detailed description",
            "technology_stack": ["tech1", "tech2"],
            "business_model": "model_type",
            "target_market": "market_description",
            "confidence_score": 0.8,
            "reasoning": "explanation"
        }}
        """
        
        try:
            await self.rate_limiter.acquire()
            
            response = await self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean JSON response
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
            
            return json.loads(content)
            
        except Exception as e:
            logger.debug(f"AI classification failed for {company_name}: {e}")
            return {}
    
    def _get_rule_based_classification(self, text: str) -> Dict:
        """Get rule-based classification using keyword matching."""
        text_lower = text.lower()
        
        # Find matching sectors
        sector_scores = {}
        for sector, info in self.ai_taxonomy.items():
            score = 0
            for keyword in info['keywords']:
                if keyword in text_lower:
                    score += 1
            if score > 0:
                sector_scores[sector] = score
        
        # Get primary sector
        primary_sector = max(sector_scores.items(), key=lambda x: x[1])[0] if sector_scores else 'machine_learning'
        
        # Find business model
        business_model = 'b2b_saas'  # default
        for model, keywords in self.business_models.items():
            if any(keyword in text_lower for keyword in keywords):
                business_model = model
                break
        
        # Extract technology stack keywords
        tech_keywords = [
            'python', 'tensorflow', 'pytorch', 'opencv', 'react', 'node.js',
            'aws', 'gcp', 'azure', 'kubernetes', 'docker', 'api', 'rest',
            'graphql', 'mongodb', 'postgresql', 'redis', 'spark', 'hadoop'
        ]
        
        found_tech = [tech for tech in tech_keywords if tech in text_lower]
        
        return {
            'primary_sector': primary_sector,
            'business_model': business_model,
            'technology_stack': found_tech,
            'confidence_score': min(len(sector_scores) * 0.2, 0.8)
        }
    
    def _merge_classifications(self, ai_result: Dict, rule_result: Dict) -> SectorClassification:
        """Merge AI and rule-based classifications."""
        return SectorClassification(
            primary_sector=ai_result.get('primary_sector') or rule_result.get('primary_sector', 'machine_learning'),
            sub_sectors=ai_result.get('sub_sectors', []),
            ai_focus=ai_result.get('ai_focus', 'Artificial Intelligence Solutions'),
            technology_stack=list(set(
                ai_result.get('technology_stack', []) + 
                rule_result.get('technology_stack', [])
            )),
            business_model=ai_result.get('business_model') or rule_result.get('business_model', 'b2b_saas'),
            target_market=ai_result.get('target_market', 'Enterprise'),
            confidence_score=max(
                ai_result.get('confidence_score', 0.0),
                rule_result.get('confidence_score', 0.0)
            ),
            reasoning=ai_result.get('reasoning', 'Classification based on content analysis'),
            classification_timestamp=datetime.now().isoformat()
        )
    
    def _get_default_classification(self) -> SectorClassification:
        """Return default classification for error cases."""
        return SectorClassification(
            primary_sector='machine_learning',
            sub_sectors=[],
            ai_focus='Artificial Intelligence Solutions',
            technology_stack=[],
            business_model='b2b_saas',
            target_market='Enterprise',
            confidence_score=0.1,
            reasoning='Default classification due to analysis error',
            classification_timestamp=datetime.now().isoformat()
        )
    
    async def batch_classify(self, companies: List[Dict]) -> List[SectorClassification]:
        """Classify multiple companies in batch for efficiency."""
        classifications = []
        
        for company in companies:
            classification = await self.classify_company(
                company_name=company.get('name', ''),
                description=company.get('description', ''),
                website_content=company.get('website_content', ''),
                additional_context=company.get('additional_context', '')
            )
            classifications.append(classification)
        
        return classifications
