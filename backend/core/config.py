"""Configuration management using Pydantic Settings."""

import os
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Core API Keys
    exa_api_key: str = Field(..., env="EXA_API_KEY")
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    apify_api_key: str = Field(..., env="APIFY_API_KEY")
    serpapi_key: str = Field(..., env="SERPAPI_KEY")
    perplexity_api_key: str = Field(..., env="PERPLEXITY_API_KEY")
    anthropic_api_key: str = Field(..., env="ANTHROPIC_API_KEY") 
    
    # Data Source APIs
    crunchbase_api_key: str = Field(..., env="CRUNCHBASE_API_KEY")
    
    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_file: Optional[str] = Field(None, env="LOG_FILE")
    
    # Rate Limiting - increased for better performance
    requests_per_minute: int = Field(60, env="REQUESTS_PER_MINUTE")  
    concurrent_requests: int = Field(5, env="CONCURRENT_REQUESTS")  
    
    # Output Settings
    default_output_dir: Path = Field(
        Path("./output"), 
        env="DEFAULT_OUTPUT_DIR"
    )
    checkpoint_enabled: bool = Field(True, env="CHECKPOINT_ENABLED")
    
    # Company Discovery - Global Early Stage Focus
    default_company_limit: int = Field(50, env="DEFAULT_COMPANY_LIMIT")
    ai_categories: List[str] = Field(
        default=[
            "artificial intelligence",
            "machine learning", 
            "computer vision",
            "natural language processing",
            "robotics",
            "autonomous vehicles",
            "generative ai",
            "deep learning",
            "neural networks",
            "quantum computing",
            "edge ai"
        ],
        env="AI_CATEGORIES"
    )
    
    # Global Geographic Focus - US Market Only
    target_regions: List[str] = Field(
        default=[
            "United States", "US", "USA", "Silicon Valley", "San Francisco", 
            "New York", "Boston", "Seattle", "Austin", "Los Angeles", "Chicago"
        ],
        env="TARGET_REGIONS"
    )
    
    # Early Stage Funding Filters
    funding_stages: List[str] = Field(
        default=["pre-seed", "seed", "series-a"],
        env="FUNDING_STAGES"
    )
    min_funding_usd: int = Field(100000, env="MIN_FUNDING_USD")
    max_funding_usd: int = Field(20000000, env="MAX_FUNDING_USD") # Max $20M for Series A focus
    founded_after_year: int = Field(2020, env="FOUNDED_AFTER_YEAR")
    
    # LinkedIn Scraping
    linkedin_actor_id: str = Field(env="LINKEDIN_ACTOR_ID")
    
    # Market Analysis
    market_analysis_timeout: int = Field(60, env="MARKET_ANALYSIS_TIMEOUT")
    
    # Websets Configuration
    webset_enabled: bool = Field(True, env="WEBSET_ENABLED")
    webset_monitoring_enabled: bool = Field(False, env="WEBSET_MONITORING_ENABLED")  # Disabled by default
    webset_monitor_cron: str = Field("0 */6 * * *", env="WEBSET_MONITOR_CRON")  # Every 6 hours
    webset_max_items_per_webset: int = Field(100, env="WEBSET_MAX_ITEMS") 
    webset_enrichment_enabled: bool = Field(True, env="WEBSET_ENRICHMENT_ENABLED")
    
    # Webset Enrichment Configurations - Comprehensive company data replacement for Crunchbase
    webset_enrichments: List[dict] = Field(
        default=[
            {
                "description": "Company name",
                "format": "text"
            },
            {
                "description": "Company description and what they do",
                "format": "text"
            },
            {
                "description": "Company website URL",
                "format": "text"
            },
            {
                "description": "Company location as city, state/region, country",
                "format": "text"
            },
            {
                "description": "Current funding stage (pre-seed, seed, series-a, series-b, series-c)",
                "format": "text"
            },
            {
                "description": "Total funding raised in USD",
                "format": "text"
            },
            {
                "description": "Company founders and co-founders names",
                "format": "text"
            },
            {
                "description": "Key investors and venture capital firms",
                "format": "text"
            },
            {
                "description": "Crunchbase company profile URL",
                "format": "text"
            }
        ],
        env="WEBSET_ENRICHMENTS"
    )
    
    # Webset Search Categories for AI Companies (year will be appended dynamically)
    webset_search_categories: List[dict] = Field(
        default=[
            {
                "name": "ai_general",
                "query": "AI startups United States seed funding Series A",
                "count": 50
            },
            {
                "name": "ai_enterprise", 
                "query": "enterprise AI companies B2B SaaS United States venture capital funding",
                "count": 40
            },
            {
                "name": "ai_healthcare",
                "query": "healthcare AI startups medical technology United States digital health",
                "count": 30
            },
            {
                "name": "ai_fintech",
                "query": "fintech AI companies financial technology United States payments lending",
                "count": 30
            },
            {
                "name": "ai_autonomous",
                "query": "autonomous vehicle AI robotics United States self-driving transportation",
                "count": 25
            },
            {
                "name": "ai_generative",
                "query": "generative AI startups large language models LLM United States",
                "count": 35
            }
        ],
        env="WEBSET_SEARCH_CATEGORIES"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure output directory exists
        self.default_output_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()