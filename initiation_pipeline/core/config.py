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
    serper_api_key: str = Field(..., env="SERPER_API_KEY")
    apify_api_key: str = Field(..., env="APIFY_API_KEY")
    perplexity_api_key: str = Field(..., env="PERPLEXITY_API_KEY")
    
    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_file: Optional[str] = Field(None, env="LOG_FILE")
    
    # Rate Limiting
    requests_per_minute: int = Field(30, env="REQUESTS_PER_MINUTE")
    concurrent_requests: int = Field(3, env="CONCURRENT_REQUESTS")
    
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
    
    # Global Geographic Focus
    target_regions: List[str] = Field(
        default=[
            "United States", "Europe", "Asia", "Canada", "Australia", 
            "Israel", "Singapore", "India", "South Korea", "Japan"
        ],
        env="TARGET_REGIONS"
    )
    
    # Early Stage Funding Filters
    funding_stages: List[str] = Field(
        default=["seed", "pre-seed", "series-a", "series-b"],
        env="FUNDING_STAGES"
    )
    min_funding_usd: int = Field(100000, env="MIN_FUNDING_USD")
    max_funding_usd: int = Field(100000000, env="MAX_FUNDING_USD")
    founded_after_year: int = Field(2018, env="FOUNDED_AFTER_YEAR")
    
    # LinkedIn Scraping
    linkedin_actor_id: str = Field(env="LINKEDIN_ACTOR_ID")
    
    # Market Analysis
    market_analysis_timeout: int = Field(60, env="MARKET_ANALYSIS_TIMEOUT")
    
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
