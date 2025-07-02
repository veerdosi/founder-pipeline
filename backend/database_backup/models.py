"""Database models based on specification requirements."""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, JSON, DECIMAL
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from typing import Optional, List, Dict, Any
from datetime import datetime

Base = declarative_base()


class Company(Base):
    """Companies table as specified in architecture."""
    __tablename__ = "companies"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text)
    website = Column(String(500))
    funding_total = Column(DECIMAL(15, 2))  # Total funding in USD
    valuation = Column(DECIMAL(15, 2))  # Current valuation in USD
    stage = Column(String(50))  # Funding stage
    founded_date = Column(DateTime)
    industry = Column(String(100))
    location = Column(String(255))
    last_funding_date = Column(DateTime)
    investor_list = Column(JSON)  # Array of investor names
    
    # Additional fields from current implementation
    short_description = Column(Text)
    linkedin_url = Column(String(500))
    crunchbase_url = Column(String(500))
    city = Column(String(100))
    region = Column(String(100))
    country = Column(String(100))
    ai_focus = Column(String(255))
    sector = Column(String(100))
    categories = Column(JSON)  # Array of categories
    source_url = Column(String(500))
    extraction_date = Column(DateTime, default=func.now())
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    founders = relationship("Founder", back_populates="company")


class Founder(Base):
    """Founders table as specified in architecture."""
    __tablename__ = "founders"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    l_level = Column(String(20))  # L1-L10 classification
    confidence_score = Column(Float)  # 0.0 to 1.0
    linkedin_url = Column(String(500))
    github_url = Column(String(500))
    crunchbase_url = Column(String(500))
    experience_years = Column(Integer)
    exit_history = Column(JSON)  # Array of exit information
    patents_count = Column(Integer, default=0)
    last_updated = Column(DateTime, default=func.now())
    verification_status = Column(String(50), default="pending")  # pending, verified, failed
    
    # Additional profile fields
    title = Column(String(255))
    location = Column(String(255))
    about = Column(Text)
    estimated_age = Column(Integer)
    
    # Experience data (top 3)
    experience_1_title = Column(String(255))
    experience_1_company = Column(String(255))
    experience_2_title = Column(String(255))
    experience_2_company = Column(String(255))
    experience_3_title = Column(String(255))
    experience_3_company = Column(String(255))
    
    # Education data (top 2)
    education_1_school = Column(String(255))
    education_1_degree = Column(String(255))
    education_2_school = Column(String(255))
    education_2_degree = Column(String(255))
    
    # Skills (top 3)
    skill_1 = Column(String(100))
    skill_2 = Column(String(100))
    skill_3 = Column(String(100))
    
    # Classification details
    reasoning = Column(Text)
    evidence = Column(JSON)  # Array of evidence points
    verification_sources = Column(JSON)  # Array of verification sources needed
    processing_metadata = Column(JSON)  # Processing information
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    company = relationship("Company", back_populates="founders")
    data_sources = relationship("DataSource", back_populates="founder")


class DataSource(Base):
    """Data sources table as specified in architecture."""
    __tablename__ = "data_sources"
    
    id = Column(Integer, primary_key=True, index=True)
    founder_id = Column(Integer, ForeignKey("founders.id"), nullable=False)
    source_type = Column(String(100), nullable=False)  # linkedin, crunchbase, sec_edgar, etc.
    source_url = Column(String(500), nullable=False)
    data_point = Column(Text)  # What data was extracted
    verification_status = Column(String(50), default="pending")  # pending, verified, failed
    confidence_score = Column(Float)  # 0.0 to 1.0
    last_checked = Column(DateTime, default=func.now())
    
    # Additional metadata
    extraction_metadata = Column(JSON)
    error_message = Column(Text)  # If verification failed
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    founder = relationship("Founder", back_populates="data_sources")


class DiscoveryJob(Base):
    """Track company discovery jobs."""
    __tablename__ = "discovery_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(100), unique=True, nullable=False, index=True)
    status = Column(String(50), default="running")  # running, completed, failed
    job_type = Column(String(50), default="company_discovery")  # company_discovery, founder_ranking
    
    # Parameters
    parameters = Column(JSON)  # Store job parameters
    
    # Results
    companies_found = Column(Integer, default=0)
    founders_ranked = Column(Integer, default=0)
    high_confidence_count = Column(Integer, default=0)
    error_message = Column(Text)
    
    # Timing
    started_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime)
    execution_time_seconds = Column(Float)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class Alert(Base):
    """Alert system for L6+ founder discoveries."""
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    alert_type = Column(String(50), nullable=False)  # founder_discovery, funding_update, etc.
    severity = Column(String(20), default="info")  # info, warning, critical
    
    # Content
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    
    # Related entities
    founder_id = Column(Integer, ForeignKey("founders.id"))
    company_id = Column(Integer, ForeignKey("companies.id"))
    
    # Status
    sent = Column(Boolean, default=False)
    sent_at = Column(DateTime)
    delivery_status = Column(String(50))  # pending, delivered, failed
    
    # Metadata
    metadata = Column(JSON)  # Additional context data
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class MonitoringSource(Base):
    """Track monitoring sources and their status."""
    __tablename__ = "monitoring_sources"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    url = Column(String(500), nullable=False)
    source_type = Column(String(50), nullable=False)  # media, accelerator, vc, patent, stealth
    check_frequency_minutes = Column(Integer, nullable=False)
    
    # Status
    active = Column(Boolean, default=True)
    last_checked = Column(DateTime)
    last_success = Column(DateTime)
    consecutive_failures = Column(Integer, default=0)
    
    # Metrics
    total_companies_found = Column(Integer, default=0)
    total_checks = Column(Integer, default=0)
    success_rate = Column(Float, default=0.0)
    
    # Configuration
    rate_limit_per_minute = Column(Integer, default=30)
    timeout_seconds = Column(Integer, default=60)
    custom_headers = Column(JSON)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
