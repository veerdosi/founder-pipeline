# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI company discovery and enrichment pipeline that combines multiple data sources to create comprehensive company profiles for investment analysis. The system specializes in finding early-stage AI companies and enriching them with LinkedIn profiles, market analysis, and funding data.

## Development Commands

### Running the Pipeline
```bash
# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env with your API keys

# Run complete pipeline (50 companies, founder-focused)
python -m initiation_pipeline.cli run

# Run with market analysis
python -m initiation_pipeline.cli run --companies 50 --analysis

# Debug mode
python -m initiation_pipeline.cli run --log-level DEBUG --verbose
```

### Individual Pipeline Steps
```bash
# Company discovery only
python -m initiation_pipeline.cli companies --limit 30 --output companies.csv

# Profile enrichment only
python -m initiation_pipeline.cli profiles --input companies.csv --output profiles.csv

# Market analysis only
python -m initiation_pipeline.cli market-analysis --input companies.csv --output analysis.csv
```

### Testing
No formal test suite is currently implemented. Test individual components by running them with small datasets:
```bash
# Test with minimal data
python -m initiation_pipeline.cli run --companies 1 --log-level DEBUG
```

## Architecture Overview

### Service-Oriented Architecture
The codebase follows a modular design with clear separation of concerns:

- **`initiation_pipeline/cli.py`**: Main CLI interface using Typer
- **`initiation_pipeline/core/`**: Configuration, logging, utilities
- **`initiation_pipeline/services/`**: Core business logic services
- **`initiation_pipeline/models/`**: Pydantic data models
- **`initiation_pipeline/utils/`**: Shared utilities

### Key Services

1. **`services/pipeline.py`** (`InitiationPipeline`): Main orchestrator that coordinates all services with checkpointing and error handling

2. **Data Discovery**: 
   - `company_discovery.py`: Discovers AI companies using Exa search API
   - `crunchbase_integration.py`: Enhances with Crunchbase funding data

3. **Data Enrichment**:
   - `profile_enrichment.py`: LinkedIn profile scraping via Apify
   - `market_analysis.py`: Market research using Perplexity AI
   - `sector_classification.py`: AI-powered company categorization

4. **Data Processing**:
   - `data_fusion.py`: Intelligent multi-source data fusion with conflict resolution
   - `metrics_extraction.py`: Financial metrics extraction from content

### Data Flow Pattern
```
Company Discovery → Data Fusion → Profile Enrichment → Market Analysis → Export
```

The pipeline uses sophisticated data fusion to combine information from multiple sources with confidence weighting and quality scoring.

## Configuration

### Required API Keys (.env file)
```bash
# Core APIs (all required)
EXA_API_KEY=your_exa_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  
SERPER_API_KEY=your_serper_api_key_here
APIFY_API_KEY=your_apify_api_key_here
PERPLEXITY_API_KEY=your_perplexity_api_key_here

# Data Source APIs
CRUNCHBASE_API_KEY=your_crunchbase_api_key_here

# LinkedIn Scraping
LINKEDIN_ACTOR_ID=your_apify_linkedin_actor_id
```

### Key Settings (initiation_pipeline/core/config.py)
- Rate limiting: 30 requests/minute, 3 concurrent requests
- Default company limit: 50
- Focus on early-stage funding (pre-seed to Series A)
- Global regions: US, Europe, Asia-Pacific
- Founded after 2020

## Working with the Codebase

### Adding New Data Sources
1. Create new service class in `services/`
2. Implement async methods following existing patterns
3. Add to data fusion pipeline in `data_fusion.py`
4. Update configuration in `core/config.py`

### Modifying AI Categories
Edit `ai_categories` list in `core/config.py` to adjust company discovery focus.

### Checkpointing System
The pipeline automatically saves progress at each major step:
- `pipeline_companies.pkl`: After company discovery
- `pipeline_fused.pkl`: After data fusion
- `pipeline_profiles.pkl`: After profile enrichment
- `pipeline_market.pkl`: After market analysis

Interrupted runs automatically resume from the last checkpoint.

### Error Handling
- Each service implements graceful degradation
- API failures don't stop the entire pipeline
- Data quality scores indicate reliability
- Rate limiting prevents API throttling

## Data Models

Key Pydantic models in `models/`:
- `Company`: Basic company information
- `LinkedInProfile`: Executive profile data
- `MarketMetrics`: Market analysis results
- `EnrichedCompany`: Complete company profile with all enrichments
- `FusedCompanyData`: Multi-source data fusion results

## Output Formats

The pipeline exports comprehensive datasets with:
- Company information (name, description, funding, location)
- Executive profiles (LinkedIn data, experience, education)
- Market metrics (size, growth, sentiment, timing)
- Data quality scores and confidence metrics

Export formats: CSV (default), JSON