# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Backend
- **Install dependencies**: `pip install -r requirements.txt`
- **Start backend server**: `uvicorn backend.api.main:app --reload`
- **Activate virtual environment**: `source venv/bin/activate` (if using venv)

### Frontend  
- **Install dependencies**: `cd frontend && npm install`
- **Start development server**: `cd frontend && npm run dev`
- **Build for production**: `cd frontend && npm run build`
- **Lint code**: `cd frontend && npm run lint`

### Full Application
The application runs as two separate services:
1. Backend API server on `http://localhost:8000` 
2. Frontend React app on `http://localhost:3000` (in development)

## Architecture Overview

### Data Pipeline Flow
The Initiation Pipeline is an AI-powered founder discovery and ranking system with a 6-stage checkpointed pipeline:

1. **Company Discovery** (`backend/core/data/company_discovery.py`) - Uses Exa API to find early-stage AI companies
2. **Data Fusion** (`backend/core/data/data_fusion.py`) - Enriches companies with Crunchbase data
3. **Profile Enrichment** (`backend/core/data/profile_enrichment.py`) - Finds and scrapes LinkedIn profiles using Apify
4. **Founder Intelligence** - Gathers additional intelligence on founders
5. **Founder Ranking** (`backend/core/ranking/ranking_service.py`) - Uses Claude Sonnet 4 to classify founders on L1-L10 experience scale
6. **Market Analysis** (`backend/core/analysis/market_analysis.py`) - Generates market reports using Perplexity AI

### Core Components

#### Pipeline Orchestration
- **Main Pipeline** (`backend/core/pipeline.py`) - Central orchestrator using `InitiationPipeline` class
- **Checkpoint System** (`backend/utils/checkpoint_manager.py`) - Robust checkpointing allows resuming interrupted pipelines
- Each stage saves progress to `checkpoints/` directory with job-specific IDs

#### API Layer
- **FastAPI Backend** (`backend/api/main.py`) - RESTful API with endpoints for pipeline execution, data export, and market analysis
- **CORS enabled** for frontend communication on localhost:3000
- **Job-based architecture** tracks pipeline state across requests

#### Frontend Architecture
- **React + TypeScript** with Vite build system
- **Key Components**:
  - `Pipeline.tsx` - Main pipeline execution interface
  - `MarketAnalysis.tsx` - Company market analysis reports
  - `CheckpointManager.tsx` - Resume from saved checkpoints
  - `DataTable.tsx` - Display and export company/founder data
- **State Management** using TanStack Query for API calls

### Configuration System
- **Settings** (`backend/core/config.py`) - Pydantic-based configuration with environment variables
- **Required API Keys** in `.env` file:
  - `ANTHROPIC_API_KEY` - For founder ranking (Claude Sonnet 4)
  - `PERPLEXITY_API_KEY` - For market analysis 
  - `EXA_API_KEY` - For company discovery
  - `CRUNCHBASE_API_KEY` - For company data enhancement
  - `APIFY_API_KEY` - For LinkedIn profile scraping
  - `OPENAI_API_KEY` - For data extraction
  - `SERPER_API_KEY` - For search validation

### Data Models
- **Core Models** (`backend/models.py`) - Pydantic models for Company, LinkedInProfile, EnrichedCompany
- **Ranking Models** (`backend/core/ranking/models.py`) - L1-L10 founder classification system
- **TypeScript Interfaces** (`frontend/src/interfaces.ts`) - Frontend type definitions

### Key Patterns

#### Checkpoint-Driven Development
- All pipeline stages support checkpointing for reliability
- Use `checkpoint_manager.create_job_id()` for new pipeline runs
- Resume functionality via `CheckpointedPipelineRunner`

#### Batch Processing
- Company discovery and ranking operate in configurable batches
- Rate limiting built into all external API calls
- Concurrent processing with `settings.concurrent_requests` control

#### Error Handling
- Comprehensive logging with Rich console output
- Graceful degradation when individual companies/founders fail
- Pipeline continues with partial results rather than failing completely

## Working with the Codebase

### Adding New Data Sources
1. Create new service in `backend/core/data/`
2. Follow existing patterns from `ExaCompanyDiscovery` or `LinkedInEnrichmentService`
3. Add configuration to `backend/core/config.py`
4. Integrate into main pipeline flow in `backend/core/pipeline.py`

### Modifying Ranking Logic
- Ranking prompts are in `backend/core/ranking/prompts.py`
- L1-L10 classification logic in `backend/core/ranking/ranking_service.py`
- Update ranking models in `backend/core/ranking/models.py` if changing output format

### Frontend Development
- Components follow consistent patterns with TypeScript interfaces
- All API calls use TanStack Query for caching and error handling  
- CSS modules for component-specific styling
- Data tables use TanStack Table for sorting/filtering

### Testing Pipeline Changes
- Use small `limit` values for faster testing during development
- Leverage checkpoint system to test individual pipeline stages
- Monitor `output/` directory for CSV exports
- Check logs for detailed execution information