# AI Company Discovery Pipeline

A comprehensive CLI tool for discovering early-stage AI companies with advanced analytics, multi-source data fusion, and intelligent market analysis.

## 🚀 Features

### Core Capabilities
- **Multi-Source Company Discovery**: Find AI companies using Exa, Crunchbase, and other premium data sources
- **Advanced Metrics Extraction**: AI-powered extraction of funding, valuation, and operational metrics
- **Intelligent Sector Classification**: Precise AI-powered categorization into detailed technology sectors
- **Multi-Source Data Fusion**: Combine and validate data from multiple sources for maximum accuracy
- **LinkedIn Profile Enrichment**: Extract comprehensive founder and executive profiles
- **Market Analysis**: Get market metrics, sentiment analysis, and timing scores using current year data
- **Data Export**: Export to CSV, JSON formats with comprehensive data fields
- **Async Processing**: Fast parallel processing with intelligent rate limiting
- **Smart Checkpointing**: Resume interrupted operations seamlessly

### Enhanced Analytics
- **Financial Intelligence**: Funding amounts, stages, valuations, revenue data
- **Operational Metrics**: Employee counts, customer metrics, growth indicators  
- **Technology Stack Analysis**: Identify and categorize technology frameworks
- **Competitive Intelligence**: Market positioning and competitive advantages
- **Data Quality Scoring**: Confidence and completeness metrics for each company

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

1. Set up your environment variables:

```bash
cp .env.example .env
# Edit .env with your API keys (see API Keys section below)
```

2. Run the enhanced pipeline:

```bash
python -m initiation_pipeline.cli run --companies 50 --output ./output/results.csv
```

3. Or run individual steps:

```bash
# Find companies only
python -m initiation_pipeline.cli companies --limit 30 --output companies.csv

# Find profiles for existing companies
python -m initiation_pipeline.cli profiles --input companies.csv --output profiles.csv

# Analyze markets for companies
python -m initiation_pipeline.cli analyze --input companies.csv --output analysis.csv
```

## API Keys Required

### Core APIs (Required)
- **EXA_API_KEY**: Company discovery and web search
- **OPENAI_API_KEY**: AI-powered analysis and classification
- **SERPER_API_KEY**: Search and validation
- **APIFY_API_KEY**: LinkedIn profile scraping
- **PERPLEXITY_API_KEY**: Market analysis and research

### Data Source APIs (Recommended)
- **CRUNCHBASE_API_KEY**: Enhanced company data and validation
```

## CLI Commands & Options

### Complete Pipeline

```bash
python -m initiation_pipeline.cli run [OPTIONS]
```

**Options:**

- `--companies, -c`: Number of companies to find (default: 50)
- `--output, -o`: Output file path
- `--format, -f`: Output format (csv, json) (default: csv)
- `--no-profiles`: Skip LinkedIn profile enrichment
- `--no-analysis`: Skip market analysis
- `--checkpoint`: Checkpoint file prefix (default: pipeline)

### Company Discovery Only

```bash
python -m initiation_pipeline.cli companies [OPTIONS]
```

**Options:**

- `--limit, -l`: Number of companies to find (default: 30)
- `--output, -o`: Output CSV file
- `--category`: AI categories to focus on (can be used multiple times)
- `--region`: Geographic regions to focus on (can be used multiple times)

### Profile Enrichment

```bash
python -m initiation_pipeline.cli profiles [OPTIONS]
```

**Options:**

- `--input, -i`: CSV file with companies (required)
- `--output, -o`: Output CSV file

### Market Analysis

```bash
python -m initiation_pipeline.cli analyze [OPTIONS]
```

**Options:**

- `--input, -i`: CSV file with companies (required)
- `--output, -o`: Output CSV file

### Global Options

- `--version`: Show version and exit
- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR)
- `--verbose, -v`: Enable verbose output

## Configuration

All configuration is done via environment variables or CLI options. See `.env.example` for available options.

## API Keys Required

- **EXA_API_KEY**: For company discovery
- **OPENAI_API_KEY**: For data processing
- **SERPER_API_KEY**: For search functionality
- **APIFY_API_KEY**: For LinkedIn profile scraping
- **PERPLEXITY_API_KEY**: For market analysis

## Output Formats

The pipeline outputs comprehensive data including:

- Company information (name, description, funding, etc.)
- Founder/executive profiles with LinkedIn data
- Market metrics (size, CAGR, timing scores)
- Regional sentiment analysis
- Competitor analysis

## Examples

```bash
# Basic run with 20 companies
python -m initiation_pipeline.cli run --companies 20

# Export as JSON
python -m initiation_pipeline.cli run --companies 30 --format json --output results.json

# Skip profiles, only do market analysis
python -m initiation_pipeline.cli run --companies 25 --no-profiles

# Find companies in specific categories
python -m initiation_pipeline.cli companies --category "computer vision" --category "robotics" --limit 20

# Resume from checkpoint
python -m initiation_pipeline.cli run --companies 100 --checkpoint my_run
```

## Development

```bash
# Install dev dependencies
pip install -r requirements.txt

# Run tests
python test_pipeline.py

# Format code
black src/
isort src/

# Type checking
mypy src/
```

## License

MIT License
