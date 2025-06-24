# AI Company Discovery Pipeline

A comprehensive CLI tool for discovering early-stage AI companies and building rich founder datasets with LinkedIn profiles, company intelligence, and optional market analysis.

## 🎯 Primary Focus: Founder Dataset Creation

This pipeline is optimized for creating comprehensive founder datasets by discovering AI companies and enriching them with detailed LinkedIn profiles of founders and key executives.

## 🚀 Features

### Core Capabilities

- **Multi-Source Company Discovery**: Find AI companies using Exa, Crunchbase, and other premium data sources
- **LinkedIn Profile Enrichment**: Extract comprehensive founder and executive profiles with experience, education, and skills
- **Advanced Data Fusion**: Combine and validate data from multiple sources for maximum accuracy
- **Intelligent Sector Classification**: AI-powered categorization into detailed technology sectors
- **Optional Market Analysis**: Add market metrics, sentiment analysis, and timing scores when needed
- **Smart Checkpointing**: Resume interrupted operations seamlessly
- **Flexible Export**: Export to CSV, JSON formats with comprehensive founder data

### Founder Data Intelligence

- **Complete LinkedIn Profiles**: Headlines, experience history, education, skills, location
- **Company Context**: AI focus, technology stack, funding stage, business model
- **Founder Relationships**: Map founders to their companies with detailed context
- **Data Quality Scoring**: Confidence and completeness metrics for each profile

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

2. Run the complete pipeline (optimized for founder data):

```bash
python -m initiation_pipeline.cli run
```

This will:
- ✅ Discover 50 AI companies
- ✅ Enrich with founder LinkedIn profiles  
- ❌ Skip market analysis (default - faster, focused on founders)
- 📁 Export comprehensive founder dataset

## 🔄 Flexible Pipeline Workflows

### 1. **Default: Founder-Focused Pipeline** (Recommended)
```bash
# Fast pipeline focused on founder data
python -m initiation_pipeline.cli run --companies 50
```
- ✅ Company discovery + data fusion
- ✅ LinkedIn profile enrichment
- ❌ Skips market analysis (faster)
- 🎯 **Perfect for building founder datasets**

### 2. **Complete Pipeline with Market Analysis**
```bash
# Full pipeline including market metrics
python -m initiation_pipeline.cli run --companies 50 --analysis
```
- ✅ Everything + market intelligence

### 3. **Incremental: Add Market Analysis Later**
```bash
# First: Run founder-focused pipeline
python -m initiation_pipeline.cli run

# Later: Add market analysis to existing data
python -m initiation_pipeline.cli market-analysis --input ./output/ai_companies_[timestamp].csv --output ./output/final_with_market.csv
```
- 🚀 **Best of both worlds**: Fast founder data + optional market metrics

### 4. **Individual Pipeline Steps**

#### Company Discovery Only
```bash
python -m initiation_pipeline.cli companies --limit 30 --output companies.csv
```

#### Profile Enrichment Only
```bash
python -m initiation_pipeline.cli profiles --input companies.csv --output profiles.csv
```

#### Market Analysis Only
```bash
python -m initiation_pipeline.cli market-analysis --input companies.csv --output analysis.csv
```

## API Keys Required

### Core APIs (Required)

- **EXA_API_KEY**: Company discovery and web search
- **OPENAI_API_KEY**: AI-powered analysis and classification  
- **SERPER_API_KEY**: Search and validation
- **APIFY_API_KEY**: LinkedIn profile scraping
- **PERPLEXITY_API_KEY**: Market analysis (only if using market analysis)

### Data Source APIs (Recommended)

- **CRUNCHBASE_API_KEY**: Enhanced company data and validation

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
- `--no-analysis`: Skip market analysis (default: True - focused on founders)
- `--analysis`: Include market analysis
- `--checkpoint`: Checkpoint file prefix (default: pipeline)

### Company Discovery

```bash
python -m initiation_pipeline.cli companies [OPTIONS]
```

**Options:**
- `--limit, -l`: Number of companies to find (default: 30)
- `--output, -o`: Output CSV file
- `--category`: AI categories to focus on
- `--region`: Geographic regions to focus on

### Profile Enrichment  

```bash
python -m initiation_pipeline.cli profiles [OPTIONS]
```

**Options:**
- `--input, -i`: CSV file with companies (required)
- `--output, -o`: Output CSV file

### Market Analysis Enhancement

```bash
python -m initiation_pipeline.cli market-analysis [OPTIONS]
```

**Options:**
- `--input, -i`: CSV file with companies (required)  
- `--output, -o`: Output CSV file

Add market metrics to existing company data without re-running profile enrichment.

## Output Data Structure

### Founder Dataset Fields

**Company Information:**
- Company name, description, website, founded year
- AI focus, sector classification, technology stack
- Business model, target market, funding stage
- Location, employee count, revenue data

**Founder Profiles:**
- Person name, LinkedIn URL, current title/role  
- Professional headline, location, about section
- Work experience (up to 5 positions with titles and companies)
- Education (up to 3 schools with degrees)
- Skills (up to 5 key skills)
- Estimated age, company association

**Optional Market Data:**
- Market size, CAGR, competitor analysis
- Regional sentiment (US, SEA), timing scores  
- Funding landscape, momentum indicators

## Example Workflows

### Founder Research Use Cases

```bash
# Quick founder dataset for 30 AI startups
python -m initiation_pipeline.cli run --companies 30

# Large founder database with market context
python -m initiation_pipeline.cli run --companies 100 --analysis

# Focus on specific AI sectors
python -m initiation_pipeline.cli companies --category "computer vision" --category "nlp" --limit 50
python -m initiation_pipeline.cli profiles --input companies.csv --output founder_profiles.csv

# Add market intelligence to existing founder data
python -m initiation_pipeline.cli market-analysis --input founder_profiles.csv --output complete_dataset.csv
```

### Pipeline Management

```bash
# Resume interrupted runs
python -m initiation_pipeline.cli run --companies 200 --checkpoint large_run

# Export different formats
python -m initiation_pipeline.cli run --companies 50 --format json --output founders.json

# Verbose logging for debugging
python -m initiation_pipeline.cli run --log-level DEBUG --verbose
```

## Smart Checkpointing

The pipeline automatically saves progress at each major step:
- `pipeline_companies.pkl`: After company discovery
- `pipeline_fused.pkl`: After data fusion  
- `pipeline_profiles.pkl`: After profile enrichment
- `pipeline_market.pkl`: After market analysis

Interrupted runs automatically resume from the last completed checkpoint.

## Performance & Rate Limiting

- **Parallel Processing**: Optimized async processing with intelligent batching
- **Rate Limiting**: Built-in delays to respect API limits  
- **Error Handling**: Graceful fallbacks for failed requests
- **Timeout Management**: Configurable timeouts for long-running operations

## Data Quality

- **Multi-Source Validation**: Cross-reference data across sources
- **Confidence Scoring**: Quality metrics for each data point
- **Fallback Handling**: Graceful degradation when sources fail
- **Data Completeness**: Track and report data coverage

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run with debug logging
python -m initiation_pipeline.cli run --log-level DEBUG

# Check API key configuration
python -m initiation_pipeline.cli run --companies 1
```

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure all required keys are set in `.env`
2. **Rate Limiting**: Reduce batch sizes or add delays
3. **Memory Issues**: Process smaller batches for large datasets
4. **LinkedIn Blocks**: Use residential proxies or reduce request frequency

### Getting Help

- Check logs with `--log-level DEBUG --verbose`
- Review checkpoint files in `./output/` directory
- Ensure sufficient API credits for chosen data sources

## License

MIT License