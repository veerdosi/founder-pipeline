"""Command Line Interface for the Initiation Pipeline."""

import asyncio
import time
from pathlib import Path
from typing import List, Optional

import typer
from rich.table import Table
from rich.panel import Panel

from .core import (
    setup_logging,
    validate_api_keys,
    console,
    save_to_csv,
    save_to_json,
    settings
)
from .services import InitiationPipeline


app = typer.Typer(
    name="initiation",
    help="AI Company Discovery Pipeline - Find early-stage AI companies with market analysis",
    no_args_is_help=True
)


def version_callback(value: bool):
    """Show version information."""
    if value:
        console.print("Initiation Pipeline v0.1.0")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, 
        "--version", 
        callback=version_callback, 
        help="Show version and exit"
    ),
    log_level: str = typer.Option(
        "INFO", 
        "--log-level", 
        help="Set logging level"
    ),
    verbose: bool = typer.Option(
        False, 
        "--verbose", 
        "-v", 
        help="Enable verbose output"
    )
):
    """
    AI Company Discovery Pipeline
    
    Find early-stage AI companies with comprehensive market analysis and LinkedIn profile enrichment.
    """
    if verbose:
        log_level = "DEBUG"
    
    # Set log level
    settings.log_level = log_level
    setup_logging()
    
    # Validate API keys
    if not validate_api_keys():
        console.print("❌ API key validation failed. Please check your .env file.")
        raise typer.Exit(1)


@app.command()
def run(
    companies: int = typer.Option(
        50, 
        "--companies", 
        "-c", 
        help="Number of companies to find"
    ),
    output: Path = typer.Option(
        None, 
        "--output", 
        "-o", 
        help="Output file path"
    ),
    format: str = typer.Option(
        "csv", 
        "--format", 
        "-f", 
        help="Output format (csv, json)"
    ),
    no_profiles: bool = typer.Option(
        False, 
        "--no-profiles", 
        help="Skip LinkedIn profile enrichment"
    ),
    no_analysis: bool = typer.Option(
        False, 
        "--no-analysis", 
        help="Skip market analysis"
    ),
    checkpoint_prefix: str = typer.Option(
        "pipeline", 
        "--checkpoint", 
        help="Checkpoint file prefix"
    )
):
    """
    Run the complete AI company discovery pipeline.
    
    This command will:
    1. Find AI companies using multiple data sources
    2. Enrich with LinkedIn profiles (unless --no-profiles)
    3. Perform market analysis (unless --no-analysis)
    4. Export results to specified format
    """
    console.print(Panel.fit(
        "[bold blue]🚀 AI Company Discovery Pipeline[/bold blue]\n"
        f"Finding {companies} companies with full analysis",
        border_style="blue"
    ))
    
    # Set default output path
    if not output:
        timestamp = int(time.time())
        output = settings.default_output_dir / f"ai_companies_{timestamp}.{format}"
    
    # Ensure output directory exists
    output.parent.mkdir(parents=True, exist_ok=True)
    
    async def run_pipeline():
        pipeline = InitiationPipeline()
        
        fused_companies = await pipeline.run_complete_pipeline(
            company_limit=companies,
            include_profiles=not no_profiles,
            include_market_analysis=not no_analysis,
            checkpoint_prefix=checkpoint_prefix
        )
        
        # Export results using the new pipeline export method
        pipeline.export_results(fused_companies, output, format)
        
        # Show summary (stats are printed by the pipeline now)
        return fused_companies
    
    # Run the pipeline
    result = asyncio.run(run_pipeline())
    
    console.print(f"\n✅ Pipeline completed successfully!")
    console.print(f"📁 Results saved to: {output}")


@app.command()
def companies(
    limit: int = typer.Option(
        30, 
        "--limit", 
        "-l", 
        help="Number of companies to find"
    ),
    output: Path = typer.Option(
        None, 
        "--output", 
        "-o", 
        help="Output CSV file"
    ),
    categories: List[str] = typer.Option(
        None, 
        "--category", 
        help="AI categories to focus on"
    ),
    regions: List[str] = typer.Option(
        None, 
        "--region", 
        help="Geographic regions to focus on"
    )
):
    """
    Find AI companies only (no profile enrichment or market analysis).
    
    This is useful for quickly getting a list of companies that can be
    processed later with other commands.
    """
    console.print(Panel.fit(
        "[bold green]🔍 Company Discovery Only[/bold green]\n"
        f"Finding {limit} AI companies",
        border_style="green"
    ))
    
    # Set default output
    if not output:
        timestamp = int(time.time())
        output = settings.default_output_dir / f"companies_{timestamp}.csv"
    
    output.parent.mkdir(parents=True, exist_ok=True)
    
    async def find_companies():
        pipeline = InitiationPipeline()
        companies = await pipeline.run_company_discovery_only(
            limit=limit,
            categories=categories,
            regions=regions
        )
        
        # Convert to records for CSV
        records = []
        for company in companies:
            record = {
                "name": company.name,
                "description": company.description,
                "website": str(company.website) if company.website else None,
                "founded_year": company.founded_year,
                "funding_total_usd": company.funding_total_usd,
                "funding_stage": company.funding_stage,
                "founders": "|".join(company.founders) if company.founders else None,
                "city": company.city,
                "region": company.region,
                "country": company.country,
                "ai_focus": company.ai_focus,
                "sector": company.sector,
                "source_url": company.source_url
            }
            records.append(record)
        
        save_to_csv(records, output)
        
        # Show summary table
        table = Table(title="Companies Found")
        table.add_column("Name", style="cyan")
        table.add_column("AI Focus", style="magenta")
        table.add_column("Funding Stage", style="green")
        table.add_column("Location", style="yellow")
        
        for company in companies[:10]:  # Show first 10
            table.add_row(
                company.name or "N/A",
                company.ai_focus or "N/A",
                str(company.funding_stage) if company.funding_stage else "N/A",
                company.city or "N/A"
            )
        
        if len(companies) > 10:
            table.add_row("...", "...", "...", "...")
        
        console.print(table)
        
        return companies
    
    companies_result = asyncio.run(find_companies())
    
    console.print(f"\n✅ Found {len(companies_result)} companies")
    console.print(f"📁 Saved to: {output}")


@app.command()
def profiles(
    input_file: Path = typer.Option(
        ..., 
        "--input", 
        "-i", 
        help="CSV file with companies"
    ),
    output: Path = typer.Option(
        None, 
        "--output", 
        "-o", 
        help="Output CSV file"
    )
):
    """
    Find LinkedIn profiles for companies from a CSV file.
    
    The input CSV should have at least 'name' and 'description' columns.
    """
    if not input_file.exists():
        console.print(f"❌ Input file not found: {input_file}")
        raise typer.Exit(1)
    
    console.print(Panel.fit(
        "[bold yellow]👤 Profile Enrichment[/bold yellow]\n"
        f"Processing companies from {input_file.name}",
        border_style="yellow"
    ))
    
    # Set default output
    if not output:
        output = input_file.parent / f"{input_file.stem}_with_profiles.csv"
    
    async def enrich_profiles():
        import pandas as pd
        from .models import Company
        
        # Load companies from CSV
        df = pd.read_csv(input_file)
        companies = []
        
        for _, row in df.iterrows():
            company = Company(
                name=row.get('name', ''),
                description=row.get('description', ''),
                founded_year=row.get('founded_year'),
                ai_focus=row.get('ai_focus', ''),
                founders=row.get('founders', '').split('|') if row.get('founders') else []
            )
            companies.append(company)
        
        pipeline = InitiationPipeline()
        enriched_companies = await pipeline.run_profile_enrichment_only(companies)
        
        # Convert to CSV records
        records = []
        for enriched in enriched_companies:
            if enriched.profiles:
                for profile in enriched.profiles:
                    record = {
                        # Company data
                        "company_name": enriched.company.name,
                        "description": enriched.company.description,
                        "ai_focus": enriched.company.ai_focus,
                        
                        # Profile data
                        "person_name": profile.person_name,
                        "linkedin_url": str(profile.linkedin_url),
                        "role": profile.role,
                        "headline": profile.headline,
                        "location": profile.location,
                        "estimated_age": profile.estimated_age
                    }
                    records.append(record)
            else:
                # Company without profiles
                record = {
                    "company_name": enriched.company.name,
                    "description": enriched.company.description,
                    "ai_focus": enriched.company.ai_focus,
                    "person_name": None,
                    "linkedin_url": None,
                    "role": None
                }
                records.append(record)
        
        save_to_csv(records, output)
        return enriched_companies
    
    result = asyncio.run(enrich_profiles())
    
    profiles_found = sum(len(ec.profiles) for ec in result)
    console.print(f"\n✅ Found {profiles_found} LinkedIn profiles")
    console.print(f"📁 Saved to: {output}")


@app.command()
def analyze(
    input_file: Path = typer.Option(
        ..., 
        "--input", 
        "-i", 
        help="CSV file with companies"
    ),
    output: Path = typer.Option(
        None, 
        "--output", 
        "-o", 
        help="Output CSV file"
    )
):
    """
    Perform market analysis for companies from a CSV file.
    
    The input CSV should have company data including 'name', 'ai_focus' or 'sector'.
    """
    if not input_file.exists():
        console.print(f"❌ Input file not found: {input_file}")
        raise typer.Exit(1)
    
    console.print(Panel.fit(
        "[bold red]📊 Market Analysis[/bold red]\n"
        f"Analyzing companies from {input_file.name}",
        border_style="red"
    ))
    
    # Set default output
    if not output:
        output = input_file.parent / f"{input_file.stem}_with_analysis.csv"
    
    async def analyze_markets():
        import pandas as pd
        from .models import Company
        
        # Load companies from CSV
        df = pd.read_csv(input_file)
        companies = []
        
        for _, row in df.iterrows():
            company = Company(
                name=row.get('name', ''),
                description=row.get('description', ''),
                founded_year=row.get('founded_year'),
                ai_focus=row.get('ai_focus', ''),
                sector=row.get('sector', '')
            )
            companies.append(company)
        
        pipeline = InitiationPipeline()
        analyzed_companies = await pipeline.run_market_analysis_only(companies)
        
        # Convert to CSV records
        records = []
        for enriched in analyzed_companies:
            metrics = enriched.market_metrics
            record = {
                # Company data
                "company_name": enriched.company.name,
                "description": enriched.company.description,
                "ai_focus": enriched.company.ai_focus,
                "sector": enriched.company.sector,
                
                # Market metrics
                "market_size_billion": metrics.market_size_billion if metrics else None,
                "cagr_percent": metrics.cagr_percent if metrics else None,
                "timing_score": metrics.timing_score if metrics else None,
                "us_sentiment": metrics.us_sentiment if metrics else None,
                "sea_sentiment": metrics.sea_sentiment if metrics else None,
                "competitor_count": metrics.competitor_count if metrics else None,
                "total_funding_billion": metrics.total_funding_billion if metrics else None,
                "momentum_score": metrics.momentum_score if metrics else None,
                "market_stage": metrics.market_stage if metrics else None,
                "confidence_score": metrics.confidence_score if metrics else None
            }
            records.append(record)
        
        save_to_csv(records, output)
        return analyzed_companies
    
    result = asyncio.run(analyze_markets())
    
    analyzed_count = sum(1 for ec in result if ec.market_metrics)
    console.print(f"\n✅ Analyzed {analyzed_count} company markets")
    console.print(f"📁 Saved to: {output}")


if __name__ == "__main__":
    app()
