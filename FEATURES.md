# Application Features and API Usage

This document outlines the key features of the Initiation Pipeline application, organized by the main tabs in the user interface. It also details the external APIs used to power each functionality.

---

## 1. Pipeline Tab

This tab is the control center for running the primary discovery workflow, which finds and ranks early-stage AI companies based on their founding year.

### Key Features

-   **Year-Based Discovery**: Initiates a pipeline run to discover companies founded in a specific year.
-   **Start Fresh / Resume**: Users can start a new pipeline from scratch or resume an interrupted job from a saved checkpoint, preventing redundant work.
-   **Real-time Progress Tracking**: A visual tracker displays the current stage of the pipeline (e.g., Company Discovery, Founder Analysis) and its progress.
-   **Results Display**: Once complete, the discovered companies and ranked founders are shown in interactive data tables.
-   **Data Export**: Final data for both companies and founders can be exported to CSV files in the `output/` directory.

### APIs Used

-   **Exa API**: Powers the initial **Company Discovery** by searching the web for AI companies matching the specified founding year.
-   **Crunchbase API**: Used during the **Data Fusion** stage to enrich company profiles with structured data like funding, investors, and location.
-   **Apify API**: Facilitates **Founder Profile Enrichment** by scraping LinkedIn profiles for detailed professional and educational histories.
-   **Anthropic API (Claude Models)**: Drives the **Founder Ranking** service, analyzing founder profiles to classify their experience on an L1-L10 scale.
-   **Serper API**: Provides real-time **Google Search validation** to verify facts and enhance data accuracy throughout the pipeline.

---

## 2. Accelerators Tab

This tab provides a specialized workflow to discover and analyze AI/ML companies that have graduated from top-tier startup accelerators.

### Key Features

-   **Targeted Accelerator Search**: Allows users to select one or more accelerators (Y Combinator, Techstars, 500 Global) to source companies.
-   **Full Pipeline Execution**: Runs the discovered accelerator companies through the entire data enrichment and founder ranking pipeline.
-   **Separate Checkpoints**: Manages checkpoints independently from the main pipeline, allowing users to resume accelerator-specific jobs.
-   **Results and Export**: Displays results in the same interactive tables and allows for CSV export, just like the main pipeline.

### APIs Used

-   **Internal Web Scraping & Discovery**: The initial discovery of accelerator companies is handled by an internal service (`AcceleratorCompanyDiscovery`) that processes public data from accelerator websites.
-   **Crunchbase, Apify, Anthropic, Serper APIs**: Once the initial list of companies is discovered, this pipeline uses the same set of external APIs as the main pipeline for data fusion, profile enrichment, ranking, and verification.

---

## 3. Market Analysis Tab

This tab offers on-demand, in-depth market intelligence for any company that has been discovered by either the main or accelerator pipelines.

### Key Features

-   **Company Selection**: Users can select any discovered company from a dropdown menu to initiate an analysis.
-   **Comprehensive Report Generation**: Generates a detailed market analysis report covering market size, growth projections (CAGR), competitive landscape, investment climate, and key trends.
-   **Quantitative & Qualitative Insights**: The report includes numerical scores for market timing and momentum, as well as narrative analysis of risks and opportunities.
-   **PDF Export**: The generated market analysis can be exported as a professionally formatted PDF document for easy sharing and archiving.

### APIs Used

-   **Perplexity API**: The primary engine for the **Market Analysis** feature. It conducts real-time, in-depth research to generate the comprehensive report.
-   **OpenAI API**: Used for supplementary **data extraction and analysis** to structure the information returned from market research into a clean, digestible format.
-   **Internal API**: The dropdown list of companies is populated by the application's internal `/api/companies/list` endpoint, which serves data from completed pipeline runs.