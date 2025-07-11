# Initiation Pipeline: AI-Powered Founder Discovery and Ranking

A comprehensive, multi-source data processing pipeline to discover, analyze, and rank early-stage AI companies and their founders. This system uses an advanced L1-L10 experience classification framework, real-time data enhancement, and multi-source verification to build detailed founder datasets.

## Table of Contents

- [Key Features](#-key-features)
- [System Architecture](#️-system-architecture)
- [Data Pipeline Workflow](#-data-pipeline-workflow)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [API Keys](#-required-api-keys)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)

## ✨ Key Features

- **Automated Company Discovery**: Identifies early-stage AI companies from multiple sources like Exa and Crunchbase.
- **Comprehensive Data Enrichment**: Gathers detailed information on companies and founders, including financials, educational background, and professional history from sources like LinkedIn, SEC EDGAR, and university databases.
- **AI-Powered Founder Ranking**: Utilizes a sophisticated L1-L10 framework to classify founder experience, leveraging large language models (Claude Sonnet 4) for nuanced analysis.
- **Multi-Source Verification**: Cross-references data from various authoritative sources to ensure accuracy and assign confidence scores.
- **Web-Based Interface**: A modern React frontend provides a user-friendly interface for running the pipeline, viewing results, and performing market analysis.
- **Smart Checkpointing**: Robust checkpointing system allows for resuming interrupted pipelines, saving time and resources.
- **Market Analysis**: On-demand market analysis for discovered companies, providing insights into market size, growth, and competition.

## 🏗️ System Architecture

The Initiation Pipeline is composed of two main components: a Python backend and a React frontend.

### Backend

The backend is built with **FastAPI** and is responsible for the core data processing and analysis. It consists of several key modules:

- **Data Collectors**: A set of services that gather data from various external APIs and web sources.
- **Data Fusion Service**: Merges and de-duplicates data from multiple sources to create a unified view of each company.
- **Profile Enrichment**: Finds and scrapes LinkedIn profiles for founders to gather professional history.
- **Founder Ranking Service**: Uses AI to classify founders based on the L1-L10 framework.
- **Market Analysis**: Generates market reports using Perplexity AI.
- **Checkpoint Manager**: Saves and loads the state of the pipeline at various stages.

### Frontend

The frontend is a **React** application built with **Vite**. It provides a user interface for:

- **Running the Pipeline**: Initiating company discovery and founder ranking tasks.
- **Viewing Results**: Displaying discovered companies and ranked founders in a clear and interactive way.
- **Market Analysis**: Selecting a company and generating a detailed market analysis report.
- **Checkpoint Management**: Resuming the pipeline from a previous checkpoint.

## 📊 Data Pipeline Workflow

The data flows through the system in the following stages:

1.  **Company Discovery**: The pipeline starts by discovering a list of companies based on user-defined criteria (e.g., industry, location, founding year).
2.  **Data Fusion & Enhancement**: The initial list of companies is enriched with additional data from Crunchbase and other sources.
3.  **Founder Profile Enrichment**: The system identifies the founders of each company and scrapes their LinkedIn profiles for detailed professional and educational history.
4.  **Founder Intelligence Collection**: Further intelligence is gathered on each founder to prepare for ranking.
5.  **Founder Ranking**: The enriched founder profiles are passed to the AI-powered ranking service, which classifies each founder on the L1-L10 scale.
6.  **Results Display**: The final results, including the list of companies and ranked founders, are displayed in the web interface and can be exported to CSV.

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Node.js 16+ and npm
- API keys for the services listed in the `.env.example` file.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/initiation-pipeline.git
    cd initiation-pipeline
    ```

2.  **Set up the environment:**
    ```bash
    cp .env.example .env
    ```
    Edit the `.env` file and add your API keys.

3.  **Set up the backend:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

4.  **Set up the frontend:**
    ```bash
    cd frontend
    npm install
    ```

### Running the Application

1.  **Start the backend server:**
    ```bash
    source venv/bin/activate
    uvicorn backend.api.main:app --reload
    ```

2.  **Start the frontend development server:**
    ```bash
    cd frontend
    npm run dev
    ```

The application will be available at `http://localhost:3000`.

## 💻 Usage

The web interface provides two main tabs: "Pipeline" and "Market Analysis".

### Pipeline Tab

- **Run a new pipeline:** Select "Start Fresh", choose a founding year, and click "Run Pipeline".
- **Resume a pipeline:** Select "Resume from Checkpoint", choose a checkpoint from the dropdown, and click "Run Pipeline".
- **View results:** Once the pipeline is complete, the discovered companies and ranked founders will be displayed in sortable tables.
- **Export results:** Click the "Export Companies" or "Export Founders" buttons to download the results as CSV files.

### Market Analysis Tab

- **Select a company:** Choose a company from the dropdown list.
- **Run analysis:** Click "Run Market Analysis" to generate a report.
- **View report:** The market analysis report will be displayed with key metrics and scores.
- **Export to PDF:** Click "Export to PDF" to save the report.

## 🔑 Required API Keys

The following API keys are required for the pipeline to function correctly. They should be placed in the `.env` file.

- `ANTHROPIC_API_KEY`: For founder ranking using Claude Sonnet 4.
- `PERPLEXITY_API_KEY`: For market analysis and real-time fact-checking.
- `OPENAI_API_KEY`: For data extraction and analysis.
- `EXA_API_KEY`: For company discovery and web search.
- `CRUNCHBASE_API_KEY`: For financial data and company intelligence.
- `APIFY_API_KEY`: For scraping LinkedIn profiles.
- `SERPER_API_KEY`: For real-time Google Search validation.

## 📁 Project Structure

```
/
├── backend/
│   ├── api/            # FastAPI application
│   ├── core/           # Core business logic
│   │   ├── analysis/   # Data analysis modules
│   │   ├── data/       # Data collection and fusion
│   │   └── ranking/    # Founder ranking service
│   └── utils/          # Utility functions
├── frontend/
│   ├── src/
│   │   ├── components/ # React components
│   │   ├── interfaces.ts # TypeScript interfaces
│   │   └── App.tsx     # Main application component
├── jupyter/            # Jupyter notebooks for data exploration
├── output/             # Default directory for exported files
├── checkpoints/        # Saved pipeline checkpoints
└── requirements.txt    # Python dependencies
```

## 🛠️ Technologies Used

- **Backend**: Python, FastAPI, Pandas, httpx, Anthropic API, OpenAI API, Exa API
- **Frontend**: React, TypeScript, Vite, TanStack Query, Recharts, Lucide React
- **Data Storage**: Filesystem for checkpointing and CSV/JSON for exports.

## 🚨 Troubleshooting

- **Missing API Keys**: Ensure all required API keys are present in the `.env` file.
- **Rate Limiting**: If you encounter 429 errors, adjust the `REQUESTS_PER_MINUTE` and `CONCURRENT_REQUESTS` settings in `backend/core/config.py`.
- **Frontend Not Connecting**: Make sure the backend server is running on `http://localhost:8000`.

## 📄 License

This project is licensed under the MIT License.