# Initiation Pipeline: AI-Powered Company Enrichment and Founder Ranking

This repository contains the source code for the Initiation Pipeline, a streamlined 3-stage data processing system designed to enrich Crunchbase company data and rank their founders. The system processes direct Crunchbase CSV inputs, adds market analysis and sector classification, then discovers and ranks founders using an advanced L1-L10 experience classification framework.

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

## 🏗️ System Architecture

The Initiation Pipeline is composed of a Python backend and a React frontend, which work together to provide a seamless user experience.

### Backend

The backend is built with **FastAPI** and serves as the engine for all data processing and analysis. Its key modules include:

- **Company Enrichment**: Processes Crunchbase CSV files, filters active companies, and enriches with sector classification, market analysis, and AI-powered funding stage detection.
- **Profile Enrichment**: Finds and scrapes LinkedIn profiles to gather founders' professional history.
- **Founder Ranking Service**: Uses AI to classify founders based on the L1-L10 experience framework.
- **Market Analysis**: Generates market reports using Perplexity AI.
- **Funding Stage Detection**: Uses ChatGPT to accurately determine company funding stages.
- **Checkpoint Manager**: Saves and loads the pipeline's state, enabling resumability.

### Frontend

The frontend is a **React** application built with **Vite** and **TypeScript**. It provides a user-friendly interface for:

- **Pipeline Execution**: Initiating company discovery and founder ranking tasks.
- **Results Visualization**: Displaying discovered companies and ranked founders in interactive tables.
- **Market Analysis**: Generating detailed market analysis reports for selected companies.
- **Checkpoint Management**: Resuming the pipeline from a previously saved state.

## 📊 Data Pipeline Workflow

The data pipeline processes information in 3 sequential stages:

### Stage 1: Company Enrichment

- **Input**: Direct Crunchbase CSV files (automatically selected by year from `/input` folder)
- **Processing**:
  - Filters out closed companies
  - Maps CSV data to internal Company model
  - Adds AI-powered sector classification
  - Performs market analysis using Perplexity AI
  - Detects accurate funding stages using ChatGPT
- **Output**: Companies CSV with enriched data including market metrics

### Stage 2: Profile Enrichment

- **Input**: Enriched company data from Stage 1
- **Processing**:
  - Identifies company founders from company data
  - Scrapes LinkedIn profiles for professional and educational history
  - Enhances profiles with media coverage and financial data
- **Output**: Enriched founder profiles linked to companies

### Stage 3: Founder Ranking

- **Input**: Enriched founder profiles from Stage 2
- **Processing**: AI-powered ranking service classifies each founder on the L1-L10 experience scale
- **Output**: Founders CSV with ranking data and confidence scores

### Results Export

Both companies and founders data are exported to CSV files with exact column specifications for further analysis.

## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

- **Python 3.9+**
- **Node.js 16+** and **npm**
- **API Keys** for the services listed in the `.env.example` file.

### Installation

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/your-username/initiation-pipeline.git
    cd initiation-pipeline
    ```

2.  **Configure Environment Variables:**
    Create a `.env` file from the example and add your API keys. This file is crucial for the application to connect to external services.

    ```bash
    cp .env.example .env
    ```

    _Note: The pipeline will not function correctly without valid API keys._

3.  **Set Up the Backend:**
    It is highly recommended to use a Python virtual environment to manage dependencies.

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

4.  **Set Up the Frontend:**
    Navigate to the `frontend` directory and install the necessary npm packages.
    ```bash
    cd frontend
    npm install
    ```

### Running the Application

1.  **Start the Backend Server:**
    With your virtual environment activated (`source venv/bin/activate`), run the following command. The backend API will be accessible at `http://localhost:8000`.

    ```bash
    uvicorn backend.api.main:app --reload
    ```

2.  **Start the Frontend Application:**
    In a separate terminal, navigate to the `frontend` directory and start the development server. The web interface will be available at `http://localhost:3000`.
    ```bash
    cd frontend
    npm run dev
    ```

## 💻 Usage

The web interface is organized into two main sections: **Pipeline** and **Market Analysis**.

### Pipeline Tab

This tab is the primary interface for processing and ranking companies.

- **To run a new pipeline:** Select "Start Fresh," specify a founding year (which automatically selects the corresponding Crunchbase CSV file from the `/input` folder), and click "Run Pipeline."
- **To resume a pipeline:** If a previous run was interrupted, select "Resume from Checkpoint," choose a checkpoint from the dropdown menu, and click "Run Pipeline." The system will pick up from the last completed stage.
- **Input Data:** The pipeline automatically processes Crunchbase CSV files (e.g., `2024companies.csv`) from the `/input` folder, filtering out closed companies and enriching the data.
- **Viewing and Exporting Results:** Once the pipeline completes, the enriched companies and ranked founders will appear in sortable tables. Companies CSV is exported after Stage 1, and Founders CSV after Stage 3.

### Market Analysis Tab

This tab allows you to generate in-depth market reports for any company discovered by the pipeline.

- **Select a company** from the dropdown list.
- Click **"Run Market Analysis"** to generate a comprehensive report, including market size, growth trends, and competitive landscape.
- The report will be displayed on the screen, and you can **export it to a PDF** for offline use.

## 🔑 Required API Keys

The following API keys are required and must be configured in the `.env` file:

- `ANTHROPIC_API_KEY`: For founder ranking using Claude models.
- `PERPLEXITY_API_KEY`: For market analysis and real-time research.
- `OPENAI_API_KEY`: For sector classification and funding stage detection.
- `APIFY_API_KEY`: For scraping LinkedIn profiles.
- `SERPAPI_KEY`: For real-time Google Search validation.

## 📁 Project Structure

The project is organized into a backend and a frontend directory, with additional folders for supporting files.

```
/
├── backend/
│   ├── api/            # FastAPI application, defines API endpoints.
│   ├── core/           # Core business logic of the pipeline.
│   │   ├── analysis/   # Market analysis, sector classification, funding stage detection.
│   │   ├── data/       # Company enrichment and profile enrichment services.
│   │   └── ranking/    # Founder ranking service and related prompts.
│   └── utils/          # Utility functions (e.g., checkpointing, rate limiting).
├── frontend/
│   ├── src/
│   │   ├── components/ # Reusable React components for the UI.
│   │   ├── interfaces.ts # TypeScript type definitions for data models.
│   │   └── App.tsx     # Main application component and routing setup.
├── input/              # Crunchbase CSV files organized by year (e.g., 2024companies.csv).
├── output/             # Default directory for exported CSV and PDF files.
├── checkpoints/        # Stores saved pipeline states for resumability.
└── requirements.txt    # Python dependencies for the backend.
```

## 🛠️ Technologies Used

- **Backend**: Python, FastAPI, Pandas, httpx, Anthropic API, OpenAI API, Perplexity API
- **Frontend**: React, TypeScript, Vite, TanStack Query, Recharts, Lucide React
- **Data Storage**: Filesystem for checkpointing and CSV exports.

## 🚨 Troubleshooting

If you encounter issues, consider the following solutions:

- **Missing API Keys**: The application will fail if API keys are missing. Ensure all required keys are present in the `.env` file and that it is correctly named.
- **CORS Errors**: If the frontend cannot connect to the backend, you may see a CORS error in your browser's developer console. Ensure the backend is running on `http://localhost:8000` and that the `allow_origins` setting in `backend/api/main.py` includes the frontend's address (`http://localhost:3000`).
- **Rate Limiting**: If you encounter `429 Too Many Requests` errors, the application is making too many calls to an external API. You can adjust the rate limits in `backend/core/config.py`.
- **Python Dependencies**: If you encounter `ModuleNotFoundError` errors, ensure your virtual environment is activated and that all dependencies have been installed correctly with `pip install -r requirements.txt`.
- **Frontend Not Connecting**: Verify that the backend server is running and accessible at `http://localhost:8000`. Check for any error messages in the terminal where you launched the backend.
