# AI Founder Discovery & Ranking Pipeline

A comprehensive web application for discovering early-stage AI companies and building detailed founder datasets with **L1-L10 experience classification**, LinkedIn profiles, company intelligence, and real-time verification.

## 🎯 Primary Focus: L1-L10 Founder Classification

This pipeline implements a sophisticated **L1-L10 founder experience framework** based on Carnegie Mellon research, automatically classifying founders from L1 (Nascent) to L10 (Legendary Entrepreneurs) with multi-source verification, all managed through a user-friendly web interface.

## ✨ Features

- **Web-Based UI**: A modern React frontend to manage discovery and ranking jobs.
- **L1-L10 Founder Classification**: Automated ranking using Claude Sonnet 4 with specific, verifiable thresholds.
- **Real-Time Verification**: Perplexity-powered fact-checking for stale data before ranking.
- **Multi-Source Company Discovery**: Leverages Exa, Crunchbase, and other data sources for comprehensive discovery.
- **LinkedIn Profile Enrichment**: Fetches and structures founder profiles from LinkedIn using Apify.
- **Advanced Data Fusion**: Validates and combines data from multiple sources for a holistic view.
- **Smart Checkpointing**: The backend uses a robust checkpointing system to resume interrupted operations without losing progress.

### L1-L10 Ranking Intelligence

- **L10**: Multiple IPOs >$1B (Legendary Entrepreneurs)
- **L9**: 1 IPO >$1B, building second company (Transformational Leaders)
- **L8**: Built 1+ unicorn companies (Proven Unicorn Builders)
- **L7**: 2+ exits >$100M (Elite Serial Entrepreneurs)
- **L6**: Groundbreaking innovation recognition (Market Innovators)
- **L5**: Companies with >$50M funding (Growth-Stage Entrepreneurs)
- **L4**: $10M-$100M exits or C-level roles (Proven Operators)
- **L3**: 10+ years experience, PhD, or senior roles (Technical Veterans)
- **L2**: Accelerator graduates, 2-5 years experience (Early-Stage)
- **L1**: <2 years experience, first-time founders (Nascent)

## 🏗️ Architecture

The application is a monorepo with two main parts:

- **`frontend/`**: A **React (Vite)** single-page application that provides the user interface.
- **`backend/`**: A **FastAPI** server that exposes a REST API to run the data processing and analysis pipelines.

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.9+
- Node.js 16+ and npm
- API keys for all required services (see below)

### 2. Setup

First, clone the repository and set up your environment variables.

```bash
git clone <repository_url>
cd <repository_folder>

# Copy the example environment file
cp .env.example .env

# Edit the .env file with your API keys
# nano .env or code .env
```

### 3. Run the Backend

Open a terminal and run the following commands to start the FastAPI server.

```bash
# Install Python dependencies from the root directory
pip install -r requirements.txt

# Start the server (will run on http://localhost:8000)
uvicorn backend.web:app --reload
```

### 4. Run the Frontend

Open a **new terminal** and run these commands to start the React development server.

```bash
# Navigate to the frontend directory
cd frontend

# Install Node.js dependencies
npm install

# Start the development server (will open http://localhost:3000)
npm run dev
```

Your browser should open to `http://localhost:3000`, and the application will be ready to use. The frontend is automatically proxied to the backend, so API calls will work seamlessly.

## 💻 How to Use the Application

1.  **Dashboard**: Provides an at-a-glance overview of your discovery and ranking activities.
2.  **Company Discovery**: Configure parameters like company limit, categories, and sources. Start a discovery job and see the results in the table. You can export the discovered companies to a CSV file.
3.  **Founder Ranking**: Upload a CSV of founders (generated from the discovery step) to begin the L1-L10 classification process. The results will show each founder's assigned level, confidence score, and the reasoning behind the classification.

## 🔑 API Keys Required

You must provide the following API keys in your `.env` file for the pipeline to function correctly.

### Core APIs (Required)

- **`EXA_API_KEY`**: Company discovery and web search.
- **`OPENAI_API_KEY`**: AI-powered analysis and data extraction.
- **`SERPER_API_KEY`**: Real-time Google Search for validation.
- **`APIFY_API_KEY`**: LinkedIn profile scraping.
- **`ANTHROPIC_API_KEY`**: Claude Sonnet 4 for L1-L10 founder classification.
- **`PERPLEXITY_API_KEY`**: Real-time verification and market analysis.

### Data Source APIs (Recommended)

- **`CRUNCHBASE_API_KEY`**: Enhanced company data and validation.

## 🔧 Troubleshooting

- **API Key Errors**: The application will fail if any required API keys are missing or invalid. Double-check your `.env` file. The backend terminal will show which key is missing on startup.
- **CORS Errors**: Ensure the backend is running and the frontend's proxy in `vite.config.ts` is correctly configured for the backend's address (`http://localhost:8000`).
- **Long-Running Tasks**: Discovery and ranking can take time. Check the terminal running the backend for progress logs and potential errors.

## License

MIT License
