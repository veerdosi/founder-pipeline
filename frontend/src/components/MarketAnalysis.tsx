import React from 'react';
import { Company, MarketAnalysisData } from '../interfaces';
import MarketAnalysisSkeleton from './MarketAnalysisSkeleton';
import ErrorDisplay from './ErrorDisplay';
import './MarketAnalysis.css';

interface MarketAnalysisProps {
  companies: Company[];
  selectedCompany: string;
  setSelectedCompany: (companyName: string) => void;
  marketAnalysis: MarketAnalysisData | null;
  isAnalyzing: boolean;
  analysisError: string | null;
  runMarketAnalysis: () => void;
  exportToPDF: () => void;
  clearAnalysisError: () => void;
}

const MarketAnalysis: React.FC<MarketAnalysisProps> = ({
  companies,
  selectedCompany,
  setSelectedCompany,
  marketAnalysis,
  isAnalyzing,
  analysisError,
  runMarketAnalysis,
  exportToPDF,
  clearAnalysisError,
}) => {
  return (
    <div className="animate-fade-in">
      <div className="card">
        <h2>Market Analysis</h2>
        <p>Generate comprehensive market analysis for any company from your latest pipeline results.</p>
        
        <div className="pipeline-grid">
          <div className="form-group">
            <label className="form-label">Select Company</label>
            {companies.length > 0 ? (
              <select
                value={selectedCompany}
                onChange={(e) => setSelectedCompany(e.target.value)}
                className="form-select"
                disabled={isAnalyzing}
              >
                <option value="">-- Select a company --</option>
                {companies.map(company => (
                  <option key={company.id} value={company.name}>
                    {company.name} - {company.sector} ({company.founded_year})
                  </option>
                ))}
              </select>
            ) : (
              <div className="form-select" style={{ color: 'var(--text-tertiary)'}}>
                No companies available. Run the pipeline first.
              </div>
            )}
          </div>
          <button
            onClick={runMarketAnalysis}
            disabled={isAnalyzing || !selectedCompany || companies.length === 0}
            className="btn btn-primary"
          >
            {isAnalyzing ? (
              <>
                <span className="animate-spin">🔄</span>
                Analyzing...
              </>
            ) : (
              '📈 Generate Analysis'
            )}
          </button>
        </div>
      </div>

      <ErrorDisplay error={analysisError} onClear={clearAnalysisError} />

      {isAnalyzing && <MarketAnalysisSkeleton />}

      {marketAnalysis && !isAnalyzing && (
        <div className="card animate-fade-in">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h2>Market Analysis Results</h2>
            <button onClick={exportToPDF} className="btn btn-success pdf-export-button">
              📄 Export PDF
            </button>
          </div>
          
          <div className="analysis-company-header">
            <h3 className="analysis-company-name">{marketAnalysis.company_name}</h3>
            <div className="analysis-company-details">
              <span>Sector: {marketAnalysis.sector}</span>
              <span>Founded: {marketAnalysis.founded_year}</span>
            </div>
          </div>

          <div className="market-analysis-grid">
            <div className="analysis-card">
              <h4 className="analysis-card-header">Market Size & Growth</h4>
              <div className="analysis-metric">
                <span>Market Size:</span>
                <span className="analysis-metric-value">${marketAnalysis.market_size_billion.toFixed(1)}B</span>
              </div>
              <div className="analysis-metric">
                <span>CAGR:</span>
                <span className="analysis-metric-value">{marketAnalysis.cagr_percent.toFixed(1)}%</span>
              </div>
              <div className="analysis-metric">
                <span>Stage:</span>
                <span className="analysis-metric-value" style={{textTransform: 'capitalize'}}>{marketAnalysis.market_stage}</span>
              </div>
            </div>

            <div className="analysis-card">
              <h4 className="analysis-card-header">Sentiment & Timing</h4>
              <div className="analysis-metric">
                <span>Timing Score:</span>
                <span className="analysis-metric-value">{marketAnalysis.timing_score.toFixed(1)}/5</span>
              </div>
              <div className="analysis-metric">
                <span>US Sentiment:</span>
                <span className="analysis-metric-value">{marketAnalysis.us_sentiment.toFixed(1)}/5</span>
              </div>
              <div className="analysis-metric">
                <span>Asia Sentiment:</span>
                <span className="analysis-metric-value">{marketAnalysis.sea_sentiment.toFixed(1)}/5</span>
              </div>
              <div className="analysis-metric">
                <span>Momentum:</span>
                <span className="analysis-metric-value">{marketAnalysis.momentum_score.toFixed(1)}/5</span>
              </div>
            </div>

            <div className="analysis-card">
              <h4 className="analysis-card-header">Competition & Funding</h4>
              <div className="analysis-metric">
                <span>Competitors:</span>
                <span className="analysis-metric-value">{marketAnalysis.competitor_count}</span>
              </div>
              <div className="analysis-metric">
                <span>Total Funding:</span>
                <span className="analysis-metric-value">${marketAnalysis.total_funding_billion.toFixed(1)}B</span>
              </div>
            </div>
          </div>

          <div className="analysis-footer">
            Generated on {new Date(marketAnalysis.analysis_date).toLocaleString()}
          </div>
        </div>
      )}
    </div>
  );
};

export default MarketAnalysis;
