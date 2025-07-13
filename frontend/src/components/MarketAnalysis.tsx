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
        
        <div className="market-analysis-controls">
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
          <div className="market-analysis-header">
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

          {/* Comprehensive Text Analysis */}
          {marketAnalysis.market_overview && (
            <div className="analysis-section">
              <h4 className="analysis-section-header">Market Overview</h4>
              <p className="analysis-text">{marketAnalysis.market_overview}</p>
            </div>
          )}

          {marketAnalysis.growth_drivers && (
            <div className="analysis-section">
              <h4 className="analysis-section-header">Growth Drivers</h4>
              <p className="analysis-text">{marketAnalysis.growth_drivers}</p>
            </div>
          )}

          {marketAnalysis.competitive_landscape && (
            <div className="analysis-section">
              <h4 className="analysis-section-header">Competitive Landscape</h4>
              <p className="analysis-text">{marketAnalysis.competitive_landscape}</p>
            </div>
          )}

          {marketAnalysis.timing_analysis && (
            <div className="analysis-section">
              <h4 className="analysis-section-header">Market Timing Analysis</h4>
              <p className="analysis-text">{marketAnalysis.timing_analysis}</p>
            </div>
          )}

          {marketAnalysis.technology_trends && (
            <div className="analysis-section">
              <h4 className="analysis-section-header">Technology Trends</h4>
              <p className="analysis-text">{marketAnalysis.technology_trends}</p>
            </div>
          )}

          {marketAnalysis.investment_climate && (
            <div className="analysis-section">
              <h4 className="analysis-section-header">Investment Climate</h4>
              <p className="analysis-text">{marketAnalysis.investment_climate}</p>
            </div>
          )}

          {marketAnalysis.risk_assessment && (
            <div className="analysis-section">
              <h4 className="analysis-section-header">Risk Assessment</h4>
              <p className="analysis-text">{marketAnalysis.risk_assessment}</p>
            </div>
          )}

          {marketAnalysis.strategic_recommendations && (
            <div className="analysis-section">
              <h4 className="analysis-section-header">Strategic Recommendations</h4>
              <p className="analysis-text">{marketAnalysis.strategic_recommendations}</p>
            </div>
          )}

          {/* Structured Insights */}
          <div className="insights-grid">
            {marketAnalysis.key_trends && marketAnalysis.key_trends.length > 0 && (
              <div className="insight-card">
                <h4 className="insight-header">Key Trends</h4>
                <ul className="insight-list">
                  {marketAnalysis.key_trends.map((trend, index) => (
                    <li key={index}>{trend}</li>
                  ))}
                </ul>
              </div>
            )}

            {marketAnalysis.major_players && marketAnalysis.major_players.length > 0 && (
              <div className="insight-card">
                <h4 className="insight-header">Major Players</h4>
                <ul className="insight-list">
                  {marketAnalysis.major_players.map((player, index) => (
                    <li key={index}>{player}</li>
                  ))}
                </ul>
              </div>
            )}

            {marketAnalysis.opportunities && marketAnalysis.opportunities.length > 0 && (
              <div className="insight-card">
                <h4 className="insight-header">Opportunities</h4>
                <ul className="insight-list">
                  {marketAnalysis.opportunities.map((opportunity, index) => (
                    <li key={index}>{opportunity}</li>
                  ))}
                </ul>
              </div>
            )}

            {marketAnalysis.threats && marketAnalysis.threats.length > 0 && (
              <div className="insight-card">
                <h4 className="insight-header">Threats</h4>
                <ul className="insight-list">
                  {marketAnalysis.threats.map((threat, index) => (
                    <li key={index}>{threat}</li>
                  ))}
                </ul>
              </div>
            )}

            {marketAnalysis.emerging_technologies && marketAnalysis.emerging_technologies.length > 0 && (
              <div className="insight-card">
                <h4 className="insight-header">Emerging Technologies</h4>
                <ul className="insight-list">
                  {marketAnalysis.emerging_technologies.map((tech, index) => (
                    <li key={index}>{tech}</li>
                  ))}
                </ul>
              </div>
            )}

            {marketAnalysis.barriers_to_entry && marketAnalysis.barriers_to_entry.length > 0 && (
              <div className="insight-card">
                <h4 className="insight-header">Barriers to Entry</h4>
                <ul className="insight-list">
                  {marketAnalysis.barriers_to_entry.map((barrier, index) => (
                    <li key={index}>{barrier}</li>
                  ))}
                </ul>
              </div>
            )}
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
