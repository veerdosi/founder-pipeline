import React from 'react';
import { PipelineResults } from '../interfaces';

interface StatusCardsProps {
  results: PipelineResults | null;
  downloadCSV: (type: 'companies' | 'founders') => void;
}

const StatusCards: React.FC<StatusCardsProps> = ({ results, downloadCSV }) => {
  if (!results) {
    return null;
  }

  return (
    <div className="card animate-fade-in">
      <h2>Results</h2>
      <div className="results-grid">
        <div className="result-card">
          <div className="result-number">{results.companies.length}</div>
          <div className="result-label mb-4">Companies Found</div>
          <button
            onClick={() => downloadCSV('companies')}
            className="btn btn-success"
          >
            💾 Save Companies CSV
          </button>
        </div>
        <div className="result-card">
          <div className="result-number">{results.founders.length}</div>
          <div className="result-label mb-4">Founders Analyzed</div>
          <button
            onClick={() => downloadCSV('founders')}
            className="btn btn-success"
            disabled={results.founders.length === 0}
          >
            💾 Save Founders CSV
          </button>
        </div>
      </div>
    </div>
  );
};

export default StatusCards;