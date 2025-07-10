import React from 'react';
import './MarketAnalysis.css';

const MarketAnalysisSkeleton: React.FC = () => {
  return (
    <div className="card animate-pulse">
      <div className="h-8 bg-gray-200 rounded w-3/4 mb-6"></div>
      <div className="mb-6">
        <div className="h-6 bg-gray-200 rounded w-1/2 mb-2"></div>
        <div className="grid grid-cols-2 gap-4 text-sm text-gray-600">
          <div className="h-4 bg-gray-200 rounded w-1/3"></div>
          <div className="h-4 bg-gray-200 rounded w-1/4"></div>
        </div>
      </div>
      <div className="market-analysis-grid">
        <div className="analysis-card h-32"></div>
        <div className="analysis-card h-32"></div>
        <div className="analysis-card h-32"></div>
      </div>
    </div>
  );
};

export default MarketAnalysisSkeleton;
