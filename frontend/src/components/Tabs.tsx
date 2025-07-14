import React from 'react';
import './Tabs.css';

interface TabsProps {
  activeTab: 'pipeline' | 'accelerators' | 'market-analysis';
  setActiveTab: (tab: 'pipeline' | 'accelerators' | 'market-analysis') => void;
}

const Tabs: React.FC<TabsProps> = ({ activeTab, setActiveTab }) => {
  return (
    <div className="tabs-container">
      <button
        onClick={() => setActiveTab('pipeline')}
        className={`tab-button ${activeTab === 'pipeline' ? 'active' : ''}`}
      >
        📊 Pipeline
      </button>
      <button
        onClick={() => setActiveTab('accelerators')}
        className={`tab-button ${activeTab === 'accelerators' ? 'active' : ''}`}
      >
        🚀 Accelerators
      </button>
      <button
        onClick={() => setActiveTab('market-analysis')}
        className={`tab-button ${activeTab === 'market-analysis' ? 'active' : ''}`}
      >
        📈 Market Analysis
      </button>
    </div>
  );
};

export default Tabs;
