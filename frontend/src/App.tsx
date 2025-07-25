import { useState, useEffect } from 'react';
import Header from './components/Header';
import Tabs from './components/Tabs';
import Pipeline from './components/Pipeline';
import MarketAnalysis from './components/MarketAnalysis';
import { PipelineStatus, PipelineResults, Company, MarketAnalysisData, CheckpointInfo } from './interfaces';
import './App.css';

export default function App() {
  const [activeTab, setActiveTab] = useState<'pipeline' | 'market-analysis'>('pipeline');
  const [year, setYear] = useState('2025');
  const [isRunning, setIsRunning] = useState(false);
  const [steps, setSteps] = useState<PipelineStatus[]>([]);
  const [results, setResults] = useState<PipelineResults | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [startMode, setStartMode] = useState<'fresh' | 'resume'>('fresh');
  const [checkpoints, setCheckpoints] = useState<CheckpointInfo[]>([]);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<string>('');


  // Market Analysis State
  const [companies, setCompanies] = useState<Company[]>([]);
  const [selectedCompany, setSelectedCompany] = useState<string>('');
  const [marketAnalysis, setMarketAnalysis] = useState<MarketAnalysisData | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisError, setAnalysisError] = useState<string | null>(null);

  const updateStep = (stepName: string, status: PipelineStatus['status'], message?: string) => {
    setSteps(prev => {
      const existing = prev.find(s => s.step === stepName);
      if (existing) {
        existing.status = status;
        existing.message = message;
        return [...prev];
      }
      return [...prev, { step: stepName, status, message }];
    });
  };

  const initializePipelineSteps = (mode: 'fresh' | 'resume', checkpointData?: CheckpointInfo) => {
    const baseSteps = [
      { step: 'Company Enrichment', status: 'pending' as const, message: 'Enriching company data with funding, sector, and details' },
      { step: 'Profile Enrichment', status: 'pending' as const, message: 'Enriching founder profiles and LinkedIn data' },
      { step: 'Founder Ranking', status: 'pending' as const, message: 'Ranking founders based on AI analysis' }
    ];

    if (mode === 'resume' && checkpointData) {
      const stageMap: { [key: string]: number } = {
        'enriched_companies': 0,
        'profiles': 1,
        'rankings': 2
      };
      
      const completedStageIndex = stageMap[checkpointData.latest_stage] || -1;
      
      return baseSteps.map((step, index) => ({
        ...step,
        status: index <= completedStageIndex ? 'completed' as const : 'pending' as const,
        message: index <= completedStageIndex 
          ? `✅ Completed (from checkpoint)` 
          : step.message
      }));
    }
    
    return baseSteps;
  };


  const loadCheckpoints = async () => {
    try {
      const response = await fetch(`/api/checkpoints?job_type=main_pipeline`);
      if (response.ok) {
        const checkpointList = await response.json();
        setCheckpoints(checkpointList);
        if (checkpointList.length > 0 && !selectedCheckpoint) {
          // Sort by completion percentage (highest first) and pick the most advanced
          const sortedByCompletion = [...checkpointList].sort((a, b) => b.completion_percentage - a.completion_percentage);
          setSelectedCheckpoint(sortedByCompletion[0].id);
        }
      }
    } catch (error) {
      console.error('Failed to load checkpoints:', error);
    }
  };

  useEffect(() => {
    loadCheckpoints();
    const interval = setInterval(loadCheckpoints, 30000);
    return () => clearInterval(interval);
  }, [activeTab]); // Reload checkpoints when tab changes

  useEffect(() => {
    if (startMode === 'resume' && checkpoints.length > 0 && !selectedCheckpoint) {
      // Sort by completion percentage (highest first) and pick the most advanced
      const sortedByCompletion = checkpoints.sort((a, b) => b.completion_percentage - a.completion_percentage);
      setSelectedCheckpoint(sortedByCompletion[0].id);
    }
  }, [startMode, checkpoints]);

  useEffect(() => {
    if (activeTab === 'market-analysis') {
      loadCompanies();
    }
  }, [activeTab]);

  const loadCompanies = async () => {
    try {
      const response = await fetch('/api/companies/list');
      if (response.ok) {
        const companiesList = await response.json();
        setCompanies(companiesList);
        if (companiesList.length > 0 && !selectedCompany) {
          setSelectedCompany(companiesList[0].name);
        }
      } else {
        console.error('Failed to load companies');
      }
    } catch (error) {
      console.error('Failed to load companies:', error);
    }
  };

  const runMarketAnalysis = async () => {
    if (!selectedCompany) {
      setAnalysisError('Please select a company');
      return;
    }

    setIsAnalyzing(true);
    setAnalysisError(null);
    setMarketAnalysis(null);

    try {
      const response = await fetch(`/api/companies/${encodeURIComponent(selectedCompany)}/market-analysis`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`);
      }

      const result = await response.json();
      if (result.status === 'success') {
        setMarketAnalysis(result.data);
      } else {
        throw new Error(result.message || 'Analysis failed');
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error occurred';
      setAnalysisError(message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const exportToPDF = () => {
    if (!marketAnalysis) return;
    
    const printWindow = window.open('', '_blank');
    if (!printWindow) return;
    
    // Helper function to safely display text or fallback
    const safeText = (text: string | undefined) => text || 'Not available';
    
    // Helper function to render list items
    const renderList = (items: string[] | undefined) => {
      if (!items || items.length === 0) return '<li>Not available</li>';
      return items.map(item => `<li>${item}</li>`).join('');
    };
    
    printWindow.document.write(`
      <html>
        <head>
          <title>Market Analysis - ${marketAnalysis.company_name}</title>
          <style>
            body { 
              font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
              margin: 30px; 
              line-height: 1.6; 
              color: #333;
            }
            .header { 
              text-align: center; 
              margin-bottom: 40px; 
              border-bottom: 3px solid #2563eb;
              padding-bottom: 20px;
            }
            .company-name { 
              color: #2563eb; 
              margin-bottom: 10px; 
            }
            .section { 
              margin-bottom: 30px; 
              break-inside: avoid;
            }
            .section h3 { 
              color: #2563eb; 
              border-bottom: 2px solid #e5e7eb; 
              padding-bottom: 8px; 
              margin-bottom: 15px;
            }
            .metrics-grid {
              display: grid;
              grid-template-columns: repeat(2, 1fr);
              gap: 15px;
              margin-bottom: 20px;
            }
            .metric { 
              display: flex; 
              justify-content: space-between; 
              margin: 8px 0; 
              padding: 8px 12px;
              background: #f8fafc;
              border-radius: 6px;
            }
            .score { 
              font-weight: bold; 
              color: #1d4ed8;
            }
            .text-content {
              background: #f9fafb;
              padding: 20px;
              border-radius: 8px;
              margin: 15px 0;
              border-left: 4px solid #2563eb;
            }
            .insights-grid {
              display: grid;
              grid-template-columns: repeat(2, 1fr);
              gap: 20px;
              margin-top: 20px;
            }
            .insight-card {
              background: #f8fafc;
              padding: 15px;
              border-radius: 8px;
              border: 1px solid #e5e7eb;
            }
            .insight-card h4 {
              color: #2563eb;
              margin-bottom: 10px;
              border-bottom: 1px solid #e5e7eb;
              padding-bottom: 5px;
            }
            .insight-card ul {
              margin: 0;
              padding-left: 20px;
            }
            .insight-card li {
              margin-bottom: 5px;
            }
            .footer {
              margin-top: 40px;
              padding-top: 20px;
              border-top: 2px solid #e5e7eb;
              text-align: center;
              color: #6b7280;
              font-size: 0.9em;
            }
            @media print {
              .section { page-break-inside: avoid; }
              .insights-grid { grid-template-columns: 1fr; }
              .metrics-grid { grid-template-columns: 1fr; }
            }
          </style>
        </head>
        <body>
          <div class="header">
            <h1>Comprehensive Market Analysis Report</h1>
            <h2 class="company-name">${marketAnalysis.company_name}</h2>
            <p><strong>Sector:</strong> ${marketAnalysis.sector} | <strong>Founded:</strong> ${marketAnalysis.founded_year}</p>
            <p>Generated on ${new Date(marketAnalysis.analysis_date).toLocaleDateString()}</p>
          </div>
          
          <div class="section">
            <h3>📊 Key Metrics Summary</h3>
            <div class="metrics-grid">
              <div class="metric">
                <span>Market Size:</span>
                <span class="score">$${marketAnalysis.market_size_billion.toFixed(1)}B</span>
              </div>
              <div class="metric">
                <span>CAGR:</span>
                <span class="score">${marketAnalysis.cagr_percent.toFixed(1)}%</span>
              </div>
              <div class="metric">
                <span>Market Stage:</span>
                <span class="score">${marketAnalysis.market_stage}</span>
              </div>
              <div class="metric">
                <span>Timing Score:</span>
                <span class="score">${marketAnalysis.timing_score.toFixed(1)}/5</span>
              </div>
              <div class="metric">
                <span>US Sentiment:</span>
                <span class="score">${marketAnalysis.us_sentiment.toFixed(1)}/5</span>
              </div>
              <div class="metric">
                <span>Asia Sentiment:</span>
                <span class="score">${marketAnalysis.sea_sentiment.toFixed(1)}/5</span>
              </div>
              <div class="metric">
                <span>Competitors:</span>
                <span class="score">${marketAnalysis.competitor_count}</span>
              </div>
              <div class="metric">
                <span>Total Funding:</span>
                <span class="score">$${marketAnalysis.total_funding_billion.toFixed(1)}B</span>
              </div>
            </div>
          </div>

          ${marketAnalysis.market_overview ? `
          <div class="section">
            <h3>🌍 Market Overview</h3>
            <div class="text-content">
              ${safeText(marketAnalysis.market_overview).replace(/\n/g, '<br>')}
            </div>
          </div>
          ` : ''}

          ${marketAnalysis.growth_drivers ? `
          <div class="section">
            <h3>📈 Growth Drivers</h3>
            <div class="text-content">
              ${safeText(marketAnalysis.growth_drivers).replace(/\n/g, '<br>')}
            </div>
          </div>
          ` : ''}

          ${marketAnalysis.competitive_landscape ? `
          <div class="section">
            <h3>🏟️ Competitive Landscape</h3>
            <div class="text-content">
              ${safeText(marketAnalysis.competitive_landscape).replace(/\n/g, '<br>')}
            </div>
          </div>
          ` : ''}

          ${marketAnalysis.timing_analysis ? `
          <div class="section">
            <h3>⏰ Market Timing Analysis</h3>
            <div class="text-content">
              ${safeText(marketAnalysis.timing_analysis).replace(/\n/g, '<br>')}
            </div>
          </div>
          ` : ''}

          ${marketAnalysis.technology_trends ? `
          <div class="section">
            <h3>💡 Technology Trends</h3>
            <div class="text-content">
              ${safeText(marketAnalysis.technology_trends).replace(/\n/g, '<br>')}
            </div>
          </div>
          ` : ''}

          ${marketAnalysis.investment_climate ? `
          <div class="section">
            <h3>💰 Investment Climate</h3>
            <div class="text-content">
              ${safeText(marketAnalysis.investment_climate).replace(/\n/g, '<br>')}
            </div>
          </div>
          ` : ''}

          ${marketAnalysis.risk_assessment ? `
          <div class="section">
            <h3>⚠️ Risk Assessment</h3>
            <div class="text-content">
              ${safeText(marketAnalysis.risk_assessment).replace(/\n/g, '<br>')}
            </div>
          </div>
          ` : ''}

          ${marketAnalysis.strategic_recommendations ? `
          <div class="section">
            <h3>🎯 Strategic Recommendations</h3>
            <div class="text-content">
              ${safeText(marketAnalysis.strategic_recommendations).replace(/\n/g, '<br>')}
            </div>
          </div>
          ` : ''}

          <div class="section">
            <h3>📋 Market Insights</h3>
            <div class="insights-grid">
              ${marketAnalysis.key_trends && marketAnalysis.key_trends.length > 0 ? `
              <div class="insight-card">
                <h4>🔥 Key Trends</h4>
                <ul>${renderList(marketAnalysis.key_trends)}</ul>
              </div>
              ` : ''}
              
              ${marketAnalysis.major_players && marketAnalysis.major_players.length > 0 ? `
              <div class="insight-card">
                <h4>🏢 Major Players</h4>
                <ul>${renderList(marketAnalysis.major_players)}</ul>
              </div>
              ` : ''}
              
              ${marketAnalysis.opportunities && marketAnalysis.opportunities.length > 0 ? `
              <div class="insight-card">
                <h4>🚀 Opportunities</h4>
                <ul>${renderList(marketAnalysis.opportunities)}</ul>
              </div>
              ` : ''}
              
              ${marketAnalysis.threats && marketAnalysis.threats.length > 0 ? `
              <div class="insight-card">
                <h4>⚡ Threats</h4>
                <ul>${renderList(marketAnalysis.threats)}</ul>
              </div>
              ` : ''}
              
              ${marketAnalysis.emerging_technologies && marketAnalysis.emerging_technologies.length > 0 ? `
              <div class="insight-card">
                <h4>🔬 Emerging Technologies</h4>
                <ul>${renderList(marketAnalysis.emerging_technologies)}</ul>
              </div>
              ` : ''}
              
              ${marketAnalysis.barriers_to_entry && marketAnalysis.barriers_to_entry.length > 0 ? `
              <div class="insight-card">
                <h4>🚧 Barriers to Entry</h4>
                <ul>${renderList(marketAnalysis.barriers_to_entry)}</ul>
              </div>
              ` : ''}
            </div>
          </div>

          <div class="footer">
            <p><strong>Analysis Quality:</strong> Confidence Score: ${(marketAnalysis.confidence_score * 100).toFixed(0)}% | Execution Time: ${marketAnalysis.execution_time.toFixed(1)}s</p>
            <p>This report was generated using AI-powered market analysis. Data should be validated with additional sources.</p>
          </div>
        </body>
      </html>
    `);
    printWindow.document.close();
    printWindow.print();
  };


  const runPipeline = async () => {
    if (startMode === 'fresh' && !year) {
      setError('Please select a year');
      return;
    }

    if (startMode === 'resume' && !selectedCheckpoint) {
      setError('Please select a checkpoint to resume from');
      return;
    }

    setIsRunning(true);
    setError(null);
    setResults(null);
    
    const checkpointData = checkpoints.find(cp => cp.id === selectedCheckpoint);
    const initialSteps = initializePipelineSteps(startMode, checkpointData);
    setSteps(initialSteps);
    

    try {
      if (startMode === 'resume') {
        const stageMap: { [key: string]: string } = {
          'enriched_companies': 'Company Enrichment',
          'profiles': 'Profile Enrichment', 
          'rankings': 'Founder Ranking'
        };
        
        const checkpointInfo = checkpoints.find(cp => cp.id === selectedCheckpoint);
        const nextStage = checkpointInfo ? stageMap[checkpointInfo.latest_stage] : null;
        
        if (nextStage) {
          updateStep(nextStage, 'running', `🔄 Resuming from ${checkpointInfo?.latest_stage}...`);
        }
        
        const resumeResponse = await fetch(`/api/pipeline/resume/${selectedCheckpoint}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' }
        });

        if (!resumeResponse.ok) {
          throw new Error(`Resume failed: ${resumeResponse.statusText}`);
        }

        const resumeResult = await resumeResponse.json();
        
        if (nextStage) {
          updateStep(nextStage, 'completed', '✅ Resume completed');
        }
        
        const [companiesData, foundersData] = await Promise.all([
          fetch('/api/companies').then(r => r.json()),
          fetch('/api/founders/rankings').then(r => r.json()).catch(() => [])
        ]);

        setResults({
          companies: companiesData,
          founders: foundersData,
          jobId: resumeResult.jobId
        });

        setSteps(prev => prev.map(step => ({ 
          ...step, 
          status: 'completed' as const, 
          message: '✅ Completed' 
        })));
        
      } else {
        updateStep('Company Enrichment', 'running', '🔄 Enriching company data...');
        
        const pipelineResponse = await fetch('/api/pipeline/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            year: parseInt(year)
          })
        });

        if (!pipelineResponse.ok) {
          throw new Error(`Pipeline failed: ${pipelineResponse.statusText}`);
        }

        const pipelineResult = await pipelineResponse.json();
        
        setSteps(prev => prev.map(step => ({ 
          ...step, 
          status: 'completed' as const, 
          message: '✅ Completed' 
        })));
        
        const [companiesData, foundersData] = await Promise.all([
          fetch('/api/companies').then(r => r.json()),
          fetch('/api/founders/rankings').then(r => r.json()).catch(() => [])
        ]);

        setResults({
          companies: companiesData,
          founders: foundersData,
          jobId: pipelineResult.jobId
        });
        
        try {
          await downloadCSV('companies');
          if (foundersData.length > 0) {
            await downloadCSV('founders');
          }
        } catch (exportError) {
          console.error('Auto-export failed:', exportError);
        }
      }

      await loadCheckpoints();

    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error occurred';
      setError(message);
      
      setSteps(prev => prev.map(step => 
        step.status === 'running' 
          ? { ...step, status: 'error' as const, message: `❌ ${message}` }
          : step
      ));
    } finally {
      setIsRunning(false);
    }
  };

  const downloadCSV = async (type: 'companies' | 'founders') => {
    try {
      const endpoint = type === 'companies' 
        ? '/api/companies/export'
        : '/api/founders/rankings/export';
      
      const response = await fetch(endpoint);
      if (!response.ok) throw new Error(`Export failed: ${response.statusText}`);
      
      const result = await response.json();
      
      if (result.status === 'success') {
        setSuccess(`✅ ${result.message}`);
        setTimeout(() => setSuccess(null), 5000);
      } else {
        throw new Error(result.message || 'Export failed');
      }
    } catch (err) {
      setError(`Export failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
    }
  };

  const clearError = () => setError(null);
  const clearAnalysisError = () => setAnalysisError(null);

  return (
    <div className="container">
      <Header />
      <p className="app-header-subtitle">AI-powered company and founder discovery platform</p>
      <Tabs activeTab={activeTab} setActiveTab={setActiveTab} />

      <main className="app-main-content">
        {activeTab === 'pipeline' && (
          <Pipeline
            year={year}
            setYear={setYear}
            isRunning={isRunning}
            steps={steps}
            results={results}
            error={error}
            success={success}
            startMode={startMode}
            setStartMode={setStartMode}
            checkpoints={checkpoints}
            selectedCheckpoint={selectedCheckpoint}
            setSelectedCheckpoint={setSelectedCheckpoint}
            runPipeline={runPipeline}
            downloadCSV={downloadCSV}
            clearError={clearError}
          />
        )}


        {activeTab === 'market-analysis' && (
          <MarketAnalysis
            companies={companies}
            selectedCompany={selectedCompany}
            setSelectedCompany={setSelectedCompany}
            marketAnalysis={marketAnalysis}
            isAnalyzing={isAnalyzing}
            analysisError={analysisError}
            runMarketAnalysis={runMarketAnalysis}
            exportToPDF={exportToPDF}
            clearAnalysisError={clearAnalysisError}
          />
        )}
      </main>
    </div>
  );
}