import React, { useState, useEffect } from 'react';
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

  const loadCheckpoints = async () => {
    try {
      const response = await fetch('/api/checkpoints');
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
  }, []);

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
    
    printWindow.document.write(`
      <html>
        <head>
          <title>Market Analysis - ${marketAnalysis.company_name}</title>
          <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .header { text-align: center; margin-bottom: 30px; }
            .section { margin-bottom: 20px; }
            .metric { display: flex; justify-content: space-between; margin: 10px 0; }
            .score { font-weight: bold; }
          </style>
        </head>
        <body>
          <div class="header">
            <h1>Market Analysis Report</h1>
            <h2>${marketAnalysis.company_name}</h2>
            <p>Generated on ${new Date(marketAnalysis.analysis_date).toLocaleDateString()}</p>
          </div>
          
          <div class="section">
            <h3>Company Information</h3>
            <div class="metric"><span>Sector:</span><span>${marketAnalysis.sector}</span></div>
            <div class="metric"><span>Founded:</span><span>${marketAnalysis.founded_year}</span></div>
          </div>
          
          <div class="section">
            <h3>Market Size & Growth</h3>
            <div class="metric"><span>Market Size:</span><span class="score">${marketAnalysis.market_size_billion.toFixed(1)}B</span></div>
            <div class="metric"><span>CAGR:</span><span class="score">${marketAnalysis.cagr_percent.toFixed(1)}%</span></div>
            <div class="metric"><span>Market Stage:</span><span>${marketAnalysis.market_stage}</span></div>
          </div>
          
          <div class="section">
            <h3>Market Sentiment & Timing</h3>
            <div class="metric"><span>Timing Score:</span><span class="score">${marketAnalysis.timing_score.toFixed(1)}/5</span></div>
            <div class="metric"><span>US Sentiment:</span><span class="score">${marketAnalysis.us_sentiment.toFixed(1)}/5</span></div>
            <div class="metric"><span>Asia Sentiment:</span><span class="score">${marketAnalysis.sea_sentiment.toFixed(1)}/5</span></div>
            <div class="metric"><span>Momentum Score:</span><span class="score">${marketAnalysis.momentum_score.toFixed(1)}/5</span></div>
          </div>
          
          <div class="section">
            <h3>Competition & Funding</h3>
            <div class="metric"><span>Competitor Count:</span><span>${marketAnalysis.competitor_count}</span></div>
            <div class="metric"><span>Total Funding:</span><span>${marketAnalysis.total_funding_billion.toFixed(1)}B</span></div>
          </div>
          
          <div class="section">
            <h3>Analysis Quality</h3>
            <div class="metric"><span>Confidence Score:</span><span class="score">${(marketAnalysis.confidence_score * 100).toFixed(0)}%</span></div>
            <div class="metric"><span>Execution Time:</span><span class="score">${marketAnalysis.execution_time.toFixed(1)}s</span></div>
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
    setSteps([]);

    try {
      if (startMode === 'resume') {
        updateStep('Resuming Pipeline', 'running', `Resuming from checkpoint: ${selectedCheckpoint}`);
        
        const resumeResponse = await fetch(`/api/pipeline/resume/${selectedCheckpoint}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' }
        });

        if (!resumeResponse.ok) {
          throw new Error(`Resume failed: ${resumeResponse.statusText}`);
        }

        const resumeResult = await resumeResponse.json();
        updateStep('Resuming Pipeline', 'completed', resumeResult.message);

        updateStep('Loading Results', 'running');
        const [companiesData, foundersData] = await Promise.all([
          fetch('/api/companies').then(r => r.json()),
          fetch('/api/founders/rankings').then(r => r.json()).catch(() => [])
        ]);

        setResults({
          companies: companiesData,
          founders: foundersData,
          jobId: resumeResult.jobId
        });

        updateStep('Loading Results', 'completed', 'Pipeline resumed successfully');
        
      } else {
        updateStep('Company Discovery', 'running', 'Searching for companies...');
        
        const pipelineResponse = await fetch('/api/pipeline/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            year: parseInt(year),
            limit: 100
          })
        });

        if (!pipelineResponse.ok) {
          throw new Error(`Pipeline failed: ${pipelineResponse.statusText}`);
        }

        const pipelineResult = await pipelineResponse.json();
        updateStep('Company Discovery', 'completed', `Found ${pipelineResult.companiesFound} companies`);

        updateStep('Loading Company Data', 'running');
        const companiesResponse = await fetch('/api/companies');
        const companies = await companiesResponse.json();
        updateStep('Loading Company Data', 'completed', `Loaded ${companies.length} companies`);

        const totalFounders = pipelineResult.foundersFound;
        if (totalFounders > 0) {
          updateStep('Founder Analysis', 'running', `Analyzing ${totalFounders} founders...`);
          
          const founderProfiles = companies.flatMap(company => 
            company.founders.map(founderName => ({
              name: founderName,
              company_name: company.name,
              linkedin_url: '',
              bio: company.description || ''
            }))
          );

          if (founderProfiles.length > 0) {
            const csvContent = [
              'name,company_name,linkedin_url,bio',
              ...founderProfiles.map(p => 
                `"${p.name}","${p.company_name}","${p.linkedin_url}","${p.bio.replace(/"/g, '""')}"`
              )
            ].join('\n');

            const formData = new FormData();
            const blob = new Blob([csvContent], { type: 'text/csv' });
            formData.append('file', blob, 'founders.csv');

            const rankingResponse = await fetch('/api/founders/rank', {
              method: 'POST',
              body: formData
            });

            if (rankingResponse.ok) {
              const rankingResult = await rankingResponse.json();
              updateStep('Founder Analysis', 'completed', `Ranked ${rankingResult.foundersFound} founders`);
            } else {
              updateStep('Founder Analysis', 'error', 'Ranking failed');
              throw new Error('Founder ranking failed');
            }
          } else {
            updateStep('Founder Analysis', 'completed', 'No founders to analyze');
          }
        } else {
          updateStep('Founder Analysis', 'completed', 'No founders found');
        }

        updateStep('Preparing Results', 'running');
        const [companiesData, foundersData] = await Promise.all([
          fetch('/api/companies').then(r => r.json()),
          fetch('/api/founders/rankings').then(r => r.json()).catch(() => [])
        ]);

        setResults({
          companies: companiesData,
          founders: foundersData,
          jobId: pipelineResult.jobId
        });

        updateStep('Preparing Results', 'completed', 'Pipeline completed successfully');
        
        updateStep('Saving Results', 'running', 'Saving CSV files...');
        try {
          await downloadCSV('companies');
          if (foundersData.length > 0) {
            await downloadCSV('founders');
          }
          updateStep('Saving Results', 'completed', 'CSV files saved to output folder');
        } catch (exportError) {
          updateStep('Saving Results', 'error', 'Failed to save CSV files');
          console.error('Auto-export failed:', exportError);
        }
      }

      await loadCheckpoints();

    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error occurred';
      setError(message);
      updateStep('Pipeline', 'error', message);
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