import React, { useState, useEffect } from 'react'

interface PipelineStatus {
  step: string
  status: 'pending' | 'running' | 'completed' | 'error'
  message?: string
}

interface PipelineResults {
  companies: any[]
  founders: any[]
  jobId: string
}

interface Company {
  id: string
  name: string
  sector: string
  founded_year: string | number
  ai_focus: string
}

interface MarketAnalysisData {
  company_name: string
  sector: string
  founded_year: number
  market_size_billion: number
  cagr_percent: number
  timing_score: number
  us_sentiment: number
  sea_sentiment: number
  competitor_count: number
  total_funding_billion: number
  momentum_score: number
  market_stage: string
  confidence_score: number
  analysis_date: string
  execution_time: number
}

interface CheckpointInfo {
  id: string
  created_at: string
  companies_count: number
  completion_percentage: number
  stages_completed: number
}

export default function App() {
  const [activeTab, setActiveTab] = useState<'pipeline' | 'market-analysis'>('pipeline')
  const [year, setYear] = useState('2025')
  const [isRunning, setIsRunning] = useState(false)
  const [steps, setSteps] = useState<PipelineStatus[]>([])
  const [results, setResults] = useState<PipelineResults | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [startMode, setStartMode] = useState<'fresh' | 'resume'>('fresh')
  const [checkpoints, setCheckpoints] = useState<CheckpointInfo[]>([])
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<string>('')
  
  // Market Analysis State
  const [companies, setCompanies] = useState<Company[]>([])
  const [selectedCompany, setSelectedCompany] = useState<string>('')
  const [marketAnalysis, setMarketAnalysis] = useState<MarketAnalysisData | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisError, setAnalysisError] = useState<string | null>(null)

  const updateStep = (stepName: string, status: PipelineStatus['status'], message?: string) => {
    setSteps(prev => {
      const existing = prev.find(s => s.step === stepName)
      if (existing) {
        existing.status = status
        existing.message = message
        return [...prev]
      }
      return [...prev, { step: stepName, status, message }]
    })
  }

  const loadCheckpoints = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/checkpoints')
      if (response.ok) {
        const checkpointList = await response.json()
        setCheckpoints(checkpointList)
        if (checkpointList.length > 0 && !selectedCheckpoint) {
          setSelectedCheckpoint(checkpointList[0].id)
        }
      }
    } catch (error) {
      console.error('Failed to load checkpoints:', error)
    }
  }

  // Load checkpoints on component mount and when pipeline completes
  useEffect(() => {
    loadCheckpoints()
    // Auto-refresh checkpoints every 30 seconds
    const interval = setInterval(loadCheckpoints, 30000)
    return () => clearInterval(interval)
  }, [])

  // Auto-select most recent checkpoint when switching to resume mode
  useEffect(() => {
    if (startMode === 'resume' && checkpoints.length > 0 && !selectedCheckpoint) {
      setSelectedCheckpoint(checkpoints[0].id)
    }
  }, [startMode, checkpoints])

  // Load companies list when switching to market analysis tab
  useEffect(() => {
    if (activeTab === 'market-analysis') {
      loadCompanies()
    }
  }, [activeTab])

  const loadCompanies = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/companies/list')
      if (response.ok) {
        const companiesList = await response.json()
        setCompanies(companiesList)
        if (companiesList.length > 0 && !selectedCompany) {
          setSelectedCompany(companiesList[0].name)
        }
      } else {
        console.error('Failed to load companies')
      }
    } catch (error) {
      console.error('Failed to load companies:', error)
    }
  }

  const runMarketAnalysis = async () => {
    if (!selectedCompany) {
      setAnalysisError('Please select a company')
      return
    }

    setIsAnalyzing(true)
    setAnalysisError(null)
    setMarketAnalysis(null)

    try {
      const response = await fetch(`http://localhost:8000/api/companies/${encodeURIComponent(selectedCompany)}/market-analysis`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      })

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`)
      }

      const result = await response.json()
      if (result.status === 'success') {
        setMarketAnalysis(result.data)
      } else {
        throw new Error(result.message || 'Analysis failed')
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error occurred'
      setAnalysisError(message)
    } finally {
      setIsAnalyzing(false)
    }
  }

  const exportToPDF = () => {
    if (!marketAnalysis) return
    
    // Simple PDF-like export using window.print for now
    // In a real implementation, you'd use a library like jsPDF
    const printWindow = window.open('', '_blank')
    if (!printWindow) return
    
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
            <div class="metric"><span>Market Size:</span><span class="score">$${marketAnalysis.market_size_billion.toFixed(1)}B</span></div>
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
            <div class="metric"><span>Total Funding:</span><span>$${marketAnalysis.total_funding_billion.toFixed(1)}B</span></div>
          </div>
          
          <div class="section">
            <h3>Analysis Quality</h3>
            <div class="metric"><span>Confidence Score:</span><span class="score">${(marketAnalysis.confidence_score * 100).toFixed(0)}%</span></div>
            <div class="metric"><span>Execution Time:</span><span>${marketAnalysis.execution_time.toFixed(1)}s</span></div>
          </div>
        </body>
      </html>
    `)
    printWindow.document.close()
    printWindow.print()
  }

  const runPipeline = async () => {
    if (startMode === 'fresh' && !year) {
      setError('Please select a year')
      return
    }

    if (startMode === 'resume' && !selectedCheckpoint) {
      setError('Please select a checkpoint to resume from')
      return
    }

    setIsRunning(true)
    setError(null)
    setResults(null)
    setSteps([])

    try {
      if (startMode === 'resume') {
        // Resume from checkpoint
        updateStep('Resuming Pipeline', 'running', `Resuming from checkpoint: ${selectedCheckpoint}`)
        
        const resumeResponse = await fetch(`http://localhost:8000/api/pipeline/resume/${selectedCheckpoint}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' }
        })

        if (!resumeResponse.ok) {
          throw new Error(`Resume failed: ${resumeResponse.statusText}`)
        }

        const resumeResult = await resumeResponse.json()
        updateStep('Resuming Pipeline', 'completed', resumeResult.message)

        // Load final results
        updateStep('Loading Results', 'running')
        const [companiesData, foundersData] = await Promise.all([
          fetch('http://localhost:8000/api/companies').then(r => r.json()),
          fetch('http://localhost:8000/api/founders/rankings').then(r => r.json()).catch(() => [])
        ])

        setResults({
          companies: companiesData,
          founders: foundersData,
          jobId: resumeResult.jobId
        })

        updateStep('Loading Results', 'completed', 'Pipeline resumed successfully')
        
      } else {
        // Fresh pipeline execution
        updateStep('Company Discovery', 'running', 'Searching for companies...')
        
        const pipelineResponse = await fetch('http://localhost:8000/api/pipeline/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            year: parseInt(year),
            limit: 100
          })
        })

        if (!pipelineResponse.ok) {
          throw new Error(`Pipeline failed: ${pipelineResponse.statusText}`)
        }

        const pipelineResult = await pipelineResponse.json()
        updateStep('Company Discovery', 'completed', `Found ${pipelineResult.companiesFound} companies`)

        // Step 2: Get Companies Data
        updateStep('Loading Company Data', 'running')
        const companiesResponse = await fetch('http://localhost:8000/api/companies')
        const companies = await companiesResponse.json()
        updateStep('Loading Company Data', 'completed', `Loaded ${companies.length} companies`)

        // Step 3: Founder Ranking - REQUIRED STEP
        const totalFounders = pipelineResult.foundersFound
        if (totalFounders > 0) {
          updateStep('Founder Analysis', 'running', `Analyzing ${totalFounders} founders...`)
          
          // Extract founder profiles from companies for ranking
          const founderProfiles = companies.flatMap(company => 
            company.founders.map(founderName => ({
              name: founderName,
              company_name: company.name,
              linkedin_url: '',
              bio: company.description || ''
            }))
          )

          if (founderProfiles.length > 0) {
            // Create CSV content for ranking
            const csvContent = [
              'name,company_name,linkedin_url,bio',
              ...founderProfiles.map(p => 
                `"${p.name}","${p.company_name}","${p.linkedin_url}","${p.bio.replace(/"/g, '""')}"`
              )
            ].join('\n')

            const formData = new FormData()
            const blob = new Blob([csvContent], { type: 'text/csv' })
            formData.append('file', blob, 'founders.csv')

            const rankingResponse = await fetch('http://localhost:8000/api/founders/rank', {
              method: 'POST',
              body: formData
            })

            if (rankingResponse.ok) {
              const rankingResult = await rankingResponse.json()
              updateStep('Founder Analysis', 'completed', `Ranked ${rankingResult.foundersFound} founders`)
            } else {
              updateStep('Founder Analysis', 'error', 'Ranking failed')
              throw new Error('Founder ranking failed')
            }
          } else {
            updateStep('Founder Analysis', 'completed', 'No founders to analyze')
          }
        } else {
          updateStep('Founder Analysis', 'completed', 'No founders found')
        }

        // Step 4: Get Final Results
        updateStep('Preparing Results', 'running')
        const [companiesData, foundersData] = await Promise.all([
          fetch('http://localhost:8000/api/companies').then(r => r.json()),
          fetch('http://localhost:8000/api/founders/rankings').then(r => r.json()).catch(() => [])
        ])

        setResults({
          companies: companiesData,
          founders: foundersData,
          jobId: pipelineResult.jobId
        })

        updateStep('Preparing Results', 'completed', 'Pipeline completed successfully')
        
        // Automatically export CSVs
        updateStep('Saving Results', 'running', 'Saving CSV files...')
        try {
          await downloadCSV('companies')
          if (foundersData.length > 0) {
            await downloadCSV('founders')
          }
          updateStep('Saving Results', 'completed', 'CSV files saved to output folder')
        } catch (exportError) {
          updateStep('Saving Results', 'error', 'Failed to save CSV files')
          console.error('Auto-export failed:', exportError)
        }
      }

      // Reload checkpoints after completion
      await loadCheckpoints()

    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error occurred'
      setError(message)
      updateStep('Pipeline', 'error', message)
    } finally {
      setIsRunning(false)
    }
  }

  const downloadCSV = async (type: 'companies' | 'founders') => {
    try {
      const endpoint = type === 'companies' 
        ? 'http://localhost:8000/api/companies/export'
        : 'http://localhost:8000/api/founders/rankings/export'
      
      const response = await fetch(endpoint)
      if (!response.ok) throw new Error(`Export failed: ${response.statusText}`)
      
      const result = await response.json()
      
      if (result.status === 'success') {
        setSuccess(`✅ ${result.message}`)
        setTimeout(() => setSuccess(null), 5000) // Clear after 5 seconds
      } else {
        throw new Error(result.message || 'Export failed')
      }
    } catch (err) {
      setError(`Export failed: ${err instanceof Error ? err.message : 'Unknown error'}`)
    }
  }

  const getStepIcon = (status: PipelineStatus['status']) => {
    switch (status) {
      case 'completed': return '✅'
      case 'running': return '🔄'
      case 'error': return '❌'
      default: return '⏳'
    }
  }

  return (
    <div className="container animate-fade-in">
      <h1>Initiation Pipeline</h1>
      <p className="text-center mb-8">AI-powered company and founder discovery platform</p>
      
      {/* Tab Navigation */}
      <div className="card mb-6">
        <div className="flex border-b">
          <button
            onClick={() => setActiveTab('pipeline')}
            className={`px-6 py-3 font-medium transition-colors ${
              activeTab === 'pipeline' 
                ? 'text-blue-600 border-b-2 border-blue-600' 
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            📊 Pipeline
          </button>
          <button
            onClick={() => setActiveTab('market-analysis')}
            className={`px-6 py-3 font-medium transition-colors ${
              activeTab === 'market-analysis' 
                ? 'text-blue-600 border-b-2 border-blue-600' 
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            📈 Market Analysis
          </button>
        </div>
      </div>

      {activeTab === 'pipeline' && (
        <>
        {/* Start Options Section */}
        <div className="card mb-6">
          <h2>Start Options</h2>
          <div className="mb-4">
            <div className="flex gap-4 mb-4">
              <label className="flex items-center">
                <input 
                  type="radio" 
                  value="fresh" 
                  checked={startMode === 'fresh'}
                  onChange={(e) => setStartMode(e.target.value as 'fresh' | 'resume')}
                  disabled={isRunning}
                  className="mr-2"
                />
                Start Fresh
              </label>
              <label className="flex items-center">
                <input 
                  type="radio" 
                  value="resume" 
                  checked={startMode === 'resume'}
                  onChange={(e) => setStartMode(e.target.value as 'fresh' | 'resume')}
                  disabled={isRunning || checkpoints.length === 0}
                  className="mr-2"
                />
                Resume from Checkpoint
              </label>
            </div>

            {startMode === 'resume' && checkpoints.length > 0 && (
              <div className="form-group">
                <label className="form-label">Select Checkpoint</label>
                <select
                  value={selectedCheckpoint}
                  onChange={(e) => setSelectedCheckpoint(e.target.value)}
                  className="form-select"
                  disabled={isRunning}
                >
                  <option value="">-- Select a checkpoint --</option>
                  {checkpoints.map(cp => (
                    <option key={cp.id} value={cp.id}>
                      {new Date(cp.created_at).toLocaleDateString()} {new Date(cp.created_at).toLocaleTimeString()} - 
                      {cp.companies_count} companies - 
                      {cp.completion_percentage.toFixed(0)}% complete - 
                      {cp.stages_completed} stages done
                    </option>
                  ))}
                </select>
                {selectedCheckpoint && (
                  <div className="text-sm text-gray-600 mt-2">
                    💡 This will resume from where the selected pipeline left off
                  </div>
                )}
              </div>
            )}

            {startMode === 'resume' && checkpoints.length === 0 && (
              <p className="text-gray-500">No checkpoints available. Start fresh to create a new pipeline.</p>
            )}
          </div>
        </div>

        {/* Year Input Section */}
        <div className="card mb-6">
          <h2>Company Foundation Year</h2>
          <div className="flex flex-col md:flex-row gap-4 items-end">
            <div className="form-group w-full">
              <label className="form-label">
                Year
              </label>
              <select
                value={year}
                onChange={(e) => setYear(e.target.value)}
                className="form-select"
                disabled={isRunning || startMode === 'resume'}
              >
                {Array.from({ length: new Date().getFullYear() - 2000 + 1 }, (_, i) => 2000 + i).map(year => (
                  <option key={year} value={year}>{year}</option>
                ))}
              </select>
            </div>
            <button
              onClick={runPipeline}
              disabled={isRunning || (startMode === 'resume' && (!selectedCheckpoint || checkpoints.length === 0))}
              className="btn btn-primary"
            >
              {isRunning ? (
                <>
                  <span className="animate-spin">🔄</span>
                  {startMode === 'resume' ? 'Resuming...' : 'Running...'}
                </>
              ) : startMode === 'resume' ? (
                checkpoints.length === 0 ? 'No Checkpoints Available' : 'Resume Pipeline'
              ) : (
                'Run Pipeline'
              )}
            </button>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="status-message status-error animate-fade-in">
            <p>{error}</p>
          </div>
        )}

        {/* Success Display */}
        {success && (
          <div className="status-message status-success animate-fade-in">
            <p>{success}</p>
          </div>
        )}

        {/* Pipeline Steps */}
        {steps.length > 0 && (
          <div className="progress-container animate-fade-in">
            <h2>Pipeline Progress</h2>
            <div className="gap-2">
              {steps.map((step, index) => (
                <div key={index} className={`progress-step ${step.status}`}>
                  <span style={{ fontSize: '1.5rem', marginRight: '1rem' }}>{getStepIcon(step.status)}</span>
                  <div className="w-full">
                    <div style={{ fontWeight: 600 }}>{step.step}</div>
                    {step.message && (
                      <div style={{ fontSize: '0.875rem', opacity: 0.8 }}>{step.message}</div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Results Section */}
        {results && (
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
        )}
        </>
      )}

      {activeTab === 'market-analysis' && (
        <>
        {/* Market Analysis Section */}
        <div className="card mb-6">
          <h2>Market Analysis</h2>
          <p className="text-gray-600 mb-4">
            Generate comprehensive market analysis for any company from your latest pipeline results.
          </p>
          
          <div className="flex flex-col md:flex-row gap-4 items-end">
            <div className="form-group w-full">
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
                <div className="form-select bg-gray-100 text-gray-500">
                  No companies available. Run the pipeline first to get company data.
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

        {/* Analysis Error Display */}
        {analysisError && (
          <div className="status-message status-error animate-fade-in">
            <p>{analysisError}</p>
          </div>
        )}

        {/* Market Analysis Results */}
        {marketAnalysis && (
          <div className="card animate-fade-in">
            <div className="flex justify-between items-center mb-6">
              <h2>Market Analysis Results</h2>
              <button
                onClick={exportToPDF}
                className="btn btn-success"
              >
                📄 Export PDF
              </button>
            </div>
            
            <div className="mb-6">
              <h3 className="text-xl font-semibold mb-2">{marketAnalysis.company_name}</h3>
              <div className="grid grid-cols-2 gap-4 text-sm text-gray-600">
                <div>Sector: {marketAnalysis.sector}</div>
                <div>Founded: {marketAnalysis.founded_year}</div>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {/* Market Size & Growth */}
              <div className="bg-blue-50 p-4 rounded-lg">
                <h4 className="font-semibold text-blue-800 mb-3">Market Size & Growth</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span>Market Size:</span>
                    <span className="font-bold">${marketAnalysis.market_size_billion.toFixed(1)}B</span>
                  </div>
                  <div className="flex justify-between">
                    <span>CAGR:</span>
                    <span className="font-bold">{marketAnalysis.cagr_percent.toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Stage:</span>
                    <span className="font-bold capitalize">{marketAnalysis.market_stage}</span>
                  </div>
                </div>
              </div>

              {/* Market Sentiment & Timing */}
              <div className="bg-green-50 p-4 rounded-lg">
                <h4 className="font-semibold text-green-800 mb-3">Sentiment & Timing</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span>Timing Score:</span>
                    <span className="font-bold">{marketAnalysis.timing_score.toFixed(1)}/5</span>
                  </div>
                  <div className="flex justify-between">
                    <span>US Sentiment:</span>
                    <span className="font-bold">{marketAnalysis.us_sentiment.toFixed(1)}/5</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Asia Sentiment:</span>
                    <span className="font-bold">{marketAnalysis.sea_sentiment.toFixed(1)}/5</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Momentum:</span>
                    <span className="font-bold">{marketAnalysis.momentum_score.toFixed(1)}/5</span>
                  </div>
                </div>
              </div>

              {/* Competition & Funding */}
              <div className="bg-yellow-50 p-4 rounded-lg">
                <h4 className="font-semibold text-yellow-800 mb-3">Competition & Funding</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span>Competitors:</span>
                    <span className="font-bold">{marketAnalysis.competitor_count}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Total Funding:</span>
                    <span className="font-bold">${marketAnalysis.total_funding_billion.toFixed(1)}B</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Analysis Quality */}
            <div className="mt-6 p-4 bg-gray-50 rounded-lg">
              <h4 className="font-semibold text-gray-800 mb-3">Analysis Quality</h4>
              <div className="grid grid-cols-2 gap-4">
                <div className="flex justify-between">
                  <span>Confidence Score:</span>
                  <span className="font-bold">{(marketAnalysis.confidence_score * 100).toFixed(0)}%</span>
                </div>
                <div className="flex justify-between">
                  <span>Execution Time:</span>
                  <span className="font-bold">{marketAnalysis.execution_time.toFixed(1)}s</span>
                </div>
              </div>
            </div>

            <div className="mt-4 text-xs text-gray-500">
              Generated on {new Date(marketAnalysis.analysis_date).toLocaleString()}
            </div>
          </div>
        )}
        </>
      )}
    </div>
  )
}
