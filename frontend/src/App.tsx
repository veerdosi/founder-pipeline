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

interface CheckpointInfo {
  id: string
  created_at: string
  companies_count: number
  completion_percentage: number
  stages_completed: number
}

export default function App() {
  const [year, setYear] = useState('2025')
  const [isRunning, setIsRunning] = useState(false)
  const [steps, setSteps] = useState<PipelineStatus[]>([])
  const [results, setResults] = useState<PipelineResults | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [startMode, setStartMode] = useState<'fresh' | 'resume'>('fresh')
  const [checkpoints, setCheckpoints] = useState<CheckpointInfo[]>([])
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<string>('')

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
    </div>
  )
}
