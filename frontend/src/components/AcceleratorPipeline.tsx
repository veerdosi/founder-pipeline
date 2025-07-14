import React from 'react';
import { PipelineStatus, PipelineResults, CheckpointInfo } from '../interfaces';
import CheckpointManager from './CheckpointManager';
import ProgressTracker from './ProgressTracker';
import StatusCards from './StatusCards';
import ErrorDisplay from './ErrorDisplay';
import './Pipeline.css';

interface AcceleratorPipelineProps {
  selectedAccelerators: string[];
  setSelectedAccelerators: (accelerators: string[]) => void;
  isRunning: boolean;
  steps: PipelineStatus[];
  results: PipelineResults | null;
  error: string | null;
  success: string | null;
  startMode: 'fresh' | 'resume';
  setStartMode: (mode: 'fresh' | 'resume') => void;
  checkpoints: CheckpointInfo[];
  selectedCheckpoint: string;
  setSelectedCheckpoint: (checkpointId: string) => void;
  runAcceleratorPipeline: () => void;
  downloadCSV: (type: 'companies' | 'founders') => void;
  clearError: () => void;
}

const AcceleratorPipeline: React.FC<AcceleratorPipelineProps> = ({
  selectedAccelerators,
  setSelectedAccelerators,
  isRunning,
  steps,
  results,
  error,
  success,
  startMode,
  setStartMode,
  checkpoints,
  selectedCheckpoint,
  setSelectedCheckpoint,
  runAcceleratorPipeline,
  downloadCSV,
  clearError,
}) => {
  const availableAccelerators = [
    { id: 'yc', name: 'Y Combinator', description: 'Leading startup accelerator' },
    { id: 'techstars', name: 'Techstars', description: 'Worldwide network of startup accelerators' },
    { id: '500co', name: '500 Global', description: 'Early-stage venture fund and accelerator' }
  ];

  const toggleAccelerator = (acceleratorId: string) => {
    if (isRunning) return;
    
    if (selectedAccelerators.includes(acceleratorId)) {
      setSelectedAccelerators(selectedAccelerators.filter(id => id !== acceleratorId));
    } else {
      setSelectedAccelerators([...selectedAccelerators, acceleratorId]);
    }
  };

  const selectAllAccelerators = () => {
    if (isRunning) return;
    setSelectedAccelerators(availableAccelerators.map(acc => acc.id));
  };

  const clearAllAccelerators = () => {
    if (isRunning) return;
    setSelectedAccelerators([]);
  };

  return (
    <div className="animate-fade-in">
      <div className="pipeline-controls">
        <CheckpointManager
          startMode={startMode}
          setStartMode={setStartMode}
          checkpoints={checkpoints}
          selectedCheckpoint={selectedCheckpoint}
          setSelectedCheckpoint={setSelectedCheckpoint}
          isRunning={isRunning}
        />

        <div className="card">
          <h2>Accelerator Selection</h2>
          <p className="form-description">
            Select which accelerators to search for AI/ML companies. The pipeline will verify company status, 
            fetch Crunchbase data, and run the full discovery and ranking process.
          </p>
          
          <div className="accelerator-selection">
            <div className="accelerator-controls">
              <button
                onClick={selectAllAccelerators}
                disabled={isRunning || startMode === 'resume'}
                className="btn btn-secondary btn-sm"
              >
                Select All
              </button>
              <button
                onClick={clearAllAccelerators}
                disabled={isRunning || startMode === 'resume'}
                className="btn btn-secondary btn-sm"
              >
                Clear All
              </button>
            </div>
            
            <div className="accelerator-grid">
              {availableAccelerators.map(accelerator => (
                <div
                  key={accelerator.id}
                  className={`accelerator-card ${
                    selectedAccelerators.includes(accelerator.id) ? 'selected' : ''
                  } ${isRunning || startMode === 'resume' ? 'disabled' : ''}`}
                  onClick={() => toggleAccelerator(accelerator.id)}
                >
                  <div className="accelerator-header">
                    <div className="accelerator-checkbox">
                      <input
                        type="checkbox"
                        checked={selectedAccelerators.includes(accelerator.id)}
                        onChange={() => toggleAccelerator(accelerator.id)}
                        disabled={isRunning || startMode === 'resume'}
                      />
                    </div>
                    <h3>{accelerator.name}</h3>
                  </div>
                  <p className="accelerator-description">{accelerator.description}</p>
                </div>
              ))}
            </div>
          </div>

          <div className="pipeline-grid">
            <button
              onClick={runAcceleratorPipeline}
              disabled={
                isRunning || 
                (startMode === 'fresh' && selectedAccelerators.length === 0) ||
                (startMode === 'resume' && (!selectedCheckpoint || checkpoints.length === 0))
              }
              className="btn btn-primary"
            >
              {isRunning ? (
                <>
                  <span className="animate-spin">🔄</span>
                  {startMode === 'resume' ? 'Resuming...' : 'Running...'}
                </>
              ) : startMode === 'resume' ? 'Resume Pipeline' : 'Run Accelerator Pipeline'}
            </button>
            
            {startMode === 'fresh' && (
              <div className="pipeline-info">
                <p className="pipeline-summary">
                  {selectedAccelerators.length === 0 
                    ? 'Select at least one accelerator to proceed'
                    : `Selected: ${selectedAccelerators.map(id => 
                        availableAccelerators.find(acc => acc.id === id)?.name
                      ).join(', ')}`
                  }
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      <ErrorDisplay error={error} onClear={clearError} />

      {success && (
        <div className="status-message status-success">
          <p>{success}</p>
        </div>
      )}

      <ProgressTracker steps={steps} />
      <StatusCards results={results} downloadCSV={downloadCSV} />
    </div>
  );
};

export default AcceleratorPipeline;