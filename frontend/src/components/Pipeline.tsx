import React from 'react';
import { PipelineStatus, PipelineResults, CheckpointInfo } from '../interfaces';
import CheckpointManager from './CheckpointManager';
import ProgressTracker from './ProgressTracker';
import StatusCards from './StatusCards';
import ErrorDisplay from './ErrorDisplay';
import './Pipeline.css';

interface PipelineProps {
  year: string;
  setYear: (year: string) => void;
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
  runPipeline: () => void;
  downloadCSV: (type: 'companies' | 'founders') => void;
  clearError: () => void;
}

const Pipeline: React.FC<PipelineProps> = ({
  year,
  setYear,
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
  runPipeline,
  downloadCSV,
  clearError,
}) => {
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
          <h2>Execution Controls</h2>
          <div className="pipeline-grid">
            <div className="form-group">
              <label className="form-label">Company Foundation Year</label>
              <select
                value={year}
                onChange={(e) => setYear(e.target.value)}
                className="form-select"
                disabled={isRunning || startMode === 'resume'}
              >
                {Array.from({ length: new Date().getFullYear() - 2000 + 1 }, (_, i) => 2000 + i).map(y => (
                  <option key={y} value={y}>{y}</option>
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
              ) : startMode === 'resume' ? 'Resume Pipeline' : 'Run Pipeline'}
            </button>
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

export default Pipeline;