import React from 'react';
import { CheckpointInfo } from '../interfaces';
import './CheckpointManager.css';

interface CheckpointManagerProps {
  startMode: 'fresh' | 'resume';
  setStartMode: (mode: 'fresh' | 'resume') => void;
  checkpoints: CheckpointInfo[];
  selectedCheckpoint: string;
  setSelectedCheckpoint: (checkpointId: string) => void;
  isRunning: boolean;
}

const CheckpointManager: React.FC<CheckpointManagerProps> = ({
  startMode,
  setStartMode,
  checkpoints,
  selectedCheckpoint,
  setSelectedCheckpoint,
  isRunning,
}) => {
  return (
    <div className="card">
      <h2>Start Options</h2>
      <div className="checkpoint-options">
        <label className="checkpoint-radio">
          <input
            type="radio"
            value="fresh"
            checked={startMode === 'fresh'}
            onChange={(e) => setStartMode(e.target.value as 'fresh' | 'resume')}
            disabled={isRunning}
          />
          Start Fresh
        </label>
        <label className="checkpoint-radio">
          <input
            type="radio"
            value="resume"
            checked={startMode === 'resume'}
            onChange={(e) => setStartMode(e.target.value as 'fresh' | 'resume')}
            disabled={isRunning || checkpoints.length === 0}
          />
          Resume from Checkpoint
        </label>
      </div>

      {startMode === 'resume' && (
        <div className="checkpoint-details">
          {checkpoints.length > 0 ? (
            <div className="form-group">
              <label className="form-label">Select Checkpoint</label>
              <select
                value={selectedCheckpoint}
                onChange={(e) => setSelectedCheckpoint(e.target.value)}
                className="form-select"
                disabled={isRunning}
              >
                <option value="">-- Select a checkpoint --</option>
                {checkpoints.map(cp => {
                  const stageDisplay = {
                    'enriched_companies': 'Company Enrichment',
                    'profiles': 'Profile Enrichment',
                    'rankings': 'Founder Ranking'
                  }[cp.latest_stage] || cp.latest_stage;
                  
                  return (
                    <option key={cp.id} value={cp.id}>
                      {new Date(cp.created_at).toLocaleString()} - Year {cp.foundation_year} - {stageDisplay} ({cp.completion_percentage.toFixed(0)}% complete)
                    </option>
                  );
                })}
              </select>
              {selectedCheckpoint && (() => {
                const selected = checkpoints.find(cp => cp.id === selectedCheckpoint);
                if (!selected) return null;
                
                const stageDisplay = {
                  'enriched_companies': 'Company Enrichment',
                  'profiles': 'Profile Enrichment',
                  'rankings': 'Founder Ranking'
                }[selected.latest_stage] || selected.latest_stage;
                
                const nextStages = {
                  'enriched_companies': ['Profile Enrichment', 'Founder Ranking'],
                  'profiles': ['Founder Ranking'],
                  'rankings': []
                }[selected.latest_stage] || [];
                
                return (
                  <div className="checkpoint-info">
                    <p className="checkpoint-detail">
                      <strong>📊 Progress:</strong> {selected.completion_percentage.toFixed(0)}% complete
                    </p>
                    <p className="checkpoint-detail">
                      <strong>✅ Latest completed stage:</strong> {stageDisplay}
                    </p>
                    {nextStages.length > 0 && (
                      <p className="checkpoint-detail">
                        <strong>🔄 Will resume with:</strong> {nextStages.join(', ')}
                      </p>
                    )}
                    {nextStages.length === 0 && (
                      <p className="checkpoint-detail checkpoint-complete">
                        ✨ This pipeline is already complete!
                      </p>
                    )}
                  </div>
                );
              })()}
            </div>
          ) : (
            <p className="checkpoint-tip">No checkpoints available. Run a fresh pipeline to create one.</p>
          )}
        </div>
      )}
    </div>
  );
};

export default CheckpointManager;
