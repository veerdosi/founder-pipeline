import React from 'react';
import { PipelineStatus } from '../interfaces';
import './ProgressTracker.css';

interface ProgressTrackerProps {
  steps: PipelineStatus[];
}

const ProgressTracker: React.FC<ProgressTrackerProps> = ({ steps }) => {
  if (steps.length === 0) {
    return null;
  }

  const getStepIcon = (status: PipelineStatus['status']) => {
    switch (status) {
      case 'completed': return '✅';
      case 'running': return '🔄';
      case 'error': return '❌';
      default: return '⏳';
    }
  };

  return (
    <div className="card progress-tracker-container animate-fade-in">
      <h2>Pipeline Progress</h2>
      <div>
        {steps.map((step, index) => (
          <div key={index} className={`progress-step ${step.status}`}>
            <div className={`progress-step-icon ${step.status === 'running' ? 'animate-spin' : ''}`}>
              {getStepIcon(step.status)}
            </div>
            <div className="progress-step-info">
              <div className="progress-step-title">{step.step}</div>
              {step.message && (
                <div className="progress-step-message">{step.message}</div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ProgressTracker;
