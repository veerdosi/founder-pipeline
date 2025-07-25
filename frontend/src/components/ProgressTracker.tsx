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

  const getProgressPercentage = () => {
    const totalSteps = steps.length;
    if (totalSteps === 0) return 0;
    
    const completedSteps = steps.filter(step => step.status === 'completed').length;
    const runningSteps = steps.filter(step => step.status === 'running').length;
    
    // Calculate base progress from completed steps
    let baseProgress = (completedSteps / totalSteps) * 100;
    
    // Add partial progress for running steps
    if (runningSteps > 0) {
      const runningStepProgress = (1 / totalSteps) * 30; // Give 30% partial credit to running steps
      baseProgress += runningStepProgress;
    }
    
    return Math.min(Math.round(baseProgress), 100);
  };

  return (
    <div className="card progress-tracker-container animate-fade-in">
      <div className="progress-header">
        <h2>Pipeline Progress</h2>
        <div className="progress-percentage">
          {getProgressPercentage()}% Complete
        </div>
      </div>
      <div className="progress-bar-container">
        <div className="progress-bar">
          <div 
            className="progress-bar-fill" 
            style={{ width: `${getProgressPercentage()}%` }}
          ></div>
        </div>
      </div>
      <div>
        {steps.map((step, index) => (
          <div key={index} className={`progress-step ${step.status}`}>
            <div className={`progress-step-icon ${step.status === 'running' ? 'animate-spin' : ''}`}>
              {getStepIcon(step.status)}
            </div>
            <div className="progress-step-info">
              <div className="progress-step-title">{step.step}</div>
              {step.message && (
                <div className="progress-step-message">
                  {step.message}
                  {step.status === 'running' && (
                    <span className="progress-dots">...</span>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ProgressTracker;
