import React from 'react';

interface ErrorDisplayProps {
  error: string | null;
  onClear?: () => void;
}

const ErrorDisplay: React.FC<ErrorDisplayProps> = ({ error, onClear }) => {
  if (!error) {
    return null;
  }

  return (
    <div className="status-message status-error">
      <span>{error}</span>
      {onClear && (
        <button onClick={onClear} className="btn-clear">
          &times;
        </button>
      )}
    </div>
  );
};

export default ErrorDisplay;
