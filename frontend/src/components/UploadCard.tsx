import React, { useRef, useState } from 'react';
import { apiService } from '../api';
import { PredictResponse } from '../types';
import FeedbackModal from './FeedbackModal';

interface UploadCardProps {
  onPrediction?: (result: PredictResponse) => void;
}

const UploadCard: React.FC<UploadCardProps> = ({ onPrediction }) => {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showFeedback, setShowFeedback] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (selectedFile: File) => {
    if (!selectedFile.type.startsWith('image/')) {
      setError('Please select an image file');
      return;
    }

    setFile(selectedFile);
    setError(null);
    setPrediction(null);

    // Create preview
    const reader = new FileReader();
    reader.onload = (e) => {
      setPreview(e.target?.result as string);
    };
    reader.readAsDataURL(selectedFile);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) {
      handleFileSelect(droppedFile);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      handleFileSelect(selectedFile);
    }
  };

  const handlePredict = async () => {
    if (!file) return;

    setIsLoading(true);
    setError(null);

    try {
      const result = await apiService.predictImage(file);
      setPrediction(result);
      onPrediction?.(result);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Prediction failed');
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setPreview(null);
    setPrediction(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const formatConfidence = (prob: number) => {
    return `${(prob * 100).toFixed(1)}%`;
  };

  return (
    <>
      <div className="card h-100">
        <div className="card-header">
          <h5 className="card-title mb-0">Upload ASL Image</h5>
        </div>
        <div className="card-body">
          {!file ? (
            <div
              className="border-dashed border-3 border-secondary rounded p-5 text-center"
              style={{ cursor: 'pointer', minHeight: '200px' }}
              onClick={() => fileInputRef.current?.click()}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
            >
              <div className="d-flex flex-column align-items-center justify-content-center h-100">
                <svg
                  width="48"
                  height="48"
                  fill="currentColor"
                  className="bi bi-cloud-upload text-muted mb-3"
                  viewBox="0 0 16 16"
                >
                  <path
                    fillRule="evenodd"
                    d="M4.406 1.342A5.53 5.53 0 0 1 8 0c2.69 0 4.923 2 5.166 4.579C14.758 4.804 16 6.137 16 7.773 16 9.569 14.502 11 12.687 11H10a.5.5 0 0 1 0-1h2.688C13.979 10 15 8.988 15 7.773c0-1.216-1.02-2.228-2.313-2.228h-.5v-.5C12.188 2.825 10.328 1 8 1a4.53 4.53 0 0 0-3.941 2.341z"
                  />
                  <path
                    fillRule="evenodd"
                    d="M7.646 4.146a.5.5 0 0 1 .708 0l3 3a.5.5 0 0 1-.708.708L8.5 5.707V14.5a.5.5 0 0 1-1 0V5.707L5.354 7.854a.5.5 0 1 1-.708-.708l3-3z"
                  />
                </svg>
                <p className="mb-2">
                  <strong>Drag & drop an image here</strong>
                </p>
                <p className="text-muted">or click to select a file</p>
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileInput}
                style={{ display: 'none' }}
              />
            </div>
          ) : (
            <div>
              <div className="text-center mb-3">
                <img
                  src={preview || ''}
                  alt="Preview"
                  className="img-fluid rounded"
                  style={{ maxHeight: '200px' }}
                />
              </div>
              <p className="text-center text-muted">{file.name}</p>

              {prediction && (
                <div className="mt-4">
                  <h6>Prediction Results:</h6>
                  {prediction.top3.map((item, index) => (
                    <div key={index} className="mb-2">
                      <div className="d-flex justify-content-between align-items-center mb-1">
                        <span className={index === 0 ? 'fw-bold' : ''}>
                          {item.label}
                        </span>
                        <span className="text-muted">
                          {formatConfidence(item.prob)}
                        </span>
                      </div>
                      <div className="progress" style={{ height: '8px' }}>
                        <div
                          className={`progress-bar ${index === 0 ? 'bg-success' : 'bg-info'}`}
                          role="progressbar"
                          style={{ width: `${item.prob * 100}%` }}
                          aria-valuenow={item.prob * 100}
                          aria-valuemin={0}
                          aria-valuemax={100}
                        ></div>
                      </div>
                    </div>
                  ))}
                  
                  <div className="mt-3 text-center">
                    <p className="text-muted small">
                      Processing time: {prediction.latency_ms.toFixed(0)}ms
                    </p>
                    <button
                      className="btn btn-outline-primary btn-sm"
                      onClick={() => setShowFeedback(true)}
                    >
                      Was this correct?
                    </button>
                  </div>
                </div>
              )}

              {error && (
                <div className="alert alert-danger mt-3" role="alert">
                  {error}
                </div>
              )}

              <div className="d-flex gap-2 mt-3">
                <button
                  className="btn btn-primary flex-fill"
                  onClick={handlePredict}
                  disabled={isLoading || !!prediction}
                >
                  {isLoading ? (
                    <>
                      <span
                        className="spinner-border spinner-border-sm me-2"
                        role="status"
                        aria-hidden="true"
                      ></span>
                      Predicting...
                    </>
                  ) : (
                    'Predict'
                  )}
                </button>
                <button
                  className="btn btn-outline-secondary"
                  onClick={handleReset}
                >
                  Reset
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {prediction && (
        <FeedbackModal
          show={showFeedback}
          onHide={() => setShowFeedback(false)}
          uploadId={prediction.upload_id}
          prediction={prediction}
        />
      )}
    </>
  );
};

export default UploadCard;
