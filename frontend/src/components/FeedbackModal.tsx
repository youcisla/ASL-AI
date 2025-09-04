import React, { useEffect, useState } from 'react';
import { apiService } from '../api';
import { FeedbackRequest, PredictResponse } from '../types';

interface FeedbackModalProps {
  show: boolean;
  onHide: () => void;
  uploadId: string;
  prediction: PredictResponse;
}

const FeedbackModal: React.FC<FeedbackModalProps> = ({
  show,
  onHide,
  uploadId,
  prediction,
}) => {
  const [isCorrect, setIsCorrect] = useState<boolean | null>(null);
  const [correctLabel, setCorrectLabel] = useState<string>('');
  const [notes, setNotes] = useState<string>('');
  const [labels, setLabels] = useState<string[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (show) {
      // Load labels when modal opens
      loadLabels();
      // Reset form
      setIsCorrect(null);
      setCorrectLabel('');
      setNotes('');
      setError(null);
    }
  }, [show]);

  const loadLabels = async () => {
    try {
      const labelsList = await apiService.getLabels();
      setLabels(labelsList);
    } catch (err) {
      setError('Failed to load labels');
    }
  };

  const handleSubmit = async () => {
    if (isCorrect === null) {
      setError('Please select whether the prediction was correct');
      return;
    }

    setIsSubmitting(true);
    setError(null);

    try {
      const feedbackData: FeedbackRequest = {
        upload_id: uploadId,
        is_correct: isCorrect,
        correct_label: !isCorrect && correctLabel ? correctLabel : undefined,
        notes: notes || undefined,
      };

      await apiService.submitFeedback(feedbackData);
      onHide();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to submit feedback');
    } finally {
      setIsSubmitting(false);
    }
  };

  if (!show) return null;

  return (
    <div className="modal show d-block" style={{ backgroundColor: 'rgba(0,0,0,0.5)' }}>
      <div className="modal-dialog">
        <div className="modal-content">
          <div className="modal-header">
            <h5 className="modal-title">Feedback on Prediction</h5>
            <button
              type="button"
              className="btn-close"
              onClick={onHide}
              aria-label="Close"
            ></button>
          </div>
          
          <div className="modal-body">
            <div className="mb-3">
              <h6>Predicted: <span className="text-primary">{prediction.top3[0].label}</span></h6>
              <p className="text-muted">
                Confidence: {(prediction.top3[0].prob * 100).toFixed(1)}%
              </p>
            </div>

            <div className="mb-3">
              <label className="form-label">Was this prediction correct?</label>
              <div>
                <div className="form-check form-check-inline">
                  <input
                    className="form-check-input"
                    type="radio"
                    name="isCorrect"
                    id="correct-yes"
                    checked={isCorrect === true}
                    onChange={() => setIsCorrect(true)}
                  />
                  <label className="form-check-label" htmlFor="correct-yes">
                    Yes
                  </label>
                </div>
                <div className="form-check form-check-inline">
                  <input
                    className="form-check-input"
                    type="radio"
                    name="isCorrect"
                    id="correct-no"
                    checked={isCorrect === false}
                    onChange={() => setIsCorrect(false)}
                  />
                  <label className="form-check-label" htmlFor="correct-no">
                    No
                  </label>
                </div>
              </div>
            </div>

            {isCorrect === false && (
              <div className="mb-3">
                <label htmlFor="correctLabel" className="form-label">
                  What is the correct label?
                </label>
                <select
                  className="form-select"
                  id="correctLabel"
                  value={correctLabel}
                  onChange={(e) => setCorrectLabel(e.target.value)}
                >
                  <option value="">Select correct label...</option>
                  {labels.map((label) => (
                    <option key={label} value={label}>
                      {label}
                    </option>
                  ))}
                </select>
              </div>
            )}

            <div className="mb-3">
              <label htmlFor="notes" className="form-label">
                Additional notes (optional)
              </label>
              <textarea
                className="form-control"
                id="notes"
                rows={3}
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="Any additional feedback..."
              ></textarea>
            </div>

            {error && (
              <div className="alert alert-danger" role="alert">
                {error}
              </div>
            )}
          </div>

          <div className="modal-footer">
            <button
              type="button"
              className="btn btn-secondary"
              onClick={onHide}
              disabled={isSubmitting}
            >
              Cancel
            </button>
            <button
              type="button"
              className="btn btn-primary"
              onClick={handleSubmit}
              disabled={isSubmitting || isCorrect === null}
            >
              {isSubmitting ? (
                <>
                  <span
                    className="spinner-border spinner-border-sm me-2"
                    role="status"
                    aria-hidden="true"
                  ></span>
                  Submitting...
                </>
              ) : (
                'Submit Feedback'
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FeedbackModal;
