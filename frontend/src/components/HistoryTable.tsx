import React, { useEffect, useState } from 'react';
import { apiService } from '../api';
import { HistoryItem, HistoryResponse } from '../types';

const HistoryTable: React.FC = () => {
  const [history, setHistory] = useState<HistoryResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 20;

  useEffect(() => {
    loadHistory();
  }, [currentPage]);

  const loadHistory = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const offset = (currentPage - 1) * itemsPerPage;
      const result = await apiService.getHistory(itemsPerPage, offset);
      setHistory(result);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load history');
    } finally {
      setIsLoading(false);
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  const formatConfidence = (prob: number) => {
    return `${(prob * 100).toFixed(1)}%`;
  };

  const getTotalPages = () => {
    if (!history) return 0;
    return Math.ceil(history.total / itemsPerPage);
  };

  const handlePageChange = (page: number) => {
    setCurrentPage(page);
  };

  if (isLoading) {
    return (
      <div className="d-flex justify-content-center align-items-center" style={{ minHeight: '200px' }}>
        <div className="spinner-border" role="status">
          <span className="visually-hidden">Loading...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="alert alert-danger" role="alert">
        {error}
      </div>
    );
  }

  if (!history || history.items.length === 0) {
    return (
      <div className="text-center py-5">
        <p className="text-muted">No predictions found. Upload an image to get started!</p>
      </div>
    );
  }

  return (
    <div>
      <div className="d-flex justify-content-between align-items-center mb-3">
        <h4>Prediction History</h4>
        <span className="text-muted">
          Showing {history.items.length} of {history.total} predictions
        </span>
      </div>

      <div className="table-responsive">
        <table className="table table-hover">
          <thead>
            <tr>
              <th>Image</th>
              <th>Filename</th>
              <th>Date</th>
              <th>Top Prediction</th>
              <th>Confidence</th>
              <th>All Predictions</th>
            </tr>
          </thead>
          <tbody>
            {history.items.map((item: HistoryItem) => (
              <tr key={item.upload_id}>
                <td>
                  <img
                    src={apiService.getImageUrl(item.file_path)}
                    alt={item.filename}
                    className="img-thumbnail"
                    style={{ width: '60px', height: '60px', objectFit: 'cover' }}
                    onError={(e) => {
                      const target = e.target as HTMLImageElement;
                      target.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHJlY3Qgd2lkdGg9IjYwIiBoZWlnaHQ9IjYwIiBmaWxsPSIjRjVGNUY1Ii8+CjxwYXRoIGQ9Ik0yMCAyMEg0MFY0MEgyMFYyMFoiIGZpbGw9IiNEREREREQiLz4KPC9zdmc+';
                    }}
                  />
                </td>
                <td>
                  <span className="text-truncate d-inline-block" style={{ maxWidth: '150px' }}>
                    {item.filename}
                  </span>
                </td>
                <td>{formatDate(item.created_at)}</td>
                <td>
                  <span className="badge bg-primary">{item.top1.label}</span>
                </td>
                <td>{formatConfidence(item.top1.prob)}</td>
                <td>
                  <div
                    className="d-inline-block"
                    data-bs-toggle="tooltip"
                    data-bs-placement="top"
                    title={item.top3.map(p => `${p.label}: ${formatConfidence(p.prob)}`).join(', ')}
                  >
                    <small className="text-muted">
                      {item.top3.length} predictions
                    </small>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {getTotalPages() > 1 && (
        <nav aria-label="History pagination">
          <ul className="pagination justify-content-center">
            <li className={`page-item ${currentPage === 1 ? 'disabled' : ''}`}>
              <button
                className="page-link"
                onClick={() => handlePageChange(currentPage - 1)}
                disabled={currentPage === 1}
              >
                Previous
              </button>
            </li>
            
            {[...Array(getTotalPages())].map((_, index) => {
              const page = index + 1;
              const showPage = page === 1 || page === getTotalPages() || 
                             (page >= currentPage - 2 && page <= currentPage + 2);
              
              if (!showPage) {
                if (page === currentPage - 3 || page === currentPage + 3) {
                  return (
                    <li key={page} className="page-item disabled">
                      <span className="page-link">...</span>
                    </li>
                  );
                }
                return null;
              }
              
              return (
                <li key={page} className={`page-item ${page === currentPage ? 'active' : ''}`}>
                  <button
                    className="page-link"
                    onClick={() => handlePageChange(page)}
                  >
                    {page}
                  </button>
                </li>
              );
            })}
            
            <li className={`page-item ${currentPage === getTotalPages() ? 'disabled' : ''}`}>
              <button
                className="page-link"
                onClick={() => handlePageChange(currentPage + 1)}
                disabled={currentPage === getTotalPages()}
              >
                Next
              </button>
            </li>
          </ul>
        </nav>
      )}
    </div>
  );
};

export default HistoryTable;
