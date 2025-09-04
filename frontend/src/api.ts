import axios from 'axios';
import { config } from './config';
import { FeedbackRequest, HealthResponse, HistoryResponse, PredictResponse } from './types';

const api = axios.create({
  baseURL: config.apiBase,
  timeout: 30000,
});

export const apiService = {
  // Health check
  async getHealth(): Promise<HealthResponse> {
    const response = await api.get('/health');
    return response.data;
  },

  // Get labels
  async getLabels(): Promise<string[]> {
    const response = await api.get('/api/labels');
    return response.data;
  },

  // Predict image
  async predictImage(file: File): Promise<PredictResponse> {
    const formData = new FormData();
    formData.append('image', file);
    
    const response = await api.post('/api/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    return response.data;
  },

  // Submit feedback
  async submitFeedback(feedback: FeedbackRequest): Promise<{ message: string }> {
    const response = await api.post('/api/feedback', feedback);
    return response.data;
  },

  // Get history
  async getHistory(limit: number = 50, offset: number = 0): Promise<HistoryResponse> {
    const response = await api.get('/api/history', {
      params: { limit, offset },
    });
    return response.data;
  },

  // Get static image URL
  getImageUrl(filePath: string): string {
    return `${config.apiBase}/static/${encodeURIComponent(filePath.split('/').pop() || '')}`;
  },
};
