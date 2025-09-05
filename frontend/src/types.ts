export interface PredictionItem {
  label: string;
  prob: number;
}

export interface PredictResponse {
  upload_id: string;
  prediction_id: string;
  top3: PredictionItem[];
  model_type?: string;
  model: {
    name: string;
    input_size: number[];
    preprocess: string;
  };
  latency_ms: number;
}

export interface FeedbackRequest {
  upload_id: string;
  is_correct: boolean;
  correct_label?: string;
  notes?: string;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  num_labels: number;
  db: string;
}

export interface HistoryItem {
  upload_id: string;
  file_path: string;
  filename: string;
  created_at: string;
  top1: PredictionItem;
  top3: PredictionItem[];
}

export interface HistoryResponse {
  items: HistoryItem[];
  total: number;
  limit: number;
  offset: number;
}
