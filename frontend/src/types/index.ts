export interface PredictionResult {
  variant: string;
  model_type: string;
  pathogenicity_score: number;
  benign_score: number;
  confidence: number;
  prediction: "pathogenic" | "benign";
  inference_time_ms: number;
}

export interface BatchPredictionResult {
  results: PredictionResult[];
  count: number;
}

export interface ModelInfo {
  id: string;
  name: string;
  description: string;
  parameters: string;
  type: string;
  architecture?: string;
  attention?: string;
}

export interface HealthStatus {
  status: string;
  device: string;
  version: string;
  service: string;
}
