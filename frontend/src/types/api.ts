export interface User {
  id: number;
  username: string;
  email: string;
  role: 'farmer' | 'admin' | string;
  is_active?: boolean;
  created_at?: string;
}

export interface TokenResponse {
  access_token: string;
  refresh_token?: string | null;
  expires_in?: number | null;
  token_type: string;
  user: User;
}

export interface DetectionBox {
  bbox: number[];
  normalized_bbox: number[];
  class_id: number;
  class_name: string;
  confidence: number;
}

export interface RecommendationsPayload {
  pathogen_or_cause?: string;
  severity_tier?: string;
  confidence_warning?: string | null;
  immediate_first_steps?: string[];
  preventive_management?: string[];
  critical_warning?: string | null;
  [key: string]: unknown;
}

export interface PredictionDetail {
  id: number;
  user_id: number;
  image_url: string;
  predicted_class: string;
  confidence: number;
  all_probabilities?: Record<string, number> | null;
  affected_area_percentage?: number | null;
  severity_level: string;
  severity_label?: string | null;
  thresholds_verified: boolean;
  model_version: string;
  status: string;
  created_at: string;
  latency_ms?: number | null;

  // Detection outputs
  detection_status: string;
  detection_model_version: string;
  detection_count: number;
  detections: DetectionBox[];
  detection_image_url?: string | null;
  detection_notes?: string | null;

  // Segmentation outputs
  segmentation_status: string;
  segmentation_model_version: string;
  leaf_pixels?: number | null;
  disease_pixels?: number | null;
  healthy_pixels?: number | null;
  overlay_url?: string | null;
  leaf_mask_url?: string | null;
  disease_mask_url?: string | null;

  // Grad-CAM outputs
  gradcam_heatmap_url?: string | null;
  gradcam_overlay_url?: string | null;
  gradcam_comparison_url?: string | null;

  // Recommendations & Notes
  recommendations: RecommendationsPayload;
  uncertainty_notes?: string | null;
  inference_device?: string | null;
  warning?: string | null;

  // Cotton Leaf Verification & Quality Gate
  is_cotton_leaf?: boolean | null;
  verification_status?: string;
  verification_confidence?: number | null;
  rejection_reason?: string | null;
  user_message?: string | null;
  quality_metrics?: Record<string, any> | null;
  probabilities?: Record<string, number> | null;
}

export interface PredictionHistoryItem {
  id: number;
  image_url: string;
  predicted_class: string;
  confidence: number;
  affected_area_percentage?: number | null;
  severity_level: string;
  created_at: string;
  detection_count?: number | null;
  gradcam_comparison_url?: string | null;
  is_cotton_leaf?: boolean | null;
  verification_status?: string;
  rejection_reason?: string | null;
}

export interface PaginatedHistoryResponse {
  items: PredictionHistoryItem[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface HistoryQueryParams {
  search?: string;
  disease?: string;
  severity?: string;
  start_date?: string;
  end_date?: string;
  page?: number;
  page_size?: number;
}

export interface GradCAMResponse {
  predicted_class: string;
  confidence: number;
  target_class: string;
  original_url: string;
  heatmap_url: string;
  overlay_url: string;
  comparison_url: string;
  metadata_url: string;
  target_layer: string;
  model_architecture: string;
  device: string;
  scientific_disclaimer: string;
  all_probabilities: Record<string, number>;
}

export interface HardwareTelemetry {
  cuda_available: boolean;
  device_name?: string;
  total_vram_mb?: number;
  allocated_vram_mb?: number;
  reserved_vram_mb?: number;
}

export interface ModelRegistryEntry {
  name: string;
  architecture?: string;
  version?: string | null;
  checkpoint?: string | null;
  checkpoint_sha256?: string | null;
  status: 'trained' | 'not_trained' | 'checkpoint_missing' | string;
  test_accuracy?: number | null;
  macro_f1?: number | null;
  map50?: number | null;
  map50_95?: number | null;
  dice?: number | null;
  iou?: number | null;
  provenance_verified?: boolean;
  trained_on?: string;
  annotation_requirements?: string;
  last_updated?: string | null;
  classes?: string[];
}

export interface HealthModelsResponse {
  registry: {
    schema_version: string;
    generated_at_utc: string;
    models: {
      classifier: ModelRegistryEntry;
      detector: ModelRegistryEntry;
      segmentor: ModelRegistryEntry;
      [key: string]: ModelRegistryEntry;
    };
  };
  hardware: HardwareTelemetry;
  models: {
    classifier: ModelRegistryEntry;
    detector: ModelRegistryEntry;
    segmentor: ModelRegistryEntry;
    [key: string]: ModelRegistryEntry;
  };
}

export interface BenchmarkReport {
  benchmark_timestamp_utc?: string;
  hardware?: {
    device?: string;
    gpu_name?: string;
    total_vram_mb?: number;
  };
  models?: {
    classifier_efficientnet_b0?: {
      status: string;
      live_validation_benchmark?: {
        validation_metrics?: {
          accuracy: number;
          macro_precision: number;
          macro_recall: number;
          macro_f1: number;
        };
        single_image_mean_latency_ms?: number;
        peak_gpu_memory_mb?: number;
        gradcam_single_image_latency_ms?: number;
      };
      pre_audited_holdout_test_metrics?: {
        overall_performance?: {
          accuracy: number;
          macro_precision: number;
          macro_recall: number;
          macro_f1: number;
        };
      };
    };
    detector_yolov8n?: {
      status: string;
      metadata?: {
        ultralytics_installed?: boolean;
        license?: string;
        annotation_requirements?: string;
      };
    };
    segmentor_unet_resnet34?: {
      status: string;
      metadata?: {
        library?: string;
        annotation_requirements?: string;
      };
    };
  };
}

export interface PredictionJobCreateResponse {
  job_id: string;
  status: string;
  stream_url: string;
  total_steps: number;
}

export interface PredictionJobStatusResponse {
  job_id: string;
  status: string;
  current_stage: string;
  step: number;
  total_steps: number;
  prediction_id?: number | null;
  error_code?: string | null;
  error_message?: string | null;
  created_at: string;
  completed_at?: string | null;
}

export interface SseProgressData {
  stage: string;
  status: string;
  step: number;
  total_steps: number;
  result_summary?: Record<string, unknown>;
}

export interface SseCompleteData {
  job_id: string;
  prediction_id: number;
  redirect?: string;
  result: PredictionDetail;
}

export interface SseErrorData {
  job_id?: string;
  stage?: string;
  code?: string;
  message: string;
}

export interface GeneratedReportMetadata {
  status: string;
  prediction_id: number;
  report_id: number;
  pdf_filename: string;
  file_size_bytes: number;
  download_url: string;
}
