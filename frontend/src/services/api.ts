import axios, { AxiosRequestConfig } from 'axios';
import {
  BenchmarkReport,
  GeneratedReportMetadata,
  GradCAMResponse,
  HealthModelsResponse,
  HistoryQueryParams,
  PaginatedHistoryResponse,
  PredictionDetail,
  PredictionJobCreateResponse,
  PredictionJobStatusResponse,
  SseCompleteData,
  SseErrorData,
  SseProgressData,
  TokenResponse,
  User,
} from '../types/api';
import { parseSseChunks } from '../utils/sse';
import { resolveMediaUrl } from '../utils/validation';

// Relative base URL leverages Vite's dev proxy (/api and /artifacts -> http://127.0.0.1:8000)
export const API_BASE_URL = import.meta.env.VITE_API_URL || '';

export function mediaUrl(path?: string | null): string {
  return resolveMediaUrl(path, API_BASE_URL);
}

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  withCredentials: true,
});

const TOKEN_KEY = 'token';
const REFRESH_TOKEN_KEY = 'refresh_token';

export function getStoredToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function getStoredRefreshToken(): string | null {
  return localStorage.getItem(REFRESH_TOKEN_KEY);
}

export function setStoredSessionTokens(accessToken: string, refreshToken?: string | null): void {
  localStorage.setItem(TOKEN_KEY, accessToken);
  if (refreshToken) {
    localStorage.setItem(REFRESH_TOKEN_KEY, refreshToken);
  }
}

export function clearStoredSessionTokens(): void {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(REFRESH_TOKEN_KEY);
}

let inFlightRefresh: Promise<string | null> | null = null;

async function refreshAccessToken(): Promise<string | null> {
  if (inFlightRefresh) return inFlightRefresh;
  const refreshToken = getStoredRefreshToken();
  inFlightRefresh = (async () => {
    try {
      const res = await axios.post<TokenResponse>(
        `${API_BASE_URL}/api/auth/refresh`,
        refreshToken ? { refresh_token: refreshToken } : {},
        { withCredentials: true }
      );
      if (res.data?.access_token) {
        setStoredSessionTokens(res.data.access_token, res.data.refresh_token);
        return res.data.access_token;
      }
      return null;
    } catch {
      clearStoredSessionTokens();
      return null;
    } finally {
      inFlightRefresh = null;
    }
  })();
  return inFlightRefresh;
}

apiClient.interceptors.request.use((config) => {
  const token = getStoredToken();
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

apiClient.interceptors.response.use(
  (response) => response,
  async (error) => {
    const original = error.config as (AxiosRequestConfig & { _retry?: boolean }) | undefined;
    const status = error.response?.status;
    const url = String(original?.url || '');
    const isAuthEndpoint =
      url.includes('/api/auth/login') ||
      url.includes('/api/auth/register') ||
      url.includes('/api/auth/refresh');

    if (status === 401 && original && !original._retry && !isAuthEndpoint && getStoredRefreshToken()) {
      original._retry = true;
      const freshToken = await refreshAccessToken();
      if (freshToken) {
        original.headers = {
          ...(original.headers || {}),
          Authorization: `Bearer ${freshToken}`,
        };
        return apiClient(original);
      }
    }
    return Promise.reject(error);
  }
);

export type { HistoryQueryParams };

export interface StreamAnalysisCallbacks {
  onProgress?: (event: SseProgressData) => void;
  onComplete?: (payload: SseCompleteData) => void;
  onError?: (err: SseErrorData) => void;
  signal?: AbortSignal;
}

export const api = {
  // Auth
  login: async (username_or_email: string, password: string): Promise<TokenResponse> => {
    const res = await apiClient.post<TokenResponse>('/api/auth/login', { username_or_email, password });
    return res.data;
  },
  register: async (username: string, email: string, password: string): Promise<User> => {
    const res = await apiClient.post<User>('/api/auth/register', {
      username,
      email,
      password,
      role: 'farmer',
    });
    return res.data;
  },
  logout: async (): Promise<void> => {
    const refreshToken = getStoredRefreshToken();
    try {
      if (getStoredToken() || refreshToken) {
        await apiClient.post('/api/auth/logout', refreshToken ? { refresh_token: refreshToken } : {});
      }
    } catch {
      // Clear local credentials even if server session already expired
    } finally {
      clearStoredSessionTokens();
    }
  },
  getMe: async (): Promise<User> => {
    const res = await apiClient.get<User>('/api/auth/me');
    return res.data;
  },

  // Health, Model Registry & Verified Phase 4 Benchmark
  getHealth: async () => {
    const res = await apiClient.get('/api/health');
    return res.data;
  },
  getHealthModels: async (): Promise<HealthModelsResponse> => {
    const res = await apiClient.get<HealthModelsResponse>('/api/health/models');
    return res.data;
  },
  getBenchmarkReport: async (): Promise<BenchmarkReport | null> => {
    try {
      const res = await apiClient.get<BenchmarkReport>('/artifacts/metrics/benchmark_report.json');
      return res.data;
    } catch {
      return null;
    }
  },

  // Synchronous & Asynchronous SSE Analysis
  analyzeImage: async (file: File, segAlpha: number = 0.4): Promise<PredictionDetail> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('seg_alpha', segAlpha.toString());
    const res = await apiClient.post<PredictionDetail>('/api/predictions/analyze', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return res.data;
  },
  analyzeImageAsync: async (file: File, segAlpha: number = 0.4): Promise<PredictionJobCreateResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('seg_alpha', segAlpha.toString());
    const res = await apiClient.post<PredictionJobCreateResponse>('/api/predictions/analyze/async', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return res.data;
  },
  getJobStatus: async (jobId: string): Promise<PredictionJobStatusResponse> => {
    const res = await apiClient.get<PredictionJobStatusResponse>(`/api/predictions/jobs/${jobId}`);
    return res.data;
  },
  streamAnalysisJob: async (
    jobId: string,
    callbacks: StreamAnalysisCallbacks = {}
  ): Promise<PredictionDetail> => {
    const token = getStoredToken();
    const headers: Record<string, string> = {
      Accept: 'text/event-stream',
    };
    if (token) {
      headers.Authorization = `Bearer ${token}`;
    }

    const response = await fetch(`${API_BASE_URL}/api/predictions/analyze/stream/${jobId}`, {
      method: 'GET',
      headers,
      credentials: 'include',
      signal: callbacks.signal,
    });

    if (!response.ok) {
      let msg = `SSE stream failed with HTTP ${response.status}`;
      try {
        const errJson = await response.json();
        msg = errJson?.error?.message || errJson?.detail || msg;
      } catch {
        // Keep HTTP error summary
      }
      throw new Error(msg);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('ReadableStream is unavailable in this browser.');
    }

    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    let completedResult: PredictionDetail | null = null;
    let streamError: SseErrorData | null = null;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const { events, remainder } = parseSseChunks(buffer);
      buffer = remainder;

      for (const evt of events) {
        let parsed: any = null;
        try {
          parsed = JSON.parse(evt.data);
        } catch {
          continue;
        }
        if (evt.event === 'progress') {
          callbacks.onProgress?.(parsed as SseProgressData);
        } else if (evt.event === 'complete') {
          const completePayload = parsed as SseCompleteData;
          completedResult = completePayload.result;
          callbacks.onComplete?.(completePayload);
        } else if (evt.event === 'error') {
          streamError = parsed as SseErrorData;
          callbacks.onError?.(streamError);
        }
      }
      if (completedResult || streamError) break;
    }

    if (streamError) {
      throw new Error(streamError.message || 'Streaming inference failed.');
    }
    if (completedResult) {
      return completedResult;
    }

    // Fallback check against persisted job row if stream closed immediately after completion
    for (let attempt = 0; attempt < 15; attempt++) {
      if (callbacks.signal?.aborted) {
        throw new Error('Analysis cancelled.');
      }
      const statusRow = await api.getJobStatus(jobId);
      if (statusRow.status === 'completed' && statusRow.prediction_id) {
        return api.getPredictionDetail(statusRow.prediction_id);
      }
      if (statusRow.status === 'failed') {
        throw new Error(statusRow.error_message || 'Background analysis failed.');
      }
      await new Promise((resolve) => setTimeout(resolve, 600));
    }
    throw new Error('Analysis stream ended before emitting a final result.');
  },
  classifyImage: async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    const res = await apiClient.post('/api/classification/predict', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return res.data;
  },
  getGradCAM: async (
    file: File,
    targetClass?: number,
    alpha: number = 0.4,
    colormap: string = 'jet'
  ): Promise<GradCAMResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    if (targetClass !== undefined) {
      formData.append('target_class', targetClass.toString());
    }
    formData.append('alpha', alpha.toString());
    formData.append('colormap', colormap);
    const res = await apiClient.post<GradCAMResponse>('/api/explainability/gradcam', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return res.data;
  },

  // History & Detail Retrieval
  getPredictionDetail: async (id: number): Promise<PredictionDetail> => {
    const res = await apiClient.get<PredictionDetail>(`/api/history/${id}`);
    return res.data;
  },
  deleteHistoryItem: async (id: number) => {
    const res = await apiClient.delete(`/api/history/${id}`);
    return res.data;
  },
  exportHistoryCsv: async (params: HistoryQueryParams = {}) => {
    const cleanParams: Record<string, string> = {};
    if (params.search) cleanParams.search = params.search;
    if (params.disease && params.disease !== 'ALL') cleanParams.disease = params.disease;
    if (params.severity && params.severity !== 'ALL') cleanParams.severity = params.severity;
    if (params.start_date) cleanParams.start_date = params.start_date;
    if (params.end_date) cleanParams.end_date = params.end_date;
    const res = await apiClient.get('/api/history/export/csv', {
      params: cleanParams,
      responseType: 'blob',
    });
    const blobUrl = window.URL.createObjectURL(new Blob([res.data], { type: 'text/csv;charset=utf-8;' }));
    const link = document.createElement('a');
    link.href = blobUrl;
    link.setAttribute('download', 'cotton_leaf_history.csv');
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(blobUrl);
  },
  getPaginatedHistory: async (params: HistoryQueryParams = {}): Promise<PaginatedHistoryResponse> => {
    const cleanParams: Record<string, string | number> = {};
    if (params.search) cleanParams.search = params.search;
    if (params.disease && params.disease !== 'ALL') cleanParams.disease = params.disease;
    if (params.severity && params.severity !== 'ALL') cleanParams.severity = params.severity;
    if (params.start_date) cleanParams.start_date = params.start_date;
    if (params.end_date) cleanParams.end_date = params.end_date;
    cleanParams.page = params.page || 1;
    cleanParams.page_size = params.page_size || 10;
    const res = await apiClient.get<PaginatedHistoryResponse>('/api/history', { params: cleanParams });
    return res.data;
  },

  // Dashboard
  getDashboardStats: async () => {
    const res = await apiClient.get('/api/dashboard/stats');
    return res.data;
  },

  // Authenticated PDF Report Generation & Download
  generateReport: async (id: number): Promise<GeneratedReportMetadata> => {
    const res = await apiClient.post<GeneratedReportMetadata>(`/api/reports/${id}/generate`);
    return res.data;
  },
  downloadReport: async (id: number) => {
    const res = await apiClient.get(`/api/reports/${id}`, { responseType: 'blob' });
    const blobUrl = window.URL.createObjectURL(new Blob([res.data], { type: 'application/pdf' }));
    const link = document.createElement('a');
    link.href = blobUrl;
    link.setAttribute('download', `cottonguard_report_${id}.pdf`);
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(blobUrl);
  },

  // Admin Management
  getAdminOverview: async () => {
    const res = await apiClient.get('/api/admin/overview');
    return res.data;
  },
  getAdminUsers: async () => {
    const res = await apiClient.get('/api/admin/users');
    return res.data;
  },
  getAdminModels: async () => {
    const res = await apiClient.get('/api/admin/models');
    return res.data;
  },
  updateUserRole: async (userId: number, role: 'farmer' | 'admin') => {
    const res = await apiClient.patch(`/api/admin/users/${userId}/role`, { role });
    return res.data;
  },
  updateUserStatus: async (userId: number, is_active: boolean) => {
    const res = await apiClient.patch(`/api/admin/users/${userId}/status`, { is_active });
    return res.data;
  },
  getAuditLogs: async (action?: string, limit: number = 50) => {
    const params: Record<string, string | number> = { limit };
    if (action && action !== 'ALL') params.action = action;
    const res = await apiClient.get('/api/admin/audit-logs', { params });
    return res.data;
  },
  getAdminSystem: async () => {
    const res = await apiClient.get('/api/admin/system');
    return res.data;
  },
};
