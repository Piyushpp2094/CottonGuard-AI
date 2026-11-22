import React, { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../services/api';
import { HealthModelsResponse, PredictionDetail, SseProgressData } from '../types/api';
import { PredictionResultsView } from '../components/PredictionResultsView';
import { ModelStatusBadge } from '../components/StatusBadges';
import {
  extractApiErrorMessage,
  formatModelStatusLabel,
  MAX_IMAGE_DIM,
  MAX_UPLOAD_SIZE_MB,
  MIN_IMAGE_DIM,
  validateImageDimensions,
  validateUploadMetadata,
} from '../utils/validation';
import {
  UploadCloud,
  AlertTriangle,
  RefreshCw,
  CheckCircle2,
  Circle,
  Loader2,
  XCircle,
  ExternalLink,
} from 'lucide-react';

const PIPELINE_STAGES: Array<{ key: string; label: string; step: number }> = [
  { key: 'preprocessing', label: 'Image Validation & Preprocessing', step: 1 },
  { key: 'classification', label: 'EfficientNet-B0 7-Class Classification', step: 2 },
  { key: 'gradcam', label: 'Grad-CAM Saliency Map', step: 3 },
  { key: 'detection', label: 'YOLOv8n Lesion Localization Check', step: 4 },
  { key: 'segmentation', label: 'U-Net ResNet34 Segmentation Check', step: 5 },
  { key: 'severity', label: 'Provisional 6-Tier Severity Evaluation', step: 6 },
  { key: 'recommendations', label: 'ICAR-CICR Advisories & Database Save', step: 7 },
];

export const AnalyzePage: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [imageDims, setImageDims] = useState<{ width: number; height: number } | null>(null);
  const [segAlpha, setSegAlpha] = useState<number>(0.4);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionDetail | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);

  // Real-time SSE progress tracking state
  const [jobId, setJobId] = useState<string | null>(null);
  const [stageMap, setStageMap] = useState<Record<string, SseProgressData>>({});
  const [currentStep, setCurrentStep] = useState<number>(0);
  const [modelHealth, setModelHealth] = useState<HealthModelsResponse | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    api
      .getHealthModels()
      .then((res) => setModelHealth(res))
      .catch(() => {});
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  const handleFileSelect = (selectedFile: File) => {
    const metaCheck = validateUploadMetadata({
      name: selectedFile.name,
      size: selectedFile.size,
      type: selectedFile.type,
    });
    if (!metaCheck.valid) {
      setError(metaCheck.error || 'Unsupported file.');
      return;
    }

    const objectUrl = URL.createObjectURL(selectedFile);
    const probe = new Image();
    probe.onload = () => {
      const dimCheck = validateImageDimensions(probe.naturalWidth, probe.naturalHeight);
      if (!dimCheck.valid) {
        URL.revokeObjectURL(objectUrl);
        setError(dimCheck.error || 'Invalid image resolution.');
        return;
      }
      setError(null);
      setFile(selectedFile);
      setPreview(objectUrl);
      setImageDims({ width: probe.naturalWidth, height: probe.naturalHeight });
      setResult(null);
      setStageMap({});
      setCurrentStep(0);
    };
    probe.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      setError('Selected file could not be decoded as a valid image.');
    };
    probe.src = objectUrl;
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleCancelStream = () => {
    abortRef.current?.abort();
    abortRef.current = null;
    setLoading(false);
  };

  const handleAnalyze = async () => {
    if (!file) return;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setLoading(true);
    setError(null);
    setStageMap({});
    setCurrentStep(0);
    setJobId(null);

    try {
      const created = await api.analyzeImageAsync(file, segAlpha);
      setJobId(created.job_id);
      const finalDetail = await api.streamAnalysisJob(created.job_id, {
        signal: controller.signal,
        onProgress: (evt) => {
          setCurrentStep((prev) => Math.max(prev, evt.step));
          setStageMap((prev) => ({ ...prev, [evt.stage]: evt }));
        },
        onError: (errEvt) => {
          setError(errEvt.message || 'Streaming pipeline reported an error.');
        },
      });
      setResult(finalDetail);
    } catch (err) {
      if (!controller.signal.aborted) {
        setError(extractApiErrorMessage(err, 'Analysis failed. Verify backend server is online and retry.'));
      }
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null;
      }
      setLoading(false);
    }
  };

  const resetAll = () => {
    abortRef.current?.abort();
    if (preview) {
      URL.revokeObjectURL(preview);
    }
    setFile(null);
    setPreview(null);
    setImageDims(null);
    setResult(null);
    setError(null);
    setLoading(false);
    setJobId(null);
    setStageMap({});
    setCurrentStep(0);
  };

  const detStatus = formatModelStatusLabel(modelHealth?.models?.detector?.status);
  const segStatus = formatModelStatusLabel(modelHealth?.models?.segmentor?.status);
  const progressPct = Math.min(100, Math.round((currentStep / PIPELINE_STAGES.length) * 100));

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 space-y-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
            Cotton Leaf Diagnostic Analysis
          </h1>
          <p className="text-slate-500 text-sm mt-1">
            Streams real-time SSE diagnostic stages using EfficientNet-B0 + Grad-CAM and reports active module status honestly
          </p>
        </div>

        {/* Live Module Availability Pills */}
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-white border border-slate-200">
            <span className="font-semibold text-slate-700">Classifier</span>
            <ModelStatusBadge status="TRAINED" />
          </span>
          <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-white border border-slate-200">
            <span className="font-semibold text-slate-700">YOLOv8n</span>
            <ModelStatusBadge status={detStatus} />
          </span>
          <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-white border border-slate-200">
            <span className="font-semibold text-slate-700">U-Net ResNet34</span>
            <ModelStatusBadge status={segStatus} />
          </span>
        </div>
      </div>

      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-2xl flex flex-wrap items-center justify-between text-red-700 text-sm gap-4">
          <div className="flex items-start space-x-3">
            <AlertTriangle className="w-5 h-5 flex-shrink-0 mt-0.5" />
            <span>{error}</span>
          </div>
          {file && !loading && (
            <button
              type="button"
              onClick={handleAnalyze}
              className="px-3 py-1.5 bg-red-600 text-white rounded-xl text-xs font-semibold hover:bg-red-700 flex-shrink-0"
            >
              Retry Analysis
            </button>
          )}
        </div>
      )}

      {!result ? (
        <div className="bg-white rounded-3xl border border-slate-200 p-6 sm:p-8 shadow-sm space-y-6">
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragOver(true);
            }}
            onDragLeave={() => setIsDragOver(false)}
            onDrop={handleDrop}
            onClick={() => !loading && fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-2xl p-8 sm:p-12 text-center transition flex flex-col items-center justify-center ${
              loading ? 'cursor-not-allowed opacity-80 border-slate-200 bg-slate-50/40' : 'cursor-pointer'
            } ${
              isDragOver ? 'border-brand-500 bg-brand-50/60' : 'border-slate-300 hover:border-brand-400 hover:bg-slate-50'
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp"
              className="hidden"
              disabled={loading}
              onChange={(e) => e.target.files && e.target.files[0] && handleFileSelect(e.target.files[0])}
            />

            {preview ? (
              <div className="space-y-3">
                <img
                  src={preview}
                  alt="Selected Leaf Preview"
                  className="max-h-72 w-auto mx-auto rounded-xl object-contain shadow-md"
                />
                <div className="flex flex-wrap items-center justify-center gap-2 text-xs text-slate-600 font-medium">
                  <span>{file?.name}</span>
                  <span>•</span>
                  <span>{((file?.size || 0) / 1024).toFixed(1)} KB</span>
                  {imageDims && (
                    <>
                      <span>•</span>
                      <span>
                        {imageDims.width}×{imageDims.height} px
                      </span>
                    </>
                  )}
                  <span className="text-brand-700 font-semibold">— Click or drop to replace</span>
                </div>
              </div>
            ) : (
              <>
                <div className="w-16 h-16 bg-brand-50 text-brand-600 rounded-2xl flex items-center justify-center mb-4">
                  <UploadCloud className="w-8 h-8" />
                </div>
                <h3 className="text-lg font-bold text-slate-900">Drop cotton leaf image here, or browse</h3>
                <p className="text-xs text-slate-500 mt-1.5 max-w-md">
                  Supports JPG, JPEG, PNG, and WEBP up to {MAX_UPLOAD_SIZE_MB} MB ({MIN_IMAGE_DIM}px–{MAX_IMAGE_DIM}px resolution). Clear natural lighting yields the best reliability.
                </p>
              </>
            )}
          </div>

          {preview && (
            <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-2">
              <div className="flex items-center space-x-3 text-xs text-slate-600">
                <span className="font-semibold">Overlay Blend Alpha: {Math.round(segAlpha * 100)}%</span>
                <input
                  type="range"
                  min={0.1}
                  max={0.9}
                  step={0.1}
                  value={segAlpha}
                  disabled={loading}
                  onChange={(e) => setSegAlpha(Number(e.target.value))}
                  className="accent-brand-600"
                />
              </div>
              <div className="flex gap-3 w-full sm:w-auto justify-end">
                {loading ? (
                  <button
                    type="button"
                    onClick={handleCancelStream}
                    className="px-5 py-2.5 rounded-xl border border-red-200 text-red-700 hover:bg-red-50 font-semibold text-sm transition flex items-center"
                  >
                    <XCircle className="w-4 h-4 mr-1.5" /> Cancel
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={resetAll}
                    className="px-5 py-2.5 rounded-xl border border-slate-200 text-slate-600 hover:bg-slate-50 font-semibold text-sm transition"
                  >
                    Clear
                  </button>
                )}
                <button
                  type="button"
                  onClick={handleAnalyze}
                  disabled={loading}
                  className="px-6 py-2.5 rounded-xl bg-brand-600 hover:bg-brand-700 text-white font-semibold text-sm shadow-md shadow-brand-500/20 transition disabled:opacity-50 flex items-center justify-center"
                >
                  {loading ? (
                    <>
                      <RefreshCw className="w-4 h-4 mr-2 animate-spin" /> Streaming Analysis...
                    </>
                  ) : (
                    'Analyze Foliar Health (SSE)'
                  )}
                </button>
              </div>
            </div>
          )}

          {/* Real-Time SSE Progress Panel */}
          {(loading || Object.keys(stageMap).length > 0) && (
            <div className="bg-slate-900 text-white rounded-2xl p-5 sm:p-6 border border-slate-800 space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <div className="text-xs font-bold uppercase tracking-wider text-brand-400">
                    Real-Time SSE Diagnostic Stream {jobId ? `• Job ${jobId.slice(0, 8)}` : ''}
                  </div>
                  <div className="text-sm font-semibold text-slate-200 mt-0.5">
                    Step {currentStep} of {PIPELINE_STAGES.length} ({progressPct}%)
                  </div>
                </div>
                <div className="w-full sm:w-56 bg-slate-800 h-2.5 rounded-full overflow-hidden">
                  <div
                    className="bg-brand-500 h-full transition-all duration-300"
                    style={{ width: `${progressPct}%` }}
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5 text-xs">
                {PIPELINE_STAGES.map((stageDef) => {
                  const evt = stageMap[stageDef.key];
                  const isDone = evt?.status === 'complete';
                  const isRunning = evt?.status === 'running';
                  const summary = evt?.result_summary as Record<string, any> | undefined;

                  return (
                    <div
                      key={stageDef.key}
                      className={`p-3 rounded-xl border flex items-center justify-between gap-2 ${
                        isDone
                          ? 'bg-emerald-950/40 border-emerald-700/50 text-emerald-100'
                          : isRunning
                          ? 'bg-brand-950/50 border-brand-500/60 text-white'
                          : 'bg-slate-800/50 border-slate-700/60 text-slate-400'
                      }`}
                    >
                      <div className="flex items-center space-x-2.5 min-w-0">
                        {isDone ? (
                          <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0" />
                        ) : isRunning ? (
                          <Loader2 className="w-4 h-4 text-brand-400 animate-spin flex-shrink-0" />
                        ) : (
                          <Circle className="w-4 h-4 text-slate-600 flex-shrink-0" />
                        )}
                        <span className="truncate font-medium">
                          {stageDef.step}. {stageDef.label}
                        </span>
                      </div>
                      {summary && (
                        <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-950/60 text-slate-200 flex-shrink-0">
                          {summary.predicted_class
                            ? `${summary.predicted_class} (${Math.round(Number(summary.confidence) * 100)}%)`
                            : summary.status || summary.severity_level || (summary.resolution ? `${summary.resolution[0]}×${summary.resolution[1]}` : 'done')}
                        </span>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="space-y-4">
          <div className="flex justify-end">
            <Link
              to={`/results/${result.id}`}
              className="inline-flex items-center text-xs font-semibold text-brand-700 hover:text-brand-800 bg-brand-50 border border-brand-200 px-3 py-1.5 rounded-xl"
            >
              Open Shareable Result View <ExternalLink className="w-3.5 h-3.5 ml-1.5" />
            </Link>
          </div>
          <PredictionResultsView
            result={result}
            sourceFile={file}
            onReset={resetAll}
            resetLabel="Scan Another Leaf"
          />
        </div>
      )}
    </div>
  );
};
