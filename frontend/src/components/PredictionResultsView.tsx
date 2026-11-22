import React, { useState } from 'react';
import { api, mediaUrl } from '../services/api';
import { GeneratedReportMetadata, PredictionDetail } from '../types/api';
import {
  extractApiErrorMessage,
  PROVISIONAL_SEVERITY_TIERS,
  VERIFIED_CLASSIFIER_METRICS,
} from '../utils/validation';
import { ModelStatusBadge, ProvisionalSeverityBadge } from './StatusBadges';
import {
  CheckCircle,
  AlertTriangle,
  FileText,
  RefreshCw,
  Info,
  Target,
  Layers,
  Eye,
  BarChart3,
  ShieldAlert,
  Clock,
  Download,
} from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const CLASS_LABELS = [
  'Bacterial Blight',
  'Curl Virus',
  'Healthy Leaf',
  'Herbicide Growth Damage',
  'Leaf Hopper Jassids',
  'Leaf Redding',
  'Leaf Variegation',
];

interface PredictionResultsViewProps {
  result: PredictionDetail;
  sourceFile?: File | null;
  onReset?: () => void;
  resetLabel?: string;
}

export const PredictionResultsView: React.FC<PredictionResultsViewProps> = ({
  result,
  sourceFile = null,
  onReset,
  resetLabel = 'Analyze Another Leaf',
}) => {
  const [activeTab, setActiveTab] = useState<'classification' | 'detection' | 'segmentation' | 'gradcam'>('gradcam');
  const [gradcamView, setGradcamView] = useState<'comparison' | 'overlay' | 'heatmap' | 'original'>('comparison');
  const [camOpacity, setCamOpacity] = useState<number>(45);
  const [targetCamClass, setTargetCamClass] = useState<string>('auto');
  const [colormap, setColormap] = useState<string>('jet');
  const [camLoading, setCamLoading] = useState(false);
  const [camError, setCamError] = useState<string | null>(null);
  const [customCam, setCustomCam] = useState<any | null>(null);
  const [overlayOpacity, setOverlayOpacity] = useState<number>(60);
  const [generatingPdf, setGeneratingPdf] = useState(false);
  const [downloadingPdf, setDownloadingPdf] = useState(false);
  const [reportNotice, setReportNotice] = useState<GeneratedReportMetadata | null>(null);
  const [reportError, setReportError] = useState<string | null>(null);

  const handleUpdateGradCam = async (nextClass: string, nextCmap: string) => {
    if (!sourceFile) return;
    setTargetCamClass(nextClass);
    setColormap(nextCmap);
    setCamLoading(true);
    setCamError(null);
    try {
      const targetIdx = nextClass === 'auto' ? undefined : Number(nextClass);
      const camRes = await api.getGradCAM(sourceFile, targetIdx, camOpacity / 100, nextCmap);
      setCustomCam(camRes);
    } catch (err) {
      setCamError(extractApiErrorMessage(err, 'Failed to recompute Grad-CAM saliency map.'));
    } finally {
      setCamLoading(false);
    }
  };

  const activeHeatmapUrl = mediaUrl(customCam?.heatmap_url || result.gradcam_heatmap_url);
  const activeOverlayUrl = mediaUrl(customCam?.overlay_url || result.gradcam_overlay_url);
  const activeComparisonUrl = mediaUrl(customCam?.comparison_url || result.gradcam_comparison_url);
  const resolvedImageUrl = mediaUrl(result.image_url);
  const activeTargetLabel = customCam?.target_class || result.predicted_class;

  const handleGeneratePdf = async () => {
    setGeneratingPdf(true);
    setReportError(null);
    try {
      const meta = await api.generateReport(result.id);
      setReportNotice(meta);
    } catch (err) {
      setReportError(extractApiErrorMessage(err, 'Failed to generate PDF diagnostic report.'));
    } finally {
      setGeneratingPdf(false);
    }
  };

  const handleDownloadReport = async () => {
    setDownloadingPdf(true);
    setReportError(null);
    try {
      await api.downloadReport(result.id);
    } catch (err) {
      setReportError(extractApiErrorMessage(err, 'Failed to download PDF diagnostic report.'));
    } finally {
      setDownloadingPdf(false);
    }
  };

  const probData = result.all_probabilities
    ? Object.entries(result.all_probabilities)
        .map(([name, val]) => ({
          name: name.replace('Herbicide Growth Damage', 'Herbicide').replace('Leaf Hopper Jassids', 'Jassids'),
          fullName: name,
          pct: Number((Number(val) * 100).toFixed(1)),
        }))
        .sort((a, b) => b.pct - a.pct)
    : [];

  const isSegAvailable = result.segmentation_status === 'SUCCESS' || result.segmentation_status === 'EXPERIMENTAL' || result.overlay_url !== null;
  const isDetAvailable = result.detection_status === 'SUCCESS' || result.detection_status === 'EXPERIMENTAL' || (result.detections && result.detections.length > 0);

  // GATE SAFEGUARD 1: REJECTED INPUT VIEW
  if (result.is_cotton_leaf === false || result.status?.startsWith('REJECTED')) {
    const isQualityFailure = result.status === 'REJECTED_QUALITY';
    const isNonPlant = result.status === 'REJECTED_NON_PLANT';
    const title = isQualityFailure
      ? 'Image Quality Insufficient'
      : isNonPlant
      ? 'Not a Cotton Leaf (Non-Plant Detected)'
      : 'Not a Cotton Leaf (Other Foliage Detected)';

    return (
      <div className="bg-white rounded-3xl border border-rose-200 p-6 sm:p-8 shadow-sm space-y-6">
        <div className="p-4 bg-rose-50 border border-rose-200 rounded-2xl flex items-start space-x-3.5">
          <ShieldAlert className="w-6 h-6 text-rose-600 flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <div className="flex flex-wrap items-center gap-2 mb-1">
              <span className="text-xs font-bold px-2.5 py-0.5 bg-rose-600 text-white rounded-full">
                {result.status}
              </span>
              <span className="text-xs font-semibold text-rose-700">
                Decision Gate: Analysis Halted
              </span>
            </div>
            <h3 className="text-xl font-bold text-rose-900">{title}</h3>
            <p className="text-sm text-rose-800 mt-1">
              {result.rejection_reason || result.user_message || 'The uploaded photograph could not be verified as a valid cotton leaf.'}
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
          <div className="text-center bg-slate-50 p-4 rounded-2xl border border-slate-200">
            <span className="text-xs font-bold text-slate-500 uppercase tracking-wider block mb-2">
              Uploaded Image (Rejected)
            </span>
            <img
              src={mediaUrl(result.image_url)}
              alt="Uploaded rejected input"
              className="max-h-64 mx-auto rounded-xl object-contain border-2 border-rose-300 shadow-sm"
            />
          </div>

          <div className="space-y-4">
            <h4 className="text-sm font-bold text-slate-800 uppercase tracking-wider">
              Verification Gate Safeguards
            </h4>
            <ul className="space-y-2 text-sm text-slate-600">
              <li className="flex items-start">
                <span className="text-rose-500 font-bold mr-2">✕</span>
                <span><strong>No False Disease Diagnostics:</strong> Disease classification models are blocked from evaluating invalid or non-cotton inputs.</span>
              </li>
              <li className="flex items-start">
                <span className="text-rose-500 font-bold mr-2">✕</span>
                <span><strong>No Saliency Artifacts:</strong> Grad-CAM heatmaps and U-Net segmentation masks are suppressed to prevent misleading visual cues.</span>
              </li>
              <li className="flex items-start">
                <span className="text-rose-500 font-bold mr-2">✕</span>
                <span><strong>Agronomic Advisory Protected:</strong> Chemical recommendations and treatment schedules are only generated for verified cotton crops.</span>
              </li>
            </ul>

            <div className="pt-2 flex flex-wrap gap-3">
              {onReset && (
                <button
                  type="button"
                  onClick={onReset}
                  className="px-5 py-2.5 bg-brand-600 hover:bg-brand-700 text-white rounded-xl font-bold text-sm shadow transition flex items-center"
                >
                  <RefreshCw className="w-4 h-4 mr-2" /> Scan Another Leaf
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  }

  // GATE SAFEGUARD 2: UNCERTAIN INPUT VIEW
  if (result.is_cotton_leaf === null || result.status === 'UNCERTAIN_IMAGE') {
    return (
      <div className="bg-white rounded-3xl border border-amber-200 p-6 sm:p-8 shadow-sm space-y-6">
        <div className="p-4 bg-amber-50 border border-amber-200 rounded-2xl flex items-start space-x-3.5">
          <AlertTriangle className="w-6 h-6 text-amber-600 flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <div className="flex flex-wrap items-center gap-2 mb-1">
              <span className="text-xs font-bold px-2.5 py-0.5 bg-amber-600 text-white rounded-full">
                UNCERTAIN_IMAGE
              </span>
              <span className="text-xs font-semibold text-amber-800">
                Verification Gate: Low Confidence
              </span>
            </div>
            <h3 className="text-xl font-bold text-amber-900">Cotton Leaf Verification Uncertain</h3>
            <p className="text-sm text-amber-800 mt-1">
              {result.rejection_reason || result.user_message || 'The system could not verify with sufficient confidence that this is a cotton leaf.'}
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
          <div className="text-center bg-slate-50 p-4 rounded-2xl border border-slate-200">
            <span className="text-xs font-bold text-slate-500 uppercase tracking-wider block mb-2">
              Uploaded Image (Uncertain)
            </span>
            <img
              src={mediaUrl(result.image_url)}
              alt="Uploaded input"
              className="max-h-64 mx-auto rounded-xl object-contain border-2 border-amber-300 shadow-sm"
            />
          </div>

          <div className="space-y-4">
            <h4 className="text-sm font-bold text-slate-800 uppercase tracking-wider">
              Recommendations for Clear Identification
            </h4>
            <ul className="space-y-2 text-sm text-slate-600">
              <li className="flex items-start">
                <span className="text-brand-600 font-bold mr-2">✓</span>
                <span>Capture a close-up photo of a single cotton leaf against the natural plant background.</span>
              </li>
              <li className="flex items-start">
                <span className="text-brand-600 font-bold mr-2">✓</span>
                <span>Ensure good natural daytime lighting without harsh flash glare.</span>
              </li>
              <li className="flex items-start">
                <span className="text-brand-600 font-bold mr-2">✓</span>
                <span>Hold the camera steady to avoid motion blur.</span>
              </li>
            </ul>

            <div className="pt-2 flex flex-wrap gap-3">
              {onReset && (
                <button
                  type="button"
                  onClick={onReset}
                  className="px-5 py-2.5 bg-brand-600 hover:bg-brand-700 text-white rounded-xl font-bold text-sm shadow transition flex items-center"
                >
                  <RefreshCw className="w-4 h-4 mr-2" /> Try Another Photo
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-3xl border border-slate-200 p-6 sm:p-8 shadow-sm space-y-8">
      {/* Header Summary */}
      <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center pb-6 border-b border-slate-100 gap-4">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs font-bold px-2.5 py-1 bg-brand-50 text-brand-700 border border-brand-200 rounded-full">
              Scan #{result.id}
            </span>
            <span className="text-xs font-medium px-2.5 py-1 bg-slate-100 text-slate-600 rounded-full">
              {result.model_version}
            </span>
            <span className="text-xs font-bold px-2.5 py-1 bg-emerald-50 text-emerald-700 border border-emerald-200 rounded-full flex items-center">
              <CheckCircle className="w-3.5 h-3.5 mr-1 text-emerald-600" /> Cotton Leaf Verified
            </span>
            <ModelStatusBadge status="TRAINED" />
            {result.inference_device && (
              <span className="text-[11px] font-mono px-2 py-0.5 bg-slate-100 text-slate-600 rounded-md">
                {result.inference_device}
              </span>
            )}
            {result.latency_ms && (
              <span className="inline-flex items-center text-xs text-slate-500">
                <Clock className="w-3.5 h-3.5 mr-1" /> {result.latency_ms.toFixed(0)} ms
              </span>
            )}
          </div>
          <h2 className="text-2xl sm:text-3xl font-extrabold text-slate-900 mt-2">{result.predicted_class}</h2>
          <p className="text-sm text-slate-500 mt-0.5">
            {result.recommendations?.pathogen_or_cause || 'EfficientNet-B0 deep learning foliar condition evaluation'}
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2.5">
          {onReset && (
            <button
              type="button"
              onClick={onReset}
              className="px-4 py-2 border border-slate-200 hover:bg-slate-50 text-slate-700 rounded-xl font-semibold text-sm transition flex items-center"
            >
              <RefreshCw className="w-4 h-4 mr-1.5" /> {resetLabel}
            </button>
          )}
          <button
            type="button"
            onClick={handleGeneratePdf}
            disabled={generatingPdf}
            className="px-4 py-2 border border-brand-200 bg-brand-50 hover:bg-brand-100 text-brand-800 rounded-xl font-semibold text-sm transition flex items-center disabled:opacity-60"
          >
            <FileText className="w-4 h-4 mr-1.5" />
            {generatingPdf ? 'Generating...' : 'Generate PDF'}
          </button>
          <button
            type="button"
            onClick={handleDownloadReport}
            disabled={downloadingPdf}
            className="px-4 py-2 bg-brand-600 hover:bg-brand-700 text-white rounded-xl font-semibold text-sm shadow-sm transition flex items-center disabled:opacity-60"
          >
            <Download className="w-4 h-4 mr-1.5" />
            {downloadingPdf ? 'Downloading...' : 'Download PDF'}
          </button>
        </div>
      </div>

      {reportNotice && (
        <div className="p-3.5 bg-emerald-50 border border-emerald-200 rounded-2xl text-emerald-800 text-xs flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center space-x-2">
            <CheckCircle className="w-4 h-4 text-emerald-600 flex-shrink-0" />
            <span>
              Report prepared cleanly: <strong>{reportNotice.pdf_filename}</strong> (
              {(reportNotice.file_size_bytes / 1024).toFixed(1)} KB).
            </span>
          </div>
          <button
            type="button"
            onClick={handleDownloadReport}
            className="font-bold underline hover:text-emerald-950"
          >
            Download Now
          </button>
        </div>
      )}

      {reportError && (
        <div className="p-3.5 bg-red-50 border border-red-200 rounded-2xl text-red-700 text-xs flex items-center space-x-2">
          <AlertTriangle className="w-4 h-4 flex-shrink-0" />
          <span>{reportError}</span>
        </div>
      )}

      {result.warning && (
        <div className="p-3.5 bg-amber-50 border border-amber-200 rounded-2xl text-amber-800 text-xs flex items-center space-x-2">
          <AlertTriangle className="w-4 h-4 flex-shrink-0" />
          <span>{result.warning}</span>
        </div>
      )}

      {/* 4 Core Distinction Cards (Confidence vs Detection vs Affected Area vs Provisional Severity) */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="p-5 bg-slate-50 rounded-2xl border border-slate-200/80 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">
              Classification Confidence
            </span>
            <ModelStatusBadge status="TRAINED" />
          </div>
          <div className="text-2xl font-extrabold text-slate-900">{(result.confidence * 100).toFixed(1)}%</div>
          <div className="w-full bg-slate-200 h-2 rounded-full overflow-hidden">
            <div className="bg-brand-600 h-full rounded-full" style={{ width: `${Math.min(100, result.confidence * 100)}%` }} />
          </div>
          <div className="text-[11px] text-slate-500">
            Softmax score (Benchmark: {VERIFIED_CLASSIFIER_METRICS.validationAccuracyPct} val /{' '}
            {VERIFIED_CLASSIFIER_METRICS.untouchedTestAccuracyPct} test)
          </div>
        </div>

        <div className="p-5 bg-slate-50 rounded-2xl border border-slate-200/80 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">
              {result.detection_model_version ? 'Disease Detection' : 'Object Detection'}
            </span>
            <ModelStatusBadge status={result.detection_status === 'EXPERIMENTAL' ? 'EXPERIMENTAL' : (isDetAvailable ? 'TRAINED' : 'NOT_TRAINED')} />
          </div>
          <div className="text-2xl font-extrabold text-slate-900">
            {isDetAvailable ? `${result.detection_count} Box(es)` : 'NOT_TRAINED'}
          </div>
          <div className="text-[11px] text-slate-500">
            {isDetAvailable
              ? result.detection_model_version
              : 'Awaiting CVAT bounding-box annotations (no synthetic boxes generated)'}
          </div>
        </div>

        <div className="p-5 bg-slate-50 rounded-2xl border border-slate-200/80 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">
              U-Net ResNet34 Area
            </span>
            <ModelStatusBadge status={result.segmentation_status === 'EXPERIMENTAL' ? 'EXPERIMENTAL' : (isSegAvailable ? 'TRAINED' : 'NOT_TRAINED')} />
          </div>
          <div className="text-2xl font-extrabold text-slate-900">
            {result.affected_area_percentage !== null && result.affected_area_percentage !== undefined
              ? `${result.affected_area_percentage}%`
              : 'NOT_TRAINED'}
          </div>
          <div className="text-[11px] text-slate-500">
            {result.leaf_pixels !== null && result.leaf_pixels !== undefined
              ? `${result.disease_pixels?.toLocaleString()} / ${result.leaf_pixels?.toLocaleString()} leaf px`
              : 'Awaiting CVAT pixel masks for U-Net ResNet34 training'}
          </div>
        </div>

        <div className="p-5 bg-slate-50 rounded-2xl border border-slate-200/80 space-y-2">
          <div className="flex items-center justify-between gap-1">
            <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">
              Severity Assessment
            </span>
            <ProvisionalSeverityBadge validated={Boolean(result.thresholds_verified)} />
          </div>
          <div className="text-xl font-extrabold text-slate-900">{result.severity_level}</div>
          <div className="text-[11px] text-slate-500 leading-snug">
            {result.severity_level === 'UNAVAILABLE'
              ? 'Provisional assessment blocked until trained U-Net mask output exists.'
              : result.thresholds_verified
              ? 'Agronomically verified category.'
              : 'PROVISIONAL (category_validated=false) — requires field calibration.'}
          </div>
        </div>
      </div>

      {/* Explicit Provisional Severity Disclaimer Banner */}
      <div className="p-4 bg-amber-50 border border-amber-200 rounded-2xl text-amber-900 text-xs space-y-2">
        <div className="flex items-start space-x-2.5">
          <ShieldAlert className="w-4 h-4 text-amber-600 flex-shrink-0 mt-0.5" />
          <div>
            <strong>PROVISIONAL SEVERITY NOTICE (category_validated = {String(Boolean(result.thresholds_verified))}):</strong>{' '}
            6-tier literature severity thresholds are provisional digital baselines pending field calibration against manually scored cotton leaves. Confidence % measures classifier probability, not physical damage area.
          </div>
        </div>
        <div className="flex flex-wrap gap-1.5 pl-6">
          {PROVISIONAL_SEVERITY_TIERS.map((tier) => (
            <span
              key={tier.label}
              className="px-2 py-0.5 rounded-md bg-white/80 border border-amber-200/80 text-[10px] font-semibold text-amber-900"
            >
              {tier.label}: {tier.range}
            </span>
          ))}
        </div>
      </div>

      {/* Multi-Model Workspace Tabs */}
      <div className="space-y-4">
        <div className="flex flex-wrap gap-2 border-b border-slate-200 pb-3">
          <button
            type="button"
            onClick={() => setActiveTab('gradcam')}
            className={`px-4 py-2 rounded-xl text-sm font-semibold flex items-center transition ${
              activeTab === 'gradcam' ? 'bg-slate-900 text-white shadow-sm' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            }`}
          >
            <Eye className="w-4 h-4 mr-1.5" /> Grad-CAM Explainability
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('classification')}
            className={`px-4 py-2 rounded-xl text-sm font-semibold flex items-center transition ${
              activeTab === 'classification' ? 'bg-slate-900 text-white shadow-sm' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            }`}
          >
            <BarChart3 className="w-4 h-4 mr-1.5" /> Classification Probabilities
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('detection')}
            className={`px-4 py-2 rounded-xl text-sm font-semibold flex items-center transition ${
              activeTab === 'detection' ? 'bg-slate-900 text-white shadow-sm' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            }`}
          >
            <Target className="w-4 h-4 mr-1.5" /> Faster R-CNN / YOLO Detection
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('segmentation')}
            className={`px-4 py-2 rounded-xl text-sm font-semibold flex items-center transition ${
              activeTab === 'segmentation' ? 'bg-slate-900 text-white shadow-sm' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            }`}
          >
            <Layers className="w-4 h-4 mr-1.5" /> U-Net ResNet34 Segmentation
          </button>
        </div>

        {/* TAB 1: GRAD-CAM EXPLAINABILITY */}
        {activeTab === 'gradcam' && (
          <div className="bg-slate-900 text-white p-6 rounded-3xl border border-slate-800 space-y-5">
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
              <div>
                <span className="text-xs font-bold uppercase tracking-wider px-2.5 py-1 bg-brand-500/20 text-brand-400 border border-brand-500/30 rounded-full">
                  EfficientNet-B0 features[-1] • Target: {activeTargetLabel}
                </span>
                <h3 className="text-lg sm:text-xl font-bold mt-2">Grad-CAM Saliency Map Visualization</h3>
              </div>

              <div className="flex flex-wrap bg-slate-800 p-1 rounded-xl text-xs font-semibold">
                {(['comparison', 'overlay', 'heatmap', 'original'] as const).map((view) => (
                  <button
                    key={view}
                    type="button"
                    onClick={() => setGradcamView(view)}
                    className={`px-3 py-1.5 rounded-lg capitalize transition ${
                      gradcamView === view ? 'bg-brand-600 text-white shadow' : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    {view === 'comparison' ? '3-Panel View' : view}
                  </button>
                ))}
              </div>
            </div>

            {camError && (
              <div className="p-3 bg-red-500/15 border border-red-500/40 rounded-xl text-red-200 text-xs">
                {camError}
              </div>
            )}

            {(sourceFile || gradcamView === 'overlay') && (
              <div className="flex flex-wrap items-center justify-between gap-4 bg-slate-800/70 px-4 py-3 rounded-2xl border border-slate-700/80 text-xs">
                {gradcamView === 'overlay' && (
                  <div className="flex items-center space-x-3">
                    <span className="font-semibold text-slate-300">Heatmap Opacity: {camOpacity}%</span>
                    <input
                      type="range"
                      min={10}
                      max={95}
                      value={camOpacity}
                      onChange={(e) => setCamOpacity(Number(e.target.value))}
                      className="accent-brand-500"
                    />
                  </div>
                )}

                {sourceFile && (
                  <div className="flex flex-wrap items-center gap-3">
                    <label className="flex items-center space-x-2 text-slate-300">
                      <span>Target Class:</span>
                      <select
                        value={targetCamClass}
                        disabled={camLoading}
                        onChange={(e) => handleUpdateGradCam(e.target.value, colormap)}
                        className="bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1 text-white"
                      >
                        <option value="auto">Predicted ({result.predicted_class})</option>
                        {CLASS_LABELS.map((cls, idx) => (
                          <option key={cls} value={idx.toString()}>
                            {cls}
                          </option>
                        ))}
                      </select>
                    </label>

                    <label className="flex items-center space-x-2 text-slate-300">
                      <span>Colormap:</span>
                      <select
                        value={colormap}
                        disabled={camLoading}
                        onChange={(e) => handleUpdateGradCam(targetCamClass, e.target.value)}
                        className="bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1 text-white"
                      >
                        <option value="jet">jet</option>
                        <option value="turbo">turbo</option>
                        <option value="viridis">viridis</option>
                      </select>
                    </label>
                    {camLoading && <span className="text-brand-400 font-semibold">Updating saliency...</span>}
                  </div>
                )}
              </div>
            )}

            <div className="bg-slate-950/80 p-4 rounded-2xl border border-slate-800 flex flex-col items-center justify-center min-h-[280px]">
              {gradcamView === 'comparison' && activeComparisonUrl && (
                <img
                  src={activeComparisonUrl}
                  alt="Grad-CAM 3-Panel Comparison"
                  className="w-full max-h-[440px] object-contain rounded-xl border border-slate-800"
                />
              )}
              {gradcamView === 'comparison' && !activeComparisonUrl && (
                <img src={resolvedImageUrl} alt="Original leaf" className="max-h-80 object-contain rounded-xl" />
              )}
              {gradcamView === 'overlay' && (
                activeHeatmapUrl ? (
                  <div className="relative inline-block">
                    <img src={resolvedImageUrl} alt="Original Base" className="max-h-96 mx-auto rounded-xl object-contain" />
                    <img
                      src={activeHeatmapUrl}
                      alt="Grad-CAM Blended Overlay"
                      style={{ opacity: camOpacity / 100 }}
                      className="absolute inset-0 w-full h-full object-contain rounded-xl"
                    />
                  </div>
                ) : (
                  <img
                    src={activeOverlayUrl || resolvedImageUrl}
                    alt="Grad-CAM Overlay"
                    className="max-h-96 mx-auto rounded-xl object-contain"
                  />
                )
              )}
              {gradcamView === 'heatmap' && (
                <img
                  src={activeHeatmapUrl || resolvedImageUrl}
                  alt="Grad-CAM Heatmap"
                  className="max-h-96 mx-auto rounded-xl object-contain"
                />
              )}
              {gradcamView === 'original' && (
                <img
                  src={resolvedImageUrl}
                  alt="Original Photograph"
                  className="max-h-96 mx-auto rounded-xl object-contain"
                />
              )}
            </div>

            <div className="p-3.5 bg-amber-500/10 border border-amber-500/30 rounded-2xl text-amber-200 text-xs flex items-start space-x-2.5">
              <Info className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
              <span>
                <strong className="text-amber-300">SCIENTIFIC NOTICE:</strong> Grad-CAM highlights image regions that contributed to the EfficientNet-B0 classification decision. It is an interpretability aid only, not pixel-level segmentation or severity measurement.
              </span>
            </div>
          </div>
        )}

        {/* TAB 2: CLASSIFICATION PROBABILITIES */}
        {activeTab === 'classification' && (
          <div className="p-6 bg-slate-50 rounded-3xl border border-slate-200 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h3 className="text-lg font-bold text-slate-900">7-Class Softmax Probability Breakdown</h3>
              <span className="text-xs font-semibold text-slate-600">
                Audited Classifier • Val {VERIFIED_CLASSIFIER_METRICS.validationAccuracyPct} • Test{' '}
                {VERIFIED_CLASSIFIER_METRICS.untouchedTestAccuracyPct} • Macro F1{' '}
                {VERIFIED_CLASSIFIER_METRICS.testMacroF1Pct}
              </span>
            </div>
            {probData.length > 0 ? (
              <div className="h-72 bg-white p-4 rounded-2xl border border-slate-100">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={probData} layout="vertical" margin={{ top: 5, right: 30, left: 40, bottom: 5 }}>
                    <XAxis type="number" domain={[0, 100]} unit="%" tick={{ fontSize: 11 }} />
                    <YAxis type="category" dataKey="name" width={110} tick={{ fontSize: 12 }} />
                    <Tooltip formatter={(val: any) => [`${val}%`, 'Probability']} />
                    <Bar dataKey="pct" radius={[0, 6, 6, 0]}>
                      {probData.map((entry, idx) => (
                        <Cell key={entry.fullName} fill={idx === 0 ? '#059669' : '#94a3b8'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <p className="text-sm text-slate-500">Full per-class probability log unavailable for this historical record.</p>
            )}
          </div>
        )}

        {/* TAB 3: YOLOV8N DETECTION */}
        {activeTab === 'detection' && (
          <div className="p-6 bg-slate-50 rounded-3xl border border-slate-200 space-y-5">
            {isDetAvailable ? (
              <>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="bg-white p-4 rounded-2xl border border-slate-100 text-center">
                    <p className="text-xs font-bold text-slate-500 mb-2">Original Input Photograph</p>
                    <img src={resolvedImageUrl} alt="Original" className="max-h-72 mx-auto rounded-xl object-contain" />
                  </div>
                  <div className="bg-white p-4 rounded-2xl border border-slate-100 text-center">
                    <p className="text-xs font-bold text-slate-500 mb-2">YOLO Localized Disease Regions</p>
                    <img
                      src={mediaUrl(result.detection_image_url) || resolvedImageUrl}
                      alt="YOLO Detections"
                      className="max-h-72 mx-auto rounded-xl object-contain"
                    />
                  </div>
                </div>
                {result.detections && result.detections.length > 0 && (
                  <div className="overflow-x-auto bg-white rounded-2xl border border-slate-200">
                    <table className="w-full text-left text-xs text-slate-700">
                      <thead className="bg-slate-50 text-slate-500 uppercase border-b border-slate-200">
                        <tr>
                          <th className="py-2.5 px-4">Region #</th>
                          <th className="py-2.5 px-4">Detected Disease Class</th>
                          <th className="py-2.5 px-4">Confidence</th>
                          <th className="py-2.5 px-4">Bounding Box [x1, y1, x2, y2]</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100">
                        {result.detections.map((d, idx) => (
                          <tr key={idx}>
                            <td className="py-2.5 px-4 font-bold">#{idx + 1}</td>
                            <td className="py-2.5 px-4 font-semibold">{d.class_name}</td>
                            <td className="py-2.5 px-4">{(d.confidence * 100).toFixed(1)}%</td>
                            <td className="py-2.5 px-4 font-mono">{d.bbox.join(', ')}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </>
            ) : (
              <div className="bg-white p-8 rounded-2xl border border-dashed border-slate-300 text-center space-y-3">
                <Target className="w-9 h-9 text-amber-500 mx-auto" />
                <div className="flex justify-center">
                  <ModelStatusBadge status="NOT_TRAINED" />
                </div>
                <h4 className="text-base font-bold text-slate-800">
                  {result.detection_model_version ? `${result.detection_model_version}` : 'Lesion Detection Status: NOT_TRAINED'}
                </h4>
                <p className="text-sm text-slate-500 max-w-xl mx-auto">
                  {result.detection_notes ||
                    'Verified CVAT Diseased_Region bounding-box annotations are required before YOLOv8n training and activation. No synthetic boxes are fabricated.'}
                </p>
              </div>
            )}
          </div>
        )}

        {/* TAB 4: U-NET RESNET34 SEGMENTATION & SEVERITY PIXEL BREAKDOWN */}
        {activeTab === 'segmentation' && (
          <div className="p-6 bg-slate-50 rounded-3xl border border-slate-200 space-y-6">
            {isSegAvailable ? (
              <>
                <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                  <div className="grid grid-cols-3 gap-3 w-full sm:w-auto text-center">
                    <div className="bg-white px-4 py-2.5 rounded-xl border border-slate-200">
                      <div className="text-[11px] font-semibold text-slate-500">Total Leaf px</div>
                      <div className="font-bold text-slate-900">{result.leaf_pixels?.toLocaleString() ?? '—'}</div>
                    </div>
                    <div className="bg-white px-4 py-2.5 rounded-xl border border-slate-200">
                      <div className="text-[11px] font-semibold text-emerald-600">Healthy px</div>
                      <div className="font-bold text-emerald-700">{result.healthy_pixels?.toLocaleString() ?? '—'}</div>
                    </div>
                    <div className="bg-white px-4 py-2.5 rounded-xl border border-slate-200">
                      <div className="text-[11px] font-semibold text-red-600">Diseased px</div>
                      <div className="font-bold text-red-700">{result.disease_pixels?.toLocaleString() ?? '—'}</div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-3 bg-white px-4 py-2 rounded-xl border border-slate-200">
                    <label className="text-xs font-semibold text-slate-600">Overlay Opacity: {overlayOpacity}%</label>
                    <input
                      type="range"
                      min={10}
                      max={100}
                      value={overlayOpacity}
                      onChange={(e) => setOverlayOpacity(Number(e.target.value))}
                      className="accent-brand-600"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                  <div className="bg-white p-3 rounded-2xl border border-slate-200 text-center">
                    <p className="text-xs font-semibold text-slate-500 mb-2">1. Original</p>
                    <img src={resolvedImageUrl} alt="Original" className="max-h-52 mx-auto rounded-lg object-contain" />
                  </div>
                  <div className="bg-white p-3 rounded-2xl border border-slate-200 text-center">
                    <p className="text-xs font-semibold text-slate-500 mb-2">2. Leaf Mask</p>
                    <img src={mediaUrl(result.leaf_mask_url)} alt="Leaf mask" className="max-h-52 mx-auto rounded-lg object-contain" />
                  </div>
                  <div className="bg-white p-3 rounded-2xl border border-slate-200 text-center">
                    <p className="text-xs font-semibold text-slate-500 mb-2">3. Disease Mask</p>
                    <img src={mediaUrl(result.disease_mask_url)} alt="Disease mask" className="max-h-52 mx-auto rounded-lg object-contain" />
                  </div>
                  <div className="bg-white p-3 rounded-2xl border border-slate-200 text-center">
                    <p className="text-xs font-semibold text-slate-500 mb-2">4. Interactive Overlay</p>
                    <div className="relative inline-block">
                      <img src={resolvedImageUrl} alt="Base" className="max-h-52 mx-auto rounded-lg object-contain" />
                      {result.overlay_url && (
                        <img
                          src={mediaUrl(result.overlay_url)}
                          alt="Overlay"
                          style={{ opacity: overlayOpacity / 100 }}
                          className="absolute inset-0 w-full h-full object-contain rounded-lg"
                        />
                      )}
                    </div>
                  </div>
                </div>
              </>
            ) : (
              <div className="bg-white p-8 rounded-2xl border border-dashed border-slate-300 text-center space-y-3">
                <Layers className="w-9 h-9 text-amber-500 mx-auto" />
                <div className="flex justify-center">
                  <ModelStatusBadge status="NOT_TRAINED" />
                </div>
                <h4 className="text-base font-bold text-slate-800">
                  U-Net ResNet34 3-Class Segmentation Status: NOT_TRAINED
                </h4>
                <p className="text-sm text-slate-500 max-w-xl mx-auto">
                  {result.uncertainty_notes ||
                    'Requires human-annotated CVAT pixel masks in data/annotations/ before U-Net ResNet34 training and 2D severity % calculation become active.'}
                </p>
              </div>
            )}
          </div>
        )}
      </div>

      {/* ICAR-CICR Agronomic Recommendations */}
      {result.recommendations && (
        <div className="space-y-5 pt-4 border-t border-slate-100">
          <h3 className="text-xl font-bold text-slate-900">Agronomic Guidance (ICAR-CICR Aligned)</h3>

          {result.recommendations.confidence_warning && (
            <div className="p-4 bg-amber-50 border border-amber-200 rounded-2xl flex items-start space-x-3 text-amber-800 text-sm">
              <AlertTriangle className="w-5 h-5 flex-shrink-0 mt-0.5" />
              <span>{result.recommendations.confidence_warning}</span>
            </div>
          )}

          {result.recommendations.immediate_first_steps && (
            <div>
              <h4 className="text-sm font-bold text-slate-800 mb-2">Recommended Field Scouting Actions:</h4>
              <ul className="space-y-2">
                {result.recommendations.immediate_first_steps.map((step: string, i: number) => (
                  <li key={i} className="flex items-start text-sm text-slate-700 bg-slate-50 p-3 rounded-xl border border-slate-100">
                    <CheckCircle className="w-4 h-4 text-brand-600 mr-2.5 flex-shrink-0 mt-0.5" />
                    <span>{step}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {result.recommendations.preventive_management && (
            <div>
              <h4 className="text-sm font-bold text-slate-800 mb-2">Preventive Cultural Practices:</h4>
              <ul className="space-y-2">
                {result.recommendations.preventive_management.map((prev: string, i: number) => (
                  <li key={i} className="flex items-start text-sm text-slate-700 bg-slate-50 p-3 rounded-xl border border-slate-100">
                    <CheckCircle className="w-4 h-4 text-slate-500 mr-2.5 flex-shrink-0 mt-0.5" />
                    <span>{prev}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {result.recommendations.critical_warning && (
            <div className="p-4 bg-red-50 border border-red-200 rounded-2xl flex items-start space-x-3 text-red-900 text-sm">
              <ShieldAlert className="w-5 h-5 flex-shrink-0 mt-0.5 text-red-600" />
              <div>
                <strong>Important Safety Warning: </strong>
                {result.recommendations.critical_warning}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
