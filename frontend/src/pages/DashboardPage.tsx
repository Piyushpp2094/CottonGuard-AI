import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { api, mediaUrl } from '../services/api';
import { BenchmarkReport, HealthModelsResponse } from '../types/api';
import { ModelStatusBadge, VerifiedClassifierMetricsStrip } from '../components/StatusBadges';
import {
  extractApiErrorMessage,
  formatModelStatusLabel,
  VERIFIED_CLASSIFIER_METRICS,
} from '../utils/validation';
import {
  ScanLine,
  CheckCircle2,
  AlertTriangle,
  Cpu,
  FileText,
  ArrowUpRight,
  Clock,
  ShieldAlert,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  LineChart,
  Line,
  CartesianGrid,
} from 'recharts';

const COLORS = ['#059669', '#ef4444', '#f97316', '#eab308', '#8b5cf6', '#ec4899', '#06b6d4'];

export const DashboardPage: React.FC = () => {
  const [stats, setStats] = useState<any>(null);
  const [modelsHealth, setModelsHealth] = useState<HealthModelsResponse | null>(null);
  const [benchmark, setBenchmark] = useState<BenchmarkReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      api.getDashboardStats(),
      api.getHealthModels().catch(() => null),
      api.getBenchmarkReport(),
    ])
      .then(([statsRes, healthRes, benchRes]) => {
        setStats(statsRes);
        setModelsHealth(healthRes);
        setBenchmark(benchRes);
      })
      .catch((err) => setError(extractApiErrorMessage(err, 'Failed to load dashboard telemetry.')))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[50vh]">
        <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-brand-600" />
      </div>
    );
  }

  const chartData = stats?.condition_distribution
    ? Object.entries(stats.condition_distribution).map(([name, count]) => ({
        name: name.replace('Herbicide Growth Damage', 'Herbicide').replace('Leaf Hopper Jassids', 'Jassids'),
        fullName: name,
        count,
      }))
    : [];

  const hw = modelsHealth?.hardware || stats?.model_readiness?.hardware;
  const registryModels = modelsHealth?.models;
  const runtimeModels = stats?.model_readiness?.models;

  const benchClassifier = benchmark?.models?.classifier_efficientnet_b0;
  const liveValAcc = benchClassifier?.live_validation_benchmark?.validation_metrics?.accuracy;
  const testAcc =
    benchClassifier?.pre_audited_holdout_test_metrics?.overall_performance?.accuracy ??
    registryModels?.classifier?.test_accuracy;
  const testF1 =
    benchClassifier?.pre_audited_holdout_test_metrics?.overall_performance?.macro_f1 ??
    registryModels?.classifier?.macro_f1;

  const valPctStr = liveValAcc ? `${(liveValAcc * 100).toFixed(2)}%` : VERIFIED_CLASSIFIER_METRICS.validationAccuracyPct;
  const testPctStr = testAcc ? `${(testAcc * 100).toFixed(2)}%` : VERIFIED_CLASSIFIER_METRICS.untouchedTestAccuracyPct;
  const f1PctStr = testF1 ? `${(testF1 * 100).toFixed(2)}%` : VERIFIED_CLASSIFIER_METRICS.testMacroF1Pct;

  const moduleCards: Array<{
    key: string;
    name: string;
    status: 'TRAINED' | 'NOT_TRAINED' | 'ACTIVE';
    detail: string;
  }> = [
    {
      key: 'classifier',
      name: registryModels?.classifier?.name || 'EfficientNet-B0',
      status: formatModelStatusLabel(registryModels?.classifier?.status, runtimeModels?.classification?.ready),
      detail: `7-class classifier • Val ${valPctStr} • Test ${testPctStr} • Macro F1 ${f1PctStr}`,
    },
    {
      key: 'explainability',
      name: 'Grad-CAM (features[-1])',
      status: runtimeModels?.explainability?.ready ? 'ACTIVE' : 'NOT_TRAINED',
      detail: 'Interactive saliency interpretability map (not pixel segmentation)',
    },
    {
      key: 'detector',
      name: registryModels?.detector?.name || 'YOLOv8n',
      status: formatModelStatusLabel(registryModels?.detector?.status, runtimeModels?.detection?.ready),
      detail:
        registryModels?.detector?.status === 'trained'
          ? 'Trained bounding-box lesion detector'
          : 'NOT_TRAINED — requires verified CVAT bounding-box annotations',
    },
    {
      key: 'segmentor',
      name: registryModels?.segmentor?.name || 'UNet-ResNet34',
      status: formatModelStatusLabel(registryModels?.segmentor?.status, runtimeModels?.segmentation?.ready),
      detail:
        registryModels?.segmentor?.status === 'trained'
          ? 'Trained 3-class leaf & lesion segmentor'
          : 'NOT_TRAINED — requires verified CVAT pixel masks (512×512)',
    },
  ];

  const handleDownloadPdf = async (predictionId: number) => {
    setError(null);
    try {
      await api.downloadReport(predictionId);
    } catch (err) {
      setError(extractApiErrorMessage(err, `Failed to download report for Scan #${predictionId}.`));
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 py-8 space-y-8">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
            CottonGuard AI Farmer Dashboard
          </h1>
          <p className="text-slate-500 text-sm mt-1">
            Real-time foliar metrics, audited classifier benchmarks, and honest module readiness telemetry
          </p>
        </div>
        <Link
          to="/analyze"
          className="flex items-center px-5 py-2.5 bg-brand-600 hover:bg-brand-700 text-white font-semibold rounded-xl shadow-sm transition"
        >
          <ScanLine className="w-4 h-4 mr-2" /> Analyze New Leaf
        </Link>
      </div>

      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-2xl text-sm text-red-700 flex items-center space-x-2">
          <AlertTriangle className="w-4 h-4 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Top 4 Real KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
          <div className="flex justify-between items-start">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Total Analyses</span>
            <div className="p-2 bg-slate-100 rounded-xl text-slate-700">
              <ScanLine className="w-5 h-5" />
            </div>
          </div>
          <div className="text-3xl font-bold text-slate-900 mt-3">{stats?.total_analyzed ?? 0}</div>
          <div className="text-xs text-slate-500 mt-1">Logged in local database</div>
        </div>

        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
          <div className="flex justify-between items-start">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Healthy Leaves</span>
            <div className="p-2 bg-emerald-50 rounded-xl text-emerald-600">
              <CheckCircle2 className="w-5 h-5" />
            </div>
          </div>
          <div className="text-3xl font-bold text-emerald-600 mt-3">{stats?.healthy_count ?? 0}</div>
          <div className="text-xs text-slate-500 mt-1">Asymptomatic foliage</div>
        </div>

        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
          <div className="flex justify-between items-start">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Diseased / Stressed</span>
            <div className="p-2 bg-red-50 rounded-xl text-red-600">
              <AlertTriangle className="w-5 h-5" />
            </div>
          </div>
          <div className="text-3xl font-bold text-red-600 mt-3">{stats?.diseased_count ?? 0}</div>
          <div className="text-xs text-slate-500 mt-1">Pathogen / pest / abiotic flags</div>
        </div>

        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
          <div className="flex justify-between items-start">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Inference Engine</span>
            <div className="p-2 bg-brand-50 rounded-xl text-brand-600">
              <Cpu className="w-5 h-5" />
            </div>
          </div>
          <div className="text-base font-bold text-slate-900 mt-3 truncate">
            {hw?.cuda_available ? hw.device_name : 'CPU Fallback'}
          </div>
          <div className={`text-xs font-medium mt-1 ${hw?.cuda_available ? 'text-emerald-600' : 'text-amber-600'}`}>
            {hw?.cuda_available ? `CUDA Active • ${hw.total_vram_mb} MB VRAM` : 'Running on CPU'}
          </div>
        </div>
      </div>

      {/* Verified Classifier Benchmark Strip + Module Registry */}
      <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h2 className="text-base font-bold text-slate-900">Verified EfficientNet-B0 Benchmark &amp; Module Readiness</h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Audited checkpoint metrics alongside live registry status for detector &amp; segmentor gates
            </p>
          </div>
        </div>

        <VerifiedClassifierMetricsStrip
          validationAccPct={valPctStr}
          testAccPct={testPctStr}
          testMacroF1Pct={f1PctStr}
          compact
        />

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {moduleCards.map((mod) => (
            <div key={mod.key} className="p-4 rounded-xl bg-slate-50 border border-slate-200/80 space-y-1.5">
              <div className="flex items-center justify-between gap-2">
                <span className="text-[11px] font-bold uppercase text-slate-500">{mod.key}</span>
                <ModelStatusBadge status={mod.status} />
              </div>
              <div className="text-sm font-bold text-slate-900">{mod.name}</div>
              <div className="text-[11px] text-slate-600 leading-snug">{mod.detail}</div>
            </div>
          ))}
        </div>

        <div className="p-3.5 bg-amber-50 border border-amber-200 rounded-xl text-amber-900 text-xs flex items-start space-x-2">
          <ShieldAlert className="w-4 h-4 text-amber-600 flex-shrink-0 mt-0.5" />
          <span>
            <strong>Severity Estimation Disclaimer:</strong> Severity grading uses provisional digital thresholds (<code>category_validated=false</code>) and stays <code>UNAVAILABLE</code> until U-Net ResNet34 is trained on verified CVAT pixel masks.
          </span>
        </div>
      </div>

      {/* Charts: Condition Distribution & Trends */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
          <h2 className="text-lg font-bold text-slate-900 mb-4">Disease Distribution</h2>
          {chartData.length > 0 ? (
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 25 }}>
                  <XAxis dataKey="name" tick={{ fontSize: 11 }} angle={-18} textAnchor="end" />
                  <YAxis allowDecimals={false} />
                  <Tooltip formatter={(value: any) => [value, 'Scans']} />
                  <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                    {chartData.map((_, index) => (
                      <Cell key={index} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="h-64 flex items-center justify-center text-slate-400 text-sm">
              Analyze a leaf sample to populate distribution charts.
            </div>
          )}
        </div>

        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
          <h2 className="text-lg font-bold text-slate-900 mb-4">Analysis Activity Trend</h2>
          {stats?.trends && stats.trends.length > 0 ? (
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={stats.trends} margin={{ top: 10, right: 15, left: -20, bottom: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Line type="monotone" dataKey="total" name="Total Scans" stroke="#059669" strokeWidth={2.5} />
                  <Line type="monotone" dataKey="diseased" name="Diseased" stroke="#ef4444" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="h-64 flex items-center justify-center text-slate-400 text-sm">
              No chronological scan trend recorded yet.
            </div>
          )}
        </div>
      </div>

      {/* Recent Predictions & Reports */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-lg font-bold text-slate-900">Recent Predictions</h2>
            <Link to="/history" className="text-xs font-semibold text-brand-600 hover:text-brand-700 flex items-center">
              Full History <ArrowUpRight className="w-3.5 h-3.5 ml-0.5" />
            </Link>
          </div>
          <div className="space-y-3">
            {stats?.recent_predictions && stats.recent_predictions.length > 0 ? (
              stats.recent_predictions.map((p: any) => (
                <div key={p.id} className="p-3 bg-slate-50 rounded-xl flex justify-between items-center border border-slate-100">
                  <Link to={`/results/${p.id}`} className="flex items-center space-x-3 group min-w-0">
                    <img src={mediaUrl(p.image_url)} alt="Scan" className="w-10 h-10 rounded-lg object-cover border border-slate-200 flex-shrink-0" />
                    <div className="min-w-0">
                      <div className="font-semibold text-sm text-slate-900 group-hover:text-brand-700 truncate">
                        {p.predicted_class}
                      </div>
                      <div className="text-xs text-slate-500">
                        Conf: {(p.confidence * 100).toFixed(1)}% • Severity: {p.severity_level}
                      </div>
                    </div>
                  </Link>
                  <button
                    type="button"
                    onClick={() => handleDownloadPdf(p.id)}
                    className="p-2 text-slate-600 hover:text-brand-600 rounded-lg hover:bg-white transition flex-shrink-0"
                    title="Download PDF Report"
                  >
                    <FileText className="w-4 h-4" />
                  </button>
                </div>
              ))
            ) : (
              <div className="text-slate-400 text-sm text-center py-8">No predictions recorded yet.</div>
            )}
          </div>
        </div>

        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
          <h2 className="text-lg font-bold text-slate-900 mb-4">Recent Generated PDF Reports</h2>
          <div className="space-y-3">
            {stats?.recent_reports && stats.recent_reports.length > 0 ? (
              stats.recent_reports.map((r: any) => (
                <div key={r.id} className="p-3 bg-slate-50 rounded-xl flex justify-between items-center border border-slate-100">
                  <div>
                    <div className="font-semibold text-sm text-slate-900">
                      Report for Scan #{r.prediction_id} ({r.predicted_class})
                    </div>
                    <div className="text-xs text-slate-500 flex items-center mt-0.5">
                      <Clock className="w-3 h-3 mr-1" /> {new Date(r.generated_at).toLocaleString()} •{' '}
                      {(r.file_size_bytes / 1024).toFixed(1)} KB
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => handleDownloadPdf(r.prediction_id)}
                    className="px-3 py-1.5 rounded-lg bg-white border border-slate-200 text-xs font-semibold text-slate-700 hover:text-brand-700 hover:border-brand-300"
                  >
                    Download
                  </button>
                </div>
              ))
            ) : (
              <div className="text-slate-400 text-sm text-center py-8">
                No PDF reports generated yet. Download a report from any prediction result.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
