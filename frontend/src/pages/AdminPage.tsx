import React, { useEffect, useState } from 'react';
import { api, mediaUrl } from '../services/api';
import { useAuth } from '../context/AuthContext';
import { BenchmarkReport } from '../types/api';
import { ModelStatusBadge, VerifiedClassifierMetricsStrip } from '../components/StatusBadges';
import {
  extractApiErrorMessage,
  formatModelStatusLabel,
  VERIFIED_CLASSIFIER_METRICS,
} from '../utils/validation';
import {
  Cpu,
  Database,
  Users,
  Activity,
  ShieldAlert,
  FileText,
  BarChart3,
  History,
  CheckCircle2,
  ShieldCheck,
} from 'lucide-react';

export const AdminPage: React.FC = () => {
  const { user } = useAuth();
  const [overview, setOverview] = useState<any>(null);
  const [users, setUsers] = useState<any[]>([]);
  const [system, setSystem] = useState<any>(null);
  const [adminModels, setAdminModels] = useState<any>(null);
  const [benchmark, setBenchmark] = useState<BenchmarkReport | null>(null);
  const [auditLogs, setAuditLogs] = useState<any[]>([]);
  const [auditFilter, setAuditFilter] = useState<string>('ALL');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const loadAdminData = async (nextAuditFilter: string = auditFilter) => {
    setLoading(true);
    setError(null);
    try {
      const [ov, usr, sys, mdl, bench, logs] = await Promise.all([
        api.getAdminOverview(),
        api.getAdminUsers(),
        api.getAdminSystem(),
        api.getAdminModels(),
        api.getBenchmarkReport(),
        api.getAuditLogs(nextAuditFilter, 40),
      ]);
      setOverview(ov);
      setUsers(usr);
      setSystem(sys);
      setAdminModels(mdl);
      setBenchmark(bench);
      setAuditLogs(logs);
    } catch (err) {
      setError(extractApiErrorMessage(err, 'Failed to load administrator telemetry.'));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (user?.role === 'admin') {
      loadAdminData();
    } else {
      setLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user]);

  const handleToggleRole = async (targetUser: any) => {
    const nextRole = targetUser.role === 'admin' ? 'farmer' : 'admin';
    try {
      await api.updateUserRole(targetUser.id, nextRole);
      await loadAdminData();
    } catch (err) {
      setError(extractApiErrorMessage(err, 'Role update failed.'));
    }
  };

  const handleToggleStatus = async (targetUser: any) => {
    const nextStatus = !targetUser.is_active;
    try {
      await api.updateUserStatus(targetUser.id, nextStatus);
      await loadAdminData();
    } catch (err) {
      setError(extractApiErrorMessage(err, 'Account status update failed.'));
    }
  };

  const handleAuditFilterChange = async (nextAction: string) => {
    setAuditFilter(nextAction);
    try {
      const logs = await api.getAuditLogs(nextAction, 40);
      setAuditLogs(logs);
    } catch (err) {
      setError(extractApiErrorMessage(err, 'Failed to filter audit logs.'));
    }
  };

  if (user?.role !== 'admin') {
    return (
      <div className="max-w-3xl mx-auto px-4 py-16 text-center space-y-4">
        <ShieldAlert className="w-12 h-12 text-amber-500 mx-auto" />
        <h1 className="text-2xl font-bold text-slate-900">Administrator Role Required</h1>
        <p className="text-sm text-slate-600">
          Your account (<strong>{user?.username}</strong>) currently holds role <code>{user?.role}</code>.
          Sign in with an administrator account to inspect user accounts, model provenance, and system monitoring telemetry.
        </p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[50vh]">
        <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-brand-600" />
      </div>
    );
  }

  const hw = overview?.system_status?.hardware;
  const registryModels = adminModels?.registry?.models;
  const runtimeModels = adminModels?.runtime?.models || overview?.system_status?.models;
  const ds = system?.dataset_summary;
  const clsBench = system?.classifier_benchmark;
  const perClass = system?.classifier_full_metrics?.per_class_metrics;
  const metricCharts = system?.metric_charts || {};
  const leakageVerdict = system?.leakage_report?.verdict || system?.leakage_report?.status;
  const provenanceVerdict = system?.provenance_report?.verdict || system?.provenance_report?.status;
  const ann = system?.annotation_summary;

  const benchClassifier = benchmark?.models?.classifier_efficientnet_b0;
  const valAcc = benchClassifier?.live_validation_benchmark?.validation_metrics?.accuracy;
  const testAcc =
    clsBench?.accuracy ??
    benchClassifier?.pre_audited_holdout_test_metrics?.overall_performance?.accuracy ??
    registryModels?.classifier?.test_accuracy;
  const macroF1 =
    clsBench?.macro_f1 ??
    benchClassifier?.pre_audited_holdout_test_metrics?.overall_performance?.macro_f1 ??
    registryModels?.classifier?.macro_f1;

  const valAccStr = valAcc ? `${(valAcc * 100).toFixed(2)}%` : VERIFIED_CLASSIFIER_METRICS.validationAccuracyPct;
  const testAccStr = testAcc ? `${(Number(testAcc) * 100).toFixed(2)}%` : VERIFIED_CLASSIFIER_METRICS.untouchedTestAccuracyPct;
  const macroF1Str = macroF1 ? `${(Number(macroF1) * 100).toFixed(2)}%` : VERIFIED_CLASSIFIER_METRICS.testMacroF1Pct;

  return (
    <div className="max-w-7xl mx-auto px-4 py-8 space-y-8">
      <div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
          CottonGuard AI Administrator Console
        </h1>
        <p className="text-slate-500 text-sm mt-1">
          Live database analytics, model provenance registry, GPU diagnostics, access control, and security audit trail
        </p>
      </div>

      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-2xl text-sm text-red-700">{error}</div>
      )}

      {/* Global Metrics */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm">
          <div className="flex items-center justify-between text-xs font-bold uppercase text-slate-500">
            <span>Registered Users</span> <Users className="w-4 h-4 text-brand-600" />
          </div>
          <div className="text-3xl font-extrabold text-slate-900 mt-2">{overview?.total_users ?? 0}</div>
        </div>
        <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm">
          <div className="flex items-center justify-between text-xs font-bold uppercase text-slate-500">
            <span>Total Platform Scans</span> <Activity className="w-4 h-4 text-brand-600" />
          </div>
          <div className="text-3xl font-extrabold text-slate-900 mt-2">{overview?.total_predictions ?? 0}</div>
        </div>
        <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm">
          <div className="flex items-center justify-between text-xs font-bold uppercase text-slate-500">
            <span>PDF Reports Saved</span> <FileText className="w-4 h-4 text-brand-600" />
          </div>
          <div className="text-3xl font-extrabold text-slate-900 mt-2">{overview?.total_reports ?? 0}</div>
        </div>
        <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm">
          <div className="flex items-center justify-between text-xs font-bold uppercase text-slate-500">
            <span>CUDA Device VRAM</span> <Cpu className="w-4 h-4 text-emerald-600" />
          </div>
          <div className="text-lg font-extrabold text-slate-900 mt-2">
            {hw?.cuda_available ? `${hw.allocated_vram_mb} / ${hw.total_vram_mb} MB` : 'CPU Mode'}
          </div>
        </div>
      </div>

      {/* Model Registry & Hardware Status */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white p-6 rounded-3xl border border-slate-200 shadow-sm space-y-4">
          <div className="flex items-center space-x-2 text-slate-900 font-bold text-lg">
            <ShieldCheck className="w-5 h-5 text-brand-600" />
            <span>Model Registry &amp; Checkpoint Provenance</span>
          </div>

          <div className="divide-y divide-slate-100 text-sm">
            <div className="flex justify-between py-2.5">
              <span className="text-slate-500">GPU Accelerator</span>
              <span className="font-semibold text-slate-800">{hw?.device_name || 'CPU Only'}</span>
            </div>

            <div className="py-3 space-y-1">
              <div className="flex items-center justify-between">
                <span className="font-semibold text-slate-800">
                  {registryModels?.classifier?.name || 'EfficientNet-B0'} (Classification)
                </span>
                <ModelStatusBadge
                  status={formatModelStatusLabel(registryModels?.classifier?.status, runtimeModels?.classification?.ready)}
                />
              </div>
              {registryModels?.classifier?.checkpoint_sha256 && (
                <div className="text-[11px] font-mono text-slate-500 break-all">
                  SHA-256: {registryModels.classifier.checkpoint_sha256}
                </div>
              )}
              {benchClassifier?.live_validation_benchmark && (
                <div className="text-[11px] text-slate-500">
                  Single-image GPU latency:{' '}
                  <strong>
                    {benchClassifier.live_validation_benchmark.single_image_mean_latency_ms} ms
                  </strong>{' '}
                  • Peak VRAM:{' '}
                  <strong>
                    {benchClassifier.live_validation_benchmark.peak_gpu_memory_mb} MB
                  </strong>
                </div>
              )}
            </div>

            <div className="py-3 space-y-1">
              <div className="flex items-center justify-between">
                <span className="font-semibold text-slate-800">
                  {registryModels?.detector?.name || 'YOLOv8n'} (Lesion Detection)
                </span>
                <ModelStatusBadge status={formatModelStatusLabel(registryModels?.detector?.status)} />
              </div>
              <div className="text-[11px] text-slate-500">
                {registryModels?.detector?.annotation_requirements ||
                  'Requires verified CVAT bounding-box annotations; Ultralytics AGPL-3.0 gated separately.'}
              </div>
            </div>

            <div className="py-3 space-y-1">
              <div className="flex items-center justify-between">
                <span className="font-semibold text-slate-800">
                  {registryModels?.segmentor?.name || 'UNet-ResNet34'} (3-Class Segmentation)
                </span>
                <ModelStatusBadge status={formatModelStatusLabel(registryModels?.segmentor?.status)} />
              </div>
              <div className="text-[11px] text-slate-500">
                {registryModels?.segmentor?.annotation_requirements ||
                  'MIT segmentation-models-pytorch; requires CVAT pixel masks before training.'}
              </div>
            </div>
          </div>
        </div>

        {/* Verified Dataset & Annotation Monitoring */}
        <div className="bg-white p-6 rounded-3xl border border-slate-200 shadow-sm space-y-4">
          <div className="flex items-center space-x-2 text-slate-900 font-bold text-lg">
            <Database className="w-5 h-5 text-brand-600" />
            <span>Dataset Splits &amp; Verified Classifier Metrics</span>
          </div>

          <div className="grid grid-cols-3 gap-3 text-center">
            <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
              <div className="text-xl font-extrabold text-slate-900">{ds?.split_image_counts?.train ?? 1492}</div>
              <div className="text-[11px] text-slate-500">Train Split</div>
            </div>
            <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
              <div className="text-xl font-extrabold text-slate-900">{ds?.split_image_counts?.validation ?? 319}</div>
              <div className="text-[11px] text-slate-500">Val Split</div>
            </div>
            <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
              <div className="text-xl font-extrabold text-emerald-600">{ds?.split_image_counts?.test ?? 321}</div>
              <div className="text-[11px] text-slate-500">Untouched Test</div>
            </div>
          </div>

          <VerifiedClassifierMetricsStrip
            validationAccPct={valAccStr}
            testAccPct={testAccStr}
            testMacroF1Pct={macroF1Str}
            compact
          />

          <div className="text-xs text-slate-600 space-y-1 pt-1">
            <div>
              Verified Segmentation Mask Pairs: <strong>{ann?.valid_pairs_count ?? 0}</strong> (Ready for U-Net training:{' '}
              <strong>{ann?.is_ready_for_segmentation_training ? 'YES' : 'NO'}</strong>)
            </div>
            <div>
              Severity Threshold Validation Status: <strong>PROVISIONAL (category_validated = false)</strong>
            </div>
          </div>
        </div>
      </div>

      {/* User Management Table */}
      <div className="bg-white rounded-3xl border border-slate-200 shadow-sm overflow-hidden">
        <div className="p-6 border-b border-slate-100">
          <h2 className="text-lg font-bold text-slate-900">User Account Management</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm text-slate-600">
            <thead className="bg-slate-50 border-b border-slate-200 text-xs uppercase text-slate-500">
              <tr>
                <th className="py-3.5 px-6">ID</th>
                <th className="py-3.5 px-6">Username</th>
                <th className="py-3.5 px-6">Email</th>
                <th className="py-3.5 px-6">Role</th>
                <th className="py-3.5 px-6">Status</th>
                <th className="py-3.5 px-6">Total Scans</th>
                <th className="py-3.5 px-6 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {users.map((u) => (
                <tr key={u.id}>
                  <td className="py-3.5 px-6 font-bold text-slate-900">#{u.id}</td>
                  <td className="py-3.5 px-6 font-semibold text-slate-900">{u.username}</td>
                  <td className="py-3.5 px-6">{u.email}</td>
                  <td className="py-3.5 px-6">
                    <span
                      className={`px-2.5 py-0.5 rounded-full text-xs font-bold uppercase ${
                        u.role === 'admin' ? 'bg-purple-100 text-purple-800' : 'bg-slate-100 text-slate-700'
                      }`}
                    >
                      {u.role}
                    </span>
                  </td>
                  <td className="py-3.5 px-6">
                    <span
                      className={`px-2.5 py-0.5 rounded-full text-xs font-bold ${
                        u.is_active ? 'bg-emerald-100 text-emerald-800' : 'bg-red-100 text-red-700'
                      }`}
                    >
                      {u.is_active ? 'Active' : 'Disabled'}
                    </span>
                  </td>
                  <td className="py-3.5 px-6">{u.prediction_count}</td>
                  <td className="py-3.5 px-6 text-right space-x-2">
                    <button
                      type="button"
                      disabled={u.id === user.id}
                      onClick={() => handleToggleRole(u)}
                      className="px-3 py-1 rounded-lg border border-slate-200 text-xs font-semibold hover:bg-slate-50 disabled:opacity-40"
                    >
                      Switch to {u.role === 'admin' ? 'Farmer' : 'Admin'}
                    </button>
                    <button
                      type="button"
                      disabled={u.id === user.id}
                      onClick={() => handleToggleStatus(u)}
                      className={`px-3 py-1 rounded-lg border text-xs font-semibold disabled:opacity-40 ${
                        u.is_active
                          ? 'border-red-200 text-red-700 hover:bg-red-50'
                          : 'border-emerald-200 text-emerald-700 hover:bg-emerald-50'
                      }`}
                    >
                      {u.is_active ? 'Disable' : 'Enable'}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Verified Classifier Per-Class Metrics & Diagnostic Charts */}
      <div className="bg-white rounded-3xl border border-slate-200 shadow-sm p-6 space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="flex items-center space-x-2 text-slate-900 font-bold text-lg">
            <BarChart3 className="w-5 h-5 text-brand-600" />
            <span>Audited Classifier Evaluation &amp; Data Integrity</span>
          </div>
          <div className="flex flex-wrap gap-2 text-xs font-semibold">
            {leakageVerdict && (
              <span className="inline-flex items-center px-3 py-1 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">
                <CheckCircle2 className="w-3.5 h-3.5 mr-1" /> Leakage Audit: {String(leakageVerdict)}
              </span>
            )}
            {provenanceVerdict && (
              <span className="inline-flex items-center px-3 py-1 rounded-full bg-brand-50 text-brand-700 border border-brand-200">
                <CheckCircle2 className="w-3.5 h-3.5 mr-1" /> Provenance: {String(provenanceVerdict).slice(0, 32)}
              </span>
            )}
          </div>
        </div>

        {perClass && (
          <div className="overflow-x-auto rounded-2xl border border-slate-200">
            <table className="w-full text-left text-xs text-slate-700">
              <thead className="bg-slate-50 border-b border-slate-200 uppercase text-slate-500">
                <tr>
                  <th className="py-3 px-4">Condition Class</th>
                  <th className="py-3 px-4">Precision</th>
                  <th className="py-3 px-4">Recall</th>
                  <th className="py-3 px-4">F1-Score</th>
                  <th className="py-3 px-4">Test Support</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {Object.entries(perClass).map(([clsName, m]: [string, any]) => (
                  <tr key={clsName}>
                    <td className="py-2.5 px-4 font-semibold text-slate-900">{clsName}</td>
                    <td className="py-2.5 px-4">{(Number(m.precision) * 100).toFixed(2)}%</td>
                    <td className="py-2.5 px-4">{(Number(m.recall) * 100).toFixed(2)}%</td>
                    <td className="py-2.5 px-4 font-bold text-brand-700">{Number(m['f1-score']).toFixed(4)}</td>
                    <td className="py-2.5 px-4">{m.support}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {Object.keys(metricCharts).length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.entries(metricCharts).map(([chartKey, url]: [string, any]) => (
              <div key={chartKey} className="bg-slate-50 p-3 rounded-2xl border border-slate-200 text-center">
                <p className="text-xs font-bold uppercase tracking-wider text-slate-600 mb-2">
                  {chartKey.replace(/_/g, ' ')}
                </p>
                <img src={mediaUrl(url)} alt={chartKey} className="w-full max-h-72 object-contain rounded-xl bg-white" />
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Security & Activity Audit Log */}
      <div className="bg-white rounded-3xl border border-slate-200 shadow-sm overflow-hidden">
        <div className="p-6 border-b border-slate-100 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="flex items-center space-x-2">
            <History className="w-5 h-5 text-brand-600" />
            <h2 className="text-lg font-bold text-slate-900">Platform Activity &amp; Security Audit Log</h2>
          </div>
          <select
            value={auditFilter}
            onChange={(e) => handleAuditFilterChange(e.target.value)}
            className="px-3 py-1.5 rounded-xl border border-slate-200 text-xs font-semibold bg-white"
          >
            <option value="ALL">All Events</option>
            <option value="LEAF_SCAN">LEAF_SCAN</option>
            <option value="USER_LOGIN">USER_LOGIN</option>
            <option value="USER_REGISTER">USER_REGISTER</option>
            <option value="PDF_REPORT_DOWNLOAD">PDF_REPORT_DOWNLOAD</option>
            <option value="HISTORY_CSV_EXPORT">HISTORY_CSV_EXPORT</option>
            <option value="PREDICTION_DELETED">PREDICTION_DELETED</option>
            <option value="USER_ROLE_CHANGE">USER_ROLE_CHANGE</option>
            <option value="USER_STATUS_CHANGE">USER_STATUS_CHANGE</option>
          </select>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-slate-600">
            <thead className="bg-slate-50 border-b border-slate-200 uppercase text-slate-500">
              <tr>
                <th className="py-3 px-5">Timestamp (UTC)</th>
                <th className="py-3 px-5">User</th>
                <th className="py-3 px-5">Action</th>
                <th className="py-3 px-5">Resource</th>
                <th className="py-3 px-5">Details</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {auditLogs.length > 0 ? (
                auditLogs.map((log) => (
                  <tr key={log.id}>
                    <td className="py-2.5 px-5 whitespace-nowrap">{new Date(log.created_at).toLocaleString()}</td>
                    <td className="py-2.5 px-5 font-semibold text-slate-900">{log.username}</td>
                    <td className="py-2.5 px-5">
                      <span className="px-2 py-0.5 rounded-full bg-slate-100 font-bold text-slate-700">{log.action}</span>
                    </td>
                    <td className="py-2.5 px-5 font-mono">{log.resource}</td>
                    <td className="py-2.5 px-5">{log.details || '—'}</td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={5} className="py-6 text-center text-slate-400">
                    No audit log entries recorded yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
