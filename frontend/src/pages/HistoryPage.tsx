import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { api, mediaUrl } from '../services/api';
import { PredictionDetail, PredictionHistoryItem } from '../types/api';
import { PredictionResultsView } from '../components/PredictionResultsView';
import { extractApiErrorMessage } from '../utils/validation';
import {
  Search,
  FileText,
  Calendar,
  Eye,
  ChevronLeft,
  ChevronRight,
  X,
  Download,
  Trash2,
  AlertTriangle,
  ExternalLink,
} from 'lucide-react';

const DISEASE_OPTIONS = [
  'ALL',
  'Bacterial Blight',
  'Curl Virus',
  'Healthy Leaf',
  'Herbicide Growth Damage',
  'Leaf Hopper Jassids',
  'Leaf Redding',
  'Leaf Variegation',
];

const SEVERITY_OPTIONS = [
  'ALL',
  'UNAVAILABLE',
  'NONE',
  'TRACE',
  'MILD',
  'MODERATE',
  'SEVERE',
  'VERY SEVERE',
  'LOW',
  'HIGH',
];

export const HistoryPage: React.FC = () => {
  const [items, setItems] = useState<PredictionHistoryItem[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [search, setSearch] = useState('');
  const [disease, setDisease] = useState('ALL');
  const [severity, setSeverity] = useState('ALL');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedDetail, setSelectedDetail] = useState<PredictionDetail | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [confirmDeleteId, setConfirmDeleteId] = useState<number | null>(null);
  const [exportingCsv, setExportingCsv] = useState(false);
  const [downloadingPdfId, setDownloadingPdfId] = useState<number | null>(null);

  const fetchHistory = async (targetPage = page) => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getPaginatedHistory({
        search,
        disease,
        severity,
        start_date: startDate || undefined,
        end_date: endDate || undefined,
        page: targetPage,
        page_size: 10,
      });
      setItems(data.items || []);
      setTotal(data.total || 0);
      setPage(data.page || 1);
      setTotalPages(data.total_pages || 1);
    } catch (err) {
      setError(extractApiErrorMessage(err, 'Failed to load diagnosis history.'));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory(1);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleFilterSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    fetchHistory(1);
  };

  const handleResetFilters = () => {
    setSearch('');
    setDisease('ALL');
    setSeverity('ALL');
    setStartDate('');
    setEndDate('');
    setLoading(true);
    setError(null);
    api
      .getPaginatedHistory({ page: 1, page_size: 10 })
      .then((data) => {
        setItems(data.items || []);
        setTotal(data.total || 0);
        setPage(data.page || 1);
        setTotalPages(data.total_pages || 1);
      })
      .catch((err) => setError(extractApiErrorMessage(err, 'Failed to reload diagnosis history.')))
      .finally(() => setLoading(false));
  };

  const handleInspectDetail = async (id: number) => {
    setLoadingDetail(true);
    setError(null);
    try {
      const detail = await api.getPredictionDetail(id);
      setSelectedDetail(detail);
    } catch (err) {
      setError(extractApiErrorMessage(err, `Failed to load breakdown for Scan #${id}.`));
    } finally {
      setLoadingDetail(false);
    }
  };

  const handleExportCsv = async () => {
    setExportingCsv(true);
    setError(null);
    try {
      await api.exportHistoryCsv({
        search,
        disease,
        severity,
        start_date: startDate || undefined,
        end_date: endDate || undefined,
      });
    } catch (err) {
      setError(extractApiErrorMessage(err, 'Failed to export prediction history CSV.'));
    } finally {
      setExportingCsv(false);
    }
  };

  const handleDownloadPdf = async (id: number) => {
    setDownloadingPdfId(id);
    setError(null);
    try {
      await api.downloadReport(id);
    } catch (err) {
      setError(extractApiErrorMessage(err, `Failed to download PDF report for Scan #${id}.`));
    } finally {
      setDownloadingPdfId(null);
    }
  };

  const handleDeleteScan = async (id: number) => {
    setError(null);
    try {
      await api.deleteHistoryItem(id);
      setConfirmDeleteId(null);
      if (selectedDetail?.id === id) setSelectedDetail(null);
      await fetchHistory(page);
    } catch (err) {
      setError(extractApiErrorMessage(err, `Failed to delete Scan #${id}.`));
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 py-8 space-y-6">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">Diagnosis History</h1>
          <p className="text-slate-500 text-sm mt-1">
            Filter scans by condition, provisional severity, or date range and inspect full AI outputs or export reports
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={handleExportCsv}
            disabled={exportingCsv || total === 0}
            className="inline-flex items-center px-3.5 py-2 rounded-xl border border-slate-200 bg-white text-xs font-semibold text-slate-700 hover:bg-slate-50 disabled:opacity-50 transition"
          >
            <Download className="w-3.5 h-3.5 mr-1.5" />
            {exportingCsv ? 'Exporting...' : 'Export CSV'}
          </button>
          <span className="text-xs font-semibold px-3 py-1.5 bg-slate-100 text-slate-700 rounded-full">
            {total} Total Record(s)
          </span>
        </div>
      </div>

      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-2xl text-sm text-red-700 flex items-center space-x-2">
          <AlertTriangle className="w-4 h-4 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Filters Toolbar */}
      <form
        onSubmit={handleFilterSubmit}
        className="bg-white p-4 rounded-2xl border border-slate-200 shadow-sm grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-6 gap-3"
      >
        <div className="relative lg:col-span-2">
          <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-3" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search class, severity, filename..."
            className="w-full pl-10 pr-3 py-2 rounded-xl border border-slate-200 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500"
          />
        </div>

        <select
          value={disease}
          onChange={(e) => setDisease(e.target.value)}
          className="px-3 py-2 rounded-xl border border-slate-200 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-brand-500"
        >
          {DISEASE_OPTIONS.map((d) => (
            <option key={d} value={d}>
              {d === 'ALL' ? 'All Conditions' : d}
            </option>
          ))}
        </select>

        <select
          value={severity}
          onChange={(e) => setSeverity(e.target.value)}
          className="px-3 py-2 rounded-xl border border-slate-200 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-brand-500"
        >
          {SEVERITY_OPTIONS.map((s) => (
            <option key={s} value={s}>
              {s === 'ALL' ? 'All Severities' : s}
            </option>
          ))}
        </select>

        <input
          type="date"
          value={startDate}
          onChange={(e) => setStartDate(e.target.value)}
          aria-label="Start date"
          className="px-3 py-2 rounded-xl border border-slate-200 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500"
        />

        <div className="flex gap-2">
          <input
            type="date"
            value={endDate}
            onChange={(e) => setEndDate(e.target.value)}
            aria-label="End date"
            className="w-full px-3 py-2 rounded-xl border border-slate-200 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500"
          />
          <button
            type="submit"
            className="px-3.5 py-2 bg-slate-900 text-white rounded-xl text-sm font-semibold hover:bg-slate-800 transition flex-shrink-0"
          >
            Apply
          </button>
          <button
            type="button"
            onClick={handleResetFilters}
            className="px-3 py-2 border border-slate-200 text-slate-600 rounded-xl text-xs font-semibold hover:bg-slate-50 transition flex-shrink-0"
          >
            Reset
          </button>
        </div>
      </form>

      {/* Table */}
      {loading ? (
        <div className="flex items-center justify-center min-h-[35vh]">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-brand-600" />
        </div>
      ) : items.length > 0 ? (
        <div className="bg-white rounded-3xl border border-slate-200 shadow-sm overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm text-slate-600">
              <thead className="bg-slate-50 border-b border-slate-200 text-xs font-semibold uppercase tracking-wider text-slate-500">
                <tr>
                  <th className="py-4 px-5">Scan &amp; Leaf</th>
                  <th className="py-4 px-5">Condition</th>
                  <th className="py-4 px-5">Confidence</th>
                  <th className="py-4 px-5">Affected Area</th>
                  <th className="py-4 px-5">Severity (Provisional)</th>
                  <th className="py-4 px-5">Timestamp</th>
                  <th className="py-4 px-5 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {items.map((item) => (
                  <tr key={item.id} className="hover:bg-slate-50 transition">
                    <td className="py-3.5 px-5 flex items-center space-x-3">
                      <img
                        src={mediaUrl(item.image_url)}
                        alt={`Leaf Scan ${item.id}`}
                        className="w-11 h-11 rounded-lg object-cover border border-slate-200"
                      />
                      <span className="font-bold text-slate-900">#{item.id}</span>
                    </td>
                    <td className="py-3.5 px-5 font-semibold text-slate-900">{item.predicted_class}</td>
                    <td className="py-3.5 px-5">{(item.confidence * 100).toFixed(1)}%</td>
                    <td className="py-3.5 px-5 text-xs">
                      {item.affected_area_percentage !== null && item.affected_area_percentage !== undefined
                        ? `${item.affected_area_percentage}%`
                        : 'NOT_TRAINED'}
                    </td>
                    <td className="py-3.5 px-5">
                      <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-slate-100 text-slate-700">
                        {item.severity_level}
                      </span>
                    </td>
                    <td className="py-3.5 px-5 text-xs text-slate-500">
                      {new Date(item.created_at).toLocaleString()}
                    </td>
                    <td className="py-3.5 px-5 text-right space-x-2 whitespace-nowrap">
                      <button
                        type="button"
                        onClick={() => handleInspectDetail(item.id)}
                        className="inline-flex items-center px-3 py-1.5 rounded-lg border border-slate-200 text-xs font-semibold text-slate-700 hover:bg-slate-100 transition"
                      >
                        <Eye className="w-3.5 h-3.5 mr-1" /> Inspect
                      </button>
                      <Link
                        to={`/results/${item.id}`}
                        className="inline-flex items-center px-2.5 py-1.5 rounded-lg border border-slate-200 text-xs font-semibold text-slate-600 hover:bg-slate-100 transition"
                        title="Open Dedicated Result Page"
                      >
                        <ExternalLink className="w-3.5 h-3.5" />
                      </Link>
                      <button
                        type="button"
                        disabled={downloadingPdfId === item.id}
                        onClick={() => handleDownloadPdf(item.id)}
                        className="inline-flex items-center px-3 py-1.5 rounded-lg border border-brand-200 bg-brand-50 text-xs font-semibold text-brand-700 hover:bg-brand-100 transition disabled:opacity-50"
                      >
                        <FileText className="w-3.5 h-3.5 mr-1" />
                        {downloadingPdfId === item.id ? '...' : 'PDF'}
                      </button>
                      {confirmDeleteId === item.id ? (
                        <span className="inline-flex items-center gap-1">
                          <button
                            type="button"
                            onClick={() => handleDeleteScan(item.id)}
                            className="px-2.5 py-1.5 rounded-lg bg-red-600 text-white text-xs font-semibold hover:bg-red-700"
                          >
                            Confirm
                          </button>
                          <button
                            type="button"
                            onClick={() => setConfirmDeleteId(null)}
                            className="px-2 py-1.5 rounded-lg border border-slate-200 text-xs font-semibold text-slate-600 hover:bg-slate-100"
                          >
                            Cancel
                          </button>
                        </span>
                      ) : (
                        <button
                          type="button"
                          onClick={() => setConfirmDeleteId(item.id)}
                          className="inline-flex items-center px-2.5 py-1.5 rounded-lg border border-red-200 text-xs font-semibold text-red-600 hover:bg-red-50 transition"
                          title="Delete Scan"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          <div className="px-6 py-4 bg-slate-50 border-t border-slate-200 flex items-center justify-between text-xs text-slate-600">
            <span>
              Page <strong>{page}</strong> of <strong>{totalPages}</strong> ({total} scans)
            </span>
            <div className="flex space-x-2">
              <button
                type="button"
                disabled={page <= 1}
                onClick={() => fetchHistory(page - 1)}
                className="px-3 py-1.5 rounded-lg border border-slate-200 bg-white disabled:opacity-40 flex items-center"
              >
                <ChevronLeft className="w-3.5 h-3.5 mr-1" /> Previous
              </button>
              <button
                type="button"
                disabled={page >= totalPages}
                onClick={() => fetchHistory(page + 1)}
                className="px-3 py-1.5 rounded-lg border border-slate-200 bg-white disabled:opacity-40 flex items-center"
              >
                Next <ChevronRight className="w-3.5 h-3.5 ml-1" />
              </button>
            </div>
          </div>
        </div>
      ) : (
        <div className="bg-white rounded-3xl border border-slate-200 p-12 text-center text-slate-400">
          <Calendar className="w-10 h-10 mx-auto mb-2 text-slate-300" />
          <h3 className="font-bold text-slate-700">No matching scans found</h3>
          <p className="text-sm mt-1">Adjust your search filters or upload a leaf photograph to analyze.</p>
        </div>
      )}

      {/* Full Stored Prediction Inspector Modal */}
      {(selectedDetail || loadingDetail) && (
        <div className="fixed inset-0 z-50 bg-slate-900/60 backdrop-blur-sm overflow-y-auto p-4 sm:p-8 flex items-start justify-center">
          <div className="max-w-6xl w-full relative my-4">
            <button
              type="button"
              onClick={() => setSelectedDetail(null)}
              className="sticky top-4 float-right z-10 p-2 bg-white border border-slate-200 rounded-full shadow-md text-slate-600 hover:text-slate-900"
              aria-label="Close Inspector"
            >
              <X className="w-5 h-5" />
            </button>
            {loadingDetail && !selectedDetail ? (
              <div className="bg-white rounded-3xl p-12 text-center">Loading scan breakdown...</div>
            ) : (
              selectedDetail && (
                <PredictionResultsView
                  result={selectedDetail}
                  onReset={() => setSelectedDetail(null)}
                  resetLabel="Close Details"
                />
              )
            )}
          </div>
        </div>
      )}
    </div>
  );
};
