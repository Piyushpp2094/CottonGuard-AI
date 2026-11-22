import React, { useEffect, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { api } from '../services/api';
import { PredictionDetail } from '../types/api';
import { PredictionResultsView } from '../components/PredictionResultsView';
import { extractApiErrorMessage } from '../utils/validation';
import { ArrowLeft, AlertTriangle } from 'lucide-react';

export const ResultDetailPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [detail, setDetail] = useState<PredictionDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const parsedId = Number(id);
    if (!parsedId || Number.isNaN(parsedId)) {
      setError('Invalid prediction record ID.');
      setLoading(false);
      return;
    }
    setLoading(true);
    setError(null);
    api
      .getPredictionDetail(parsedId)
      .then((res) => setDetail(res))
      .catch((err) => setError(extractApiErrorMessage(err, `Failed to load Scan #${parsedId}.`)))
      .finally(() => setLoading(false));
  }, [id]);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[55vh]">
        <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-brand-600" />
      </div>
    );
  }

  if (error || !detail) {
    return (
      <div className="max-w-3xl mx-auto px-4 py-16 space-y-4">
        <div className="p-6 bg-red-50 border border-red-200 rounded-2xl flex items-start space-x-3 text-red-700 text-sm">
          <AlertTriangle className="w-5 h-5 flex-shrink-0 mt-0.5" />
          <div>
            <h2 className="font-bold text-base">Unable to Load Diagnostic Record</h2>
            <p className="mt-1">{error || 'Record not found.'}</p>
          </div>
        </div>
        <Link
          to="/history"
          className="inline-flex items-center px-4 py-2 rounded-xl border border-slate-200 bg-white text-sm font-semibold text-slate-700 hover:bg-slate-50"
        >
          <ArrowLeft className="w-4 h-4 mr-1.5" /> Back to History
        </Link>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 space-y-5">
      <div className="flex items-center justify-between">
        <Link
          to="/history"
          className="inline-flex items-center text-sm font-semibold text-slate-600 hover:text-slate-900"
        >
          <ArrowLeft className="w-4 h-4 mr-1.5" /> Back to History
        </Link>
      </div>
      <PredictionResultsView
        result={detail}
        onReset={() => navigate('/analyze')}
        resetLabel="Analyze Another Leaf"
      />
    </div>
  );
};
