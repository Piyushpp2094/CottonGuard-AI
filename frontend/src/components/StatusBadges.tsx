import React from 'react';
import { CheckCircle2, Clock3, ShieldAlert } from 'lucide-react';
import { VERIFIED_CLASSIFIER_METRICS } from '../utils/validation';

export const ModelStatusBadge: React.FC<{
  status: 'TRAINED' | 'NOT_TRAINED' | 'ACTIVE' | 'EXPERIMENTAL' | string;
}> = ({ status }) => {
  const norm = status?.toUpperCase();
  if (norm === 'TRAINED' || norm === 'ACTIVE' || norm === 'SUCCESS') {
    return (
      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-[11px] font-bold tracking-wide bg-emerald-100 text-emerald-800 border border-emerald-200">
        <CheckCircle2 className="w-3 h-3 mr-1" />
        {status}
      </span>
    );
  }
  if (norm === 'EXPERIMENTAL') {
    return (
      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-[11px] font-bold tracking-wide bg-purple-100 text-purple-800 border border-purple-200">
        <Clock3 className="w-3 h-3 mr-1" />
        EXPERIMENTAL
      </span>
    );
  }
  return (
    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-[11px] font-bold tracking-wide bg-amber-100 text-amber-800 border border-amber-200">
      <Clock3 className="w-3 h-3 mr-1" />
      NOT_TRAINED
    </span>
  );
};

export const ProvisionalSeverityBadge: React.FC<{ validated?: boolean }> = ({ validated = false }) => {
  if (validated) {
    return (
      <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider bg-emerald-100 text-emerald-800 border border-emerald-200">
        Field Validated
      </span>
    );
  }
  return (
    <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider bg-amber-100 text-amber-800 border border-amber-200">
      <ShieldAlert className="w-3 h-3 mr-1" />
      Provisional • Unvalidated
    </span>
  );
};

export const VerifiedClassifierMetricsStrip: React.FC<{
  validationAccPct?: string;
  testAccPct?: string;
  testMacroF1Pct?: string;
  compact?: boolean;
}> = ({
  validationAccPct = VERIFIED_CLASSIFIER_METRICS.validationAccuracyPct,
  testAccPct = VERIFIED_CLASSIFIER_METRICS.untouchedTestAccuracyPct,
  testMacroF1Pct = VERIFIED_CLASSIFIER_METRICS.testMacroF1Pct,
  compact = false,
}) => {
  return (
    <div
      className={`grid grid-cols-1 sm:grid-cols-3 gap-3 ${
        compact ? 'text-xs' : 'text-sm'
      }`}
    >
      <div className="bg-emerald-50/80 border border-emerald-200/80 rounded-2xl px-4 py-3 flex items-center justify-between">
        <div>
          <div className="text-[11px] font-bold uppercase tracking-wider text-emerald-700">
            Validation Accuracy
          </div>
          <div className="text-[11px] text-emerald-800/80">SAR-CLD-2024 (319 imgs)</div>
        </div>
        <div className="text-xl font-extrabold text-emerald-900">{validationAccPct}</div>
      </div>

      <div className="bg-brand-50/80 border border-brand-200/80 rounded-2xl px-4 py-3 flex items-center justify-between">
        <div>
          <div className="text-[11px] font-bold uppercase tracking-wider text-brand-700">
            Untouched Test Accuracy
          </div>
          <div className="text-[11px] text-brand-800/80">Holdout Split (321 imgs)</div>
        </div>
        <div className="text-xl font-extrabold text-brand-900">{testAccPct}</div>
      </div>

      <div className="bg-slate-100/90 border border-slate-200 rounded-2xl px-4 py-3 flex items-center justify-between">
        <div>
          <div className="text-[11px] font-bold uppercase tracking-wider text-slate-600">
            Test Macro F1
          </div>
          <div className="text-[11px] text-slate-500">Balanced 7-Class F1</div>
        </div>
        <div className="text-xl font-extrabold text-slate-900">{testMacroF1Pct}</div>
      </div>
    </div>
  );
};
