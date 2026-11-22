import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../services/api';
import { HealthModelsResponse } from '../types/api';
import { formatModelStatusLabel, VERIFIED_CLASSIFIER_METRICS } from '../utils/validation';
import { ModelStatusBadge, VerifiedClassifierMetricsStrip } from '../components/StatusBadges';
import {
  Leaf,
  Cpu,
  ArrowRight,
  Target,
  Layers,
  Eye,
  FileText,
  ShieldCheck,
  Database,
  ShieldAlert,
} from 'lucide-react';

const CONDITIONS = [
  { name: 'Bacterial Blight', desc: 'Angular water-soaked leaf lesions caused by Xanthomonas citri pv. malvacearum', tag: 'Bacterial' },
  { name: 'Curl Virus', desc: 'Upward/downward leaf curling and vein thickening transmitted by whiteflies', tag: 'Viral' },
  { name: 'Healthy Leaf', desc: 'Normal foliage with optimal chlorophyll and intact vein architecture', tag: 'Normal' },
  { name: 'Herbicide Growth Damage', desc: 'Chemical epinasty, cupping, and strap-like leaf distortion', tag: 'Abiotic' },
  { name: 'Leaf Hopper Jassids', desc: 'Marginal yellowing and hopper burn caused by Amrasca biguttula', tag: 'Pest' },
  { name: 'Leaf Redding', desc: 'Anthocyanin discoloration from nitrogen/magnesium depletion or waterlogging', tag: 'Physiological' },
  { name: 'Leaf Variegation', desc: 'Irregular mosaic chimera patterns across foliar lamina', tag: 'Genetic/Mosaic' },
];

export const LandingPage: React.FC = () => {
  const [modelsData, setModelsData] = useState<HealthModelsResponse | null>(null);

  useEffect(() => {
    api
      .getHealthModels()
      .then((res) => setModelsData(res))
      .catch(() => {
        // Public fallback still stays honest when backend health endpoint is initializing
      });
  }, []);

  const clsStatus = formatModelStatusLabel(modelsData?.models?.classifier?.status, true);
  const detStatus = formatModelStatusLabel(modelsData?.models?.detector?.status);
  const segStatus = formatModelStatusLabel(modelsData?.models?.segmentor?.status);

  const aiModules: Array<{
    icon: React.ComponentType<{ className?: string }>;
    title: string;
    status: 'TRAINED' | 'NOT_TRAINED' | 'ACTIVE';
    desc: string;
  }> = [
    {
      icon: Cpu,
      title: 'EfficientNet-B0 Classifier',
      status: clsStatus,
      desc: `Verified on SAR-CLD-2024: ${VERIFIED_CLASSIFIER_METRICS.validationAccuracyPct} validation accuracy, ${VERIFIED_CLASSIFIER_METRICS.untouchedTestAccuracyPct} untouched test accuracy, and ${VERIFIED_CLASSIFIER_METRICS.testMacroF1Pct} Test Macro F1.`,
    },
    {
      icon: Eye,
      title: 'Grad-CAM Explainability',
      status: 'ACTIVE',
      desc: 'Highlights classifier receptive-field saliency on features[-1] with opacity, target-class, and colormap controls (interpretability aid only).',
    },
    {
      icon: Target,
      title: 'YOLOv8n Lesion Detector',
      status: detStatus,
      desc:
        detStatus === 'TRAINED'
          ? 'Trained bounding-box localization isolating symptomatic cotton foliar regions.'
          : 'NOT_TRAINED — gated until verified CVAT bounding-box annotations and approved training are completed.',
    },
    {
      icon: Layers,
      title: 'U-Net ResNet34 Segmentor',
      status: segStatus,
      desc:
        segStatus === 'TRAINED'
          ? 'Trained 3-class pixel segmentation extracting leaf area and diseased lamina percentage.'
          : 'NOT_TRAINED — gated until verified CVAT pixel masks (512×512) are annotated and trained.',
    },
  ];

  return (
    <div className="space-y-16 py-10">
      {/* Hero Section */}
      <section className="text-center max-w-5xl mx-auto px-4 space-y-6">
        <div className="inline-flex items-center space-x-2 px-3.5 py-1.5 bg-brand-50 border border-brand-200 rounded-full text-brand-700 text-xs font-semibold uppercase tracking-wider">
          <ShieldCheck className="w-3.5 h-3.5" />
          <span>CottonGuard AI • 100% Local Precision Agriculture • RTX 3050 CUDA Accelerated</span>
        </div>
        <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold text-slate-900 tracking-tight leading-tight">
          Leaf Disease Analysis &amp; <span className="text-brand-600">Severity Estimation</span> Platform
        </h1>
        <p className="text-lg text-slate-600 max-w-3xl mx-auto leading-relaxed">
          Transparent cotton leaf diagnostics combining EfficientNet-B0 7-class classification, live Grad-CAM saliency,
          real-time SSE pipeline streaming, ICAR-CICR advisories, and downloadable PDF field reports.
        </p>

        {/* Verified Classifier Metrics Strip */}
        <div className="max-w-4xl mx-auto pt-2 text-left">
          <VerifiedClassifierMetricsStrip />
        </div>

        <div className="pt-2 flex flex-col sm:flex-row items-center justify-center gap-4">
          <Link
            to="/analyze"
            className="w-full sm:w-auto flex items-center justify-center px-6 py-3.5 rounded-xl text-base font-semibold bg-brand-600 hover:bg-brand-700 text-white shadow-md shadow-brand-500/20 transition group"
          >
            <span>Upload &amp; Analyze Leaf</span>
            <ArrowRight className="w-4 h-4 ml-2 group-hover:translate-x-1 transition" />
          </Link>
          <Link
            to="/dashboard"
            className="w-full sm:w-auto flex items-center justify-center px-6 py-3.5 rounded-xl text-base font-semibold bg-white hover:bg-slate-50 text-slate-700 border border-slate-200 shadow-sm transition"
          >
            Open Farmer Dashboard
          </Link>
        </div>
      </section>

      {/* 4 AI Capabilities */}
      <section className="max-w-7xl mx-auto px-4">
        <div className="text-center mb-10">
          <h2 className="text-2xl sm:text-3xl font-bold text-slate-900">Honest Four-Module Computer Vision Status</h2>
          <p className="text-slate-500 mt-2">
            Every model reports its real checkpoint state directly — untransformed modules remain strictly NOT_TRAINED
          </p>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {aiModules.map((m) => {
            const Icon = m.icon;
            return (
              <div key={m.title} className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm hover:shadow-md transition space-y-3">
                <div className="flex justify-between items-center gap-2">
                  <div className="w-10 h-10 rounded-xl bg-brand-50 text-brand-700 flex items-center justify-center flex-shrink-0">
                    <Icon className="w-5 h-5" />
                  </div>
                  <ModelStatusBadge status={m.status} />
                </div>
                <h3 className="text-base font-bold text-slate-900">{m.title}</h3>
                <p className="text-xs text-slate-600 leading-relaxed">{m.desc}</p>
              </div>
            );
          })}
        </div>

        <div className="mt-6 p-4 bg-amber-50 border border-amber-200 rounded-2xl text-amber-900 text-xs flex items-start space-x-2.5">
          <ShieldAlert className="w-4 h-4 text-amber-600 flex-shrink-0 mt-0.5" />
          <span>
            <strong>Provisional Severity Notice:</strong> 6-tier severity estimation thresholds (<code>None / Trace / Mild / Moderate / Severe / Very Severe</code>) are marked <strong>provisional (category_validated=false)</strong> until calibrated against field-verified masks and agronomic ground truth.
          </span>
        </div>
      </section>

      {/* Workflow Steps */}
      <section className="bg-brand-900 text-white rounded-3xl p-8 sm:p-12 max-w-7xl mx-auto">
        <div className="text-center mb-10">
          <h2 className="text-2xl sm:text-3xl font-bold">How the Diagnostic Pipeline Works</h2>
          <p className="text-brand-200 text-sm mt-2">Real-time 7-stage SSE pipeline from foliar capture to downloadable PDF report</p>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="bg-brand-800/60 p-6 rounded-2xl border border-brand-700">
            <div className="w-9 h-9 rounded-xl bg-brand-500 flex items-center justify-center font-bold mb-3">1</div>
            <h4 className="font-semibold mb-1.5">Capture &amp; Validate</h4>
            <p className="text-xs text-brand-100 leading-relaxed">
              Upload a cotton leaf photo (JPG/PNG/WEBP &le; 10 MB); backend verifies magic bytes and bounds (100–8000 px).
            </p>
          </div>
          <div className="bg-brand-800/60 p-6 rounded-2xl border border-brand-700">
            <div className="w-9 h-9 rounded-xl bg-brand-500 flex items-center justify-center font-bold mb-3">2</div>
            <h4 className="font-semibold mb-1.5">Live SSE Vision Stream</h4>
            <p className="text-xs text-brand-100 leading-relaxed">
              Runs EfficientNet-B0 + Grad-CAM over Server-Sent Events while checking trained detector/segmentor gates.
            </p>
          </div>
          <div className="bg-brand-800/60 p-6 rounded-2xl border border-brand-700">
            <div className="w-9 h-9 rounded-xl bg-brand-500 flex items-center justify-center font-bold mb-3">3</div>
            <h4 className="font-semibold mb-1.5">Honest Severity Gate</h4>
            <p className="text-xs text-brand-100 leading-relaxed">
              Computes provisional 6-tier severity only when U-Net leaf/disease masks are active; otherwise marked UNAVAILABLE.
            </p>
          </div>
          <div className="bg-brand-800/60 p-6 rounded-2xl border border-brand-700">
            <div className="w-9 h-9 rounded-xl bg-brand-500 flex items-center justify-center font-bold mb-3">4</div>
            <h4 className="font-semibold mb-1.5">ICAR-CICR &amp; PDF Report</h4>
            <p className="text-xs text-brand-100 leading-relaxed">
              Delivers field scouting guidance and generates a structured ReportLab PDF report.
            </p>
          </div>
        </div>
      </section>

      {/* 7 Cotton Conditions */}
      <section className="max-w-7xl mx-auto px-4">
        <div className="text-center mb-10">
          <h2 className="text-2xl sm:text-3xl font-bold text-slate-900">Seven Covered Cotton Leaf Conditions</h2>
          <p className="text-slate-500 mt-2">Trained on 2,132 audited images on disk from SAR-CLD-2024 (1,492 train / 319 val / 321 test)</p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
          {CONDITIONS.map((c) => (
            <div key={c.name} className="bg-white p-5 rounded-2xl border border-slate-200 hover:shadow-md transition">
              <div className="flex justify-between items-start mb-2.5">
                <span className="text-[11px] font-semibold px-2.5 py-0.5 bg-slate-100 text-slate-600 rounded-full">{c.tag}</span>
                <Leaf className="w-4 h-4 text-brand-600" />
              </div>
              <h3 className="text-base font-bold text-slate-900 mb-1">{c.name}</h3>
              <p className="text-xs text-slate-600 leading-relaxed">{c.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Local Tech Stack */}
      <section className="max-w-7xl mx-auto px-4">
        <div className="bg-white rounded-3xl border border-slate-200 p-8 flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="space-y-2 max-w-xl">
            <div className="inline-flex items-center text-xs font-bold text-brand-700 uppercase tracking-wider">
              <Database className="w-4 h-4 mr-1.5" /> Zero-Cost Local Stack
            </div>
            <h3 className="text-xl font-bold text-slate-900">Open-Source Local Precision Agriculture Stack</h3>
            <p className="text-sm text-slate-600">
              FastAPI + SQLite + Alembic + PyTorch CUDA + React/Vite/Tailwind + Recharts + ReportLab.
              Runs completely offline with verified model provenance and zero synthetic diagnostics.
            </p>
          </div>
          <Link
            to="/auth"
            className="flex items-center px-5 py-3 rounded-xl bg-slate-900 hover:bg-slate-800 text-white text-sm font-semibold transition flex-shrink-0"
          >
            <FileText className="w-4 h-4 mr-2" /> Sign In to Start Diagnostics
          </Link>
        </div>
      </section>
    </div>
  );
};
