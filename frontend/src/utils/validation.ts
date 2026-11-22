export const MAX_UPLOAD_SIZE_MB = 10;
export const MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024;
export const MIN_IMAGE_DIM = 100;
export const MAX_IMAGE_DIM = 8000;
export const ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp'] as const;
export const ALLOWED_MIME_TYPES = ['image/jpeg', 'image/png', 'image/webp'] as const;

export const VERIFIED_CLASSIFIER_METRICS = {
  validationAccuracyPct: '98.43%',
  untouchedTestAccuracyPct: '97.82%',
  testMacroF1Pct: '97.54%',
  validationAccuracyRatio: 0.9843,
  untouchedTestAccuracyRatio: 0.9782,
  testMacroF1Ratio: 0.9754,
} as const;

export const PROVISIONAL_SEVERITY_TIERS = [
  { label: 'None', range: '0%' },
  { label: 'Trace', range: '>0% – 10%' },
  { label: 'Mild', range: '>10% – 25%' },
  { label: 'Moderate', range: '>25% – 50%' },
  { label: 'Severe', range: '>50% – 75%' },
  { label: 'Very Severe', range: '>75% – 100%' },
] as const;

export interface UploadMetadataValidationResult {
  valid: boolean;
  error?: string;
}

export function validateUploadMetadata(file: {
  name: string;
  size: number;
  type?: string;
}): UploadMetadataValidationResult {
  const lowerName = (file.name || '').toLowerCase();
  const hasAllowedExt = ALLOWED_EXTENSIONS.some((ext) => lowerName.endsWith(ext));
  const mime = (file.type || '').toLowerCase();
  const hasAllowedMime =
    mime === '' || ALLOWED_MIME_TYPES.includes(mime as (typeof ALLOWED_MIME_TYPES)[number]);

  if (!hasAllowedExt) {
    return {
      valid: false,
      error: `Unsupported image extension for "${file.name}". Allowed formats: JPG, JPEG, PNG, WEBP.`,
    };
  }
  if (!hasAllowedMime) {
    return {
      valid: false,
      error: `Unsupported image MIME type "${file.type}". Allowed: JPEG, PNG, WEBP.`,
    };
  }
  if (file.size <= 0) {
    return {
      valid: false,
      error: 'Selected image file is empty (0 bytes).',
    };
  }
  if (file.size > MAX_UPLOAD_SIZE_BYTES) {
    return {
      valid: false,
      error: `Image exceeds the ${MAX_UPLOAD_SIZE_MB} MB backend limit (${(file.size / (1024 * 1024)).toFixed(2)} MB).`,
    };
  }
  return { valid: true };
}

export function validateImageDimensions(width: number, height: number): UploadMetadataValidationResult {
  if (
    width < MIN_IMAGE_DIM ||
    height < MIN_IMAGE_DIM ||
    width > MAX_IMAGE_DIM ||
    height > MAX_IMAGE_DIM
  ) {
    return {
      valid: false,
      error: `Image resolution ${width}×${height}px is outside supported bounds (${MIN_IMAGE_DIM}×${MIN_IMAGE_DIM}px to ${MAX_IMAGE_DIM}×${MAX_IMAGE_DIM}px).`,
    };
  }
  return { valid: true };
}

export function formatModelStatusLabel(status?: string | null, ready?: boolean): 'TRAINED' | 'NOT_TRAINED' {
  if (typeof ready === 'boolean') {
    return ready ? 'TRAINED' : 'NOT_TRAINED';
  }
  const normalized = String(status || '').trim().toUpperCase();
  if (normalized === 'TRAINED' || normalized === 'SUCCESS' || normalized === 'READY') {
    return 'TRAINED';
  }
  return 'NOT_TRAINED';
}

export function extractApiErrorMessage(err: unknown, fallback: string): string {
  if (!err || typeof err !== 'object') return fallback;
  const anyErr = err as Record<string, any>;
  const data = anyErr.response?.data;
  if (data) {
    if (typeof data.error?.message === 'string' && data.error.message.trim()) {
      return data.error.message;
    }
    if (typeof data.detail === 'string' && data.detail.trim()) {
      return data.detail;
    }
    if (typeof data.detail?.message === 'string' && data.detail.message.trim()) {
      return data.detail.message;
    }
    if (Array.isArray(data.detail) && data.detail.length > 0) {
      const first = data.detail[0];
      if (typeof first?.msg === 'string') return first.msg;
    }
  }
  if (typeof anyErr.message === 'string' && anyErr.message.trim()) {
    return anyErr.message;
  }
  return fallback;
}

export function resolveMediaUrl(path?: string | null, apiBaseUrl: string = ''): string {
  if (!path) return '';
  if (/^https?:\/\//i.test(path) || path.startsWith('blob:') || path.startsWith('data:')) {
    return path;
  }
  if (!apiBaseUrl) return path;
  const cleanBase = apiBaseUrl.replace(/\/+$/, '');
  const cleanPath = path.startsWith('/') ? path : `/${path}`;
  return `${cleanBase}${cleanPath}`;
}
