import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

import {
  extractApiErrorMessage,
  formatModelStatusLabel,
  MAX_UPLOAD_SIZE_BYTES,
  PROVISIONAL_SEVERITY_TIERS,
  resolveMediaUrl,
  validateImageDimensions,
  validateUploadMetadata,
  VERIFIED_CLASSIFIER_METRICS,
} from '../src/utils/validation.ts';
import { parseSseChunks } from '../src/utils/sse.ts';

describe('CottonGuard AI Frontend Validation & Benchmarks', () => {
  it('enforces verified classifier metrics exactly', () => {
    assert.equal(VERIFIED_CLASSIFIER_METRICS.validationAccuracyPct, '98.43%');
    assert.equal(VERIFIED_CLASSIFIER_METRICS.untouchedTestAccuracyPct, '97.82%');
    assert.equal(VERIFIED_CLASSIFIER_METRICS.testMacroF1Pct, '97.54%');
    assert.equal(PROVISIONAL_SEVERITY_TIERS.length, 6);
  });

  it('validates allowed extensions, MIME types, and 10 MB size boundary', () => {
    const valid = validateUploadMetadata({ name: 'leaf.jpg', size: 256_000, type: 'image/jpeg' });
    assert.equal(valid.valid, true);

    const badExt = validateUploadMetadata({ name: 'leaf.bmp', size: 1024, type: 'image/bmp' });
    assert.equal(badExt.valid, false);
    assert.match(badExt.error || '', /Unsupported image extension/i);

    const empty = validateUploadMetadata({ name: 'leaf.png', size: 0, type: 'image/png' });
    assert.equal(empty.valid, false);

    const oversized = validateUploadMetadata({
      name: 'huge.png',
      size: MAX_UPLOAD_SIZE_BYTES + 1,
      type: 'image/png',
    });
    assert.equal(oversized.valid, false);
    assert.match(oversized.error || '', /10 MB/i);
  });

  it('enforces image resolution bounds matching backend config (100..8000 px)', () => {
    assert.equal(validateImageDimensions(224, 224).valid, true);
    assert.equal(validateImageDimensions(99, 224).valid, false);
    assert.equal(validateImageDimensions(8001, 500).valid, false);
  });

  it('maps registry model statuses honestly without mislabeling NOT_TRAINED models', () => {
    assert.equal(formatModelStatusLabel('trained'), 'TRAINED');
    assert.equal(formatModelStatusLabel('not_trained'), 'NOT_TRAINED');
    assert.equal(formatModelStatusLabel('NOT_TRAINED'), 'NOT_TRAINED');
    assert.equal(formatModelStatusLabel(undefined, false), 'NOT_TRAINED');
    assert.equal(formatModelStatusLabel(undefined, true), 'TRAINED');
  });

  it('extracts structured Phase 3 FastAPI JSON error envelopes safely', () => {
    const structured = {
      response: {
        data: {
          detail: 'compat',
          error: {
            code: 'INVALID_IMAGE_MAGIC_BYTES',
            message: 'File content signature does not match a valid JPEG, PNG, or WEBP image.',
          },
        },
      },
    };
    assert.equal(
      extractApiErrorMessage(structured, 'fallback'),
      'File content signature does not match a valid JPEG, PNG, or WEBP image.'
    );

    const validationList = {
      response: {
        data: {
          detail: [{ msg: 'Password too short' }],
        },
      },
    };
    assert.equal(extractApiErrorMessage(validationList, 'fallback'), 'Password too short');
    assert.equal(extractApiErrorMessage(null, 'fallback'), 'fallback');
  });

  it('resolves relative media URLs cleanly with or without VITE_API_URL override', () => {
    assert.equal(resolveMediaUrl('/artifacts/predictions/leaf.jpg', ''), '/artifacts/predictions/leaf.jpg');
    assert.equal(
      resolveMediaUrl('/artifacts/predictions/leaf.jpg', 'http://127.0.0.1:8000/'),
      'http://127.0.0.1:8000/artifacts/predictions/leaf.jpg'
    );
    assert.equal(resolveMediaUrl('blob:http://localhost/123', 'http://127.0.0.1:8000'), 'blob:http://localhost/123');
    assert.equal(resolveMediaUrl(null, 'http://127.0.0.1:8000'), '');
  });
});

describe('SSE Frame Parser', () => {
  it('parses incremental SSE chunks and preserves trailing partial frames', () => {
    const chunk1 =
      'event: progress\r\ndata: {"stage":"classification","status":"complete","step":2,"total_steps":7}\r\n\r\n' +
      'event: complete\r\ndata: {"prediction_id":42';

    const first = parseSseChunks(chunk1);
    assert.equal(first.events.length, 1);
    assert.equal(first.events[0].event, 'progress');
    const prog = JSON.parse(first.events[0].data);
    assert.equal(prog.stage, 'classification');
    assert.equal(prog.step, 2);

    const chunk2 = first.remainder + ',"result":{"id":42}}\r\n\r\n';
    const second = parseSseChunks(chunk2);
    assert.equal(second.events.length, 1);
    assert.equal(second.events[0].event, 'complete');
    assert.equal(JSON.parse(second.events[0].data).prediction_id, 42);
    assert.equal(second.remainder, '');
  });
});
