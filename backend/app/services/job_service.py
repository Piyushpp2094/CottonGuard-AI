"""
Asynchronous inference job manager for real-time Server-Sent Events (SSE) progress streaming.
"""

import asyncio
import json
import threading
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List

from PIL import Image

from backend.app.config import get_settings
from backend.app.database import SessionLocal
from backend.app.models import PredictionJob, User
from backend.app.services.audit_service import record_audit_event


class JobManager:
    """Tracks active prediction jobs and fans out stage events to authenticated SSE subscribers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._history: Dict[str, List[Dict[str, Any]]] = {}
        self._queues: Dict[str, List[tuple[asyncio.AbstractEventLoop, asyncio.Queue]]] = {}

    def _publish(self, job_id: str, event_name: str, payload: Dict[str, Any]) -> None:
        item = {"event": event_name, "data": payload}
        with self._lock:
            self._history.setdefault(job_id, []).append(item)
            subscribers = list(self._queues.get(job_id, []))

        for loop, q in subscribers:
            try:
                loop.call_soon_threadsafe(q.put_nowait, item)
            except RuntimeError:
                pass

    def enqueue_analysis_job(
        self,
        *,
        user_id: int,
        rgb_image: Image.Image,
        raw_bytes: bytes,
        ext: str,
        seg_alpha: float = 0.40,
    ) -> str:
        """Creates a persisted PredictionJob and starts background inference worker."""
        job_id = str(uuid.uuid4())
        db = SessionLocal()
        try:
            job = PredictionJob(
                id=job_id,
                user_id=user_id,
                status="queued",
                current_stage="preprocessing",
                step=1,
                total_steps=7,
            )
            db.add(job)
            db.commit()
        finally:
            db.close()

        self._publish(
            job_id,
            "progress",
            {
                "stage": "preprocessing",
                "status": "complete",
                "step": 1,
                "total_steps": 7,
                "result_summary": {"resolution": list(rgb_image.size)},
            },
        )

        worker = threading.Thread(
            target=self._run_worker,
            kwargs={
                "job_id": job_id,
                "user_id": user_id,
                "rgb_image": rgb_image,
                "raw_bytes": raw_bytes,
                "ext": ext,
                "seg_alpha": seg_alpha,
            },
            daemon=True,
        )
        worker.start()
        return job_id

    def _update_job_db(
        self,
        job_id: str,
        *,
        status: str,
        stage: str,
        step: int,
        prediction_id: int | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        completed: bool = False,
    ) -> None:
        db = SessionLocal()
        try:
            job = db.query(PredictionJob).filter(PredictionJob.id == job_id).first()
            if job:
                job.status = status
                job.current_stage = stage
                job.step = step
                if prediction_id is not None:
                    job.prediction_id = prediction_id
                if error_code is not None:
                    job.error_code = error_code
                if error_message is not None:
                    job.error_message = error_message
                if completed:
                    job.completed_at = datetime.utcnow()
                db.commit()
        finally:
            db.close()

    def _run_worker(
        self,
        *,
        job_id: str,
        user_id: int,
        rgb_image: Image.Image,
        raw_bytes: bytes,
        ext: str,
        seg_alpha: float,
    ) -> None:
        from backend.app.services.prediction_service import prediction_service

        def on_progress(stage: str, stage_status: str, step: int, total_steps: int, summary: Dict[str, Any] | None = None) -> None:
            self._update_job_db(job_id, status="running", stage=stage, step=step)
            payload: Dict[str, Any] = {
                "stage": stage,
                "status": stage_status,
                "step": step,
                "total_steps": total_steps,
            }
            if summary:
                payload["result_summary"] = summary
            self._publish(job_id, "progress", payload)

        db = SessionLocal()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                raise RuntimeError("Submitting user account no longer exists.")

            result = prediction_service.process_validated_image(
                rgb_image=rgb_image,
                contents=raw_bytes,
                ext=ext,
                user=user,
                db=db,
                seg_alpha=seg_alpha,
                progress_callback=on_progress,
            )
            record_audit_event(
                db,
                action="LEAF_SCAN",
                resource=f"prediction:{result['id']}",
                details=f"SSE Job {job_id}: Class={result['predicted_class']} ({result['confidence']*100:.1f}%)",
                user=user,
                commit=True,
            )
            self._update_job_db(
                job_id,
                status="completed",
                stage="complete",
                step=7,
                prediction_id=int(result["id"]),
                completed=True,
            )
            self._publish(
                job_id,
                "complete",
                {
                    "job_id": job_id,
                    "prediction_id": result["id"],
                    "redirect": f"/results/{result['id']}",
                    "result": result,
                },
            )
        except Exception as exc:
            self._update_job_db(
                job_id,
                status="failed",
                stage="error",
                step=0,
                error_code="INFERENCE_FAILED",
                error_message=str(exc),
                completed=True,
            )
            self._publish(
                job_id,
                "error",
                {
                    "job_id": job_id,
                    "stage": "pipeline",
                    "code": "INFERENCE_FAILED",
                    "message": str(exc),
                },
            )
        finally:
            db.close()

    async def subscribe(self, job_id: str) -> AsyncGenerator[Dict[str, str], None]:
        """Async generator yielding SSE event dictionaries until completion, error, or timeout."""
        def _to_json(payload: Dict[str, Any]) -> str:
            return json.dumps(
                payload,
                default=lambda obj: obj.isoformat() if hasattr(obj, "isoformat") else str(obj),
            )

        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        settings = get_settings()

        with self._lock:
            past_events = list(self._history.get(job_id, []))
            self._queues.setdefault(job_id, []).append((loop, q))

        try:
            for item in past_events:
                yield {"event": item["event"], "data": _to_json(item["data"])}
                if item["event"] in ("complete", "error"):
                    return

            deadline = loop.time() + float(settings.SSE_TIMEOUT_SECONDS)
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    yield {
                        "event": "error",
                        "data": _to_json({"job_id": job_id, "code": "SSE_TIMEOUT", "message": "Stream timed out."}),
                    }
                    return
                try:
                    item = await asyncio.wait_for(q.get(), timeout=min(remaining, 15.0))
                except asyncio.TimeoutError:
                    continue
                yield {"event": item["event"], "data": _to_json(item["data"])}
                if item["event"] in ("complete", "error"):
                    return
        finally:
            with self._lock:
                subs = self._queues.get(job_id, [])
                self._queues[job_id] = [(lp, qq) for lp, qq in subs if qq is not q]


job_manager = JobManager()
