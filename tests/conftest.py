"""
Session-level pytest isolation so API/backend tests never touch production cotton_leaf_ai.db
or overwrite workspace artifacts across multi-file test runs.
"""

import os
import sys
import tempfile
from pathlib import Path

# Ensure project root (cotton_leaf_ai) is always in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_SESSION_TMP = tempfile.TemporaryDirectory(prefix="cottonguard_pytest_session_")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{Path(_SESSION_TMP.name) / 'pytest_suite.db'}")
os.environ.setdefault("ARTIFACTS_DIR", str(Path(_SESSION_TMP.name) / "artifacts"))


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    try:
        from backend.app.database import engine
        engine.dispose()
    except Exception:
        pass
    _SESSION_TMP.cleanup()
