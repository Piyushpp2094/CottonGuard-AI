"""
Cross-platform path helpers for Cotton Leaf AI.

Allows Windows-recorded manifest or database paths (e.g. D:\\Main_Project\\...)
to resolve transparently inside Ubuntu WSL 2 (/mnt/d/Main_Project/...) without
altering the original manifests or database records.
"""

import os
import re
from pathlib import Path
from typing import Union

WINDOWS_DRIVE_RE = re.compile(r"^([A-Za-z]):[\\/](.*)$")


def get_project_root() -> Path:
    """Returns the repository root directory."""
    return Path(__file__).resolve().parents[2]


def resolve_platform_path(path_value: Union[str, Path]) -> Path:
    """
    Resolves a stored file path across Windows and WSL 2 POSIX environments.

    If running on POSIX and given a Windows drive path like 'D:\\Main_Project\\...',
    maps it to '/mnt/d/Main_Project/...' when '/mnt/<drive>' exists.
    """
    raw = str(path_value).strip()
    if not raw:
        return Path(raw)

    if os.name != "nt":
        match = WINDOWS_DRIVE_RE.match(raw)
        if match:
            drive = match.group(1).lower()
            rest = match.group(2).replace("\\", "/").lstrip("/")
            mnt_candidate = Path(f"/mnt/{drive}") / rest
            if Path(f"/mnt/{drive}").exists():
                return mnt_candidate
        return Path(raw.replace("\\", "/"))

    return Path(raw)
