"""Filesystem locations the application works with.

Paths are derived from the current user's home and the running user's media
mount point rather than hardcoded, so the app keeps working when the username,
machine, or card label changes. Each can be overridden with the matching
environment variable.
"""

import getpass
import os
from pathlib import Path
from typing import Optional


def _env_path(name: str, default: Path) -> Path:
    """Read a path from the environment, falling back to a default."""
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


# Root of the sorted library. Also the default output directory: each run
# creates its own <model>-<timestamp> folder inside it.
LIBRARY_DIR = _env_path("VILT_KAMERA_DIR", Path.home() / "Personal" / "vilt-kamera")

# Folder holding a small fixed set of files, used for test runs.
TEST_DIR = _env_path("VILT_KAMERA_TEST_DIR", LIBRARY_DIR / "test")

# Where mounted removable media shows up. udisks2 uses /run/media/<user>,
# older setups use /media/<user>.
MEDIA_ROOTS = (Path("/run/media") / getpass.getuser(), Path("/media") / getpass.getuser())


def find_camera_card() -> Optional[Path]:
    """
    Locate the camera's media folder on a mounted card.

    Returns the first DCIM subfolder found (e.g. .../DCIM/100MEDIA), or the
    DCIM folder itself when it has no subfolders. Returns None when no card
    is mounted, so callers can prompt for a path instead.
    """
    override = os.environ.get("VILT_KAMERA_CARD_DIR")
    if override:
        path = Path(override).expanduser()
        return path if path.is_dir() else None

    for root in MEDIA_ROOTS:
        if not root.is_dir():
            continue
        try:
            mounts = sorted(root.iterdir())
        except OSError:
            continue

        for mount in mounts:
            dcim = mount / "DCIM"
            if not dcim.is_dir():
                continue
            try:
                subfolders = sorted(p for p in dcim.iterdir() if p.is_dir())
            except OSError:
                continue
            return subfolders[0] if subfolders else dcim

    return None
