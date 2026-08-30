"""Filesystem locations the application works with.

Paths are derived from the current user's home and the running user's media
mount point rather than hardcoded, so the app keeps working when the username,
machine, or card label changes. Each can be overridden with the matching
environment variable.
"""

import ctypes
import getpass
import os
import string
import sys
from pathlib import Path
from typing import List, Optional


def _env_path(name: str, default: Path) -> Path:
    """Read a path from the environment, falling back to a default."""
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


# Root of the sorted library. Also the default output directory: each run
# creates its own <model>-<timestamp> folder inside it.
LIBRARY_DIR = _env_path("VILT_KAMERA_DIR", Path.home() / "Personal" / "vilt-kamera")

# Folder holding a small fixed set of files, used for test runs.
TEST_DIR = _env_path("VILT_KAMERA_TEST_DIR", LIBRARY_DIR / "test")

# Windows drive type for removable media, from GetDriveTypeW.
_DRIVE_REMOVABLE = 2


def _linux_mounts() -> List[Path]:
    """Mount points under the directories udisks2 and older setups use."""
    user = getpass.getuser()
    roots = (Path("/run/media") / user, Path("/media") / user, Path("/media"), Path("/mnt"))

    mounts = []
    for root in roots:
        if not root.is_dir():
            continue
        try:
            mounts.extend(sorted(root.iterdir()))
        except OSError:
            continue

    return mounts


def _macos_mounts() -> List[Path]:
    """Mounted volumes, which macOS gathers under a single directory."""
    volumes = Path("/Volumes")
    if not volumes.is_dir():
        return []
    try:
        return sorted(volumes.iterdir())
    except OSError:
        return []


def _windows_mounts() -> List[Path]:
    """
    Drive letters that hold removable media.

    Cards mount as their own drive rather than under a shared parent, so the
    letters are probed directly. Network and fixed drives are filtered out via
    GetDriveTypeW, falling back to every existing letter if that call is
    unavailable.
    """
    try:
        get_drive_type = ctypes.windll.kernel32.GetDriveTypeW
    except (AttributeError, OSError):
        get_drive_type = None

    mounts = []
    # C: is the system drive on essentially every install; start past it.
    for letter in string.ascii_uppercase[3:]:
        root = Path(f"{letter}:\\")
        if not root.is_dir():
            continue
        if get_drive_type and get_drive_type(str(root)) != _DRIVE_REMOVABLE:
            continue
        mounts.append(root)

    return mounts


def candidate_mounts() -> List[Path]:
    """Directories that could be the root of a mounted camera card."""
    if sys.platform == "win32":
        return _windows_mounts()
    if sys.platform == "darwin":
        return _macos_mounts()
    return _linux_mounts()


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

    for mount in candidate_mounts():
        dcim = mount / "DCIM"
        if not dcim.is_dir():
            continue
        try:
            subfolders = sorted(p for p in dcim.iterdir() if p.is_dir())
        except OSError:
            continue
        return subfolders[0] if subfolders else dcim

    return None
