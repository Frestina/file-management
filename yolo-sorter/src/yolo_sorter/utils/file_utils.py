"""File system utilities."""

import os
from typing import List, Tuple

# The extensions the sorter knows how to handle. Camera traps vary by make:
# older units write .avi, most current ones .mp4, a few .mov. Anything not
# listed here is reported as skipped rather than passed over in silence.
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
VIDEO_EXTENSIONS = (".avi", ".mp4", ".mov")


def count_files_by_type(input_dir: str) -> Tuple[int, int]:
    """
    Count image and video files in a directory.

    Args:
        input_dir: Directory to scan

    Returns:
        Tuple of (num_images, num_videos)
    """
    files = os.listdir(input_dir)
    num_images = sum(f.lower().endswith(IMAGE_EXTENSIONS) for f in files)
    num_videos = sum(f.lower().endswith(VIDEO_EXTENSIONS) for f in files)

    return num_images, num_videos


def find_unsupported_files(input_dir: str) -> List[str]:
    """
    Return the names of files the sorter will not process.

    A card holding footage in an unrecognised format would otherwise appear to
    sort cleanly while quietly leaving most of it behind, so callers report
    these to the user.

    Args:
        input_dir: Directory to scan

    Returns:
        Filenames that match neither the image nor the video extensions
    """
    known = IMAGE_EXTENSIONS + VIDEO_EXTENSIONS

    return sorted(
        f for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f))
        and not f.lower().endswith(known)
    )


def get_total_size(input_dir: str) -> int:
    """
    Return the total size in bytes of all files in the directory (non-recursive).
    
    Args:
        input_dir: Directory to calculate size for
        
    Returns:
        Total size in bytes
    """
    files = os.listdir(input_dir)
    total = 0
    
    for f in files:
        path = os.path.join(input_dir, f)
        if os.path.isfile(path):
            total += os.path.getsize(path)
    
    return total


def get_total_size_recursive(directory: str) -> int:
    """
    Return the total size in bytes of all files in the directory and subdirectories (recursive).
    
    Args:
        directory: Root directory to calculate size for
        
    Returns:
        Total size in bytes
    """
    total = 0
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                total += os.path.getsize(file_path)
            except (OSError, FileNotFoundError):
                # Skip files that can't be accessed
                pass
    
    return total