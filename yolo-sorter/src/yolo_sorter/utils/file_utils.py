"""File system utilities."""

import os
from typing import Tuple


def count_files_by_type(input_dir: str) -> Tuple[int, int]:
    """
    Count image and video files in a directory.
    
    Args:
        input_dir: Directory to scan
        
    Returns:
        Tuple of (num_images, num_videos)
    """
    image_exts = (".jpg", ".jpeg", ".png")
    video_exts = (".avi",)
    
    files = os.listdir(input_dir)
    num_images = sum(f.lower().endswith(image_exts) for f in files)
    num_videos = sum(f.lower().endswith(video_exts) for f in files)
    
    return num_images, num_videos


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