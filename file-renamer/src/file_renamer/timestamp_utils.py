"""Utilities for getting file timestamps across different platforms."""

import os
from datetime import datetime

def get_creation_time(file_path: str) -> float:
    """
    Get the original creation time of a file.
    Returns the timestamp as a float (seconds since epoch).
    """
    try:
        stat = os.stat(file_path)
        if hasattr(stat, 'st_birthtime'):
            return stat.st_birthtime
        else:
            return stat.st_mtime
    except (OSError, AttributeError):
        return get_modification_time(file_path)


def get_modification_time(file_path: str) -> float:
    """Get the modification time of a file."""
    return os.path.getmtime(file_path)


def format_timestamp(timestamp: float, format_str: str = "%y-%m-%d_%H-%M-%S") -> str:
    """Format a timestamp using the specified format string."""
    return datetime.fromtimestamp(timestamp).strftime(format_str)



