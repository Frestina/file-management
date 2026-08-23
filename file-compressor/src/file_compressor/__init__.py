"""File compression utilities using FFmpeg."""

from .compressor import FileCompressor
from .ffmpeg_utils import check_ffmpeg_available, CompressionSettings
from .exceptions import CompressionError, FFmpegNotFoundError

__version__ = "0.1.0"
__all__ = [
  "FileCompressor",
  "CompressionSettings", 
  "check_ffmpeg_available",
  "CompressionError",
  "FFmpegNotFoundError"
]