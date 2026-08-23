"""Custom exceptions for file compression."""


class CompressionError(Exception):
  """Raised when file compression fails."""
  pass


class FFmpegNotFoundError(CompressionError):
  """Raised when FFmpeg is not found or not available."""
  pass