"""Main file compression functionality."""

import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

from rich.console import Console

from file_renamer import get_creation_time
from .exceptions import CompressionError, FFmpegNotFoundError
from .ffmpeg_utils import (
    build_ffmpeg_command,
    check_ffmpeg_available,
    CompressionSettings
)


class FileCompressor:
  """Handles file compression using FFmpeg."""
  
  def __init__(self, console: Optional[Console] = None):
    """
    Initialize the FileCompressor.
    
    Args:
      console: Rich console for output (optional)
    """
    self.console = console or Console()
    self._original_file_info: Dict[str, Tuple[str, float]] = {}
  
  def is_ffmpeg_available(self) -> bool:
    """Check if FFmpeg is available."""
    return check_ffmpeg_available()
  
  def compress_video(
    self,
    input_path: str,
    output_path: str,
    settings: Optional[CompressionSettings] = None,
    store_original_info: bool = False
  ) -> bool:
    """
    Compress a video file using FFmpeg.
    
    Args:
      input_path: Path to input video file
      output_path: Path for output compressed file
      settings: Compression settings (uses defaults if None)
      store_original_info: Whether to store original file info for later use
        
    Returns:
      True if compression was successful, False otherwise
    """
    if not self.is_ffmpeg_available():
      raise FFmpegNotFoundError("FFmpeg is not available")
    
    if settings is None:
      settings = CompressionSettings.for_videos()
    
    # Store original info if requested
    if store_original_info:
      original_timestamp = get_creation_time(input_path)
      self._original_file_info[output_path] = (input_path, original_timestamp)
    
    try:
      cmd = build_ffmpeg_command(input_path, output_path, settings, is_video=True)
      result = subprocess.run(cmd, capture_output=True, check=True)
      return True
        
    except subprocess.CalledProcessError as e:
      self.console.print(f"[red]FFmpeg error compressing {input_path}: {e}[/red]")
      return False
    except Exception as e:
      self.console.print(f"[red]Error compressing {input_path}: {e}[/red]")
      return False
  
  def compress_image(
    self,
    input_path: str,
    output_path: str,
    settings: Optional[CompressionSettings] = None,
    store_original_info: bool = False
  ) -> bool:
    """
    Compress an image file using FFmpeg.
    
    Args:
      input_path: Path to input image file
      output_path: Path for output compressed file  
      settings: Compression settings (uses defaults if None)
      store_original_info: Whether to store original file info for later use
        
    Returns:
      True if compression was successful, False otherwise
    """
    if not self.is_ffmpeg_available():
      raise FFmpegNotFoundError("FFmpeg is not available")
    
    if settings is None:
      settings = CompressionSettings.for_images()
    
    # Store original info if requested
    if store_original_info:
      original_timestamp = get_creation_time(input_path)
      self._original_file_info[output_path] = (input_path, original_timestamp)
    
    try:
      cmd = build_ffmpeg_command(input_path, output_path, settings, is_video=False)
      result = subprocess.run(cmd, capture_output=True, check=True)
      return True
        
    except subprocess.CalledProcessError as e:
      self.console.print(f"[red]FFmpeg error compressing {input_path}: {e}[/red]")
      return False
    except Exception as e:
      self.console.print(f"[red]Error compressing {input_path}: {e}[/red]")
      return False
  
  def get_compressed_path(
    self,
    temp_dir: Optional[Path],
    original_path: str,
    is_video: bool = False
  ) -> str:
    """
    Get the path for a compressed file.
    
    Args:
      temp_dir: Temporary directory for compressed files (None means use original)
      original_path: Path to original file
      is_video: Whether the file is a video
        
    Returns:
      Path where compressed file should be saved
    """
    if not temp_dir:
      return original_path
    
    filename = Path(original_path).name
    if is_video:
      compressed_path = temp_dir / filename
    else:
      stem = Path(original_path).stem
      compressed_path = temp_dir / f"{stem}.jpg"
    
    return str(compressed_path)
  
  def get_original_file_info(self, compressed_path: str) -> Optional[Tuple[str, float]]:
    """
    Get original file info for a compressed file.
    
    Returns:
      Tuple of (original_path, original_timestamp) or None if not found
    """
    return self._original_file_info.get(compressed_path)
  
  def clear_original_file_info(self):
    """Clear stored original file information."""
    self._original_file_info.clear()