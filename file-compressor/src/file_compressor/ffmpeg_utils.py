"""FFmpeg utility functions and settings."""

import subprocess
from dataclasses import dataclass
from typing import List, Optional

from .exceptions import FFmpegNotFoundError


def check_ffmpeg_available() -> bool:
  """Check if FFmpeg is available in the system PATH."""
  try:
    subprocess.run(['ffmpeg', '-version'], 
                  capture_output=True, check=True)
    return True
  except (subprocess.CalledProcessError, FileNotFoundError):
    return False


@dataclass
class CompressionSettings:
  """Settings for compression operations."""
  
  # Common settings
  overwrite: bool = True
  
  # Video settings
  video_codec: Optional[str] = None
  crf: Optional[int] = None  # Constant Rate Factor (quality)
  preset: Optional[str] = None
  scale: Optional[str] = None
  
  # Image settings  
  image_quality: Optional[int] = None  # JPEG quality (1-5, lower = better)
  
  @classmethod
  def for_videos(
    cls,
    max_resolution: str = "1920:1080",
    crf: int = 23,
    preset: str = "medium",
    codec: str = "libx264"
  ) -> "CompressionSettings":
    """Create settings optimized for video compression."""
    return cls(
        video_codec=codec,
        crf=crf,
        preset=preset,
        scale=max_resolution
    )
  
  @classmethod
  def for_images(
    cls,
    max_width: int = 1920,
    quality: int = 3
  ) -> "CompressionSettings":
    """Create settings optimized for image compression."""
    return cls(
        scale=f"{max_width}:-1",  # Keep aspect ratio
        image_quality=quality
    )


def build_ffmpeg_command(
  input_path: str,
  output_path: str,
  settings: CompressionSettings,
  is_video: bool = True
) -> List[str]:
  """Build FFmpeg command based on settings and file type."""
  cmd = ['ffmpeg', '-i', input_path]
  
  if is_video:
    if settings.video_codec:
        cmd.extend(['-vcodec', settings.video_codec])
    if settings.crf is not None:
        cmd.extend(['-crf', str(settings.crf)])
    if settings.preset:
        cmd.extend(['-preset', settings.preset])
    if settings.scale:
        cmd.extend(['-vf', f'scale={settings.scale}'])
    # Fix timestamp issues for videos
    cmd.extend(['-avoid_negative_ts', 'make_zero'])
  else:
    # Image compression
    if settings.scale:
        cmd.extend(['-vf', f'scale={settings.scale}'])
    if settings.image_quality is not None:
        cmd.extend(['-q:v', str(settings.image_quality)])
  
  if settings.overwrite:
    cmd.append('-y')
      
  cmd.append(output_path)
  return cmd