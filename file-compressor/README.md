# File Compressor

A Python library for compressing images and videos using FFmpeg.

## Installation

```bash
uv add file-compressor
```

## Requirements

- FFmpeg must be installed and available in PATH
- `file-renamer` (used to read original creation timestamps)

## Usage

```python
from file_compressor import FileCompressor, CompressionSettings
from rich.console import Console

console = Console()
compressor = FileCompressor(console)

# Check if FFmpeg is available
if not compressor.is_ffmpeg_available():
    print("FFmpeg not found!")
    exit(1)

# Compress an image
settings = CompressionSettings.for_images(max_width=1920, quality=3)
success = compressor.compress_image("input.jpg", "output.jpg", settings)

# Compress a video
settings = CompressionSettings.for_videos(max_resolution="1920:1080", crf=23)
success = compressor.compress_video("input.avi", "output.avi", settings)
```
