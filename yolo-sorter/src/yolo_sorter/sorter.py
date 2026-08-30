"""Main YOLO file sorting class."""

import csv
import os
import shutil
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

from rich.console import Console
from rich.progress import Progress

from file_compressor import FileCompressor, CompressionSettings
from file_renamer import FileRenamer

from .detection.image_detector import ImageDetector
from .detection.video_detector import VideoDetector
from .ui.display import create_progress, sorting_summary, visualize_summary
from .utils.image_quality import (
  LOW_CONTRAST_THRESHOLD,
  describe_conditions,
  measure_contrast,
  measure_video_contrast,
)
from .utils.file_utils import (
  IMAGE_EXTENSIONS,
  VIDEO_EXTENSIONS,
  count_files_by_type,
  find_unsupported_files,
  get_total_size,
)
from .utils.stats import calculate_compression_stats


class YoloFileSorter:
  """Main class for sorting files using YOLO object detection."""
  
  def __init__(
      self,
      model_path: str,
      input_dir: str,
      output_dir: str,
      console: Console,
      visualize_only: bool = False,
      compress_files: bool = True,
      video_method: str = "most_frequent",
      min_confidence: float = 0.25,
      backend: str = "yolo",
      language: str = "en",
      low_contrast_threshold: float = LOW_CONTRAST_THRESHOLD,
  ) -> None:
    """Initialize the YOLO file sorter."""
    self.backend = backend
    self.language = language
    self.low_contrast_threshold = low_contrast_threshold

    model_name = (
      "speciesnet" if backend == "speciesnet"
      else os.path.splitext(os.path.basename(model_path))[0]
    )
    self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if backend == "speciesnet":
      self.model = None
    else:
      # Imported lazily so the speciesnet backend does not need ultralytics.
      from ultralytics import YOLO
      self.model = YOLO(model_path)
    self.input_dir = input_dir
    self.output_dir = Path(output_dir)
    
    if not visualize_only:
      self.base_output_dir = self.output_dir / f"{model_name}-{self.timestamp}"
      os.makedirs(self.base_output_dir, exist_ok=True)
    else:
        self.base_output_dir = None
        
    self.console = console
    self.compress_files = compress_files
    self.video_method = video_method
    self.min_confidence = min_confidence
    
    # Initialize components
    self.compressor = FileCompressor(console)
    self.renamer = FileRenamer()

    if backend == "speciesnet":
      from .detection.speciesnet_detector import (
        SpeciesNetDetector,
        SpeciesNetVideoDetector,
      )
      self.image_detector = SpeciesNetDetector(language=language)
      self.video_detector = SpeciesNetVideoDetector(self.image_detector)
    else:
      self.image_detector = ImageDetector(self.model)
      self.video_detector = VideoDetector(self.model)
    
    # Setup compression directory
    if self.compress_files and not visualize_only:
      self.temp_compressed_dir = self.base_output_dir / "temp_compressed"
      os.makedirs(self.temp_compressed_dir, exist_ok=True)
    else:
      self.temp_compressed_dir = None

    # Statistics
    if not visualize_only:
      num_images, num_videos = count_files_by_type(input_dir)
      total_bytes = get_total_size(input_dir)
      self.total_gib_input = total_bytes / (1024**3)
      self.image_count = num_images
      self.video_count = num_videos
    else:
      self.total_gib_input = 0
      self.image_count = 0
      self.video_count = 0
        
    # Per-file results, written out as manifest.csv at the end of a run.
    self.manifest: List[Dict[str, str]] = []
    self.low_contrast_count = 0

    # Counters
    self.class_counts: Dict[str, int] = {}
    self.corrupted_count = 0
    self.processing_error_count = 0
    self.compression_error_count = 0

  def sort_files(self) -> None:
      """Main method to sort all files in the input directory."""
      files = os.listdir(self.input_dir)

      self._report_unsupported_files()
      self._process_videos(files)
      self._process_images(files)
      self._cleanup_and_report()

  def _report_unsupported_files(self) -> None:
    """Warn about files in the input directory that will not be processed."""
    skipped = find_unsupported_files(self.input_dir)
    if not skipped:
      return

    self.console.print(
      f"[yellow]Skipping {len(skipped)} file(s) in an unsupported format.[/yellow] "
      f"[dim]Supported: {', '.join(IMAGE_EXTENSIONS + VIDEO_EXTENSIONS)}[/dim]"
    )

    for name in skipped[:10]:
      self.console.print(f"  [yellow]Skipped:[/yellow] {name}")

    if len(skipped) > 10:
      self.console.print(f"  [dim]...and {len(skipped) - 10} more.[/dim]")

    self.console.print("\n")

  def _process_videos(self, files: List[str]) -> None:
    """Process all video files."""
    video_files = [f for f in files if f.lower().endswith(VIDEO_EXTENSIONS)]
    if not video_files:
      return

    error_dirs = self._setup_error_directories()
    
    with create_progress(self.console) as progress:
      # Compression phase
      compressed_files = self._setup_compression_workflow(
        video_files, "videos", progress
      )
      
      # Processing phase
      video_task = progress.add_task("Processing videos...", total=len(video_files))
      
      for video_name in video_files:
        processing_path = compressed_files[video_name]
        
        self._process_single_video(
          processing_path, video_name, error_dirs, progress
        )
        progress.update(video_task, advance=1)
      
      progress.update(video_task, description="Processing complete")
    self.console.print("\n")

  def _process_single_video(
    self, 
    video_path: str, 
    video_name: str, 
    error_dirs: Dict[str, Path], 
    progress: Progress
  ) -> None:
    """Process a single video file."""
    # Check if corrupted
    if self.video_detector.is_corrupted(video_path):
      self._handle_file_error(
        video_path, error_dirs['corrupted'], 'corrupted_count',
        f"[red]Corrupted:[/red] {video_name}"
      )
      return
    
    try:
      class_name, confidence = self.video_detector.get_classification(
        video_path,
        method=self.video_method,
        min_confidence=self.min_confidence,
      )
      self._sort_file(video_path, class_name, video_name, confidence, "video")
      self.console.print(f"[green]Processed:[/green] {video_name} -> {class_name}/")
    except Exception as e:
      self._handle_file_error(
        video_path, error_dirs['processing'], 'processing_error_count',
        f"[red]Error processing {video_name}: {e}[/red]"
      )

  def _process_single_image(
    self, 
    image_path: str, 
    image_name: str, 
    error_dirs: Dict[str, Path], 
    progress: Progress
  ) -> None:
    """Process a single image file."""
    # Check if corrupted
    if self.image_detector.is_corrupted(image_path):
      self._handle_file_error(
          image_path, error_dirs['corrupted'], 'corrupted_count',
          f"[red]Corrupted:[/red] {image_name}"
      )
      return
    
    try:
      class_name, confidence = self.image_detector.get_classification(
        image_path, min_confidence=self.min_confidence
      )
      self._sort_file(image_path, class_name, image_name, confidence, "image")
      self.console.print(f"[green]Processed:[/green] {image_name} -> {class_name}/")
    except Exception as e:
      self._handle_file_error(
          image_path, error_dirs['processing'], 'processing_error_count',
          f"[red]Error processing {image_name}: {e}[/red]"
      )

  def _sort_file(
    self,
    file_path: str,
    class_name: str,
    original_name: str,
    confidence: float = 0.0,
    media_type: str = "",
  ) -> None:
    """Sort a file into the appropriate class directory."""
    class_dir = self.base_output_dir / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    
    # Get original file info if this was a compressed file
    original_info = self.compressor.get_original_file_info(file_path)
    if original_info:
      original_path, original_timestamp = original_info
    else:
      from file_renamer import get_creation_time
      original_path = file_path
      original_timestamp = get_creation_time(file_path)
    
    # Generate new filename. The extension comes from the file being copied
    # (a .png compressed to .jpg must land as .jpg), the timestamp from the
    # original.
    new_filename = self.renamer.generate_filename(
      file_path, class_name, original_timestamp
    )
    
    # Get unique path to avoid overwrites
    dest_path = self.renamer.get_unique_filename(class_dir, new_filename)
    
    # Copy file
    shutil.copy2(file_path, dest_path)
    
    # Preserve timestamps
    self._preserve_timestamps(original_path, str(dest_path))
    
    # Update counter
    self.class_counts[class_name] = self.class_counts.get(class_name, 0) + 1

    # Measured on the original, not the compressed copy, since compression
    # alters contrast.
    if media_type == "video":
      contrast = measure_video_contrast(original_path)
    else:
      contrast = measure_contrast(original_path)

    conditions = describe_conditions(contrast, self.low_contrast_threshold)
    if conditions == "low_contrast":
      self.low_contrast_count += 1

    self.manifest.append({
      "original_file": original_name,
      "species": class_name,
      "confidence": f"{confidence:.3f}",
      "captured_at": datetime.fromtimestamp(original_timestamp).isoformat(
        sep=" ", timespec="seconds"
      ),
      "media_type": media_type,
      "contrast": f"{contrast:.1f}" if contrast is not None else "",
      "conditions": conditions,
      "output_file": str(dest_path.relative_to(self.base_output_dir)),
    })

  def _preserve_timestamps(self, source_path: str, dest_path: str) -> None:
    """Preserve original file timestamps."""
    try:
      import subprocess
      stat_info = os.stat(source_path)
      os.utime(dest_path, (stat_info.st_atime, stat_info.st_mtime))
      
      if hasattr(stat_info, 'st_birthtime'):
        birth_time = stat_info.st_birthtime
        birth_dt = datetime.fromtimestamp(birth_time)
        touch_time = birth_dt.strftime("%Y%m%d%H%M.%S")
        subprocess.run(['touch', '-t', touch_time, dest_path], capture_output=True)
    except Exception as e:
      self.console.print(f"[yellow]Warning: Could not preserve timestamps: {e}[/yellow]")

  def _setup_compression_workflow(
    self, 
    files: List[str], 
    file_type: str, 
    progress: Progress
  ) -> Dict[str, str]:
    """Set up compression workflow for files."""
    compressed_files = {}
    
    if not self.compress_files:
      for file_name in files:
        src = os.path.join(self.input_dir, file_name)
        compressed_files[file_name] = src
      return compressed_files
    
    if not self.compressor.is_ffmpeg_available():
      self.console.print(f"[red]Warning: FFmpeg not found. Disabling compression for {file_type}.[/red]")
      for file_name in files:
        src = os.path.join(self.input_dir, file_name)
        compressed_files[file_name] = src
      return compressed_files
    
    compress_task = progress.add_task(f"Compressing {file_type}...", total=len(files))
    
    for file_name in files:
      progress.update(compress_task, description=f"Compressing {file_name}")
      src = os.path.join(self.input_dir, file_name)
      
      compressed_path = self.compressor.get_compressed_path(
        self.temp_compressed_dir, src, is_video=(file_type == "videos")
      )
      
      if file_type == "videos":
        settings = CompressionSettings.for_videos()
        success = self.compressor.compress_video(
            src, compressed_path, settings, store_original_info=True
        )
      else:
        settings = CompressionSettings.for_images()
        success = self.compressor.compress_image(
            src, compressed_path, settings, store_original_info=True
        )
      
      if success:
        compressed_files[file_name] = compressed_path
      else:
        compressed_files[file_name] = src
        self.compression_error_count += 1
      
      progress.update(compress_task, advance=1)
    
    return compressed_files

  def _setup_error_directories(self) -> Dict[str, Path]:
    """Create and return error directories."""
    return {
      'corrupted': self.base_output_dir / "corrupted_files",
      'processing': self.base_output_dir / "processing_errors",
      'compression': self.base_output_dir / "compression_errors"
    }

  def _handle_file_error(
    self, 
    file_path: str, 
    error_dir: Path, 
    counter_attr: str, 
    message: str
  ) -> None:
    """Handle file processing errors."""
    error_dir.mkdir(parents=True, exist_ok=True)
    dest_path = error_dir / Path(file_path).name
    
    # Make filename unique if it already exists
    counter = 1
    original_dest = dest_path
    while dest_path.exists():
      stem = original_dest.stem
      ext = original_dest.suffix
      dest_path = error_dir / f"{stem}_{counter:03d}{ext}"
      counter += 1
    
    shutil.copy2(file_path, dest_path)
    setattr(self, counter_attr, getattr(self, counter_attr) + 1)
    self.console.print(message)

  def _write_manifest(self) -> None:
    """
    Write a row per sorted file to manifest.csv.

    The folder tree answers "show me the lynx"; this answers "when, and how
    sure were you" — which is what a survey report or a compensation claim
    has to cite, and what lets someone re-check every low-confidence call
    without opening the rest.
    """
    if not self.manifest:
      return

    manifest_path = self.base_output_dir / "manifest.csv"
    fieldnames = [
      "original_file", "species", "confidence", "captured_at",
      "media_type", "contrast", "conditions", "output_file",
    ]

    try:
      with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
          sorted(self.manifest, key=lambda row: row["captured_at"])
        )
      self.console.print(
        f"[green]Wrote manifest:[/green] {manifest_path} "
        f"[dim]({len(self.manifest)} rows)[/dim]"
      )
    except OSError as e:
      self.console.print(f"[yellow]Warning: Could not write manifest: {e}[/yellow]")

  def _report_conditions(self) -> None:
    """
    Report files whose capture conditions made them hard to identify.

    Worth stating separately from the rest of the summary: it distinguishes a
    frame no model could read from one the classifier simply got nothing out
    of, which is the difference between a field problem and a software one.
    """
    if not self.low_contrast_count:
      return

    unidentified = sum(
      1 for row in self.manifest
      if row["conditions"] == "low_contrast"
      and (row["species"].startswith(("unsorted", "usortert", "review_", "usikker_")))
    )

    self.console.print(
      f"[yellow]{self.low_contrast_count} file(s) shot in poor conditions[/yellow] "
      f"[dim](contrast below {self.low_contrast_threshold:g} — fog, rain or "
      f"darkness).[/dim]"
    )
    if unidentified:
      self.console.print(
        f"  [dim]{unidentified} of those could not be identified. The frames "
        f"themselves are the limit, not the model — see manifest.csv.[/dim]"
      )

  def _cleanup_and_report(self) -> None:
    """Clean up temporary files and show final report."""
    self._write_manifest()
    self._report_conditions()

    # Clean up compressed files
    if self.temp_compressed_dir and self.temp_compressed_dir.exists():
      try:
        shutil.rmtree(self.temp_compressed_dir)
        self.console.print("[dim]Cleaned up temporary compressed files.[/dim]")
      except Exception as e:
        self.console.print(f"[yellow]Warning: Could not clean up temp files: {e}[/yellow]")
    
    # Show compression stats
    if self.compress_files and self.total_gib_input > 0:
      calculate_compression_stats(
        self.total_gib_input, self.base_output_dir, self.console
      )
    
    # Show sorting summary
    sorting_summary(
      self.console,
      self.video_count,
      self.image_count,
      self.corrupted_count,
      self.processing_error_count,
      self.class_counts,
    )

  def visualize_image(self, image: SimpleNamespace) -> None:
    """Visualize a single image with YOLO detections."""
    if self.model is None:
      raise RuntimeError(
        "--visualize draws YOLO bounding boxes and needs the yolo backend; "
        "SpeciesNet returns labels rather than boxes."
      )

    img_path = os.path.join(self.input_dir, image.name)

    # Only compress when there is a temp directory to compress into.
    # Without one the compressed path is the source path itself, and
    # FFmpeg would overwrite the original in place.
    processing_path = img_path
    if (self.compress_files and self.temp_compressed_dir
        and self.compressor.is_ffmpeg_available()):
      compressed_path = self.compressor.get_compressed_path(
        self.temp_compressed_dir, img_path
      )
      settings = CompressionSettings.for_images()
      if self.compressor.compress_image(img_path, compressed_path, settings):
        processing_path = compressed_path

    with create_progress(self.console) as progress:
      task = progress.add_task("Creating visualization...", total=1)
      result = self.model(processing_path, verbose=False)[0]
      result.show()
      progress.update(task, advance=1)

    # Save visualization
    visualize_dir = self.output_dir / "visualize"
    os.makedirs(visualize_dir, exist_ok=True)

    base_name = Path(image.name).stem
    ext = Path(image.name).suffix
    save_path = visualize_dir / image.name
    counter = 1

    while save_path.exists():
      save_path = visualize_dir / f"{base_name}_{counter}{ext}"
      counter += 1

    result.save(filename=str(save_path))

    detected_class = self.image_detector.get_class_name(processing_path)
    self.class_counts[detected_class] = self.class_counts.get(detected_class, 0) + 1

    visualize_summary(self.console, self.class_counts, save_path, detected_class)

  def _process_images(self, files: List[str]) -> None:
    """Process all image files."""
    image_files = [f for f in files if f.lower().endswith(IMAGE_EXTENSIONS)]
    if not image_files:
      return

    error_dirs = self._setup_error_directories()

    with create_progress(self.console) as progress:
      # Compression phase
      compressed_files = self._setup_compression_workflow(
        image_files, "images", progress
      )

      # Processing phase
      image_task = progress.add_task("Processing images...", total=len(image_files))

      warm = getattr(self.image_detector, "warm", None)
      if warm:
        progress.update(image_task, description="Classifying images...")
        warm(compressed_files[name] for name in image_files)

      for image_name in image_files:
        processing_path = compressed_files[image_name]

        self._process_single_image(
          processing_path, image_name, error_dirs, progress
        )
        progress.update(image_task, advance=1)

      progress.update(image_task, description="Processing complete")
    self.console.print("\n")
