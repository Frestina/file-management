"""Main entry point for YOLO sorter application."""

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console

from .config import LIBRARY_DIR
from .sorter import YoloFileSorter
from .ui.prompts import (
  default_args_prompt,
  interactive_compression_prompt,
  interactive_input_dir_prompt,
  interactive_model_prompt,
)
from .ui.display import print_arguments
from .utils.file_utils import count_files_by_type, get_total_size


class Args:
  """Configuration arguments for the application."""
  
  def __init__(
    self,
    model: str = None,
    visualize: bool = False,
    input_dir: str = None,
    output_dir: str = None,
    compress_files: bool = None
  ):
    self.model = model
    self.visualize = visualize
    self.input_dir = input_dir
    self.output_dir = output_dir or str(LIBRARY_DIR)
    self.compress_files = compress_files


def main() -> None:
  """Main application entry point."""
  parser = argparse.ArgumentParser(description="Sort images and videos using YOLO.")
  parser.add_argument("--input_dir", help="Directory with files to sort.")
  parser.add_argument(
    "--output_dir",
    default=str(LIBRARY_DIR),
    help="Directory to save sorted files.",
  )
  parser.add_argument("--model", help="YOLO model filename.")
  parser.add_argument(
    "--visualize",
    action="store_true",
    help="Visualize an image instead of sorting.",
  )
  parser.add_argument(
    "--compress",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Compress files before processing (--no-compress to disable). "
         "Prompts when not given.",
  )
  parser.add_argument(
    "--video_method",
    choices=["most_frequent", "confidence_weighted", "highest_confidence"],
    default="most_frequent",
    help="How to pick one class from a clip's sampled frames.",
  )
  parser.add_argument(
    "--backend",
    choices=["yolo", "speciesnet"],
    default="yolo",
    help="Classifier to use. On the labelled library speciesnet was correct "
         "97%% of the time when it committed, against 82%% for local yolo.",
  )
  parser.add_argument(
    "--language",
    choices=["en", "no"],
    default="en",
    help="Language for output folder names. 'no' gives Norwegian species "
         "names (elg, rådyr, gaupe). speciesnet backend only.",
  )
  parser.add_argument(
    "--low_contrast_threshold",
    type=float,
    default=30.0,
    help="Files with contrast below this are reported as shot in poor "
         "conditions (fog, rain, darkness). Annotates the manifest; never "
         "changes how a file is classified.",
  )
  parser.add_argument(
    "--min_confidence",
    type=float,
    default=0.25,
    help="Detections below this score are filed as unsorted rather than "
         "guessed at. Applies to stills and to each sampled video frame.",
  )

  parsed = parser.parse_args()
  args = Args(
    model=parsed.model,
    visualize=parsed.visualize,
    input_dir=parsed.input_dir,
    output_dir=parsed.output_dir,
    compress_files=parsed.compress
  )
  args.video_method = parsed.video_method
  args.min_confidence = parsed.min_confidence
  args.backend = parsed.backend
  args.language = parsed.language
  args.low_contrast_threshold = parsed.low_contrast_threshold

  console = Console()

  # Interactive prompts. Skip the defaults menu when the command line already
  # supplied a source, so --input_dir/--model/--no-compress are not overridden.
  if args.input_dir is None and not args.visualize:
    default_args_prompt(args, console)
  interactive_input_dir_prompt(args, console)

  if not getattr(args, "visualize", False) and args.compress_files is None:
    interactive_compression_prompt(args, console)

  if getattr(args, "visualize", False):
    interactive_model_prompt(args, console)
    _handle_visualization_mode(args, console)
    return

  # Regular sorting mode
  num_images, num_videos = count_files_by_type(args.input_dir)
  total_bytes = get_total_size(args.input_dir)
  total_gib_input = total_bytes / (1024**3)

  if num_images == 0 and num_videos == 0:
    console.print(
      "[bold red]Warning:[/bold red] No images or videos found. Exiting."
    )
    return

  console.print(
    f"[yellow]Found {num_images} images and {num_videos} videos. "
    f"({total_gib_input:.1f} GiB)[/yellow]\n"
  )

  if args.backend == "yolo":
    interactive_model_prompt(args, console)
  print_arguments(args, console)

  model_path = _resolve_model_path(args.model) if args.backend == "yolo" else ""
  
  sorter = YoloFileSorter(
    model_path=model_path,
    input_dir=args.input_dir,
    output_dir=args.output_dir,
    console=console,
    compress_files=args.compress_files,
    video_method=args.video_method,
    min_confidence=args.min_confidence,
    backend=args.backend,
    language=args.language,
    low_contrast_threshold=args.low_contrast_threshold,
  )

  try:
    console.rule("[bold cyan]Starting file sorting\n")
    sorter.sort_files()
    console.print("[green]File sorting completed![/green]\n")
      
  except Exception as e:
    console.print(f"[red]Error during processing: {e}[/red]")
    import traceback
    console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")


def _handle_visualization_mode(args: Args, console: Console) -> None:
  """Handle visualization mode."""
  model_path = _resolve_model_path(args.model)
  
  sorter = YoloFileSorter(
    model_path=model_path,
    input_dir=".",
    output_dir=args.output_dir,
    console=console,
    visualize_only=True,
    compress_files=args.compress_files
  )

  img_path = input("Enter the path to the image you want to visualize: ")
  image = SimpleNamespace(name=os.path.basename(img_path))
  sorter.input_dir = os.path.dirname(img_path) or "."
  sorter.visualize_image(image)


def _resolve_model_path(model: str) -> str:
  """Resolve the full path to the model file."""
  if os.path.isabs(model):
    return model
  
  # Look for model in the package's models directory
  package_models_dir = Path(__file__).parent.parent.parent / "models"
  model_path = package_models_dir / model
  
  if model_path.exists():
    return str(model_path)
  
  # Fall back to looking in the current working directory
  cwd_model_path = Path.cwd() / model
  if cwd_model_path.exists():
    return str(cwd_model_path)
  
  # Return as-is and let YOLO handle it
  return model


if __name__ == "__main__":
  main()
