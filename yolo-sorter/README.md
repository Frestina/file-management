# YOLO Sorter

A Python application for automatically sorting images and videos using YOLO object detection.

## Features

- Sort images and videos into folders based on detected objects
- Compress files to save space during processing
- Support for multiple YOLO models
- Interactive CLI with rich output
- Preserve original file timestamps

## Requirements

- Python 3.13
- FFmpeg (for compression)
- YOLO model files in `models/`

## Installation

```bash
uv sync
```

## Usage

```bash
# Interactive mode - prompts for source, compression and model
yolo-sort

# Fully non-interactive: every prompt is skipped when the flag is given
yolo-sort --input_dir /path/to/files --output_dir /path/to/output \
          --model yolo11x_custom.pt --compress
```

| Flag | Effect |
| --- | --- |
| `--input_dir` | Directory to sort. Skips the source prompt. |
| `--output_dir` | Where to write results. Defaults to the library directory. |
| `--model` | Model filename, resolved against `models/`. Skips the model prompt. |
| `--compress` / `--no-compress` | Force compression on or off. Prompts when neither is given. |
| `--visualize` | Annotate a single image instead of sorting. |

Each run writes to its own `<output_dir>/<model>-<timestamp>/` folder, with one
subfolder per detected class plus `unsorted/` for files with no detections, and
`corrupted_files/` / `processing_errors/` for failures.

**Input files are never modified or deleted** - they are copied, not moved.
Do not point `--output_dir` inside `--input_dir`.

## Paths

Locations live in `src/yolo_sorter/config.py` and are derived from your home
directory, so nothing is hardcoded to one machine. Override with environment
variables:

| Variable | Default |
| --- | --- |
| `VILT_KAMERA_DIR` | `~/Personal/vilt-kamera` |
| `VILT_KAMERA_TEST_DIR` | `$VILT_KAMERA_DIR/test` |
| `VILT_KAMERA_CARD_DIR` | auto-detected from `/run/media/<user>/*/DCIM/*` |

A camera card is located at runtime, so the "Memory card from camera" option
works whatever the card is labelled. It is greyed out when no card is mounted.

## Supported formats

Images: `.jpg`, `.jpeg`, `.png`. Videos: `.avi`. Matching is case-insensitive;
output extensions are normalised to lowercase.
