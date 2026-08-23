# file-management

Automatically sort wildlife trail-camera footage by the animal in it.

Point it at a memory card full of motion-triggered photos and videos, and it
files each one into a folder named after the species it detects — using a YOLO
object-detection model, with FFmpeg compression along the way to keep the
library small.

```
card/DCIM/100MEDIA/          →   vilt-kamera/yolo11x_custom-20260823_132630/
  IMG_0042.JPG                     badger/badger_2025-05-08_04-12-06.jpg
  IMG_0043.JPG                     badger/badger_2025-05-08_04-12-06_001.jpg
  MOV_0001.AVI                     deer/deer_2025-06-12_09-31-02.jpg
  ...                              moose/moose_2025-06-05_07-02-24.avi
                                   unsorted/unsorted_2025-06-01_02-38-12.avi
```

**Your source files are never modified or deleted.** Everything is copied, never
moved.

## How it works

1. **Survey** — count images and videos in the input folder, report total size.
2. **Compress** — each file is run through FFmpeg into a temporary folder
   (images scaled to 1920px wide at JPEG q3; video to 1080p, x264 CRF 23).
   Detection then runs on the *small* copy, which is much faster. The original
   path and capture timestamp are remembered so the final file is still named
   and stamped from the original.
3. **Detect** — images take the single highest-confidence detection. Videos
   sample every 30th frame (up to 300) and take the most frequently detected
   class. No detections means `unsorted`.
4. **File** — copy into `<output>/<model>-<timestamp>/<class>/`, renamed
   `class_YYYY-MM-DD_HH-MM-SS.ext`, with the original timestamps restored.
   Unreadable files go to `corrupted_files/`, failures to `processing_errors/`.
5. **Report** — delete the temp folder, print a per-class summary table and the
   overall size reduction.

Timestamps include seconds because camera traps fire in bursts; same-second
collisions get a `_001` suffix rather than overwriting.

## Layout

Three packages, wired together as editable path dependencies:

| Package | Role |
| --- | --- |
| [`yolo-sorter`](yolo-sorter/) | The CLI application. Detection, sorting, reporting. |
| [`file-compressor`](file-compressor/) | FFmpeg wrapper for shrinking images and video. |
| [`file-renamer`](file-renamer/) | Timestamp helpers and filename generation. |

## Requirements

- Python 3.13
- FFmpeg on `PATH`
- A YOLO model file (see below)

## Install

```bash
cd yolo-sorter
uv sync
```

## Usage

```bash
# Interactive — prompts for source, compression and model
yolo-sort

# Fully non-interactive; every prompt is skipped when its flag is given
yolo-sort --input_dir /path/to/files --output_dir /path/to/output \
          --model yolo11x_custom.pt --compress
```

| Flag | Effect |
| --- | --- |
| `--input_dir` | Directory to sort. Skips the source prompt. |
| `--output_dir` | Where results go. Defaults to the library directory. |
| `--model` | Model filename, resolved against `yolo-sorter/models/`. |
| `--compress` / `--no-compress` | Force compression on or off. Prompts when neither is given. |
| `--visualize` | Annotate a single image instead of sorting. |

Don't put `--output_dir` inside `--input_dir` — the temp-folder cleanup writes
there.

## Models

Model files live in `yolo-sorter/models/` and are **not** tracked in this repo
(they are large binaries). The stock weights download automatically from
Ultralytics:

| Model | Notes |
| --- | --- |
| `yolo11s.pt` | Fast, 80 general COCO classes. Good for a quick pass. |
| `yolo11x.pt` | Most accurate stock model, 80 classes, slow and memory-hungry. |
| `yolo11x_custom.pt` | Locally trained on 8 relevant classes — `cat`, `fox`, `moose`, `badger`, `lynx`, `bird`, `deer`, `dog`. Not included; train your own or substitute a stock model. |

## Configuration

Paths are derived from your home directory, not hardcoded. Override with
environment variables:

| Variable | Default |
| --- | --- |
| `VILT_KAMERA_DIR` | `~/Personal/vilt-kamera` |
| `VILT_KAMERA_TEST_DIR` | `$VILT_KAMERA_DIR/test` |
| `VILT_KAMERA_CARD_DIR` | auto-detected from `/run/media/<user>/*/DCIM/*` |

A camera card is located at runtime, so the "Memory card from camera" option
works whatever the card is labelled, and greys out when nothing is mounted.

## Supported formats

Images `.jpg` `.jpeg` `.png`, video `.avi`. Matching is case-insensitive;
output extensions are normalised to lowercase.

## Known limitations

- **Video classification is the weak spot.** Majority vote across ten sampled
  frames means a few confident wrong frames can outvote the rest. On a sample
  run, deer clips were repeatedly labelled moose while the stills were correct.
  `VideoDetector.get_class_name` accepts `method="highest_confidence"` and
  tunable `frame_sample_rate` / `max_frames`, but `sorter.py` only ever calls it
  with defaults.
- Only `.avi` video is recognised — cameras producing `.mp4` or `.mov` are
  skipped silently.
- Each file is filed under a single class, so a frame containing two species
  keeps only the winner.
- CPU-only inference is slow. A 1 GB card takes several minutes, dominated by
  x264 compression.

## Licence

Not currently licensed. All rights reserved by default — add a licence file if
you want others to reuse this.
