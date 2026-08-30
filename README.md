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
5. **Report** — delete the temp folder, write `manifest.csv`, and print a
   per-class summary table and the overall size reduction.

`manifest.csv` sits at the root of the run folder with one row per sorted file —
`original_file`, `species`, `confidence`, `captured_at`, `media_type`,
`output_file` — ordered by capture time. The folder tree answers "show me the
lynx"; the manifest answers "when, and how sure were you", which is what a
survey report has to cite and what lets you re-check every low-confidence call
without opening the rest.

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
- A YOLO model file (see below), for the default `yolo` backend

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
| `--backend` | `yolo` (default) or `speciesnet`. |
| `--language` | `en` (default) or `no` for Norwegian folder names. `speciesnet` only. |
| `--video_method` | `confidence_weighted` (default), `most_frequent` or `highest_confidence`. |
| `--min_confidence` | Below this score a file is filed as `unsorted` rather than guessed at. Applies to stills and to each sampled video frame. Default `0.25`. |

Don't put `--output_dir` inside `--input_dir` — the temp-folder cleanup writes
there.

## Backends

Two classifiers are available.

`yolo` (default) uses a local `.pt` model from `yolo-sorter/models/`.

`speciesnet` uses [SpeciesNet](https://github.com/google/cameratrapai) (Apache-2.0),
which pairs [MegaDetector](https://github.com/agentmorris/MegaDetector) for
locating the animal with a classifier covering 2000+ labels. Scored on the 164
labelled stills in the library:

| | `yolo11x_custom` | `speciesnet` |
| --- | --- | --- |
| overall exact-correct | 107/164 — 65% | 122/164 — 74% |
| correct when it commits | 107/130 — 82% | 122/126 — 97% |
| animal missed entirely | ~21% | 3% |

Per-species precision went from 55% to 100% on badger and 85% to 100% on moose.
Being two-stage is what closes the miss rate: the classifier only has to name a
crop the detector already found.

It also answers at genus or family level when it will not commit to a species.
Those land in `review_*` folders (`usikker_*` in Norwegian) rather than
`unsorted`, because "a deer, species unclear" is worth more to someone paging
through results than "no idea".

```bash
# Norwegian folder names: elg, rådyr, gaupe, grevling, usikker_hjortedyr
yolo-sort --input_dir /path/to/card --backend speciesnet --language no
```

Install it as an extra:

```bash
cd yolo-sorter
uv sync --extra speciesnet
```

Model weights download on first run. The `speciesnet` backend does not need
`ultralytics`; that import is deferred so only the `yolo` backend pulls it in.
Geofencing is set to Norway, restricting predictions to plausible species.
`--visualize` draws YOLO boxes and is `yolo`-only.

Known gap: moose recall is 63% — SpeciesNet often rolls moose up to
`review_deer` rather than committing. Lynx is untested; the six lynx stills in
the library are all fog or night, and are not a usable test set.

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

Images `.jpg` `.jpeg` `.png`, video `.avi` `.mp4` `.mov`. Matching is
case-insensitive; output extensions are normalised to lowercase. Anything else
in the input directory is listed as skipped before the run starts, rather than
passed over in silence.

## Video classification

A clip is decided by sampling every 30th frame and pooling the detections. The
default `confidence_weighted` method sums each class's confidence across those
frames, which avoids how the alternatives go wrong: a plain frame count
(`most_frequent`) lets a run of weak, wrong detections outvote a few strong
correct ones, and `highest_confidence` lets a single unlucky frame decide the
whole clip. Both remain available via `--video_method`.

## Known limitations

- Each file is filed under a single class, so a frame containing two species
  keeps only the winner.
- CPU-only inference is slow. A 1 GB card takes several minutes, dominated by
  x264 compression.

## Licence

[MIT](LICENSE).
