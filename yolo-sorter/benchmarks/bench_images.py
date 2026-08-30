"""Same comparison for stills, which have no voting step - just a threshold."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ultralytics import YOLO
from yolo_sorter.detection.image_detector import ImageDetector
from yolo_sorter.utils.file_utils import IMAGE_EXTENSIONS

LIBRARY = Path(os.environ.get("VILT_KAMERA_DIR", Path.home() / "Personal" / "vilt-kamera"))
MODEL = str(Path(__file__).resolve().parent.parent / "models" /
             os.environ.get("BENCH_MODEL", "yolo11x_custom.pt"))
CACHE = Path(__file__).parent / "image_results.json"
SPECIES = ["badger", "bird", "cat", "deer", "fox", "lynx", "moose"]

detector = ImageDetector(YOLO(MODEL))
cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}

shots = [
    (sp, p)
    for sp in SPECIES
    for p in sorted((LIBRARY / sp).iterdir())
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
]

for i, (sp, path) in enumerate(shots, 1):
    key = f"{sp}/{path.name}"
    if key in cache:
        continue
    print(f"[{i}/{len(shots)}] {key}", flush=True)
    try:
        # Threshold 0.0 so any floor can be applied afterwards from the cache.
        cache[key] = list(detector.get_classification(str(path), min_confidence=0.0))
    except Exception as e:
        print(f"  failed: {e}", flush=True)
        cache[key] = ["unsorted", 0.0]
    CACHE.write_text(json.dumps(cache))

print(f"\n{len(cache)} stills\n")
print(f"{'threshold':<12}accuracy")
for t in [0.0, 0.25, 0.35, 0.45, 0.55, 0.65]:
    correct = sum(
        (name if conf >= t else "unsorted") == key.split("/")[0]
        for key, (name, conf) in cache.items()
    )
    print(f"conf>={t:<7}{correct:>4}/{len(cache)}  {correct/len(cache):.0%}")

print("\nConfusion matrix (conf>=0.25) - rows are true labels")
labels = SPECIES + ["unsorted"]
print(f"{'':<9}" + "".join(f"{l[:7]:>9}" for l in labels))
for true in SPECIES:
    counts = {l: 0 for l in labels}
    for key, (name, conf) in cache.items():
        if key.split("/")[0] != true:
            continue
        pred = name if conf >= 0.25 else "unsorted"
        counts[pred] = counts.get(pred, 0) + 1
    print(f"{true:<9}" + "".join(f"{counts[l] or '.':>9}" for l in labels))
