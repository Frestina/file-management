"""Benchmark video voting methods against the labelled vilt-kamera library.

Detections are extracted once per clip and cached to JSON, so the three voting
methods and any confidence threshold can be compared over identical input
without re-running inference on a CPU-only box.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ultralytics import YOLO
from yolo_sorter.detection.video_detector import VideoDetector
from yolo_sorter.utils.file_utils import VIDEO_EXTENSIONS

LIBRARY = Path(os.environ.get("VILT_KAMERA_DIR", Path.home() / "Personal" / "vilt-kamera"))
MODEL = str(Path(__file__).resolve().parent.parent / "models" /
             os.environ.get("BENCH_MODEL", "yolo11x_custom.pt"))
CACHE = Path(__file__).parent / "detections.json"

# Folders named after a species the model can actually predict. barn/shed hold
# false triggers with no matching class, unknown is unlabelled.
SPECIES = ["badger", "bird", "cat", "deer", "fox", "lynx", "moose"]


def extract():
    detector = VideoDetector(YOLO(MODEL))
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}

    clips = [
        (species, p)
        for species in SPECIES
        for p in sorted(Path(LIBRARY / species).iterdir())
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]

    for i, (species, path) in enumerate(clips, 1):
        key = f"{species}/{path.name}"
        if key in cache:
            continue

        print(f"[{i}/{len(clips)}] {key}", flush=True)
        try:
            dets = detector._extract_detections(str(path), 30, 300, 0.0)
            cache[key] = [[name, float(conf)] for name, conf in dets]
        except Exception as e:
            print(f"  failed: {e}", flush=True)
            cache[key] = []

        CACHE.write_text(json.dumps(cache))

    return cache


def vote(dets, method, min_conf):
    d = VideoDetector.__new__(VideoDetector)
    kept = [(n, c) for n, c in dets if c >= min_conf]

    if method == "highest_confidence":
        return d._get_highest_confidence_class(kept)
    if method == "most_frequent":
        return d._get_most_frequent_class(kept)
    return d._get_confidence_weighted_class(kept)


def evaluate(cache):
    methods = ["most_frequent", "highest_confidence", "confidence_weighted"]
    thresholds = [0.0, 0.25, 0.35, 0.45, 0.55]

    print(f"\n{len(cache)} clips, {sum(len(v) for v in cache.values())} raw detections\n")
    print(f"{'method':<22}" + "".join(f"  conf>={t:<6}" for t in thresholds))

    best = None
    for method in methods:
        row = f"{method:<22}"
        for t in thresholds:
            correct = sum(
                vote(dets, method, t) == key.split("/")[0]
                for key, dets in cache.items()
            )
            acc = correct / len(cache)
            row += f"  {correct:>3}/{len(cache)} {acc:>4.0%}"
            if best is None or acc > best[0]:
                best = (acc, method, t)
        print(row)

    acc, method, t = best
    print(f"\nBest: {method} at conf>={t} -> {acc:.1%}")

    # Confusion matrix for the winning configuration.
    print(f"\nConfusion matrix ({method}, conf>={t}) - rows are true labels")
    labels = SPECIES + ["unsorted"]
    print(f"{'':<9}" + "".join(f"{l[:7]:>9}" for l in labels))

    for true in SPECIES:
        counts = {l: 0 for l in labels}
        for key, dets in cache.items():
            if key.split("/")[0] != true:
                continue
            pred = vote(dets, method, t)
            counts[pred] = counts.get(pred, 0) + 1
        print(f"{true:<9}" + "".join(f"{counts[l] or '.':>9}" for l in labels))


if __name__ == "__main__":
    evaluate(extract())
