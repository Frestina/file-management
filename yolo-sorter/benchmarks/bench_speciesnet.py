"""Score the SpeciesNet backend against the labelled library.

Predictions are folded to folder names with the sorter's own `fold_label`, so
this measures the code that actually ships rather than a second copy of the
mapping. A rollup to a higher taxon ("cervidae family") counts as review work,
not as a species call - which is how the product treats it too.

Results are cached, so re-running only predicts files it has not seen.

    VILT_KAMERA_DIR=/path/to/library python benchmarks/bench_speciesnet.py
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from speciesnet import DEFAULT_MODEL, SpeciesNet

from yolo_sorter.detection.speciesnet_detector import fold_label

LIBRARY = Path(os.environ.get("VILT_KAMERA_DIR", Path.home() / "Personal" / "vilt-kamera"))
CACHE = Path(__file__).parent / "speciesnet_results.json"
SPECIES = ["badger", "bird", "cat", "deer", "fox", "lynx", "moose"]
CHUNK = 10

# The library uses one coarse "deer" folder; the sorter resolves the actual
# species. Both count as a hit against a "deer" label.
EQUIVALENT = {"deer": {"deer", "roe_deer", "red_deer", "fallow_deer"}}


def matches(predicted: str, truth: str) -> bool:
    """Whether a prediction satisfies a library folder label."""
    return predicted in EQUIVALENT.get(truth, {truth})

shots = [
    (sp, p)
    for sp in SPECIES
    for p in sorted((LIBRARY / sp).iterdir())
    if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")
]

cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
todo = [(sp, p) for sp, p in shots if f"{sp}/{p.name}" not in cache]

if todo:
    model = SpeciesNet(DEFAULT_MODEL)
    print(f"predicting {len(todo)} stills", flush=True)
    # Chunked so a long CPU run checkpoints instead of losing everything.
    for i in range(0, len(todo), CHUNK):
        batch = todo[i:i + CHUNK]
        res = model.predict(filepaths=[str(p) for _, p in batch], country="NOR")
        by_path = {r["filepath"]: r for r in res["predictions"]}
        for sp, p in batch:
            r = by_path.get(str(p), {})
            cache[f"{sp}/{p.name}"] = [
                r.get("prediction", ""),
                r.get("prediction_score", 0.0),
                len(r.get("detections", [])),
            ]
        CACHE.write_text(json.dumps(cache))
        print(f"  {min(i + CHUNK, len(todo))}/{len(todo)}", flush=True)

folded = {k: fold_label(v[0]) for k, v in cache.items()}
committed = {k: v for k, v in folded.items()
             if not v.startswith("review_") and v != "unsorted"}
correct = sum(matches(v, k.split("/")[0]) for k, v in committed.items())
rollups = sum(1 for v in folded.values() if v.startswith("review_"))
blank = sum(1 for v in folded.values() if v == "unsorted")
nodet = sum(1 for v in cache.values() if v[2] == 0)

print(f"\n{len(folded)} stills\n")
print(f"committed to a species : {len(committed):>4}   correct {correct} "
      f"({correct / len(committed):.0%})" if committed else "no commits")
print(f"rollup (genus/family)  : {rollups:>4}")
print(f"blank / nothing seen   : {blank:>4}")
print(f"animal not detected    : {nodet:>4}  ({nodet / len(folded):.0%})")
print(f"overall exact-correct  : {correct}/{len(folded)}  {correct / len(folded):.0%}")

print(f"\n{'class':<9}{'recall':>13}{'precision':>22}")
for sp in SPECIES:
    truth = [k for k in folded if k.split("/")[0] == sp]
    filed = [k for k, v in folded.items() if v in EQUIVALENT.get(sp, {sp})]
    hit = sum(matches(folded[k], sp) for k in truth)
    right = sum(matches(sp_pred := folded[k], k.split("/")[0]) for k in filed)
    prec = f"{right}/{len(filed)} {right / len(filed):.0%}" if filed else "none filed"
    rec = f"{hit}/{len(truth)} {hit / len(truth):.0%}" if truth else "n/a"
    print(f"{sp:<9}{rec:>13}{prec:>22}")
