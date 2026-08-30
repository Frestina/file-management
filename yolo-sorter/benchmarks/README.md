# Benchmarks

Scoring harnesses for the classifier backends, run against a labelled library
laid out as one folder per species.

Point them at a library with `VILT_KAMERA_DIR`; it defaults to
`~/Personal/vilt-kamera`. Results are cached to JSON beside each script, so a
re-run only predicts files it has not already seen — useful, since a full pass
takes about an hour on CPU.

| script | what it measures |
| --- | --- |
| `bench_images.py` | the local YOLO model on stills, swept across confidence thresholds |
| `bench_video.py` | the local YOLO model on clips, comparing the three frame-pooling methods |
| `bench_speciesnet.py` | the SpeciesNet backend on stills |

```bash
cd yolo-sorter
.venv/bin/python benchmarks/bench_speciesnet.py
```

## Reading the results

Report **precision when the model commits** separately from overall accuracy.
A model that abstains on a fifth of the input is not a coin flip; it is a triage
tool, and conflating the two understates it badly.

`bench_speciesnet.py` folds predictions with the sorter's own `fold_label`, so
it measures shipped code rather than a second copy of the mapping. It treats
`roe_deer` / `red_deer` as satisfying a coarse `deer` folder, because the
library labels one class where the classifier resolves several.

## Capture conditions

Contrast was tested as a predictor of classifier error and does **not** work for
that: at every threshold it caught none of SpeciesNet's incorrect calls, and
above 30 it only began discarding correct ones. It does predict *abstention* —
below contrast 30 every file in the library was one the classifier declined to
identify, against roughly a fifth above it. That is why the sorter uses it to
explain unsorted files rather than to gate classifications.

## Caveats

The library's labels are curated unevenly — some folders were hand-corrected,
others were not. Treat these numbers as a **relative** comparison between
models, not as a certified accuracy figure. Before quoting a number to a
customer, hand-label a held-out set.

Lynx is not measurable from the current library: all six lynx stills are fog or
night (median contrast 27.6, against 40–57 for every other class), so they test
image conditions rather than the model.
