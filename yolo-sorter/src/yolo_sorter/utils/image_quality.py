"""Capture-condition measurement.

A camera trap fires in fog, rain and full dark, and some of what it returns is
not identifiable by any model. Measuring that up front lets the report say
"unusable frame" instead of leaving an unexplained file in `unsorted`, which is
the difference between a software failure and a field condition when someone is
citing the output in a survey.

Contrast — mean per-channel standard deviation — is the useful signal. Fog
produces frames that are *bright* but flat, so brightness alone does not
separate them: in the reference library the fogged frames sit near 99, brighter
than correctly-classified badger shots at 36. Their contrast is what collapses.

What this does not do is predict wrong answers. Measured against the labelled
library, no threshold caught any of the classifier's incorrect calls, and
thresholds above the default only discarded correct ones. It is therefore used
to annotate and to explain abstentions, never to override a classification.
"""

from pathlib import Path
from typing import Optional

# Below this, every file in the reference library was one the classifier
# declined to identify; above it, the abstention rate falls to roughly a fifth.
LOW_CONTRAST_THRESHOLD = 30.0

# Frames sampled to characterise a clip. Conditions rarely change within one
# short trigger, so a handful spread through the file is enough.
VIDEO_SAMPLES = 5


def measure_contrast(file_path: str) -> Optional[float]:
    """
    Return mean per-channel standard deviation, or None if unreadable.

    Higher is more contrast. Callers treat None as "unknown" rather than as
    poor conditions — an unreadable file is a separate problem.
    """
    from PIL import Image, ImageStat

    try:
        with Image.open(file_path) as img:
            stat = ImageStat.Stat(img.convert("RGB"))
        return sum(stat.stddev) / 3
    except Exception:
        return None


def measure_video_contrast(file_path: str, samples: int = VIDEO_SAMPLES) -> Optional[float]:
    """Return the median contrast of frames sampled across a clip."""
    import cv2
    import numpy as np

    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return None

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total > 0:
            positions = [int(total * i / (samples + 1)) for i in range(1, samples + 1)]
        else:
            positions = list(range(samples))

        values = []
        for pos in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if ret:
                values.append(float(np.std(frame)))

        cap.release()
        return float(np.median(values)) if values else None
    except Exception:
        return None


def describe_conditions(
    contrast: Optional[float], threshold: float = LOW_CONTRAST_THRESHOLD
) -> str:
    """Summarise capture conditions for a manifest row."""
    if contrast is None:
        return "unknown"
    return "low_contrast" if contrast < threshold else "ok"
