"""Image detection and classification."""

from typing import TYPE_CHECKING, Tuple

from PIL import Image

if TYPE_CHECKING:
    from ultralytics import YOLO


class ImageDetector:
    """Handles image detection and classification using YOLO."""
    
    def __init__(self, model: "YOLO"):
        """Initialize with a YOLO model."""
        self.model = model
    
    def is_corrupted(self, image_path: str) -> bool:
        """Check if an image file is corrupted."""
        try:
            with Image.open(image_path) as img:
                img.verify()
            return False
        except Exception:
            return True
    
    def get_class_name(self, image_path: str, min_confidence: float = 0.25) -> str:
        """
        Get the class name for the detection with the highest confidence.
        Returns 'unsorted' if no detections found.
        """
        return self.get_classification(image_path, min_confidence)[0]

    def get_classification(
        self, image_path: str, min_confidence: float = 0.25
    ) -> Tuple[str, float]:
        """
        Get the class name and score for the highest-confidence detection.

        Returns ('unsorted', 0.0) if nothing was detected above min_confidence.
        Filing a weak guess as though it were certain is worse than filing it as
        unsorted: a reviewer can page through one folder of maybes, but has no
        way to find a wrong call buried among the confident ones. Callers that
        report per-file results to a customer also need the score alongside the
        label, so a weak call can be told apart from a certain one.
        """
        results = self.model(image_path, verbose=False)
        return self._extract_classification_from_results(results, min_confidence)

    def _extract_classification_from_results(
        self, results, min_confidence: float = 0.0
    ) -> Tuple[str, float]:
        """Extract class name and confidence from YOLO results."""
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return "unsorted", 0.0
        
        classes = results[0].boxes.cls.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        
        if len(classes) == 0 or len(confs) == 0:
            return "unsorted", 0.0
        
        # Get class with highest confidence
        max_conf_idx = confs.argmax()
        class_id = int(classes[max_conf_idx])
        confidence = float(confs[max_conf_idx])
        
        if confidence < min_confidence:
            return "unsorted", 0.0
        
        # Convert class ID to name
        model_names = self.model.names
        if isinstance(model_names, dict) and class_id in model_names:
            return model_names[class_id], confidence
        elif isinstance(model_names, list) and class_id < len(model_names):
            return model_names[class_id], confidence
        
        return str(class_id), confidence