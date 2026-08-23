"""Image detection and classification."""

from PIL import Image
from ultralytics import YOLO


class ImageDetector:
    """Handles image detection and classification using YOLO."""
    
    def __init__(self, model: YOLO):
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
    
    def get_class_name(self, image_path: str) -> str:
        """
        Get the class name for the detection with the highest confidence.
        Returns 'unsorted' if no detections found.
        """
        results = self.model(image_path, verbose=False)
        return self._extract_class_name_from_results(results)
    
    def _extract_class_name_from_results(self, results) -> str:
        """Extract class name from YOLO results."""
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return "unsorted"
        
        classes = results[0].boxes.cls.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        
        if len(classes) == 0 or len(confs) == 0:
            return "unsorted"
        
        # Get class with highest confidence
        max_conf_idx = confs.argmax()
        class_id = int(classes[max_conf_idx])
        
        # Convert class ID to name
        model_names = self.model.names
        if isinstance(model_names, dict) and class_id in model_names:
            return model_names[class_id]
        elif isinstance(model_names, list) and class_id < len(model_names):
            return model_names[class_id]
        
        return str(class_id)