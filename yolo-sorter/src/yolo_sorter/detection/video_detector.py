"""Video detection and classification."""

import cv2
from ultralytics import YOLO
from typing import List, Tuple


class VideoDetector:
    """Handles video detection and classification using YOLO."""
    
    def __init__(self, model: YOLO):
        """Initialize with a YOLO model."""
        self.model = model
    
    def is_corrupted(self, video_path: str) -> bool:
        """Check if a video file is corrupted."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return True
            ret, _ = cap.read()
            cap.release()
            return not ret
        except Exception:
            return True
    
    def get_class_name(
        self, 
        video_path: str,
        frame_sample_rate: int = 30,
        max_frames: int = 300,
        method: str = "most_frequent"
    ) -> str:
        """
        Get the class name for a video based on frame analysis.
        
        Args:
            video_path: Path to video file
            frame_sample_rate: Sample every Nth frame
            max_frames: Maximum number of frames to process
            method: 'most_frequent' or 'highest_confidence'
            
        Returns:
            Class name or 'unsorted' if no detections
        """
        detections = self._extract_detections(video_path, frame_sample_rate, max_frames)
        
        if method == "highest_confidence":
            return self._get_highest_confidence_class(detections)
        else:
            return self._get_most_frequent_class(detections)
    
    def _extract_detections(
        self, 
        video_path: str, 
        frame_sample_rate: int, 
        max_frames: int
    ) -> List[Tuple[str, float]]:
        """Extract detections from video frames."""
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        detections = []
        names = self.model.names
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_sample_rate == 0:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.model(img, verbose=False)
                
                if (results and results[0].boxes is not None 
                    and len(results[0].boxes) > 0):
                    
                    classes = results[0].boxes.cls.cpu().numpy()
                    confs = results[0].boxes.conf.cpu().numpy()
                    
                    for i, class_id in enumerate(classes):
                        class_id = int(class_id)
                        conf = confs[i]
                        
                        # Convert class ID to name
                        if isinstance(names, dict) and class_id in names:
                            class_name = names[class_id]
                        elif isinstance(names, list) and class_id < len(names):
                            class_name = names[class_id]
                        else:
                            class_name = str(class_id)
                            
                        detections.append((class_name, conf))
            
            frame_count += 1
        
        cap.release()
        return detections
    
    def _get_most_frequent_class(self, detections: List[Tuple[str, float]]) -> str:
        """Return the most frequently detected class."""
        if not detections:
            return "unsorted"
            
        class_counter = {}
        for class_name, _ in detections:
            class_counter[class_name] = class_counter.get(class_name, 0) + 1
            
        return max(class_counter.items(), key=lambda x: x[1])[0]
    
    def _get_highest_confidence_class(self, detections: List[Tuple[str, float]]) -> str:
        """Return the class with the highest confidence score."""
        if not detections:
            return "unsorted"
            
        highest = max(detections, key=lambda x: x[1])
        return highest[0]