"""Video detection and classification."""

import cv2
from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from ultralytics import YOLO


class VideoDetector:
    """Handles video detection and classification using YOLO."""
    
    def __init__(self, model: "YOLO"):
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
        method: str = "most_frequent",
        min_confidence: float = 0.25
    ) -> str:
        """
        Get the class name for a video based on frame analysis.

        Args:
            video_path: Path to video file
            frame_sample_rate: Sample every Nth frame
            max_frames: Maximum number of frames to process
            method: 'most_frequent' (default), 'confidence_weighted' or 'highest_confidence'
            min_confidence: Discard detections below this score before voting

        Returns:
            Class name or 'unsorted' if no detections
        """
        return self.get_classification(
            video_path, frame_sample_rate, max_frames, method, min_confidence
        )[0]

    def get_classification(
        self,
        video_path: str,
        frame_sample_rate: int = 30,
        max_frames: int = 300,
        method: str = "most_frequent",
        min_confidence: float = 0.25
    ) -> Tuple[str, float]:
        """
        Get the class name for a video along with a score for that call.

        The score is the mean confidence of the frames that voted for the
        winning class, so a clip decided by one glimpse reads differently from
        one the model saw clearly throughout. Returns ('unsorted', 0.0) when
        nothing was detected.
        """
        detections = self._extract_detections(
            video_path, frame_sample_rate, max_frames, min_confidence
        )

        if method == "highest_confidence":
            class_name = self._get_highest_confidence_class(detections)
        elif method == "most_frequent":
            class_name = self._get_most_frequent_class(detections)
        else:
            class_name = self._get_confidence_weighted_class(detections)

        scores = [conf for name, conf in detections if name == class_name]
        confidence = sum(scores) / len(scores) if scores else 0.0

        return class_name, confidence

    def _extract_detections(
        self,
        video_path: str,
        frame_sample_rate: int,
        max_frames: int,
        min_confidence: float = 0.0
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
                        conf = float(confs[i])

                        if conf < min_confidence:
                            continue

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
    
    def _get_confidence_weighted_class(self, detections: List[Tuple[str, float]]) -> str:
        """
        Return the class with the highest total confidence across all frames.

        Sits between the other two methods and avoids how each of them goes
        wrong: a plain frame count lets a run of weak, wrong detections outvote
        a handful of strong correct ones, while taking the single highest score
        lets one unlucky frame decide the whole clip. Summing confidence per
        class means a frame only carries as much weight as the model's certainty
        about it.
        """
        if not detections:
            return "unsorted"

        class_scores = {}
        for class_name, conf in detections:
            class_scores[class_name] = class_scores.get(class_name, 0.0) + conf

        return max(class_scores.items(), key=lambda x: x[1])[0]

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