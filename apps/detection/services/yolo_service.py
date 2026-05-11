import cv2
import numpy as np
from ultralytics import YOLO
from apps.core.env import Env
from apps.detection.services.stats_service import StatsService

class YOLOService:
    _instance = None
    _model = None
    _stats_service = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(YOLOService, cls).__new__(cls)
            cls._model = YOLO(Env.YOLO_MODEL_PATH)
            cls._stats_service = StatsService()
        return cls._instance

    def predict(self, frame):
        """
        Runs YOLO detection with ByteTrack.
        Returns a list of detections with track IDs.
        """
        # We use track=True to enable ByteTrack
        results = self._model.track(
            source=frame, 
            persist=True, 
            conf=Env.CONFIDENCE_THRESHOLD, 
            iou=Env.IOU_THRESHOLD,
            tracker="bytetrack.yaml",
            verbose=False
        )
        
        detections = []
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            for box in boxes:
                # Get Class ID and Label
                class_id = int(box.cls[0].item())
                label = result.names[class_id]

                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Get Track ID (might be None if not tracking)
                track_id = int(box.id[0].item()) if box.id is not None else None
                
                # Get Confidence
                confidence = float(box.conf[0].item())
                
                # Register detection in stats service
                if track_id is not None:
                    self._stats_service.add_detection(track_id, label)
                
                detections.append({
                    "trackId": track_id,
                    "classId": class_id,
                    "label": label,
                    "confidence": confidence,
                    "bbox": {
                        "x": int(x1),
                        "y": int(y1),
                        "width": int(x2 - x1),
                        "height": int(y2 - y1)
                    }
                })
        
        return detections

    def get_current_stats(self):
        """Returns the last 15 minutes stats from the stats service."""
        return self._stats_service.get_stats()
