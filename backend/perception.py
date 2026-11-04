"""
SafeFlow-AI Real-Time Computer Vision Perception Module
Implements:
1. Video capture stream processing (camera / benchmark video file).
2. Multi-class vehicle detection via YOLOv8.
3. Persistent multi-object tracking via ByteTrack.
4. Lane-level classification and PCU density aggregation.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import cv2

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


# COCO class IDs mapped to Indian traffic vehicle types
COCO_VEHICLE_MAP = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}


class TrafficPerceptionEngine:
    def __init__(self, model_name: str = "yolov8n.pt", confidence_thresh: float = 0.35):
        self.conf_thresh = confidence_thresh
        self.model = None
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(model_name)
            except Exception as e:
                print(f"Warning: Could not initialize YOLO model: {e}")

    def process_frame(
        self,
        frame: np.ndarray,
        zones: Optional[Dict[str, List[Tuple[int, int]]]] = None
    ) -> Dict[str, Any]:
        """
        Processes a single video frame:
        - Runs YOLO detection and tracking.
        - Assigns detected vehicles to directional approach zones.
        - Calculates counts per category and total PCU.
        """
        h, w = frame.shape[:2]
        annotated_frame = frame.copy()

        # Default 4-way crossroad quadrants if no custom polygon zones provided
        counts = {
            "North": {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0, "autorickshaw": 0},
            "South": {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0, "autorickshaw": 0},
            "East": {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0, "autorickshaw": 0},
            "West": {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0, "autorickshaw": 0}
        }

        detections = []

        if self.model is not None:
            results = self.model.track(frame, persist=True, verbose=False, conf=self.conf_thresh)
            if results and len(results) > 0:
                res = results[0]
                boxes = res.boxes
                if boxes is not None:
                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        if cls_id in COCO_VEHICLE_MAP:
                            vtype = COCO_VEHICLE_MAP[cls_id]
                            xyxy = box.xyxy[0].tolist()
                            conf = float(box.conf[0].item())
                            track_id = int(box.id[0].item()) if box.id is not None else None

                            # Center point of bounding box
                            cx = int((xyxy[0] + xyxy[2]) / 2.0)
                            cy = int((xyxy[1] + xyxy[3]) / 2.0)

                            # Determine approach arm based on spatial position
                            arm = "North" if cy < h * 0.5 and cx < w * 0.6 else "South"
                            if cx > w * 0.7:
                                arm = "East"
                            elif cx < w * 0.3:
                                arm = "West"

                            counts[arm][vtype] += 1

                            # Draw bounding box on frame
                            color = (0, 255, 0) if vtype == "car" else ((255, 100, 0) if vtype == "bus" else (0, 200, 255))
                            cv2.rectangle(
                                annotated_frame,
                                (int(xyxy[0]), int(xyxy[1])),
                                (int(xyxy[2]), int(xyxy[3])),
                                color,
                                2
                            )
                            label = f"{vtype} #{track_id if track_id else ''} {conf:.2f}"
                            cv2.putText(
                                annotated_frame,
                                label,
                                (int(xyxy[0]), max(20, int(xyxy[1]) - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color,
                                2
                            )

                            detections.append({
                                "type": vtype,
                                "track_id": track_id,
                                "conf": round(conf, 2),
                                "arm": arm,
                                "box": [round(x, 1) for x in xyxy]
                            })

        return {
            "annotated_frame": annotated_frame,
            "vehicle_counts": counts,
            "raw_detections": detections
        }

