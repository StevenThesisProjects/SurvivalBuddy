import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

class DroneSmallObjectDetector:
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.15,
        slice_height: int = 512,
        slice_width: int = 512,
        overlap_ratio: float = 0.25,
        device: str = "cpu"
    ):
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            device=device
        )
        self.slice_h = slice_height
        self.slice_w = slice_width
        self.overlap = overlap_ratio

    def detect(self, frame: np.ndarray) -> list:
        result = get_sliced_prediction(
            frame,
            self.detection_model,
            slice_height=self.slice_h,
            slice_width=self.slice_w,
            overlap_height_ratio=self.overlap,
            overlap_width_ratio=self.overlap,
            postprocess_type="GREEDYNMM",
            postprocess_match_threshold=0.5,
            verbose=0
        )
        detections = []
        for pred in result.object_prediction_list:
            bbox = pred.bbox.to_xyxy()
            detections.append({
                'bbox': [int(b) for b in bbox],
                'score': float(pred.score.value)
            })
        return detections
