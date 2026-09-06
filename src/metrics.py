import time
import torch
import numpy as np


def compute_iou_bbox(boxA, boxB):
    """Tính IoU giữa 2 bounding box [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    denom = float(boxAArea + boxBArea - interArea)
    return interArea / denom if denom > 0 else 0.0


def calculate_st_iou(ground_truth_dict, predictions_dict):
    """Tính Spatio-Temporal IoU (STIoU) trên toàn bộ khung hình của một video."""
    all_frames = set(ground_truth_dict.keys()).union(
        set(predictions_dict.keys())
    )
    if not all_frames:
        return 0.0

    intersection_iou_sum = 0.0
    for f in all_frames:
        if f in ground_truth_dict and f in predictions_dict:
            gt_box = ground_truth_dict[f]
            pred_box = predictions_dict[f]
            intersection_iou_sum += compute_iou_bbox(gt_box, pred_box)

    return intersection_iou_sum / float(len(all_frames))


class HardwareProfiler:

    def __init__(self):
        self.start_time = 0
        self.frame_count = 0

    def start(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        self.start_time = time.time()
        self.frame_count = 0

    def update(self, count=1):
        self.frame_count += count

    def get_summary(self):
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0.0
        peak_vram = (
            torch.cuda.max_memory_allocated() / (1024**3)
            if torch.cuda.is_available()
            else 0.0
        )
        return {
            "Elapsed Time (s)": round(elapsed, 2),
            "Throughput (FPS)": round(fps, 2),
            "Peak VRAM (GB)": round(peak_vram, 2),
        }