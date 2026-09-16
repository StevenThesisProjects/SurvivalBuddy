import time
import torch
import numpy as np

def compute_bbox_intersection_and_union(boxA, boxB):
    """Tính toán diện tích phần giao (Intersection) và phần hợp (Union) của 2 Bounding Boxes."""
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    
    inter_area = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    
    union_area = areaA + areaB - inter_area
    return inter_area, union_area

def calculate_spatiotemporal_tubelet_iou(gt_tubelet: dict, pred_tubelet: dict) -> float:
    """
    Tính Spatio-Temporal Tubelet IoU (3D IoU) chuẩn học thuật CVPR/ICCV.
    gt_tubelet: {frame_idx: [x1, y1, x2, y2]}
    pred_tubelet: {frame_idx: [x1, y1, x2, y2]}
    """
    total_inter_area = 0.0
    total_union_area = 0.0
    
    all_frames = set(gt_tubelet.keys()).union(set(pred_tubelet.keys()))
    if not all_frames:
        return 0.0

    for f in all_frames:
        has_gt = f in gt_tubelet
        has_pred = f in pred_tubelet
        
        if has_gt and has_pred:
            # Nếu cả 2 đều tồn tại bbox ở frame này, cộng phần giao và phần hợp
            inter, union = compute_bbox_intersection_and_union(gt_tubelet[f], pred_tubelet[f])
            total_inter_area += inter
            total_union_area += union
        elif has_gt:
            # Frame chỉ có Ground Truth, mất hoàn toàn phần giao (False Negative)
            box = gt_tubelet[f]
            area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
            total_union_area += area
        elif has_pred:
            # Frame chỉ có dự đoán, không có Ground Truth (False Positive)
            box = pred_tubelet[f]
            area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
            total_union_area += area

    return total_inter_area / total_union_area if total_union_area > 0 else 0.0


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