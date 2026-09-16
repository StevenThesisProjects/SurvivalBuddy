import numpy as np
from src.evaluation.stiou_metric import calculate_iou

class SimpleIoUTracker:
    def __init__(self, max_lost_frames=3, min_iou_threshold=0.3):
        self.max_lost_frames = max_lost_frames
        self.min_iou_threshold = min_iou_threshold
        self.tracklet = None  # Lưu trữ hộp bao của đối tượng đang được theo dõi
        self.lost_count = 0

    def update(self, detected_boxes):
        # Trường hợp 1: Chưa có đối tượng nào được theo dõi trong các frame trước
        if self.tracklet is None:
            if len(detected_boxes) > 0:
                # Khởi tạo tracklet bằng box có độ tin cậy hoặc kích thước ổn định đầu tiên
                self.tracklet = detected_boxes[0]
                self.lost_count = 0
                return self.tracklet
            return None

        # Trường hợp 2: Đang có một đối tượng nằm trong tầm theo dõi
        if len(detected_boxes) > 0:
            # Tìm box trùng khớp nhất với vị trí frame trước bằng IoU toán học
            ious = [calculate_iou(self.tracklet, box) for box in detected_boxes]
            best_idx = np.argmax(ious)
            
            if ious[best_idx] >= self.min_iou_threshold:
                self.tracklet = detected_boxes[best_idx]
                self.lost_count = 0
                return self.tracklet

        # Trường hợp 3: Không tìm thấy box khớp ở frame hiện tại (Bị khuất bóng/Nhiễu sáng)
        self.lost_count += 1
        if self.lost_count <= self.max_lost_frames:
            return self.tracklet
        else:
            self.tracklet = None
            return None