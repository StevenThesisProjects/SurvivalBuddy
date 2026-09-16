import os
import json
import yaml
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from PIL import Image
from src.data_loader.video_reader import DroneVideoDataset
from src.matcher import ZeroShotHybridMatcher

class RescueInferencePipeline:
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)
            
        self.device = "cuda" if torch.cuda.is_available() and self.cfg['inference']['device'] == "cuda" else "cpu"
        print(f"🚀 Khởi chạy Pipeline Inference trên thiết bị: {self.device}")
        
        # 1. Khởi tải YOLO với trọng số custom đã train
        detector_path = self.cfg['model']['detector_path']
        print(f"📦 Đang tải YOLO detector từ: {detector_path}")
        self.detector = YOLO(detector_path)
        
        # 2. Khởi tạo Hybrid Matcher (CLIP + DINOv2)
        print("🔗 Đang tải Zero-Shot Hybrid Matcher (CLIP + DINOv2)...")
        self.matcher = ZeroShotHybridMatcher(
            clip_weight=0.75,
            dino_weight=0.25
        ).to(self.device)
        
        self.similarity_threshold = self.cfg['inference']['similarity_threshold']
        self.conf_threshold = self.cfg['inference']['conf_threshold']

    def run_inference_on_video(self, sample_path):
        video_id = os.path.basename(sample_path)
        try:
            dataset = DroneVideoDataset(
                sample_path, 
                target_size=self.cfg['inference']['target_size'], 
                augment=False
            )
            
            ref_images = dataset.load_reference_images()
            if not ref_images:
                print(f"⚠️ Không tìm thấy ảnh tham chiếu (Query) tại: {sample_path}")
                return {"video_id": video_id, "detections": [{"bboxes": []}]}
                
            # Trích xuất đặc trưng ảnh tham chiếu (Query)
            pil_refs = [Image.fromarray(img) for img in ref_images]
            q_clip, q_dino = self.matcher.encode_queries(pil_refs)
            q_clip, q_dino = q_clip.to(self.device), q_dino.to(self.device)
            
            detections_log = []
            
            cap = cv2.VideoCapture(dataset.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
            cap.release()

            frame_gen = dataset.get_frames_generator(batch_size=self.cfg['inference']['batch_size'])
            pbar_frames = tqdm(total=total_frames, desc=f"🎥 Quét [{video_id}]", leave=False)
            
            total_detected_boxes = 0
            passed_matcher_boxes = 0
            sim_scores_debug = []

            for batch_indices, frames in frame_gen:
                frames_list = [np.ascontiguousarray(frame) for frame in frames]
                
                # Chạy YOLO detector trên batch frame
                results = self.detector(
                    frames_list, 
                    verbose=False, 
                    conf=self.conf_threshold
                )
                
                for b_idx, idx in enumerate(batch_indices):
                    frame_res = results[b_idx]
                    
                    if frame_res.boxes is None or len(frame_res.boxes) == 0:
                        continue
                        
                    boxes = frame_res.boxes.xyxy.cpu().numpy()
                    total_detected_boxes += len(boxes)
                    
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box)
                        
                        crop = frames_list[b_idx][y1:y2, x1:x2].copy()
                        if crop.size == 0:
                            continue
                            
                        # Trích xuất đặc trưng ROI của vùng cắt
                        pil_crop = Image.fromarray(crop)
                        roi_clip, roi_dino = self.matcher.extract_crop_features(pil_crop)
                        roi_clip, roi_dino = roi_clip.to(self.device), roi_dino.to(self.device)
                        
                        # Tính độ tương đồng Hybrid
                        similarity = self.matcher.match(roi_clip, roi_dino, q_clip, q_dino)
                        sim_scores_debug.append(similarity)
                        
                        # Kiểm tra qua ngưỡng so khớp
                        if similarity >= self.similarity_threshold:
                            passed_matcher_boxes += 1
                            detections_log.append({
                                "frame": int(idx),
                                "x1": int(x1),
                                "y1": int(y1),
                                "x2": int(x2),
                                "y2": int(y2)
                            })
                
                pbar_frames.update(len(batch_indices))
                
            pbar_frames.close()
            if sim_scores_debug:
                print(f"   ↳ [{video_id}] YOLO: {total_detected_boxes} boxes | Matcher pass (>={self.similarity_threshold}): {passed_matcher_boxes} | Min Sim: {min(sim_scores_debug):.3f}, Max Sim: {max(sim_scores_debug):.3f}")
            else:
                print(f"   ↳ [{video_id}] Không tìm thấy box nào từ YOLO.")
            
            return {
                "video_id": video_id,
                "detections": [{"bboxes": detections_log}]
            }
        except Exception as e:
            import traceback
            print(f"❌ Lỗi xảy ra tại video {video_id}: {str(e)}")
            traceback.print_exc()
            return {"video_id": video_id, "detections": [{"bboxes": []}]}

    def generate_submission(self):
        test_dir = self.cfg['paths']['test_data']
        output_list = []
        
        if not os.path.exists(test_dir):
            print(f"⚠️ Chưa có dữ liệu test tại {test_dir}. Vui lòng kiểm tra lại đường dẫn.")
            return
            
        print("🎯 Bắt đầu chạy tiến trình Inference trên tập Public Test...")
        test_samples = [d for d in sorted(os.listdir(test_dir)) if os.path.isdir(os.path.join(test_dir, d))]
        
        for sample_name in tqdm(test_samples, desc="📊 Tổng tiến độ Public Test"):
            sample_path = os.path.join(test_dir, sample_name)
            res = self.run_inference_on_video(sample_path)
            if res:
                output_list.append(res)
                
        output_json_path = self.cfg['paths']['output_json']
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_list, f, indent=2)
            
        print(f"\n✅ Đã xuất file submission thành công tại: `{output_json_path}`")

if __name__ == "__main__":
    pipeline = RescueInferencePipeline()
    pipeline.generate_submission()