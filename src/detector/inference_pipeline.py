import os
import json
import yaml
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import clip
from tqdm import tqdm
from PIL import Image # import class Image từ lib PILLOW
from src.data_loader.video_reader import DroneVideoDataset

class RescueInferencePipeline:
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
            
        self.device = "cuda" if torch.cuda.is_available() and self.cfg['inference']['device'] == "cuda" else "cpu"
        print(f"Khởi chạy pipeline trên thiết bị: {self.device}")
        
        # Giai đoạn 1: Lọc thô vùng nghi ngờ tích hợp ByteTrack thông qua YOLO
        self.detector = YOLO(self.cfg['model']['detector_path'])
        
        # Giai đoạn 2: Zero-shot bằng OpenAI CLIP
        self.clip_model, self.clip_preprocess = clip.load(self.cfg['model']['clip_backbone'], device=self.device)
        
        # Bộ nhớ đệm không-thời gian (Temporal Architecture)
        self.embedding_cache = {}  # Lưu trữ CLIP embedding theo track_id: {track_id: tensor}
        self.score_cache = {}      # Lưu trữ điểm số mượt EMA theo track_id: {track_id: float_score}
        self.ema_alpha = 0.65      # Trọng số bộ lọc làm mịn thời gian (EMA - Exponential Moving Average)

    def extract_reference_embedding(self, ref_images):
        features = []
        with torch.no_grad():
            for img in ref_images:
                pil_img = Image.fromarray(img)
                img_input = self.clip_preprocess(pil_img).unsqueeze(0).to(self.device)
                feat = self.clip_model.encode_image(img_input)
                features.append(feat / feat.norm(dim=-1, keepdim=True))
                
        mean_feat = torch.cat(features, dim=0).mean(dim=0, keepdim=True)
        return mean_feat / mean_feat.norm(dim=-1, keepdim=True)

    def run_inference_on_video(self, sample_path):
        dataset = DroneVideoDataset(
            sample_path, 
            target_size=self.cfg['inference']['target_size'], 
            augment=False
        )
        
        ref_images = dataset.load_reference_images()
        if not ref_images:
            return []
            
        ref_embedding = self.extract_reference_embedding(ref_images)
        video_id = os.path.basename(sample_path)
        
        detections_log = []
        
        # Làm sạch bộ nhớ đệm cache khi chuyển sang video mới
        self.embedding_cache.clear()
        self.score_cache.clear()
        
        cap = cv2.VideoCapture(dataset.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
        cap.release()

        frame_gen = dataset.get_frames_generator(batch_size=self.cfg['inference']['batch_size'])
        pbar_frames = tqdm(total=total_frames, desc=f"Đang quét {video_id}", leave=False)
        
        for batch_indices, frames in frame_gen:
            frames_list = [np.ascontiguousarray(frame) for frame in frames]
            
            # Sử dụng .track() để gán ID không-thời gian (ByteTrack) thay vì .predict() thuần túy
            results = self.detector.track(
                frames_list, 
                verbose=False, 
                persist=True, 
                conf=self.cfg['inference']['conf_threshold']
            )
            
            for b_idx, idx in enumerate(batch_indices):
                frame_res = results[b_idx]
                
                # Kiểm tra xem có đối tượng nào được theo dõi và gán ID trong frame này không
                if frame_res.boxes is None or frame_res.boxes.id is None:
                    continue
                    
                boxes = frame_res.boxes.xyxy.cpu().numpy()
                track_ids = frame_res.boxes.id.cpu().numpy().astype(int)
                
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Cắt vùng ứng viên, thêm .copy() để giải quyết triệt để lỗi Tensor Reference gây bão hòa score
                    crop = frames_list[b_idx][y1:y2, x1:x2].copy()
                    if crop.size == 0:
                        continue
                        
                    # BƯỚC 2: Kiểm tra embedding cache để giải phóng CPU
                    if track_id in self.embedding_cache:
                        crop_embedding = self.embedding_cache[track_id]
                    else:
                        # Chỉ chạy trích xuất CLIP đúng 1 lần duy nhất khi phát hiện ID mới
                        pil_crop = Image.fromarray(crop)
                        crop_input = self.clip_preprocess(pil_crop).unsqueeze(0).to(self.device)
                        
                        with torch.no_grad():
                            feat = self.clip_model.encode_image(crop_input)
                            crop_embedding = feat / feat.norm(dim=-1, keepdim=True)
                            
                        # Lưu vào cache để tái sử dụng cho các frame sau
                        self.embedding_cache[track_id] = crop_embedding
                        
                    # Tính toán độ tương đồng Cosine Similarity 
                    raw_similarity = (crop_embedding @ ref_embedding.T).item()
                    
                    # BƯỚC 3: Áp dụng cơ chế làm mịn thời gian EMA (Temporal Smoothing)
                    if track_id in self.score_cache:
                        # Kết hợp điểm quá khứ và hiện tại theo trọng số alpha
                        smoothed_similarity = (self.ema_alpha * raw_similarity) + ((1 - self.ema_alpha) * self.score_cache[track_id])
                    else:
                        smoothed_similarity = raw_similarity
                        
                    # Cập nhật lại điểm số mượt vào bộ đếm thời gian
                    self.score_cache[track_id] = smoothed_similarity
                    
                    # Lọc qua ngưỡng toán học dựa trên điểm số đã được làm mịn
                    if smoothed_similarity >= self.cfg['inference']['similarity_threshold']:
                        detections_log.append({
                            "frame": idx,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2
                        })
            
            pbar_frames.update(len(batch_indices))
            
        pbar_frames.close()
        return {
            "video_id": video_id,
            "detections": [{"bboxes": detections_log}]
        }

    def generate_submission(self):
        test_dir = self.cfg['paths']['test_data']
        output_list = []
        
        if not os.path.exists(test_dir):
            print(f"Chưa có dữ liệu test tại {test_dir}. Hãy nạp dữ liệu vào trước.")
            return
            
        print("Bắt đầu chạy tiến trình xử lý tập Public Test...")
        test_samples = [d for d in sorted(os.listdir(test_dir)) if os.path.isdir(os.path.join(test_dir, d))]
        
        for sample_name in tqdm(test_samples, desc="Tổng tiến độ Public Test"):
            sample_path = os.path.join(test_dir, sample_name)
            res = self.run_inference_on_video(sample_path)
            if res:
                output_list.append(res)
                    
        with open(self.cfg['paths']['output_json'], 'w') as f:
            json.dump(output_list, f, indent=2)
        print(f"\nĐã xuất tệp tin thành công tại: {self.cfg['paths']['output_json']}")

if __name__ == "__main__":
    pipeline = RescueInferencePipeline()
    pipeline.generate_submission()