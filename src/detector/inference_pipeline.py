import os
import json
import yaml
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import clip
from tqdm import tqdm
from src.data_loader.video_reader import DroneVideoDataset

class RescueInferencePipeline:
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
            
        self.device = "cuda" if torch.cuda.is_available() and self.cfg['inference']['device'] == "cuda" else "cpu"
        print(f"Khởi chạy pipeline trên thiết bị: {self.device}")
        
        # Lọc thô vùng nghi ngờ bằng YOLO
        self.detector = YOLO(self.cfg['model']['detector_path'])
        
        # Zero-shot bằng OpenAI CLIP
        self.clip_model, self.clip_preprocess = clip.load(self.cfg['model']['clip_backbone'], device=self.device)
        
    def extract_reference_embedding(self, ref_images):
        features = []
        with torch.no_grad():
            for img in ref_images:
                from PIL import Image
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
        
        # Đọc tổng số frame thực tế của video 
        cap = cv2.VideoCapture(dataset.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
        cap.release()

        frame_gen = dataset.get_frames_generator(batch_size=self.cfg['inference']['batch_size'])
        
        # Thanh tiến trình tqdm cho từng khung trong video
        pbar_frames = tqdm(total=total_frames, desc=f" Đang quét {video_id}", leave=False)
        
        for batch_indices, frames in frame_gen:
            # CHuyển batch thành danh sách mảng NumPy 
            frames_list = [np.ascontiguousarray(frame) for frame in frames]
            
            results = self.detector(frames_list, verbose=False, conf=self.cfg['inference']['conf_threshold'])
            
            for b_idx, idx in enumerate(batch_indices):
                frame_res = results[b_idx]
                boxes = frame_res.boxes.xyxy.cpu().numpy()
                
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    crop = frames_list[b_idx][y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                        
                    from PIL import Image
                    pil_crop = Image.fromarray(crop)
                    crop_input = self.clip_preprocess(pil_crop).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        crop_embedding = self.clip_model.encode_image(crop_input)
                        crop_embedding = crop_embedding / crop_embedding.norm(dim=-1, keepdim=True)
                        similarity = (crop_embedding @ ref_embedding.T).item()
                        
                    if similarity >= self.cfg['inference']['similarity_threshold']:
                        detections_log.append({
                            "frame": idx,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2
                        })
            
            # Cập nhật số lượng frames 
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
        
        # Thanh tiến trình tqdm  
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