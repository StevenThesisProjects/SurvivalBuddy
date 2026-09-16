import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoImageProcessor, CLIPModel, CLIPProcessor

class NormalizedMotionFilter:
    def __init__(self, max_norm_velocity: float = 0.15):
        self.max_velocity = max_norm_velocity
        self.last_records = {}

    def is_valid(self, track_id: int, bbox: list, frame_w: int, frame_h: int, timestamp: float) -> bool:
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        diag = np.sqrt(frame_w**2 + frame_h**2)
        
        if track_id not in self.last_records:
            self.last_records[track_id] = (cx, cy, timestamp)
            return True
            
        prev_cx, prev_cy, prev_t = self.last_records[track_id]
        dt = max(1e-3, timestamp - prev_t)
        dist_px = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
        norm_v = (dist_px / diag) / dt
        
        if norm_v > self.max_velocity:
            return False
            
        self.last_records[track_id] = (cx, cy, timestamp)
        return True

    def reset(self):
        self.last_records.clear()


class ZeroShotHybridMatcher(nn.Module):
    """
    So khớp Zero-Shot kết hợp đa không gian đặc trưng (CLIP Ngữ nghĩa + DINOv2 Cấu trúc cục bộ).
    Đảm bảo pooling an toàn tuyệt đối tránh lỗi kích thước 257 token.
    """
    def __init__(
        self, 
        clip_model_name: str = "openai/clip-vit-large-patch14", 
        dino_model_name: str = "facebook/dinov2-base",
        clip_weight: float = 0.75, 
        dino_weight: float = 0.25, 
        ema_alpha: float = 0.65, 
        max_norm_velocity: float = 0.15
    ):
        super().__init__()
        self.w_clip = clip_weight
        self.w_dino = dino_weight
        self.ema_alpha = ema_alpha
        self.score_history = {}
        
        print(f"Đang tải CLIP model: {clip_model_name}...")
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        
        print(f"Đang tải DINOv2 model: {dino_model_name}...")
        self.dino_model = AutoModel.from_pretrained(dino_model_name)
        self.dino_processor = AutoImageProcessor.from_pretrained(dino_model_name)
        
        for param in self.clip_model.parameters():
            param.requires_grad = False
        for param in self.dino_model.parameters():
            param.requires_grad = False

        self.motion_filter = NormalizedMotionFilter(max_norm_velocity=max_norm_velocity)

    def _extract_clip_features(self, pixel_values):
        # Ưu tiên sử dụng get_image_features chuẩn của CLIPModel
        clip_out = self.clip_model.get_image_features(pixel_values=pixel_values)
        if isinstance(clip_out, torch.Tensor):
            feat = clip_out
        elif hasattr(clip_out, 'image_embeds') and clip_out.image_embeds is not None:
            feat = clip_out.image_embeds
        else:
            feat = clip_out[0]
            
        # Nếu vô tình là tensor 3 chiều (seq_len), ép về 2 chiều bằng mean pooling hoặc lấy token đầu
        if feat.ndim > 2:
            feat = feat[:, 0] if feat.shape[1] > 1 else feat.squeeze(1)
        return F.normalize(feat, dim=-1)

    def _extract_dino_features(self, pixel_values):
        outputs = self.dino_model(pixel_values=pixel_values)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            feat = outputs.pooler_output
        elif hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
            # Lấy [CLS] token (index 0) của DINOv2
            feat = outputs.last_hidden_state[:, 0]
        else:
            feat = outputs[0][:, 0]
            
        if feat.ndim > 2:
            feat = feat.mean(dim=1)
        return F.normalize(feat, dim=-1)

    def encode_queries(self, ref_images: list):
        device = next(self.parameters()).device
        clip_feats = []
        dino_feats = []

        for img in ref_images:
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            
            clip_inputs = self.clip_processor(images=img, return_tensors="pt").to(device)
            c_feat = self._extract_clip_features(clip_inputs['pixel_values'])
            clip_feats.append(c_feat)

            dino_inputs = self.dino_processor(images=img, return_tensors="pt").to(device)
            d_feat = self._extract_dino_features(dino_inputs['pixel_values'])
            dino_feats.append(d_feat)

        q_clip = torch.cat(clip_feats, dim=0).mean(dim=0, keepdim=True)
        q_dino = torch.cat(dino_feats, dim=0).mean(dim=0, keepdim=True)

        return F.normalize(q_clip, dim=-1), F.normalize(q_dino, dim=-1)

    def extract_crop_features(self, crop_image):
        device = next(self.parameters()).device
        if not isinstance(crop_image, Image.Image):
            crop_image = Image.fromarray(crop_image)

        clip_inputs = self.clip_processor(images=crop_image, return_tensors="pt").to(device)
        c_feat = self._extract_clip_features(clip_inputs['pixel_values'])

        dino_inputs = self.dino_processor(images=crop_image, return_tensors="pt").to(device)
        d_feat = self._extract_dino_features(dino_inputs['pixel_values'])

        return c_feat, d_feat

    def match(self, roi_clip, roi_dino, query_clip, query_dino) -> float:
        sim_clip = F.cosine_similarity(roi_clip, query_clip, dim=-1).item()
        sim_dino = F.cosine_similarity(roi_dino, query_dino, dim=-1).item()
        
        sim_clip_norm = max(0.0, sim_clip)
        sim_dino_norm = max(0.0, sim_dino)
        
        s_raw = (self.w_clip * sim_clip_norm) + (self.w_dino * sim_dino_norm)
        return s_raw
        
    def validate_trajectory(self, track_id: int, bbox: list, frame_w: int, frame_h: int, timestamp: float) -> bool:
        return self.motion_filter.is_valid(track_id, bbox, frame_w, frame_h, timestamp)

    def reset_tracking(self):
        self.motion_filter.reset()
        self.score_history.clear()