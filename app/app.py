import os
import cv2
import torch
import gradio as gr
import numpy as np
import subprocess
from pathlib import Path
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

ROOT_DIR = Path("/workspace/SurvivalBuddy").resolve()
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ==============================================================================
# 1. KHỞI TẠO CÁC MODULE KIẾN TRÚC DS-ORS
# ==============================================================================
class DroneSmallObjectDetector:
    def __init__(self, model_path: str, confidence_threshold: float = 0.20, device: str = "cpu"):
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            device=device
        )

    def detect(self, frame: np.ndarray) -> list:
        result = get_sliced_prediction(
            frame,
            self.detection_model,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.25,
            overlap_width_ratio=0.25,
            postprocess_type="GREEDYNMM",
            postprocess_match_threshold=0.5,
            verbose=0
        )
        return [{'bbox': [int(b) for b in pred.bbox.to_xyxy()], 'score': float(pred.score.value)} for pred in result.object_prediction_list]

class CrossViewAdapter(nn.Module):
    def __init__(self, clip_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, clip_dim),
        )
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.alpha * self.adapter(x)

class MultiViewQueryEncoder(nn.Module):
    def __init__(self, clip_model_name: str = "openai/clip-vit-large-patch14"):
        super().__init__()
        self.clip = CLIPVisionModelWithProjection.from_pretrained(clip_model_name)
        self.processor = CLIPImageProcessor.from_pretrained(clip_model_name)
        self.cross_view_adapter = CrossViewAdapter(clip_dim=768)
        self.proj = nn.Linear(768, 512, bias=False)
        for p in self.clip.parameters():
            p.requires_grad = False

    def forward(self, ref_images: list) -> torch.Tensor:
        inputs = self.processor(images=ref_images, return_tensors="pt")
        inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}
        with torch.no_grad():
            clip_features = self.clip(**inputs).image_embeds
            adapted = self.cross_view_adapter(clip_features)
            proj = self.proj(adapted)
            f_q = F.normalize(proj.mean(dim=0, keepdim=True), dim=-1)
        return f_q

class CrossAttentionInstanceMatcher(nn.Module):
    def __init__(self, feature_dim: int = 512, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.roi_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.roi_encoder.eval()
        for p in self.roi_encoder.parameters():
            p.requires_grad = False
        self.roi_proj = nn.Linear(384, feature_dim)

    def extract_roi(self, crop_t: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            f = self.roi_proj(self.roi_encoder(crop_t.to(self.device)))
            return F.normalize(f, dim=-1)

    def match(self, f_roi: torch.Tensor, f_q: torch.Tensor) -> float:
        score = torch.sigmoid((f_roi @ f_q.T) / np.sqrt(512)).item()
        return score

# ==============================================================================
# 2. TẢI TRỌNG SỐ MÔ HÌNH VÀ PIPELINE
# ==============================================================================
best_w = ROOT_DIR / "models" / "04_yolo11l_drone" / "training_run_img1024" / "weights" / "best.pt"
if not best_w.exists():
    best_w = ROOT_DIR / "weights" / "best_ds_ors.pt"

print(f"Đang tải detector từ: {best_w}...")
detector = DroneSmallObjectDetector(model_path=str(best_w), confidence_threshold=0.15, device=DEVICE)
query_encoder = MultiViewQueryEncoder().to(DEVICE)
matcher = CrossAttentionInstanceMatcher(device=DEVICE).to(DEVICE)

def crop_tensor(frame, bbox):
    x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), bbox[2], bbox[3]
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb).resize((224, 224))
    t = torch.tensor(np.array(pil), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (t - mean) / std

# ==============================================================================
# 3. HÀM XỬ LÝ VIDEO VÀ CHUYỂN ĐỔI CODEC H.264
# ==============================================================================
def process_video(video_file, img1, img2, img3, match_threshold, progress=gr.Progress()):
    if video_file is None:
        return None, "Vui lòng tải lên Drone Video!"
    
    ref_imgs = [Image.fromarray(img).convert("RGB") for img in [img1, img2, img3] if img is not None]
    f_q = query_encoder(ref_imgs).to(DEVICE) if ref_imgs else None
    
    cap = cv2.VideoCapture(video_file)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    temp_raw_path = "/workspace/SurvivalBuddy/outputs/temp_raw_output.mp4"
    web_ready_path = "/workspace/SurvivalBuddy/outputs/demo_output_h264.mp4"
    Path(temp_raw_path).parent.mkdir(parents=True, exist_ok=True)
    
    out = cv2.VideoWriter(temp_raw_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    frame_idx = 0
    detected_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        proposals = detector.detect(frame)
        for p in proposals:
            score = p['score']
            if f_q is not None:
                ct = crop_tensor(frame, p['bbox'])
                if ct is not None:
                    f_roi = matcher.extract_roi(ct)
                    score = matcher.match(f_roi, f_q)
            
            if score >= match_threshold:
                x1, y1, x2, y2 = p['bbox']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Target: {score:.2f}", (x1, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                detected_count += 1

        out.write(frame)
        frame_idx += 1
        if frame_idx % 25 == 0:
            progress(frame_idx / total_frames, desc=f"Đang xử lý Frame {frame_idx}/{total_frames}")

    cap.release()
    out.release()

    # Chuyển đổi mã hóa sang H.264 (yuv420p) chuẩn trình duyệt Web HTML5
    cmd = f"ffmpeg -y -i {temp_raw_path} -vcodec libx264 -crf 23 -pix_fmt yuv420p {web_ready_path}"
    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if os.path.exists(temp_raw_path):
        os.remove(temp_raw_path)

    return web_ready_path, f"Xử lý hoàn tất {frame_idx} frames! Phát hiện {detected_count} mục tiêu."

# ==============================================================================
# 4. GIAO DIỆN WEB GRADIO
# ==============================================================================
with gr.Blocks(title="SurvivalBuddy DS-ORS Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛸 SurvivalBuddy: One-Shot Drone Search & Rescue UI")
    gr.Markdown("Hệ thống định vị mục tiêu không-thời gian góc nhìn Drone sử dụng DS-ORS (YOLO11l + CLIP + DINOv2)")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Tải lên 3 Ảnh tham chiếu mặt đất (Query Images)")
            ref1 = gr.Image(label="Ảnh góc 1", type="numpy")
            ref2 = gr.Image(label="Ảnh góc 2", type="numpy")
            ref3 = gr.Image(label="Ảnh góc 3", type="numpy")
            
            gr.Markdown("### 2. Thiết lập thông số")
            thresh = gr.Slider(0.1, 0.9, value=0.30, step=0.05, label="Match Confidence Threshold")
            
        with gr.Column(scale=2):
            gr.Markdown("### 3. Tải lên Drone Video")
            input_video = gr.Video(label="Drone Video (Input)")
            btn_run = gr.Button("🚀 BẮT ĐẦU TÌM KIẾM MỤC TIÊU", variant="primary")
            
            gr.Markdown("### 4. Kết quả định vị")
            output_video = gr.Video(label="Target Localization Output")
            status_text = gr.Textbox(label="Trạng thái thực thi")

    btn_run.click(
        fn=process_video,
        inputs=[input_video, ref1, ref2, ref3, thresh],
        outputs=[output_video, status_text]
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)