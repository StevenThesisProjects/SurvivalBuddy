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
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor, AutoModel, AutoImageProcessor
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

ROOT_DIR = Path("/workspace/SurvivalBuddy").resolve()
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

class CentroidTracker:
    def __init__(self, max_disappeared=8, max_distance=160):
        self.next_id = 1
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (x1, y1, x2, y2, conf) in enumerate(rects):
            input_centroids[i] = (int((x1 + x2) / 2.0), int((y1 + y2) / 2.0))

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], rects[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = [self.objects[obj_id]["centroid"] for obj_id in object_ids]
            
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id]["centroid"] = input_centroids[col]
                self.objects[object_id]["bbox"] = rects[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(D.shape[0])).difference(used_rows)
            unused_cols = set(range(input_centroids.shape[0])).difference(used_cols)

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(input_centroids[col], rects[col])

        return self.objects

    def register(self, centroid, rect):
        self.objects[self.next_id] = {"centroid": centroid, "bbox": rect}
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

class MotionDistanceFilter:
    def __init__(self, max_pixel_distance: float = 140.0):
        self.max_pixel_distance = max_pixel_distance
        self.last_positions = {}

    def is_valid(self, track_id: int, bbox: list) -> bool:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        if track_id not in self.last_positions:
            self.last_positions[track_id] = (cx, cy)
            return True

        prev_cx, prev_cy = self.last_positions[track_id]
        distance = np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)

        if distance > self.max_pixel_distance:
            return False

        self.last_positions[track_id] = (cx, cy)
        return True

    def reset(self):
        self.last_positions.clear()

class HybridZeroShotMatcher(nn.Module):
    def __init__(self, clip_model="openai/clip-vit-large-patch14", dino_model="facebook/dinov2-base", clip_weight=0.75):
        super().__init__()
        self.clip_weight = clip_weight
        self.dino_weight = 1.0 - clip_weight
        
        self.clip = CLIPVisionModelWithProjection.from_pretrained(clip_model)
        self.clip_processor = CLIPImageProcessor.from_pretrained(clip_model)
        
        self.dino = AutoModel.from_pretrained(dino_model)
        self.dino_processor = AutoImageProcessor.from_pretrained(dino_model)
        
        for p in self.clip.parameters():
            p.requires_grad = False
        for p in self.dino.parameters():
            p.requires_grad = False

    def encode_queries(self, ref_images: list):
        # CLIP Query Feature
        c_in = self.clip_processor(images=ref_images, return_tensors="pt")
        c_in = {k: v.to(next(self.parameters()).device) for k, v in c_in.items()}
        with torch.no_grad():
            c_feat = F.normalize(self.clip(**c_in).image_embeds, dim=-1)
            q_clip = F.normalize(c_feat.mean(dim=0, keepdim=True), dim=-1)

        # DINO Query Feature
        d_in = self.dino_processor(images=ref_images, return_tensors="pt")
        d_in = {k: v.to(next(self.parameters()).device) for k, v in d_in.items()}
        with torch.no_grad():
            d_out = self.dino(**d_in)
            d_feat = F.normalize(d_out.pooler_output, dim=-1)
            q_dino = F.normalize(d_feat.mean(dim=0, keepdim=True), dim=-1)

        return q_clip, q_dino

    def extract_crop_features(self, pil_crop: Image.Image):
        c_in = self.clip_processor(images=[pil_crop], return_tensors="pt")
        c_in = {k: v.to(next(self.parameters()).device) for k, v in c_in.items()}
        with torch.no_grad():
            roi_clip = F.normalize(self.clip(**c_in).image_embeds, dim=-1)

        d_in = self.dino_processor(images=[pil_crop], return_tensors="pt")
        d_in = {k: v.to(next(self.parameters()).device) for k, v in d_in.items()}
        with torch.no_grad():
            d_out = self.dino(**d_in)
            roi_dino = F.normalize(d_out.pooler_output, dim=-1)

        return roi_clip, roi_dino

    def match(self, roi_clip, roi_dino, q_clip, q_dino) -> float:
        sim_clip = torch.sum(roi_clip * q_clip, dim=-1).item()
        sim_dino = torch.sum(roi_dino * q_dino, dim=-1).item()
        
        sim_clip = max(0.0, sim_clip)
        sim_dino = max(0.0, sim_dino)
        
        # Weighted Ensemble score
        blended = (self.clip_weight * sim_clip) + (self.dino_weight * sim_dino)
        return max(0.0, min(1.0, blended))

best_w = ROOT_DIR / "models" / "04_yolo11l_drone" / "training_run_img1024" / "weights" / "best.pt"
if not best_w.exists():
    best_w = ROOT_DIR / "weights" / "best_ds_ors.pt"

print(f"Dang tai SAHI Detection Model tu: {best_w}...")
detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8", 
    model_path=str(best_w),
    confidence_threshold=0.15,
    device=DEVICE,
)

matcher = HybridZeroShotMatcher(clip_weight=0.75).to(DEVICE)
motion_filter = MotionDistanceFilter(max_pixel_distance=140.0)
tracker = CentroidTracker(max_disappeared=8, max_distance=160)

def process_video(video_file, img1, img2, img3, match_threshold, progress=gr.Progress()):
    if video_file is None:
        return None, None, "Vui long tai len Drone Video"
    
    ref_imgs = [Image.fromarray(img).convert("RGB") for img in [img1, img2, img3] if img is not None]
    if ref_imgs:
        q_clip, q_dino = matcher.encode_queries(ref_imgs)
        q_clip = q_clip.to(DEVICE)
        q_dino = q_dino.to(DEVICE)
    else:
        q_clip, q_dino = None, None
    
    cap = cv2.VideoCapture(video_file)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    temp_raw_path = "/workspace/SurvivalBuddy/outputs/temp_raw_output.mp4"
    web_ready_path = "/workspace/SurvivalBuddy/outputs/demo_output_h264.mp4"
    snapshot_dir = "/workspace/SurvivalBuddy/outputs/snapshots"
    
    Path(temp_raw_path).parent.mkdir(parents=True, exist_ok=True)
    Path(snapshot_dir).mkdir(parents=True, exist_ok=True)
    
    out = cv2.VideoWriter(temp_raw_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    motion_filter.reset()
    score_cache = {}
    crop_cache = {}
    captured_ids = set()
    snapshots_gallery = []
    
    frame_idx = 0
    detected_count = 0
    localization_logs = []
    ema_alpha = 0.65

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_seconds = int(frame_idx / fps)
        mins = current_seconds // 60
        secs = current_seconds % 60
        timestamp_str = f"{mins:02d}:{secs:02d}"

        # SAHI Tiled Inference
        sahi_result = get_sliced_prediction(
            frame,
            detection_model,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            verbose=False
        )

        rects = []
        for obj in sahi_result.object_prediction_list:
            x1, y1, x2, y2 = obj.bbox.minx, obj.bbox.miny, obj.bbox.maxx, obj.bbox.maxy
            conf = obj.score.value
            rects.append([int(x1), int(y1), int(x2), int(y2), float(conf)])

        tracked_objects = tracker.update(rects)

        for track_id, info in tracked_objects.items():
            bbox = info["bbox"]
            conf_val = bbox[4]
            bbox_coords = bbox[:4]
            
            if not motion_filter.is_valid(track_id, bbox_coords):
                continue

            raw_sim = 0.0
            if q_clip is not None:
                x1_c, y1_c, x2_c, y2_c = max(0, bbox_coords[0]), max(0, bbox_coords[1]), min(w, bbox_coords[2]), min(h, bbox_coords[3])
                crop_bgr = frame[y1_c:y2_c, x1_c:x2_c]
                
                if crop_bgr.size > 0:
                    if track_id in crop_cache and frame_idx % 3 != 0:
                        roi_c, roi_d = crop_cache[track_id]
                    else:
                        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                        pil_crop = Image.fromarray(crop_rgb)
                        roi_c, roi_d = matcher.extract_crop_features(pil_crop)
                        crop_cache[track_id] = (roi_c, roi_d)
                        
                    raw_sim = matcher.match(roi_c, roi_d, q_clip, q_dino)

            if track_id in score_cache:
                smoothed_sim = (ema_alpha * raw_sim) + ((1.0 - ema_alpha) * score_cache[track_id])
            else:
                smoothed_sim = raw_sim
            score_cache[track_id] = smoothed_sim

            if smoothed_sim >= match_threshold:
                x1, y1, x2, y2 = bbox_coords
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                
                det_pct = conf_val * 100.0
                match_pct = smoothed_sim * 100.0
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

                label_text1 = f"ID:{track_id} | Khop: {match_pct:.1f}%"
                label_text2 = f"Toa do: [{x1},{y1},{x2},{y2}]"
                
                (tw1, th1), _ = cv2.getTextSize(label_text1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                (tw2, th2), _ = cv2.getTextSize(label_text2, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                box_w = max(tw1, tw2) + 8
                box_h = th1 + th2 + 10

                bg_y1 = max(0, y1 - box_h - 4)
                bg_y2 = y1 - 2
                bg_x2 = min(w, x1 + box_w)
                cv2.rectangle(frame, (x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)

                cv2.putText(frame, label_text1, (x1 + 4, bg_y1 + th1 + 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, label_text2, (x1 + 4, bg_y2 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

                detected_count += 1
                
                if track_id not in captured_ids:
                    captured_ids.add(track_id)
                    pad = 40
                    px1, py1 = max(0, x1 - pad), max(0, y1 - pad)
                    px2, py2 = min(w, x2 + pad), min(h, y2 + pad)
                    snap_img = frame[py1:py2, px1:px2]
                    
                    snap_filename = f"{snapshot_dir}/track_{track_id}_{mins:02d}m{secs:02d}s.jpg"
                    cv2.imwrite(snap_filename, snap_img)
                    
                    caption = f"ID: {track_id} | Thoi gian: {timestamp_str} | Khop: {match_pct:.1f}%"
                    snapshots_gallery.append((snap_filename, caption))
                
                if len(localization_logs) < 50:
                    localization_logs.append(
                        f"Thoi gian {timestamp_str} | Frame {frame_idx:04d} | ID:{track_id:02d} | "
                        f"Do khop: {match_pct:5.1f}% | YOLO: {det_pct:5.1f}% | "
                        f"x1:{x1:4d}, y1:{y1:4d}, x2:{x2:4d}, y2:{y2:4d} | Tam:({cx:4d}, {cy:4d})"
                    )

        out.write(frame)
        frame_idx += 1
        if frame_idx % 25 == 0:
            progress(frame_idx / total_frames, desc=f"Dang xu ly Frame {frame_idx}/{total_frames}")

    cap.release()
    out.release()

    cmd = (
        f"ffmpeg -y -i {temp_raw_path} "
        f"-vcodec libx264 -preset veryfast -b:v 1500k -crf 28 -pix_fmt yuv420p "
        f"-movflags +faststart {web_ready_path}"
    )
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        if os.path.exists(temp_raw_path):
            web_ready_path = temp_raw_path
    else:
        if os.path.exists(temp_raw_path):
            os.remove(temp_raw_path)

    report_text = f"Xu ly hoan tat {frame_idx} frames | Phat hien {detected_count} muc tieu hop le.\n\n"
    report_text += "CHI TIET TOA DO VA THOI GIAN PHAT HIEN (Mau 50 detections dau tien):\n"
    report_text += "=" * 100 + "\n"
    report_text += "\n".join(localization_logs) if localization_logs else "Khong tim thay muc tieu vuot nguong."

    return web_ready_path, snapshots_gallery, report_text

with gr.Blocks(title="SurvivalBuddy DS-ORS Demo") as demo:
    gr.Markdown("# SurvivalBuddy: Drone Rescue System (SAHI + Ensemble CLIP-DINOv2)")
    gr.Markdown("He thong dinh vi muc tieu khong-thoi gian goc nhin Drone ket hop YOLO11l, SAHI, CLIP va DINOv2")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Tai len 3 Anh tham chieu mat dat (Query Images)")
            ref1 = gr.Image(label="Anh goc 1", type="numpy")
            ref2 = gr.Image(label="Anh goc 2", type="numpy")
            ref3 = gr.Image(label="Anh goc 3", type="numpy")
            
            gr.Markdown("### 2. Thiet lap thong so")
            thresh = gr.Slider(0.1, 0.9, value=0.50, step=0.05, label="Match Confidence Threshold")
            
            btn_run = gr.Button("BAT DAU TIM KIEM MUC TIEU", variant="primary")
            
        with gr.Column(scale=2):
            gr.Markdown("### 3. Tai len Drone Video")
            input_video = gr.Video(label="Drone Video (Input)")
            
            gr.Markdown("### 4. Ket qua dinh vi")
            output_video = gr.Video(label="Target Localization Output")
            
            gr.Markdown("### 5. Hinh anh vat the chup tu dong")
            output_gallery = gr.Gallery(label="Anh chup muc tieu (Kem thoi gian)", columns=4, height="auto")
            
            status_text = gr.Textbox(label="Bao cao Toa do va Trang thai Dinh vi", lines=12)

    btn_run.click(
        fn=process_video,
        inputs=[input_video, ref1, ref2, ref3, thresh],
        outputs=[output_video, output_gallery, status_text]
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)