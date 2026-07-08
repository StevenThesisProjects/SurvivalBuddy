import os
import cv2
import yaml
import torch
import numpy as np
import gradio as gr
from PIL import Image
from ultralytics import YOLO
import clip

# Khởi tạo cấu hình và định vị thiết bị phần cứng
CONFIG_PATH = "configs/config.yaml"
with open(CONFIG_PATH, 'r') as f:
    cfg = yaml.safe_load(f)

device = "cuda" if torch.cuda.is_available() and cfg['inference']['device'] == "cuda" else "cpu"

# Tải các mô hình học sâu cốt lõi
detector = YOLO(cfg['model']['detector_path'])
clip_model, clip_preprocess = clip.load(cfg['model']['clip_backbone'], device=device)

# Khởi tạo Bộ nhớ đệm không-thời gian (Temporal Architecture) cho luồng Dashboard
embedding_cache = {}  # Lưu trữ CLIP embedding theo track_id: {track_id: tensor}
score_cache = {}      # Lưu trữ điểm số mượt EMA theo track_id: {track_id: float}
ema_alpha = 0.65      # Trọng số bộ lọc làm mịn thời gian (EMA - Exponential Moving Average)

def process_rescue_mission(video_path, ref_image, conf_thresh, sim_thresh):
    if video_path is None or ref_image is None:
        return None, "Vui lòng cung cấp đầy đủ Video luồng Drone và Ảnh tham chiếu đối tượng!"

    # Làm sạch bộ nhớ đệm cache trước mỗi chiến dịch quét mới
    embedding_cache.clear()
    score_cache.clear()

    # 1. Trích xuất đặc trưng (Embedding) của ảnh tham chiếu cứu hộ bằng CLIP
    pil_ref = Image.fromarray(ref_image.astype('uint8'), 'RGB')
    ref_input = clip_preprocess(pil_ref).unsqueeze(0).to(device)
    with torch.no_grad():
        ref_embedding = clip_model.encode_image(ref_input)
        ref_embedding = ref_embedding / ref_embedding.norm(dim=-1, keepdim=True)

    # 2. Mở luồng đọc và xử lý video bằng OpenCV 
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Sử dụng định dạng 'avc1' hoặc 'mp4v' chuẩn hiển thị trên Web HTML5 của Gradio
    output_video_path = "app/output_rescue_demo.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_idx = 0
    detected_count = 0
    log_messages = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Tạo một bản sao để vẽ, giữ nguyên frame gốc để trích xuất ngữ nghĩa
        annotated_frame = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Giai đoạn 1: Cho YOLO track trực tiếp trên frame gốc để tránh sai lệch scale tọa độ pixel
        results = detector.track(frame, verbose=False, persist=True, conf=conf_thresh)
        
        # Kiểm tra xem có đối tượng nào được định danh ID không
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                
                # Trích xuất vùng ảnh cắt từ ảnh RGB gốc
                crop = frame_rgb[y1:y2, x1:x2].copy()
                if crop.size == 0:
                    continue
                    
                # Giai đoạn 2: Kiểm tra bộ nhớ Cache để tái sử dụng Embedding cũ
                if track_id in embedding_cache:
                    crop_embedding = embedding_cache[track_id]
                else:
                    pil_crop = Image.fromarray(crop)
                    crop_input = clip_preprocess(pil_crop).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        feat = clip_model.encode_image(crop_input)
                        crop_embedding = feat / feat.norm(dim=-1, keepdim=True)
                    
                    embedding_cache[track_id] = crop_embedding
                
                # Tính toán độ tương đồng Cosine Similarity nguyên bản
                raw_similarity = (crop_embedding @ ref_embedding.T).item()
                
                # Áp dụng cơ chế làm mịn thời gian EMA
                if track_id in score_cache:
                    smoothed_similarity = (ema_alpha * raw_similarity) + ((1 - ema_alpha) * score_cache[track_id])
                else:
                    smoothed_similarity = raw_similarity
                    
                score_cache[track_id] = smoothed_similarity
                
                # Nếu vượt ngưỡng, tiến hành vẽ đè khung xanh lên bản sao annotated_frame
                if smoothed_similarity >= sim_thresh:
                    detected_count += 1
                    
                    if len(log_messages) < 15:
                        log_messages.append(f"[Frame {frame_idx}] ID {track_id} -> Khớp nối: {smoothed_similarity:.2f} | Tọa độ: ({x1}, {y1}) -> ({x2}, {y2})")
                    
                    # Vẽ trực tiếp lên khung hình đầu ra
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    label = f"ID {track_id}: {smoothed_similarity:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
        # Ghi khung hình đã được vẽ (annotated_frame) vào luồng video xuất bản
        out.write(annotated_frame)
        frame_idx += 1

    cap.release()
    out.release()
    
    status_msg = f"=== BÁO CÁO THỰC ĐỊA HỆ THỐNG ===\n"
    status_msg += f"• Trạng thái: Hoàn thành xử lý video chiến dịch.\n"
    status_msg += f"• Tổng số khung hình quét qua: {frame_idx}\n"
    status_msg += f"• Số vị trí không-thời gian khớp mục tiêu: {detected_count}\n\n"
    status_msg += "=== CHI TIẾT LOG ĐỊNH VỊ (TOP 15) ===\n"
    status_msg += "\n".join(log_messages) if log_messages else "Không phát hiện mục tiêu trùng khớp vượt ngưỡng toán học."
    
    return output_video_path, status_msg

# Xây dựng kiến trúc giao diện Web Dashboard bằng Gradio Blocks
with gr.Blocks(title="SurvivalBuddy - Drone Rescue AI Dashboard") as demo:
    gr.Markdown("""
    # SurvivalBuddy: Hệ Thống Phân Tầng Phát Hiện Vật Thể Cứu Hộ Từ Drone
    *Đồ án nghiên cứu & Phát triển hệ thống thị giác máy tính phục vụ tìm kiếm cứu nạn thiên tai.*
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Dữ Liệu Đầu Vào")
            input_video = gr.Video(label="Luồng Video ghi hình từ Drone (.mp4)")
            input_image = gr.Image(label="Ảnh vật thể mục tiêu cần tìm kiếm (Chụp ngang mặt đất)")
            
            with gr.Accordion("Tinh chỉnh Ngưỡng Toán Học Hệ Thống", open=True):
                conf_slider = gr.Slider(minimum=0.1, maximum=0.9, value=0.18, step=0.01, 
                                        label="YOLO Confidence Threshold (Lọc thô ứng viên)")
                sim_slider = gr.Slider(minimum=0.4, maximum=0.95, value=0.50, step=0.01, 
                                       label="CLIP Similarity Threshold (Ngưỡng khớp nối ảnh)")
                
            submit_btn = gr.Button("KÍCH HOẠT CHIẾN DỊCH TÌM KIẾM CỨU HỘ", variant="primary")
            
        with gr.Column():
            gr.Markdown("### Kết Quả Phân Tích Thực Địa")
            output_video = gr.Video(label="Luồng Video Kết Quả (Đã vẽ Bounding Box)")
            output_text = gr.Textbox(label="Trạng thái hệ thống & Log báo cáo định vị", lines=12, interactive=False)

    # Thiết lập sự kiện kích hoạt luồng xử lý
    submit_btn.click(
        fn=process_rescue_mission,
        inputs=[input_video, input_image, conf_slider, sim_slider],
        outputs=[output_video, output_text]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)