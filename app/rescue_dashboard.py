import os
import cv2
import yaml
import torch
import numpy as np
import gradio as gr
from PIL import Image
from ultralytics import YOLO
import clip

# Khởi tạo cấu hình và load model
CONFIG_PATH = "configs/config.yaml"
with open(CONFIG_PATH, 'r') as f:
    cfg = yaml.safe_load(f)

device = "cuda" if torch.cuda.is_available() and cfg['inference']['device'] == "cuda" else "cpu"

# Tải các mô hình học sâu
detector = YOLO(cfg['model']['detector_path'])
clip_model, clip_preprocess = clip.load(cfg['model']['clip_backbone'], device=device)

def process_rescue_mission(video_path, ref_image, conf_thresh, sim_thresh):
    if video_path is None or ref_image is None:
        return None, "Vui lòng cung cấp đầy đủ Video luồng Drone và Ảnh tham chiếu đối tượng!"

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
    
    # Định nghĩa luồng ghi video đầu ra
    output_video_path = "app/output_rescue_demo.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_idx = 0
    detected_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Sao lưu frame gốc định dạng RGB để đưa vào mô hình tính toán
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize tạm thời về kích thước đích để YOLO quét thô vùng nghi ngờ
        target_size = cfg['inference']['target_size']
        frame_resized = cv2.resize(frame_rgb, (target_size, target_size))
        
        # Giai đoạn 1: YOLO quét tìm các bounding box ứng viên tiềm năng
        results = detector(frame_resized, verbose=False, conf=conf_thresh)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        # Tính tỷ lệ scale ngược tọa độ về kích thước video gốc
        scale_x = width / target_size
        scale_y = height / target_size

        for box in boxes:
            # Tọa độ pixel trên ảnh resized
            rx1, ry1, rx2, ry2 = map(int, box)
            
            # Khôi phục về tọa độ gốc pixel tuyệt đối trên video nguyên bản
            x1, y1 = int(rx1 * scale_x), int(ry1 * scale_y)
            x2, y2 = int(rx2 * scale_x), int(ry2 * scale_y)
            
            crop = frame_rgb[y1:y2, x1:x2]
            if crop.size == 0:
                continue
                
            # Giai đoạn 2: Cắt vùng nghi ngờ và đẩy qua CLIP để tính Cosine Similarity
            pil_crop = Image.fromarray(crop)
            crop_input = clip_preprocess(pil_crop).unsqueeze(0).to(device)
            
            with torch.no_grad():
                crop_embedding = clip_model.encode_image(crop_input)
                crop_embedding = crop_embedding / crop_embedding.norm(dim=-1, keepdim=True)
                similarity = (crop_embedding @ ref_embedding.T).item()
            
            # Nếu vượt ngưỡng tương đồng hình ảnh, thực hiện vẽ khung bao cứu hộ lên video
            if similarity >= sim_thresh:
                detected_count += 1
                # Vẽ khung bao hình chữ nhật màu xanh lá (định dạng BGR cho OpenCV)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                # Ghi nhãn text độ tương đồng toán học
                label = f"Target: {similarity:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
        # Ghi khung hình đã xử lý vào luồng video đầu ra
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    
    status_msg = f"Hoàn thành xử lý video! Tổng số frame quét qua: {frame_idx}. Phát hiện vật thể mục tiêu trùng khớp tại {detected_count} vị trí không-thời gian."
    return output_video_path, status_msg

# Xây dựng kiến trúc giao diện Web Dashboard bằng Gradio Blocks
with gr.Blocks(title="SurvivalBuddy - Drone Rescue AI Dashboard") as demo:
    gr.Markdown("""
    # SurvivalBuddy: Hệ Thống Phân Tầng Phát Hiện Vật Thể Cứu Hộ Từ Drone
    *Đồ án nghiên cứu & Phát triển hệ thống thị giác máy tính phục vụ tìm kiếm cứu nạn thiên tai.*
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("Dữ Liệu Đầu Vào")
            input_video = gr.Video(label="Luồng Video ghi hình từ Drone (.mp4)")
            input_image = gr.Image(label="Ảnh vật thể mục tiêu cần tìm kiếm (Chụp ngang mặt đất)")
            
            with gr.Accordion("Tinh chỉnh Ngưỡng Toán Học Hệ Thống", open=False):
                conf_slider = gr.Slider(minimum=0.1, maximum=0.9, value=0.25, step=0.05, 
                                        label="YOLO Confidence Threshold (Lọc thô ứng viên)")
                sim_slider = gr.Slider(minimum=0.4, maximum=0.95, value=0.65, step=0.05, 
                                       label="CLIP Similarity Threshold (Ngưỡng khớp nối ảnh)")
                
            submit_btn = gr.Button("KÍCH HOẠT CHIẾN DỊCH TÌM KIẾM CỨU HỘ", variant="primary")
            
        with gr.Column():
            gr.Markdown("Kết Quả Phân Tích Thực Địa")
            output_video = gr.Video(label="Luồng Video Kết Quả (Đã vẽ Bounding Box)")
            output_text = gr.Textbox(label="Trạng thái hệ thống & Log báo cáo định vị", interactive=False)

    # Thiết lập sự kiện kích hoạt luồng xử lý
    submit_btn.click(
        fn=process_rescue_mission,
        inputs=[input_video, input_image, conf_slider, sim_slider],
        outputs=[output_video, output_text]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)