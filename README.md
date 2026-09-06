# Báo Cáo Tiến Độ Nghiên Cứu - SurvivalBuddy

Nhánh này lưu trữ mã nguồn thử nghiệm và kết quả Benchmark cho 3 kiến trúc phát hiện đối tượng trên chuỗi video drone cứu hộ.

---

## Ma Trận Hiệu Năng So Sánh (Benchmark Matrix)

| Chỉ số Đo lường | YOLO26n (Baseline) | Faster R-CNN V2 | RT-DETR-L |
| :--- | :--- | :--- | :--- |
| **Kiến trúc cốt lõi** | Anchor-free CNN | Two-Stage (RPN + RoIAlign) | Hybrid Encoder + DETR Decoder |
| **Throughput (Tốc độ)** | **89.98 FPS** | 13.78 FPS | **30.94 FPS** (Chuẩn Real-time) |
| **Peak VRAM Tiêu thụ** | **0.15 GB** | 4.62 GB | **1.00 GB** |
| **Validation STIoU** | **0.4569** | **0.4517** | **0.4312** |
| - *Backpack_0* | 0.4768 | 0.0813 | 0.1420 |
| - *Backpack_1* | 0.5266 | **0.7269** | **0.6913** |
| - *Person1_0* | 0.3672 | **0.5470** | **0.4603** |
| **Precision (mAP50)** | - | - | **82.84% (0.5944)** |
| **Recall (mAP50-95)** | - | - | **54.46% (0.3108)** |

---

## Thực Nghiệm Zero-Shot Cross-View Retrieval (Ablation Study)

Đánh giá năng lực định vị mục tiêu cứu hộ qua không gian vector (Embedding Matching) giữa ảnh chụp mặt đất và không ảnh UAV trên tập kiểm thử `BlackBox_01` (5,776 frames).

| Tiêu chí | Pipeline 1: YOLO11l + SigLIP Tracker | Pipeline 2: YOLO11l + SAHI + Hybrid Ensemble |
| :--- | :--- | :--- |
| **Cơ chế phát hiện** | YOLO11l (Toàn cục / Global Resize) | YOLO11l + SAHI (Lưới cắt 512x512) |
| **Kiến trúc trích xuất** | SigLIP-Base (`patch16-224`) | Hybrid CLIP (`ViT-Large/14`) + DINOv2 (`base`) |
| **Phương thức đối khớp** | Cosine Similarity (Sigmoid Loss domain) | Weighted Score Blending (75% CLIP + 25% DINOv2) |
| **Số chiều Embedding** | 768-d | 1536-d (Multimodal & Spatial Concatenation) |
| **Ngưỡng lọc (Threshold)** | 0.55 | 0.50 |
| **Detections ghi nhận** | **762** | **128** |
| **Dải điểm Similarity** | 58.6% - 65.3% | 50.1% - 58.9% |
| **Đặc tính đầu ra** | **High Recall / High Noise**: Thất thoát chi tiết vật thể nhỏ do nén ảnh toàn cục; SigLIP nhạy màu dẫn đến nhiều False Positives. | **High Precision / Low Noise**: SAHI bảo toàn độ phân giải thực; DINOv2 đóng vai trò chốt chặn topo hình học nghiêm ngặt. |

---

## Cấu Trúc Mã Nguồn Bổ Sung
- `src/metrics.py`: Module tính toán chỉ số Spatio-Temporal IoU (STIoU) và lớp đo đạc tài nguyên phần cứng `HardwareProfiler`.
- `models/01_baseline_yolo26n/`: Pipeline huấn luyện và kết quả YOLO26n.
- `models/02_visdrone_faster_rcnn/`: Pipeline huấn luyện và kết quả Faster R-CNN V2.
- `models/03_rt_detr_drone/`: Pipeline huấn luyện và kết quả RT-DETR-L.
- `app/app_siglip_tracker.py`: Giao diện WebUI Gradio tích hợp luồng phát hiện YOLO11l, ByteTrack và SigLIP.
- `app/app_vit_baseline.py`: Giao diện WebUI tích hợp SAHI Slicing (512x512) kết hợp Ensemble Blending (CLIP + DINOv2).
- `notebooks/ablation_study_crossview.ipynb`: Phân tích thống kê chi tiết, biểu đồ phân phối và luận điểm nghiên cứu về Domain Gap trong bài toán Cross-View Retrieval.