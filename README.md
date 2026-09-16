# 🚁 Báo Cáo Tiến Độ Nghiên Cứu - SurvivalBuddy: Drone-Based Emergency Rescue Pipeline

Nhánh này lưu trữ mã nguồn thử nghiệm, kết quả Benchmark và hệ thống suy luận thời gian thực cho bài toán phát hiện và định vị đối tượng cứu hộ trên chuỗi video drone.

---

## 📊 1. Ma Trận Hiệu Năng So Sánh (Benchmark Matrix)

Đánh giá hiệu năng tổng thể của 4 kiến trúc phát hiện đối tượng trên tập dữ liệu chuẩn của bài toán cứu hộ:

| Model Name | Architecture | Detector mAP@0.50 | Detector Recall | End-to-End STIoU | Peak VRAM | Latency (ms) | Throughput (FPS) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLO26-Nano (Baseline)** | One-Stage CNN | 0.3494 | 0.2961 | 0.4569 | **0.15 GB** | **11.11** | **90.0** |
| **Faster R-CNN (ResNet50)** | Two-Stage Anchor | 0.4120 | 0.3850 | 0.4517 | 4.62 GB | 72.57 | 13.8 |
| **RT-DETR Large** | Real-Time Transformer | **0.5944** | **0.5446** | 0.4312 | 1.00 GB | 32.32 | 30.9 |
| **YOLO11-Large (Global)** | One-Stage CNN | 0.4669 | 0.4441 | 0.4620 | 0.85 GB | **9.31** | **107.4** |
| **YOLO11-Large + SAHI (DS-ORS)** | Sliced Inference | **0.5812** | **0.5280** | **0.5340** | 1.45 GB | 38.20 | 26.2 |

**Phân tích sự đánh đổi (Trade-off Analysis):**
Nhìn vào bảng số liệu, mặc dù RT-DETR Large có độ chính xác mAP thuần túy nhỉnh hơn đôi chút so với luồng YOLO11l + SAHI (0.5944 so với 0.5812), nhưng nó lại thất bại trong việc duy trì bám vết dài hạn khiến chỉ số cuối cùng `End-to-End STIoU` tuột xuống mức thấp (0.4312). 
Ngược lại, kiến trúc **YOLO11-Large + SAHI (DS-ORS)** đã chứng minh được tính ưu việt khi xử lý đối tượng nhỏ xíu bị che khuất nhờ cơ chế cắt lưới, đem lại điểm tổng thể STIoU cao nhất (0.5340). Sự đánh đổi giảm thông lượng (từ 107.4 FPS xuống 26.2 FPS) hoàn toàn xứng đáng vì hệ thống vẫn bảo đảm vượt qua ngưỡng thời gian thực tiêu chuẩn (>25 FPS) để phục vụ công tác cứu hộ khẩn cấp.

---

## 📈 2. Kết Quả Thực Thi Thực Tế (Public Test Inference Results)

Hệ thống suy luận sử dụng mô hình tối ưu kết hợp detector custom (`weights/best_ds_ors.pt`) và Zero-Shot Hybrid Matcher (CLIP + DINOv2) đã hoàn thành quét toàn bộ tập Public Test với tổng cộng **2,951 bounding box** được ghi nhận trên 6 video mẫu.

![Thống kê phân phối kết quả](outputs/detection_distribution.png)

### Bảng Thống Kê Chi Tiết Theo Sample:
| Video ID / Sample | Số lượng Bounding Box phát hiện |
| :--- | :---: |
| `BlackBox_0` | 173 detections |
| `BlackBox_1` | 446 detections |
| `CardboardBox_0` | 155 detections |
| `CardboardBox_1` | 87 detections |
| `LifeJacket_0` | 574 detections |
| `LifeJacket_1` | 1516 detections |

* **Tổng số video đã xử lý:** 6
* **Tổng số bounding box mục tiêu phát hiện:** 2,951

---

## 🔬 3. Thực Nghiệm Zero-Shot Cross-View Retrieval (Ablation Study)

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

## 📂 4. Cấu Trúc Mã Nguồn Dự Án
- `src/metrics.py`: Module tính toán chỉ số Spatio-Temporal IoU (STIoU) và lớp đo đạc tài nguyên phần cứng `HardwareProfiler`.
- `src/matcher.py`: Mô-đun Zero-Shot Hybrid Matcher (CLIP + DINOv2 chuẩn CLS Token).
- `src/detector/inference_pipeline.py`: Pipeline suy luận chính tích hợp CUDA và xử lý batch.
- `models/01_baseline_yolo26n/`: Pipeline huấn luyện và kết quả YOLO26n.
- `models/02_visdrone_faster_rcnn/`: Pipeline huấn luyện và kết quả Faster R-CNN V2.
- `models/03_rt_detr_drone/`: Pipeline huấn luyện và kết quả RT-DETR-L.
- `models/04_yolo11l_drone/`: Pipeline huấn luyện YOLO11l - Detector Core chịu tải chính trong mạng Ensemble CLIP-DINOv2.
- `app/app_siglip_tracker.py`: Giao diện WebUI Gradio tích hợp luồng phát hiện YOLO11l, ByteTrack và SigLIP.
- `app/app_vit_baseline.py`: Giao diện WebUI tích hợp SAHI Slicing (512x512) kết hợp Ensemble Blending (CLIP + DINOv2).
- `notebooks/ablation_study_crossview.ipynb`: Phân tích thống kê chi tiết, biểu đồ phân phối và luận điểm nghiên cứu về Domain Gap trong bài toán Cross-View Retrieval.