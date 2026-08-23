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

## Cấu Trúc Mã Nguồn Bổ Sung
- `src/metrics.py`: Module tính toán chỉ số Spatio-Temporal IoU (STIoU) và lớp đo đạc tài nguyên phần cứng `HardwareProfiler`.
- `models/01_baseline_yolo26n/`: Pipeline huấn luyện và kết quả YOLO26n.
- `models/02_visdrone_faster_rcnn/`: Pipeline huấn luyện và kết quả Faster R-CNN V2.
- `models/03_rt_detr_drone/`: Pipeline huấn luyện và kết quả RT-DETR-L.
