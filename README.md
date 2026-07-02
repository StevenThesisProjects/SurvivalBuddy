# 🚁 SurvivalBuddy: Hệ Thống Phân Tầng Phát Hiện Vật Thể Cứu Hộ Từ Drone

**SurvivalBuddy** là một hệ thống thị giác máy tính phân tầng (*Cascaded Computer Vision Pipeline*) được thiết kế nhằm phát hiện các vật thể cứu hộ có kích thước rất nhỏ (balo, áo phao, hộp cứu thương, vật dụng sinh tồn...) từ luồng video ghi hình bởi drone, chỉ dựa trên một số lượng hạn chế ảnh tham chiếu chụp từ mặt đất. Hệ thống kết hợp giữa các mô hình học sâu hiện đại như **YOLO** và **OpenAI CLIP** cùng nhiều kỹ thuật tối ưu hóa bộ nhớ và xử lý không – thời gian, hướng đến khả năng triển khai trong các nhiệm vụ tìm kiếm cứu nạn (Search and Rescue - SAR) ngoài thực địa.

---

# 🏗️ Kiến Trúc Hệ Thống (Cascaded Pipeline)

Toàn bộ hệ thống được xây dựng theo kiến trúc xử lý hai tầng nhằm cân bằng giữa tốc độ và độ chính xác.

### Giai đoạn 1 – Agnostic Proposal Generation (YOLO)

Mô hình **YOLO** thực hiện quét toàn bộ khung hình video với mục tiêu phát hiện nhanh tất cả các vùng có khả năng chứa vật thể. Ở giai đoạn này hệ thống không quan tâm vật thể thuộc lớp nào mà chỉ tạo ra các **Object Proposals**, giúp giảm đáng kể khối lượng dữ liệu cần xử lý ở bước tiếp theo.

### Giai đoạn 2 – Zero-shot Feature Matching (OpenAI CLIP)

Các vùng ảnh được cắt (Image Crops) từ YOLO sẽ được đưa vào **OpenAI CLIP** để trích xuất vector đặc trưng (*Embedding*). Sau đó hệ thống tính toán **Cosine Similarity** giữa embedding của crop và embedding của ảnh tham chiếu mặt đất nhằm xác định xem vùng ảnh đó có thực sự là mục tiêu cứu hộ hay không.

Kiến trúc phân tầng này giúp hệ thống vừa duy trì tốc độ xử lý cao của YOLO, vừa tận dụng khả năng nhận dạng Zero-shot mạnh mẽ của CLIP đối với những vật thể chưa từng xuất hiện trong tập huấn luyện.

---

# 📁 Cấu Trúc Thư Mục Dự Án

```text
SurvivalBuddy/
├── .github/
│   ├── ISSUE_TEMPLATE/                 # Template báo lỗi và đề xuất tính năng
│   └── pull_request_template.md
│
├── app/
│   └── rescue_dashboard.py             # Dashboard trực quan sử dụng Gradio
│
├── configs/
│   └── config.yaml                     # Toàn bộ cấu hình hệ thống
│
├── data/
│   ├── public_test/                    # Bộ dữ liệu Public Test
│   └── training/                       # Dữ liệu huấn luyện và validation
│
├── src/
│   ├── data_loader/
│   │   └── video_reader.py             # OpenCV Generator tối ưu RAM
│   │
│   ├── detector/
│   │   └── inference_pipeline.py       # Pipeline YOLO + CLIP
│   │
│   ├── evaluation/
│   │   └── stiou_metric.py             # Đánh giá STIoU
│   │
│   ├── matcher/
│   │   └── ...                         # SIFT / SuperGlue Matching
│   │
│   └── tracker/
│       └── object_tracker.py           # Theo dõi vật thể theo thời gian
│
├── requirements.txt
└── README.md
```

---

# 🛠️ Cài Đặt Môi Trường

Cài đặt toàn bộ thư viện cần thiết bằng các lệnh sau:

```bash
# Cài đặt các thư viện của dự án
pip install -r requirements.txt

# OpenCV Headless dành cho môi trường Cloud/Server
pip install opencv-python-headless --force-reinstall

# Cài đặt OpenAI CLIP
pip install git+https://github.com/openai/CLIP.git
```

---

# 🚀 Hướng Dẫn Sử Dụng

## 1. Chạy Pipeline Suy Luận (Inference)

Trước khi chạy, hãy kiểm tra các tham số trong:

```text
configs/config.yaml
```

Bao gồm:

- Đường dẫn tập Train/Test
- Đường dẫn ảnh tham chiếu
- Kích thước ảnh
- Ngưỡng Cosine Similarity
- Các siêu tham số của hệ thống

Sau đó thực thi:

```bash
python3 -m src.detector.inference_pipeline
```

Pipeline sử dụng **OpenCV Generator** kết hợp với **tqdm**, giúp đọc video theo từng frame thay vì tải toàn bộ video vào RAM. Nhờ đó bộ nhớ luôn được duy trì ở mức khoảng **1.5 GB**, phù hợp với các môi trường Cloud hoặc GitHub Codespaces.

---

## 2. Đánh Giá Kết Quả (Local Validation)

Hệ thống hỗ trợ đánh giá bằng chỉ số **Spatio-Temporal IoU (STIoU)**.

### Đánh giá trên tập Training

```bash
python3 src/evaluation/stiou_metric.py \
data/training/train/annotations/annotations.json \
data/training/train_submission.json
```

### Đánh giá trên Public Test

```bash
python3 src/evaluation/stiou_metric.py \
<ground_truth.json> \
data/public_test/submission.json
```

---

## 3. Khởi Động Dashboard

Để mở giao diện trực quan:

```bash
python3 app/rescue_dashboard.py
```

Dashboard Gradio cho phép:

- Upload video drone
- Upload ảnh tham chiếu
- Điều chỉnh ngưỡng Cosine Similarity
- Hiển thị Bounding Box theo thời gian thực
- Quan sát trực tiếp kết quả phát hiện

Mặc định hệ thống chạy tại:

```text
http://localhost:7860
```

Nếu sử dụng **GitHub Codespaces**, chỉ cần mở tab **PORTS** hoặc chọn **Open in Browser** để truy cập giao diện.

---

# 📊 Kết Quả Thử Nghiệm

## Môi Trường Thực Thi

- **Platform:** GitHub Codespaces (CPU-only)
- **RAM sử dụng:** ~1.5 GB
- **Mức sử dụng bộ nhớ:** khoảng 19.6% tổng tài nguyên
- **Không xảy ra lỗi Out-of-Memory**
- **Tốc độ xử lý trung bình:** khoảng **7.9 FPS**

---

## Đánh Giá Pipeline

### Giai đoạn 1 – YOLO11 Nano

- Phát hiện nhanh các vùng ứng viên
- Xử lý tốt các vật thể kích thước nhỏ
- Tạo Proposal với chi phí tính toán thấp

### Giai đoạn 2 – OpenAI CLIP

- Thực hiện nhận dạng Zero-shot
- So khớp đặc trưng bằng Cosine Similarity
- Chịu được sự thay đổi lớn về góc nhìn giữa:
  - Ảnh tham chiếu chụp ngang từ mặt đất
  - Video drone quan sát từ trên cao
- Không cần huấn luyện lại khi bổ sung mục tiêu mới

---

# 🎯 Mục Tiêu Của Dự Án

SurvivalBuddy hướng đến việc xây dựng một hệ thống hỗ trợ tìm kiếm cứu nạn có khả năng:

- Phát hiện vật thể cứu hộ kích thước nhỏ trong video drone.
- Hoạt động với số lượng ảnh tham chiếu rất hạn chế.
- Hỗ trợ nhận dạng Zero-shot mà không cần huấn luyện lại mô hình.
- Tối ưu bộ nhớ để triển khai trên các môi trường điện toán đám mây hoặc thiết bị có tài nguyên giới hạn.
- Làm nền tảng cho các hệ thống Search and Rescue thông minh trong thực tế.