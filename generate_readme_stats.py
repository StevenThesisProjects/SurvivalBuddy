import json
import os
import matplotlib.pyplot as plt

def analyze_and_plot(json_path="outputs/submission.json"):
    if not os.path.exists(json_path):
        print(f"⚠️ Không tìm thấy file kết quả tại {json_path}. Hãy chạy inference pipeline trước!")
        return
        
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    video_ids = []
    counts = []
    
    for sample in data:
        video_id = sample.get("video_id", "Unknown")
        
        # Bóc tách chuẩn xác mảng bboxes từ cấu trúc lồng nhau của submission.json
        bboxes = []
        dets = sample.get("detections", [])
        if isinstance(dets, list) and len(dets) > 0:
            if isinstance(dets[0], dict) and "bboxes" in dets[0]:
                bboxes = dets[0]["bboxes"]
            elif isinstance(dets[0], list):
                bboxes = dets[0]
        elif isinstance(dets, dict) and "bboxes" in dets:
            bboxes = dets["bboxes"]
            
        count = len(bboxes) if isinstance(bboxes, list) else 0
        video_ids.append(video_id)
        counts.append(count)
        
    # 1. In bảng Markdown ra terminal
    print("### 📊 Bảng Thống Kê Kết Quả Nhận Diện (Public Test)\n")
    print("| Video ID / Sample | Số lượng bounding box phát hiện |")
    print("| :--- | :---: |")
    for vid, cnt in zip(video_ids, counts):
        print(f"| `{vid}` | {cnt} detections |")
    print(f"\n* **Tổng số video đã quét:** {len(video_ids)}")
    print(f"* **Tổng số bounding box mục tiêu phát hiện:** {sum(counts)}\n")
    
    # 2. Vẽ biểu đồ trực quan bằng Matplotlib với trục Y tự động co giãn theo dữ liệu thực tế
    plt.figure(figsize=(11, 6))
    bars = plt.bar(video_ids, counts, color='#2563eb', edgecolor='#1d4ed8', width=0.55, alpha=0.85)
    
    plt.xlabel('Video ID / Sample', fontsize=11, fontweight='bold')
    plt.ylabel('Số lượng Detections', fontsize=11, fontweight='bold')
    plt.title('Biểu đồ phân phối mục tiêu cứu hộ qua các Sample (Public Test)', fontsize=13, fontweight='bold', pad=15)
    plt.xticks(rotation=25, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Đảm bảo trục Y có khoảng trống phía trên cột cao nhất để hiển thị số liệu đẹp mắt
    max_val = max(counts) if counts else 10
    plt.ylim(0, max_val * 1.15 if max_val > 0 else 10)
    
    # Gắn con số trực tiếp lên đỉnh mỗi cột biểu đồ
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', color='#1e293b', fontsize=10)
                    
    plt.tight_layout()
    
    # 3. Lưu hình ảnh vào thư mục outputs
    os.makedirs("outputs", exist_ok=True)
    chart_path = "outputs/detection_distribution.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    
    print(f"📈 Đã vẽ và lưu biểu đồ thành công tại: `{chart_path}`")
    print(f"👉 Dòng Markdown để nhúng vào README.md:\n   `![Thống kê kết quả](outputs/detection_distribution.png)`\n")

if __name__ == "__main__":
    analyze_and_plot()