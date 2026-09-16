import os
import json
import cv2
import random
import shutil
from pathlib import Path
from tqdm import tqdm

def build_dataset_video_level(
    annotations_file: str,
    samples_dir: str,
    output_dir: str = "data/yolo_dataset",
    train_ratio: float = 0.7,
    seed: int = 42
):
    """
    Tạo YOLO dataset từ tập video training bằng chiến lược Video-Level Split (Ngăn chặn Data Leakage).
    Tất cả các frame của một video_id CHỈ nằm trong Train HOẶC Val set.
    """
    random.seed(seed)
    
    # 1. Khởi tạo thư mục chuẩn của YOLO
    out_path = Path(output_dir)
    for split in ['train', 'val']:
        (out_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 2. Đọc file annotation
    if not Path(annotations_file).exists():
        print(f"❌ Không tìm thấy file annotations tại: {annotations_file}")
        return

    with open(annotations_file, 'r', encoding='utf-8') as f:
        records = json.load(f)

    # 3. Gom danh sách video_id và chia theo Video-Level
    all_video_ids = [record['video_id'] for record in records]
    random.shuffle(all_video_ids)

    num_train = int(len(all_video_ids) * train_ratio)
    train_video_ids = set(all_video_ids[:num_train])
    val_video_ids = set(all_video_ids[num_train:])

    print(f"📊 [VIDEO-LEVEL SPLIT] Tổng số video: {len(all_video_ids)}")
    print(f"   • Train Videos ({len(train_video_ids)}): {sorted(list(train_video_ids))}")
    print(f"   • Val Videos   ({len(val_video_ids)}): {sorted(list(val_video_ids))}")

    # 4. Tiến hành đọc video và trích xuất Frame + BBox Label
    total_images = {'train': 0, 'val': 0}

    for record in tqdm(records, desc="Processing Videos"):
        vid = record['video_id']
        split = 'train' if vid in train_video_ids else 'val'
        vid_dir = Path(samples_dir) / vid
        video_file = vid_dir / "drone_video.mp4"

        if not video_file.exists():
            print(f"⚠️ Bỏ qua video {vid} vì không tìm thấy file drone_video.mp4")
            continue

        # Gom thông tin annotation theo frame
        frame_annos = {}
        for item in record.get('annotations', []):
            if 'bboxes' in item:
                for bbox in item['bboxes']:
                    f_num = int(bbox['frame'])
                    if f_num not in frame_annos:
                        frame_annos[f_num] = []
                    frame_annos[f_num].append(bbox)
            elif 'frame' in item:
                f_num = int(item['frame'])
                if f_num not in frame_annos:
                    frame_annos[f_num] = []
                frame_annos[f_num].append(item)

        if not frame_annos:
            continue

        cap = cv2.VideoCapture(str(video_file))
        current_frame = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if current_frame in frame_annos:
                h, w, _ = frame.shape
                img_name = f"{vid}_frame_{current_frame:06d}.jpg"
                txt_name = f"{vid}_frame_{current_frame:06d}.txt"

                img_save_path = out_path / "images" / split / img_name
                txt_save_path = out_path / "labels" / split / txt_name

                # Lưu ảnh frame
                cv2.imwrite(str(img_save_path), frame)

                # Chuyển đổi BBox sang chuẩn YOLO (Normalized x_center, y_center, width, height)
                yolo_labels = []
                for bbox in frame_annos[current_frame]:
                    x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                    
                    # Tính toán tọa độ tâm và kích thước tương đối
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    bx = (x1 + (x2 - x1) / 2) / w
                    by = (y1 + (y2 - y1) / 2) / h

                    # Single class ID = 0 (target_object)
                    yolo_labels.append(f"0 {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}")

                with open(txt_save_path, 'w') as lf:
                    lf.write("\n".join(yolo_labels))

                total_images[split] += 1

            current_frame += 1

        cap.release()

    # 5. Sinh file dataset.yaml cho YOLO
    yaml_content = f"""path: {out_path.resolve()}
train: images/train
val: images/val

names:
  0: target_object
"""
    with open(out_path / "dataset.yaml", 'w') as yf:
        yf.write(yaml_content)

    print(f"\n✅ Đã hoàn thành chia dataset chuẩn Video-Level!")
    print(f"   • Train images: {total_images['train']}")
    print(f"   • Val images:   {total_images['val']}")
    print(f"📄 File cấu hình: {out_path / 'dataset.yaml'}")

if __name__ == "__main__":
    build_dataset_video_level(
        annotations_file="data/training/train/annotations/annotations.json", 
        samples_dir="data/training/train/samples",
        output_dir="data/yolo_dataset"
    )