import os
import cv2
import numpy as np

class DroneVideoDataset:
    def __init__(self, sample_dir, target_size=640, augment=False):
        self.sample_dir = sample_dir
        self.target_size = target_size
        self.augment = augment
        
        self.video_path = os.path.join(sample_dir, "drone_video.mp4")
        self.ref_images_dir = os.path.join(sample_dir, "object_images")
        
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Không tìm thấy video tại: {self.video_path}")
            
    def load_reference_images(self):
        ref_images = []
        if not os.path.exists(self.ref_images_dir):
            return ref_images
            
        for i in range(1, 4):
            img_name = f"img_{i}.jpg"
            img_path = os.path.join(self.ref_images_dir, img_name)
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    ref_images.append(img)
        return ref_images

    def apply_motion_blur(self, image, kernel_size=15):
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        return cv2.filter2D(image, -1, kernel)

    def apply_exposure_tweak(self, image, gamma=1.2):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def get_frames_generator(self, batch_size=1):
        cap = cv2.VideoCapture(self.video_path)
        idx = 0
        
        batch_frames = []
        batch_indices = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Chuyển sang định dạng RGB 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if self.augment:
                if np.random.rand() > 0.5:
                    frame = self.apply_motion_blur(frame, kernel_size=np.random.choice([9, 15]))
                if np.random.rand() > 0.5:
                    frame = self.apply_exposure_tweak(frame, gamma=np.random.choice([0.7, 1.4]))
            
            # Hạ độ phân giải ảnh gốc khủng xuống target_size để giải phóng RAM -> chạy CPU
            h, w, _ = frame.shape
            if max(h, w) != self.target_size:
                frame = cv2.resize(frame, (self.target_size, self.target_size))
                
            batch_frames.append(frame)
            batch_indices.append(idx)
            idx += 1
            
            if len(batch_frames) == batch_size:
                yield batch_indices, np.array(batch_frames)
                batch_frames = []
                batch_indices = []
                
        cap.release()
        if batch_frames:
            yield batch_indices, np.array(batch_frames)