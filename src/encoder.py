import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

class MultiViewQueryEncoder(nn.Module):
    def __init__(self, clip_model_name: str = "openai/clip-vit-large-patch14"):
        super().__init__()
        self.clip = CLIPVisionModelWithProjection.from_pretrained(clip_model_name)
        self.processor = CLIPImageProcessor.from_pretrained(clip_model_name)

        # Đóng băng trọng số, KHÔNG sử dụng thêm bất kỳ linear projection chưa huấn luyện nào
        for param in self.clip.parameters():
            param.requires_grad = False

    def forward(self, ref_images: list) -> torch.Tensor:
        inputs = self.processor(images=ref_images, return_tensors="pt")
        inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}

        with torch.no_grad():
            # Trích xuất trực tiếp từ không gian nguyên bản của CLIP
            clip_features = self.clip(**inputs).image_embeds
            
            # Tính trung bình cộng của các ảnh tham chiếu
            f_q = clip_features.mean(dim=0, keepdim=True)
            
            # Chuẩn hóa L2
            f_q = F.normalize(f_q, dim=-1)
            
        return f_q