import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

class CrossViewAdapter(nn.Module):
    def __init__(self, clip_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, clip_dim),
        )
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.alpha * self.adapter(x)

class MultiViewQueryEncoder(nn.Module):
    def __init__(self, clip_model_name: str = "openai/clip-vit-large-patch14"):
        super().__init__()
        self.clip = CLIPVisionModelWithProjection.from_pretrained(clip_model_name)
        self.processor = CLIPImageProcessor.from_pretrained(clip_model_name)
        self.cross_view_adapter = CrossViewAdapter(clip_dim=768)
        self.proj = nn.Linear(768, 512, bias=False)

        for param in self.clip.parameters():
            param.requires_grad = False

    def forward(self, ref_images: list) -> torch.Tensor:
        inputs = self.processor(images=ref_images, return_tensors="pt")
        inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}

        with torch.no_grad():
            clip_features = self.clip(**inputs).image_embeds
            adapted_features = self.cross_view_adapter(clip_features)
            projected = self.proj(adapted_features)
            f_q = projected.mean(dim=0, keepdim=True)
            f_q = F.normalize(f_q, dim=-1)
        return f_q
