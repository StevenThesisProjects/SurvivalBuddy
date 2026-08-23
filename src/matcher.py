import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalConsistencyGate(nn.Module):
    def __init__(self, history_len: int = 5):
        super().__init__()
        self.history_len = history_len
        self.gate = nn.Linear(2, 1)

    def forward(self, s_current: float, history: list) -> float:
        if not history:
            return s_current
        s_hist_mean = sum(history[-self.history_len:]) / len(history[-self.history_len:])
        inp = torch.tensor([[s_current, s_hist_mean]], dtype=torch.float32, device=self.gate.weight.device)
        gate_w = torch.sigmoid(self.gate(inp)).item()
        return gate_w * s_current + (1.0 - gate_w) * s_hist_mean

class CrossAttentionInstanceMatcher(nn.Module):
    def __init__(self, feature_dim: int = 512, backbone: str = "dinov2_vits14", device: str = "cpu"):
        super().__init__()
        self.device = device
        self.roi_encoder = torch.hub.load('facebookresearch/dinov2', backbone)
        self.roi_encoder.eval()
        for param in self.roi_encoder.parameters():
            param.requires_grad = False

        self.roi_proj = nn.Linear(384, feature_dim)
        self.tcg = TemporalConsistencyGate(history_len=5)
        self.scale_factor = np.sqrt(feature_dim)

    def extract_roi_feature(self, roi_crop: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            dino_feat = self.roi_encoder(roi_crop.to(self.device))
            f_roi = self.roi_proj(dino_feat)
            f_roi = F.normalize(f_roi, dim=-1)
        return f_roi

    def match(self, f_roi: torch.Tensor, f_q: torch.Tensor, score_history: list) -> float:
        dot_product = (f_roi @ f_q.T) / self.scale_factor
        s_raw = torch.sigmoid(dot_product).squeeze().item()
        s_final = self.tcg(s_raw, score_history)
        return s_final
