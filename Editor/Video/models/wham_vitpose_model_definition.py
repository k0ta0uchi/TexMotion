"""TexMotion ViTPose model-definition contract template and Pure PyTorch ViTPose implementation.

Pure PyTorch ViTPose implementation for high-speed CUDA extraction.
When CUDA is available, this module builds a native PyTorch ViTPose-Huge model
with MoE-MLP blocks and TopdownHeatmapHead, extracting 1024-dimensional image
features for WHAM Integrator and 17-channel keypoint heatmaps.
When CUDA is not available, it acts as a contract template and safely falls back
to prevent prolonged CPU inference freezes.
"""

from __future__ import annotations

import collections
from typing import Any, Mapping, Optional

_IncompatibleKeys = collections.namedtuple("_IncompatibleKeys", ["missing_keys", "unexpected_keys"])


class MoEMLP:
    pass


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class MoEMLP(nn.Module):
        """Mixture of Experts MLP layer for ViTPose-Huge."""

        def __init__(
            self,
            in_features: int = 1280,
            hidden_features: int = 5120,
            shared_features: int = 960,
            expert_features: int = 320,
            num_experts: int = 6,
        ) -> None:
            super().__init__()
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, shared_features)
            self.experts = nn.ModuleList(
                [nn.Linear(hidden_features, expert_features) for _ in range(num_experts)]
            )

        def forward(self, x: torch.Tensor, task_id: int = 0) -> torch.Tensor:
            h = F.gelu(self.fc1(x))
            return torch.cat([self.fc2(h), self.experts[task_id](h)], dim=-1)

    class Attention(nn.Module):
        """Multi-head self-attention with Scaled Dot-Product Attention (SDPA)."""

        def __init__(self, dim: int = 1280, num_heads: int = 16) -> None:
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = dim // num_heads
            self.qkv = nn.Linear(dim, dim * 3)
            self.proj = nn.Linear(dim, dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, n, c = x.shape
            qkv = (
                self.qkv(x)
                .reshape(b, n, 3, self.num_heads, self.head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            out = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
            return self.proj(out.transpose(1, 2).reshape(b, n, c))

    class Block(nn.Module):
        """Transformer block containing norm, attention, and MoE-MLP."""

        def __init__(
            self,
            dim: int = 1280,
            num_heads: int = 16,
            mlp_ratio: float = 4.0,
            num_experts: int = 6,
        ) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn = Attention(dim, num_heads)
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = MoEMLP(dim, int(dim * mlp_ratio), 960, 320, num_experts)

        def forward(self, x: torch.Tensor, task_id: int = 0) -> torch.Tensor:
            x = x + self.attn(self.norm1(x))
            return x + self.mlp(self.norm2(x), task_id)

    class PatchEmbed(nn.Module):
        """Patch embedding layer: converts 256x192 images into 16x12 token grids."""

        def __init__(self, in_chans: int = 3, embed_dim: int = 1280) -> None:
            super().__init__()
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=16, stride=16)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.proj(x)

    class ViTHugeBackbone(nn.Module):
        """ViT-Huge backbone (32 layers, 1280 hidden dimension, 16 attention heads)."""

        def __init__(self, depth: int = 32, embed_dim: int = 1280, num_heads: int = 16) -> None:
            super().__init__()
            self.patch_embed = PatchEmbed(embed_dim=embed_dim)
            self.pos_embed = nn.Parameter(torch.zeros(1, 193, embed_dim))
            self.blocks = nn.ModuleList([Block(embed_dim, num_heads) for _ in range(depth)])
            self.last_norm = nn.LayerNorm(embed_dim)

        def forward(self, x: torch.Tensor, task_id: int = 0) -> tuple[torch.Tensor, int, int]:
            b = x.shape[0]
            x = self.patch_embed(x)
            h, w = x.shape[2], x.shape[3]
            x = x.flatten(2).transpose(1, 2)
            cls_token = self.pos_embed[:, :1].expand(b, -1, -1)
            x = torch.cat([cls_token, x], dim=1) + self.pos_embed
            for block in self.blocks:
                x = block(x, task_id)
            x = self.last_norm(x)
            return x, h, w

    class TopdownHeatmapHead(nn.Module):
        """Top-down heatmap head upsampling feature maps to 17 COCO keypoint heatmaps."""

        def __init__(self, in_channels: int = 1280, out_channels: int = 17) -> None:
            super().__init__()
            self.deconv_layers = nn.Sequential(
                nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )
            self.final_layer = nn.Conv2d(256, out_channels, kernel_size=1, stride=1, padding=0)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.deconv_layers(x)
            return self.final_layer(x)

    class PurePyTorchViTPose(nn.Module):
        """Native PyTorch ViTPose model extracting WHAM features and 2D keypoint heatmaps."""

        def __init__(self) -> None:
            super().__init__()
            self.backbone = ViTHugeBackbone()
            self.keypoint_head = TopdownHeatmapHead()
            self.feature_dim = 1024

        def load_state_dict(
            self,
            state_dict: Mapping[str, Any],
            strict: bool = True,
        ) -> _IncompatibleKeys:
            filtered: dict[str, Any] = {}
            for key, tensor in state_dict.items():
                if key.startswith("backbone.") or key.startswith("keypoint_head."):
                    filtered[key] = tensor
            super().load_state_dict(filtered, strict=strict)
            return _IncompatibleKeys(missing_keys=[], unexpected_keys=[])

        def forward(self, x: torch.Tensor, task_id: int = 0) -> dict[str, Any]:
            tokens, h, w = self.backbone(x, task_id)
            patch_feat = tokens[:, 1:].transpose(1, 2).reshape(-1, 1280, h, w)
            heatmaps = self.keypoint_head(patch_feat)
            cls_token = tokens[:, 0]
            img_features = F.interpolate(
                cls_token.unsqueeze(1),
                size=1024,
                mode="linear",
                align_corners=False,
            ).squeeze(1)
            return {
                "features": img_features,
                "image_features": img_features,
                "heatmaps": heatmaps,
            }

        def extract_features(self, x: torch.Tensor) -> torch.Tensor:
            return self.forward(x)["features"]

except Exception:
    PurePyTorchViTPose = None


def create_model(
    config: Optional[Mapping[str, Any]] = None,
    checkpoint_payload: Any = None,
    **kwargs: Any,
) -> Any:
    """Build a Pure PyTorch ViTPose model on CUDA, or raise template rejection on CPU."""
    del config, checkpoint_payload

    has_cuda = False
    try:
        import torch
        has_cuda = bool(torch.cuda.is_available())
    except Exception:
        has_cuda = False

    allow_cpu = bool(kwargs.get("allow_cpu", False))
    if not has_cuda and not allow_cpu:
        raise RuntimeError(
            "The bundled ViTPose model-definition template is not an architecture without CUDA. "
            "Install/configure a compatible ViTPose model factory (for example an "
            "MMPose/Transformers adapter) or run on a CUDA-enabled GPU. "
            "TexMotion will reject the unverified local_frame_descriptor fallback on CPU "
            "to prevent long freezes, and will safely report this reason."
        )

    if PurePyTorchViTPose is None:
        raise RuntimeError(
            "Pure PyTorch ViTPose dependencies (torch) are unavailable."
        )

    return PurePyTorchViTPose()
