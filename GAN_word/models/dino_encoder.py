# models/dino_encoder.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

# ---- fixed local defaults ----
DEFAULT_REPO_DIR = "/home/woody/iwi5/iwi5333h/facebookresearch_dinov2_main"
DEFAULT_CKPT = "/home/woody/iwi5/iwi5333h/model/dinov2_vitl14_pretrain.pth"
DEFAULT_ARCH = "vitl14"
DEFAULT_IN_CHANNELS = 50
DEFAULT_FINAL_SIZE: Tuple[int, int] = (8, 27)
DEFAULT_TAPS: List[int] = [4, 8, 16, 23]        # fixed block indices
DEFAULT_PROBE_SIZE: Tuple[int, int] = (48, 540)


class ImageEncoderDINOv2(nn.Module):
    """
    DINOv2 ViT encoder loaded from a LOCAL torch.hub repo that:
      • accepts arbitrary in_channels (default 50)
      • pads inputs so H,W are multiples of patch size (14)
      • returns 5 spatial feature maps, each reduced to 512 channels
      • resizes the last map to (8,27) for your generator
    """

    def __init__(
        self,
        repo_dir: str = DEFAULT_REPO_DIR,
        arch: str = DEFAULT_ARCH,                       # "vits14" | "vitb14" | "vitl14" | "vitg14"
        ckpt_path: Optional[str] = DEFAULT_CKPT,
        in_channels: int = DEFAULT_IN_CHANNELS,
        final_size: Tuple[int, int] = DEFAULT_FINAL_SIZE,
        tap_blocks: Optional[List[int]] = DEFAULT_TAPS,
        probe_size: Tuple[int, int] = DEFAULT_PROBE_SIZE,
    ):
        super().__init__()
        self.output_dim = 512
        self.final_size = final_size

        # ---- Load backbone from local torch.hub repo ----
        hub_name = f"dinov2_{arch}"
        self.model = torch.hub.load(repo_dir, hub_name, source="local", pretrained=False)

        # Remove classification head if it exists
        if hasattr(self.model, "reset_classifier"):
            self.model.reset_classifier(0)

        # ---- Load pretrained checkpoint ----
        if ckpt_path and os.path.isfile(ckpt_path):
            sd = torch.load(ckpt_path, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            elif isinstance(sd, dict) and "model" in sd:
                sd = sd["model"]
            sd = { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }
            self.model.load_state_dict(sd, strict=False)

        # ---- Modify patch embedding to accept in_channels ----
        patch = self.model.patch_embed
        old_proj = patch.proj
        new_proj = nn.Conv2d(
            in_channels,
            old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
            bias=(old_proj.bias is not None),
        )
        with torch.no_grad():
            if old_proj.weight.shape[1] == 3:
                new_proj.weight[:, :3] = old_proj.weight
                if in_channels > 3:
                    rep = old_proj.weight[:, :1].repeat(1, in_channels - 3, 1, 1)
                    new_proj.weight[:, 3:] = rep
            else:
                nn.init.kaiming_normal_(new_proj.weight, mode="fan_out", nonlinearity="relu")
                if new_proj.bias is not None:
                    nn.init.zeros_(new_proj.bias)
        patch.proj = new_proj

        self.embed_dim = self.model.embed_dim
        self.patch_size = (
            patch.patch_size if isinstance(patch.patch_size, tuple)
            else (patch.patch_size, patch.patch_size)
        )

        # ---- Fixed tap blocks ----
        self.tap_blocks = tap_blocks or DEFAULT_TAPS
        assert len(self.tap_blocks) == 4, f"Expected 4 tap blocks, got {self.tap_blocks}"

        # ---- Reducers (1 stem + 4 taps = 5 total) ----
        self.reduce_layers = nn.ModuleList(
            [nn.Conv2d(self.embed_dim, 512, kernel_size=1) for _ in range(1 + len(self.tap_blocks))]
        )

        # ---- Test forward to verify output shape ----
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *probe_size)
            _ = self.encode_with_intermediate(dummy)

    # ---------- helpers ----------
    def _pos_embed_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional embeddings (uses model’s internal function if available)."""
        if hasattr(self.model, "_pos_embed"):
            return self.model._pos_embed(x)

        # Fallback for generic ViT-style pos embedding
        B, N, C = x.shape
        if hasattr(self.model, "cls_token") and self.model.cls_token is not None:
            cls_tok = self.model.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tok, x), dim=1)
        if hasattr(self.model, "pos_embed") and self.model.pos_embed is not None:
            if self.model.pos_embed.shape[1] == x.shape[1]:
                x = x + self.model.pos_embed
        if hasattr(self.model, "pos_drop"):
            x = self.model.pos_drop(x)
        return x

    @staticmethod
    def _tokens_to_map(x_tokens: torch.Tensor, Hp: int, Wp: int) -> torch.Tensor:
        """Convert flattened patch tokens -> (B, C, Hp, Wp) feature map."""
        B, HW, C = x_tokens.shape
        x_tokens = x_tokens.transpose(1, 2).contiguous()
        return x_tokens.view(B, C, Hp, Wp)

    # ---------- forward passes ----------
    def encode_with_intermediate(self, x: torch.Tensor) -> List[torch.Tensor]:
        B, _, H, W = x.shape
        ph, pw = self.patch_size

        # Pad H,W to multiples of patch size
        pad_h = (ph - (H % ph)) % ph
        pad_w = (pw - (W % pw)) % pw
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        Hp, Wp = x.shape[-2] // ph, x.shape[-1] // pw

        # Patchify → tokens → pos embed
        tokens = self.model.patch_embed(x)
        tokens = self._pos_embed_tokens(tokens)

        results = []

        # ---- Stem output ----
        stem_tokens = tokens[:, 1:, :]
        stem_map = self._tokens_to_map(stem_tokens, Hp, Wp)
        results.append(self.reduce_layers[0](stem_map))

        # ---- Transformer blocks ----
        red_idx = 1
        for i, blk in enumerate(self.model.blocks):
            tokens = blk(tokens)
            if i in self.tap_blocks:
                spatial = tokens[:, 1:, :]
                fmap = self._tokens_to_map(spatial, Hp, Wp)
                results.append(self.reduce_layers[red_idx](fmap))
                red_idx += 1

        # ---- Resize final map ----
        results[-1] = F.interpolate(
            results[-1], size=self.final_size, mode="bilinear", align_corners=False
        )
        assert len(results) == 5, f"Expected 5 feature maps, got {len(results)}"
        return results

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.encode_with_intermediate(x)
